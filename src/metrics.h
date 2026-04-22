#ifndef _NVSHARE_METRICS_H_
#define _NVSHARE_METRICS_H_

#include <time.h>
#include <stdint.h>
#include "nvshare_types.h"

#define METRICS_SOCKET_PATH "/var/run/nvshare/metrics.sock"
#define MAX_DEVICE_ID_LEN 32

/*
 * Wire protocol version. Bump when any on-wire struct/enum changes. The Go
 * side of the metrics socket rejects messages with a different version.
 *
 * Rolling-upgrade order: deploy the device plugin (Go) BEFORE the scheduler
 * (C). A stale Go plugin receiving a new-format datagram drops it as short
 * (different size), so metrics are temporarily lost but never corrupted.
 * The reverse order (new scheduler + stale plugin) also fails safely: the
 * plugin reads the version uint32 as its old `type` field and rejects it.
 */
#define METRICS_PROTOCOL_VERSION 1u

struct pod_gpu_session {
	char namespace[POD_NAMESPACE_LEN_MAX];
	char pod_name[POD_NAME_LEN_MAX];
	char device_id[MAX_DEVICE_ID_LEN];
	uint64_t client_id;
	struct timespec start_time; /* CLOCK_MONOTONIC */
	struct pod_gpu_session *next;
};

/*
 * Type IDs are intentionally separated from the old scheduler's range
 * (100/101) so that during a rolling upgrade a stale peer is recognized as
 * "unknown" rather than silently misinterpreted (the old type 101 put an
 * epoch timestamp in the duration field — a ~54-year spike per release).
 */
enum metrics_message_type {
	METRICS_LOCK_ACQUIRED = 200,
	METRICS_LOCK_RELEASED = 201,
} __attribute__((__packed__));

/*
 * On-wire metrics message. Packed, little-endian (Linux on x86_64/arm64).
 * The Go side parses each field explicitly with binary.LittleEndian so we
 * do not rely on Go struct-alignment matching C packed alignment.
 *
 * Sizing: 4 + 4 + 254 + 254 + 32 + 8 + 8 = 564 bytes.
 */
struct metrics_message {
	uint32_t version;
	uint32_t type;
	char namespace[POD_NAMESPACE_LEN_MAX];
	char pod_name[POD_NAME_LEN_MAX];
	char device_id[MAX_DEVICE_ID_LEN];
	uint64_t client_id;
	int64_t duration_ms; /* 0 for ACQUIRED; hold duration in ms for RELEASED */
} __attribute__((__packed__));

/*
 * Locking contract: all metrics_* functions must be called with the
 * scheduler's global_mutex held. The metrics module keeps no internal
 * mutex -- its state is only ever touched from paths the scheduler has
 * already serialized.
 */
int metrics_init(void);
int metrics_lock_acquired(const struct nvshare_client *client, const char *device_id);
int metrics_lock_released(const struct nvshare_client *client, const char *device_id);
void metrics_flush_all(void);
void metrics_cleanup(void);

#endif
