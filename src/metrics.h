#ifndef _NVSHARE_METRICS_H_
#define _NVSHARE_METRICS_H_

#include <stdint.h>
#include "nvshare_types.h"

#define METRICS_SOCKET_PATH "/var/run/nvshare/metrics.sock"

/*
 * Wire protocol version. Bump when any on-wire struct/enum changes. The Go
 * side of the metrics socket rejects messages with a different version, so
 * a scheduler and a device plugin of different versions lose metrics but
 * never misinterpret them. Deploy both together.
 */
#define METRICS_PROTOCOL_VERSION 2u

/* How often the scheduler re-sends the state of every pod. */
#define METRICS_TICK_SEC 10

/* How long a pod is still reported after its last client disconnected. */
#define METRICS_POD_RETENTION_SEC 300

enum metrics_message_type {
	METRICS_POD_STATS = 300,
	METRICS_SCHED_STATS = 301,
};

enum metrics_release_reason {
	METRICS_RELEASE_QUANTUM = 0, /* after DROP_LOCK: time quantum expired */
	METRICS_RELEASE_IDLE = 1,    /* client released on its own */
	METRICS_RELEASE_EXIT = 2,    /* client disconnected while holding */
	METRICS_RELEASE_OFF = 3,     /* scheduler turned off or stopped */
	METRICS_RELEASE_REASONS = 4,
};

/*
 * On-wire messages. Packed, little-endian (Linux on x86_64/arm64). The Go
 * side parses each field explicitly, so it does not depend on Go struct
 * alignment matching C.
 *
 * Both messages carry state, not events: counters are cumulative since the
 * scheduler started and every message replaces the previous one for the
 * same pod. A lost datagram is repaired by the next one.
 */
struct metrics_pod_stats {
	uint32_t version;
	uint32_t type;
	char namespace[POD_NAMESPACE_LEN_MAX];
	char pod_name[POD_NAME_LEN_MAX];
	uint32_t clients;  /* connected clients of this pod */
	uint32_t holding;  /* 1 if one of them holds the lock */
	uint32_t waiting;  /* how many of them are queued for the lock */
	uint64_t held_ms;  /* includes the hold in progress */
	uint64_t wait_ms;  /* includes waits in progress */
	uint64_t acquisitions;
	uint64_t releases[METRICS_RELEASE_REASONS];
} __attribute__((__packed__));

struct metrics_sched_stats {
	uint32_t version;
	uint32_t type;
	uint32_t scheduler_on;
	uint32_t tq_sec;
	uint32_t clients;
	uint32_t queue_length;      /* clients waiting for the lock */
	uint64_t dropped_messages;  /* metrics datagrams that could not be sent */
} __attribute__((__packed__));

_Static_assert(sizeof(struct metrics_pod_stats) == 584, "pod stats wire size");
_Static_assert(sizeof(struct metrics_sched_stats) == 32, "sched stats wire size");

/*
 * Locking contract: all metrics_* functions must be called with the
 * scheduler's global_mutex held. The metrics module keeps no internal
 * mutex -- its state is only ever touched from paths the scheduler has
 * already serialized.
 */
void metrics_init(int scheduler_on, int tq);
void metrics_client_registered(const struct nvshare_client *client);
void metrics_client_removed(const struct nvshare_client *client);
void metrics_lock_requested(const struct nvshare_client *client);
void metrics_lock_acquired(const struct nvshare_client *client);
void metrics_drop_sent(const struct nvshare_client *client);
void metrics_lock_released(const struct nvshare_client *client);
void metrics_scheduler_state(int scheduler_on, int tq);
void metrics_tick(void);
void metrics_cleanup(void);

#endif
