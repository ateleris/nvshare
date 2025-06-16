#ifndef _NVSHARE_METRICS_H_
#define _NVSHARE_METRICS_H_

#include <time.h>
#include <stdint.h>
#include "nvshare_types.h"

#define METRICS_SOCKET_PATH "/var/run/nvshare/metrics.sock"
#define MAX_DEVICE_ID_LEN 32

struct pod_gpu_session {
	char namespace[POD_NAMESPACE_LEN_MAX];
	char pod_name[POD_NAME_LEN_MAX];
	char container[128];
	char device_id[MAX_DEVICE_ID_LEN];
	int process_id;
	uint64_t client_id;
	time_t start_time;
	time_t last_active;
	struct pod_gpu_session *next;
};

enum metrics_message_type {
	METRICS_SESSION_START = 100,
	METRICS_SESSION_END   = 101,
	METRICS_SESSION_UPDATE = 102,
} __attribute__((__packed__));

struct metrics_message {
	int32_t type;
	char namespace[POD_NAMESPACE_LEN_MAX];
	char pod_name[POD_NAME_LEN_MAX];
	char container[128];
	char device_id[MAX_DEVICE_ID_LEN];
	int32_t process_id;
	uint64_t client_id;
	int64_t timestamp;
} __attribute__((__packed__));

int metrics_init(void);
int metrics_register_session(const struct nvshare_client *client, const char *device_id, int process_id);
int metrics_unregister_session(const struct nvshare_client *client, const char *device_id);
int metrics_update_session(const struct nvshare_client *client, const char *device_id);
void metrics_cleanup(void);

#endif