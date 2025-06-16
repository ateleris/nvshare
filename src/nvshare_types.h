#ifndef _NVSHARE_TYPES_H_
#define _NVSHARE_TYPES_H_

#include <stdint.h>
#include "comm.h"

struct nvshare_client {
	int fd;
	uint64_t id;
	char pod_name[POD_NAME_LEN_MAX];
	char pod_namespace[POD_NAMESPACE_LEN_MAX];
	struct nvshare_client *next;
};

struct nvshare_request {
	struct nvshare_client *client;
	struct nvshare_request *next;
};

#endif