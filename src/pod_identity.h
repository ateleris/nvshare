#ifndef _NVSHARE_POD_IDENTITY_H_
#define _NVSHARE_POD_IDENTITY_H_

#include <stddef.h>

int pod_identity_verify(int fd, char *namespace, size_t namespace_len,
			char *name, size_t name_len);

#endif
