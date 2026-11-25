/*
 * A libnvidia-ml without a GPU, for tests of libnvshare. The GPU utilization
 * it reports is FAKE_NVML_UTIL percent, 0 by default.
 */
#include <stdlib.h>

typedef struct {
	unsigned int gpu;
	unsigned int memory;
} nvmlUtilization_t;

int nvmlInit_v2(void)
{
	return 0;
}

int nvmlDeviceGetHandleByIndex_v2(unsigned int index, void **device)
{
	(void)index;
	*device = (void *)0x1;
	return 0;
}

int nvmlDeviceGetUtilizationRates(void *device, nvmlUtilization_t *utilization)
{
	const char *value = getenv("FAKE_NVML_UTIL");

	(void)device;
	utilization->gpu = value != NULL ? (unsigned int)atoi(value) : 0;
	utilization->memory = 0;
	return 0;
}
