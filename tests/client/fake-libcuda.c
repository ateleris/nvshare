/*
 * A libcuda without a GPU, for tests of libnvshare. It has every symbol
 * libnvshare resolves, remembers which launch or copy function was called
 * last, and behaves like the driver where libnvshare depends on it: a
 * cuCtxSynchronize() during a stream capture fails and invalidates the
 * capture, and cuGetProcAddress() picks the variant of a function from the
 * version and the flags. Only one stream can capture at a time, and the
 * stream right after it has joined the capture: it is capturing too, but
 * cannot end the capture. FAKE_CUDA_* in the environment change its
 * behaviour: FAKE_CUDA_UNKNOWN_PROCS makes every cuGetProcAddress() lookup
 * return a function libnvshare has never seen, FAKE_CUDA_FAIL_PROCS makes
 * every lookup fail.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_ILLEGAL_STATE 401
#define CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED 900
#define CUDA_ERROR_STREAM_CAPTURE_INVALIDATED 901
#define CUDA_ERROR_STREAM_CAPTURE_UNMATCHED 903
#define CUDA_ERROR_UNKNOWN 999
#define PER_THREAD_DEFAULT_STREAM 2

static const char *last_call = "none";
static int sync_count;
static void *capture_stream;
static int capture_invalidated;

static int env_int(const char *name, int fallback)
{
	const char *value = getenv(name);

	return value != NULL ? atoi(value) : fallback;
}

static void sleep_ms(int ms)
{
	struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };

	nanosleep(&ts, NULL);
}

const char *fake_last_call(void)
{
	return last_call;
}

int fake_sync_count(void)
{
	return sync_count;
}

int cuInit(unsigned int flags)
{
	static int called;
	Dl_info self, found;
	int (*fn)(unsigned int);

	if (called++ || getenv("FAKE_CUDA_REENTRANT_INIT") == NULL)
		return 0;
	fn = (int (*)(unsigned int))dlsym(
		dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD), "cuInit");
	if (fn != NULL && dladdr((void *)fn, &found) != 0 &&
	    dladdr((void *)fake_last_call, &self) != 0 &&
	    found.dli_fbase != self.dli_fbase)
		return fn(flags);
	return 0;
}

int cuGetErrorString(int error, const char **str)
{
	(void)error;
	if (getenv("FAKE_CUDA_NO_ERROR_STRINGS") != NULL)
		return CUDA_ERROR_UNKNOWN;
	*str = "fake error";
	return 0;
}

int cuGetErrorName(int error, const char **str)
{
	(void)error;
	if (getenv("FAKE_CUDA_NO_ERROR_STRINGS") != NULL)
		return CUDA_ERROR_UNKNOWN;
	*str = "FAKE_ERROR";
	return 0;
}

int cuDeviceGet(int *device, int ordinal)
{
	if (getenv("FAKE_CUDA_NO_DEVICE") != NULL)
		return CUDA_ERROR_UNKNOWN;
	*device = ordinal;
	return 0;
}

int cuDeviceTotalMem_v2(size_t *bytes, int device)
{
	(void)device;
	*bytes = (size_t)env_int("FAKE_CUDA_GPU_MIB", 8192) << 20;
	return 0;
}

int cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes)
{
	cuDeviceTotalMem_v2(total_bytes, 0);
	*free_bytes = *total_bytes;
	return 0;
}

int cuMemAllocManaged(unsigned long long *dptr, size_t bytesize, unsigned int flags)
{
	static unsigned long long next = 0x10000;

	(void)bytesize;
	(void)flags;
	*dptr = next;
	next += 0x10000;
	return 0;
}

int cuMemAlloc_v2(unsigned long long *dptr, size_t bytesize)
{
	return cuMemAllocManaged(dptr, bytesize, 0);
}

int cuMemFree_v2(unsigned long long dptr)
{
	(void)dptr;
	return 0;
}

int cuMemAllocAsync(void) { return 0; }
int cuMemFreeAsync(void) { return 0; }
int cuCtxSetCurrent(void *ctx) { (void)ctx; return 0; }

int cuCtxGetCurrent(void **ctx)
{
	*ctx = (void *)0x1;
	return 0;
}

int cuCtxSynchronize(void)
{
	sync_count++;
	sleep_ms(env_int("FAKE_CUDA_SYNC_MS", 0));
	if (capture_stream != NULL) {
		capture_invalidated = 1;
		return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
	}
	return 0;
}

#define RECORDED(name)                                              \
int name(void)                                                      \
{                                                                   \
	last_call = #name;                                          \
	return getenv("FAKE_CUDA_FAIL_CALLS") != NULL ?             \
	       CUDA_ERROR_UNKNOWN : 0;                              \
}

RECORDED(cuLaunchKernel)
RECORDED(cuLaunchKernel_ptsz)
RECORDED(cuLaunchKernelEx)
RECORDED(cuLaunchKernelEx_ptsz)
RECORDED(cuLaunchCooperativeKernel)
RECORDED(cuLaunchCooperativeKernel_ptsz)
RECORDED(cuGraphLaunch)
RECORDED(cuGraphLaunch_ptsz)
RECORDED(cuMemcpy)
RECORDED(cuMemcpy_ptds)
RECORDED(cuMemcpyAsync)
RECORDED(cuMemcpyAsync_ptsz)
RECORDED(cuMemcpyDtoH_v2)
RECORDED(cuMemcpyDtoH_v2_ptds)
RECORDED(cuMemcpyDtoHAsync_v2)
RECORDED(cuMemcpyDtoHAsync_v2_ptsz)
RECORDED(cuMemcpyHtoD_v2)
RECORDED(cuMemcpyHtoD_v2_ptds)
RECORDED(cuMemcpyHtoDAsync_v2)
RECORDED(cuMemcpyHtoDAsync_v2_ptsz)
RECORDED(cuMemcpyDtoD_v2)
RECORDED(cuMemcpyDtoD_v2_ptds)
RECORDED(cuMemcpyDtoDAsync_v2)
RECORDED(cuMemcpyDtoDAsync_v2_ptsz)

int cuStreamBeginCapture_v2(void *stream, int mode)
{
	(void)mode;
	if (capture_stream != NULL)
		return CUDA_ERROR_ILLEGAL_STATE;
	capture_stream = stream;
	capture_invalidated = 0;
	return 0;
}

int cuStreamBeginCapture(void *stream)
{
	return cuStreamBeginCapture_v2(stream, 0);
}

int cuStreamEndCapture(void *stream, void **graph)
{
	if (capture_stream != NULL && stream == (char *)capture_stream + 1)
		return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
	if (capture_stream == NULL || stream != capture_stream)
		return CUDA_ERROR_ILLEGAL_STATE;
	capture_stream = NULL;
	*graph = (void *)0x1;
	return capture_invalidated ? CUDA_ERROR_STREAM_CAPTURE_INVALIDATED : 0;
}

int cuStreamIsCapturing(void *stream, int *status)
{
	if (capture_stream != NULL && (stream == capture_stream ||
	    stream == (char *)capture_stream + 1))
		*status = capture_invalidated ? 2 : 1;
	else *status = 0;
	return 0;
}

static int unknown_function(void)
{
	last_call = "unknown";
	return 0;
}

int cuGetProcAddress_v2(const char *symbol, void **pfn, int version,
			unsigned long long flags, int *status)
{
	static const char *suffixes[] = {"_v2_ptsz", "_v2_ptds", "_ptsz", "_ptds", "_v2", ""};
	const char **suffix = (flags & PER_THREAD_DEFAULT_STREAM) ? suffixes : suffixes + 4;
	void *self = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
	char name[128];

	*pfn = NULL;
	if (getenv("FAKE_CUDA_FAIL_PROCS") != NULL)
		return CUDA_ERROR_NOT_FOUND;
	if (getenv("FAKE_CUDA_UNKNOWN_PROCS") != NULL)
		*pfn = (void *)unknown_function;
	for (; *pfn == NULL && suffix < suffixes + 6; suffix++) {
		if (strcmp(symbol, "cuGetProcAddress") == 0 &&
		    strcmp(*suffix, "_v2") == 0 && version < 12000)
			continue;
		snprintf(name, sizeof(name), "%s%s", symbol, *suffix);
		*pfn = dlsym(self, name);
	}
	if (status != NULL)
		*status = *pfn == NULL;
	return 0;
}

int cuGetProcAddress(const char *symbol, void **pfn, int version,
		     unsigned long long flags)
{
	void *fn;
	int ret = cuGetProcAddress_v2(symbol, &fn, version, flags, NULL);

	if (ret == 0 && fn == NULL)
		ret = CUDA_ERROR_NOT_FOUND;
	if (ret == 0)
		*pfn = fn;
	return ret;
}
