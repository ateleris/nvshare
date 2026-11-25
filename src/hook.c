/*
 * Copyright (c) 2023 Georgios Alexopoulos
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * Hook CUDA function calls.
 */

/*
 * Defining _GNU_SOURCE allows us to call dlvsym().
 *
 * More on _GNU_SOURCE: https://stackoverflow.com/a/5583764
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif /* _GNU_SOURCE */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>

#include "comm.h"
#include "common.h"
#include "cuda_defs.h"
#include "client.h"
#include "utlist.h"

#define ENV_NVSHARE_ENABLE_SINGLE_OVERSUB "NVSHARE_ENABLE_SINGLE_OVERSUB"
#define ENV_NVSHARE_USE_ORIGINAL_MEMINFO "NVSHARE_USE_ORIGINAL_MEMINFO"

#define MEMINFO_RESERVE_MIB 1536           /* MiB */
#define KERN_SYNC_DURATION_BIG 10          /* seconds */
#define KERN_SYNC_WINDOW_STEPDOWN_THRESH 1 /* seconds */
#define KERN_SYNC_WINDOW_MAX 2048          /* Pending Kernels */

static void *real_dlsym_225(void *handle, const char *symbol);
static void load_cuda_symbols(void *cuda_handle);
static void *hook_for_symbol(const char *symbol);
static void *hook_for_proc(const char *symbol);
static void *hook_for_real(void *fn);

cuCtxSynchronize_func real_cuCtxSynchronize = NULL;
cuLaunchKernel_func real_cuLaunchKernel = NULL;
cuMemcpy_func real_cuMemcpy = NULL;
cuMemcpyAsync_func real_cuMemcpyAsync = NULL;
cuMemcpyDtoH_func real_cuMemcpyDtoH = NULL;
cuMemcpyDtoHAsync_func real_cuMemcpyDtoHAsync = NULL;
cuMemcpyHtoD_func real_cuMemcpyHtoD = NULL;
cuMemcpyHtoDAsync_func real_cuMemcpyHtoDAsync = NULL;
cuMemcpyDtoD_func real_cuMemcpyDtoD = NULL;
cuMemcpyDtoDAsync_func real_cuMemcpyDtoDAsync = NULL;
cuMemAllocAsync_func real_cuMemAllocAsync = NULL;
cuMemFreeAsync_func real_cuMemFreeAsync = NULL;
cuGetProcAddress_func real_cuGetProcAddress = NULL;
cuGetProcAddress_v2_func real_cuGetProcAddress_v2 = NULL;
cuMemAllocManaged_func real_cuMemAllocManaged = NULL;
cuMemFree_func real_cuMemFree = NULL;
cuMemGetInfo_func real_cuMemGetInfo = NULL;
cuDeviceGet_func real_cuDeviceGet = NULL;
cuDeviceTotalMem_func real_cuDeviceTotalMem = NULL;
cuGetErrorString_func real_cuGetErrorString = NULL;
cuGetErrorName_func real_cuGetErrorName = NULL;
cuCtxSetCurrent_func real_cuCtxSetCurrent = NULL;
cuCtxGetCurrent_func real_cuCtxGetCurrent = NULL;
cuInit_func real_cuInit = NULL;

static cuLaunchKernel_func real_cuLaunchKernel_ptsz = NULL;
static cuMemcpy_func real_cuMemcpy_ptds = NULL;
static cuMemcpyAsync_func real_cuMemcpyAsync_ptsz = NULL;
static cuMemcpyDtoH_func real_cuMemcpyDtoH_v2_ptds = NULL;
static cuMemcpyDtoHAsync_func real_cuMemcpyDtoHAsync_v2_ptsz = NULL;
static cuMemcpyHtoD_func real_cuMemcpyHtoD_v2_ptds = NULL;
static cuMemcpyHtoDAsync_func real_cuMemcpyHtoDAsync_v2_ptsz = NULL;
static cuMemcpyDtoD_func real_cuMemcpyDtoD_v2_ptds = NULL;
static cuMemcpyDtoDAsync_func real_cuMemcpyDtoDAsync_v2_ptsz = NULL;
static cuLaunchKernelEx_func real_cuLaunchKernelEx = NULL;
static cuLaunchKernelEx_func real_cuLaunchKernelEx_ptsz = NULL;
static cuLaunchCooperativeKernel_func real_cuLaunchCooperativeKernel = NULL;
static cuLaunchCooperativeKernel_func real_cuLaunchCooperativeKernel_ptsz = NULL;
static cuGraphLaunch_func real_cuGraphLaunch = NULL;
static cuGraphLaunch_func real_cuGraphLaunch_ptsz = NULL;
static cuStreamBeginCapture_func real_cuStreamBeginCapture = NULL;
static cuStreamBeginCapture_func real_cuStreamBeginCapture_ptsz = NULL;
static cuStreamBeginCapture_v2_func real_cuStreamBeginCapture_v2 = NULL;
static cuStreamBeginCapture_v2_func real_cuStreamBeginCapture_v2_ptsz = NULL;
static cuStreamBeginCaptureToGraph_func real_cuStreamBeginCaptureToGraph = NULL;
static cuStreamBeginCaptureToGraph_func real_cuStreamBeginCaptureToGraph_ptsz = NULL;
static cuStreamEndCapture_func real_cuStreamEndCapture = NULL;
static cuStreamEndCapture_func real_cuStreamEndCapture_ptsz = NULL;
static cuStreamIsCapturing_func real_cuStreamIsCapturing = NULL;

pthread_mutex_t capture_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t capture_cv = PTHREAD_COND_INITIALIZER;
int active_captures = 0;

static void *libcuda_handle = NULL;
static void *libcuda_base = NULL;

nvmlDeviceGetUtilizationRates_func real_nvmlDeviceGetUtilizationRates = NULL;
nvmlInit_func real_nvmlInit = NULL;
nvmlDeviceGetHandleByIndex_func real_nvmlDeviceGetHandleByIndex = NULL;

size_t nvshare_size_mem_allocatable = 0;
size_t sum_allocated = 0;

int kern_since_sync = 0;
int pending_kernel_window = 1;
pthread_mutex_t kcount_mutex;

int enable_single_oversub = 0;
int use_original_meminfo = 0;
int nvml_ok = 1;

/* Representation of a CUDA memory allocation */
struct cuda_mem_allocation {
	CUdeviceptr ptr;
	size_t size;
	struct cuda_mem_allocation *next;
};

/* Linked list that holds all memory allocations of current application. */
struct cuda_mem_allocation *cuda_allocation_list = NULL;

/* Initializaters will be executed only once per client application */
static pthread_once_t init_libnvshare_done = PTHREAD_ONCE_INIT;
static pthread_once_t init_done = PTHREAD_ONCE_INIT;

/* Load real CUDA {Driver API, NVML} functions and bootstrap auxiliary stuff. */
static void bootstrap_cuda(void)
{
	char *error;
	void *cuda_handle;
	void *nvml_handle;
	Dl_info cuda_info;

	true_or_exit(pthread_mutex_init(&kcount_mutex, NULL) == 0);

	nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
	if (!nvml_handle) {
		error = dlerror();
		log_debug("%s", error);
		nvml_ok = 0;
	} else {
		dlerror();
		real_nvmlDeviceGetUtilizationRates =
			(nvmlDeviceGetUtilizationRates_func)real_dlsym_225(nvml_handle,
			CUDA_SYMBOL_STRING(nvmlDeviceGetUtilizationRates));
		error = dlerror();
		if (error != NULL) {
			log_debug("%s", error);
			nvml_ok = 0;
		}
		real_nvmlInit = (nvmlInit_func)
		real_dlsym_225(nvml_handle,CUDA_SYMBOL_STRING(nvmlInit));
		error = dlerror();
		if (error != NULL) {
			log_debug("%s", error);
			nvml_ok = 0;
		}
		real_nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_func)
		real_dlsym_225(nvml_handle,
			CUDA_SYMBOL_STRING(nvmlDeviceGetHandleByIndex));
		error = dlerror();
		if (error != NULL) {
			log_debug("%s", error);
			nvml_ok = 0;
		}
	}
	if (nvml_ok) log_debug("Found NVML");
	else log_debug("Could not find NVML");

	cuda_handle = dlopen("libcuda.so", RTLD_LAZY);
	if (!cuda_handle) {
		error = dlerror();
		log_fatal("%s", error);
	}
	load_cuda_symbols(cuda_handle);

	if (dladdr((void *)real_cuInit, &cuda_info) == 0)
		log_fatal("dladdr() failed for the real cuInit");
	libcuda_handle = cuda_handle;
	libcuda_base = cuda_info.dli_fbase;
}


/* Append a new CUDA memory allocation at the end of the list. */
static void insert_cuda_allocation(CUdeviceptr dptr, size_t bytesize)
{
	struct cuda_mem_allocation *allocation;


	sum_allocated += bytesize;
	log_debug("Total allocated memory on GPU is %.2f MiB",
		  toMiB(sum_allocated));

	true_or_exit(allocation = malloc(sizeof(*allocation)));

	allocation->ptr = dptr;
	allocation->size = bytesize;
	allocation->next = NULL;
	LL_APPEND(cuda_allocation_list, allocation);

	report_mem_growth();
}

/* Remove a CUDA memory allocation given the pointer it starts at */
static void remove_cuda_allocation(CUdeviceptr rm_ptr)
{
	struct cuda_mem_allocation *tmp, *a;


	LL_FOREACH_SAFE(cuda_allocation_list, a, tmp) {
		if (a->ptr == rm_ptr) {
			sum_allocated -= a->size;
			log_debug("Total allocated memory on GPU is %.2f MiB",
				  toMiB(sum_allocated));
			LL_DELETE(cuda_allocation_list, a);
			free(a);
		}
	}
}


/* Toggle debug mode and single process oversubscription based on envvars */
static void initialize_libnvshare(void)
{
	char *value;
	value = getenv(ENV_NVSHARE_DEBUG);
	if (value != NULL)
		__debug = 1;
	value = getenv(ENV_NVSHARE_ENABLE_SINGLE_OVERSUB);
	if (value != NULL) {
		enable_single_oversub = 1;
		log_warn("Enabling GPU memory oversubscription for this"
		         " application");
	}
	value = getenv(ENV_NVSHARE_USE_ORIGINAL_MEMINFO);
	if (value != NULL) {
		use_original_meminfo = 1;
		log_warn("Using original mem info without removing %d MiB", MEMINFO_RESERVE_MIB);
	}

	bootstrap_cuda();
}


/*
 * Check the return value of a CUDA Driver API function call for errors.
 *
 * Interpret using the Driver API functions:
 * - cuGetErrorString
 * - cuGetErrorName
 */
void cuda_driver_check_error(CUresult err, const char *func_name)
{
	if (err != CUDA_SUCCESS) {
		const char *err_string = "unknown";
		const char *err_name = "unknown";
		if (real_cuGetErrorString(err, &err_string) != CUDA_SUCCESS)
			err_string = "unknown";
		if (real_cuGetErrorName(err, &err_name) != CUDA_SUCCESS)
			err_name = "unknown";
		log_warn("%s returned %s: %s",
	                 func_name, err_name, err_string);
	}
}


/*
 * Since we're interposing dlsym() in libnvshare, we use dlvsym() to obtain the
 * address of the real dlsym function.
 *
 * Depending on glibc version, we look for the appropriate symbol.
 *
 * Some context on the implementation:
 *
 * glibc 2.34 remove the internal __libc_dlsym() symbol that NVIDIA uses in
 * their cuHook example:
 * https://github.com/phrb/intro-cuda/blob/d38323b81cd799dc09179e2ef27aa8f81b6dac40/src/cuda-samples/7_CUDALibraries/cuHook/libcuhook.cpp#L43
 *
 * One solution, discussed in apitrace's repo is to use dlvsym(), which also
 * takes a version string as a 3rd argument, in order to obtain the real
 * dlsym().
 *
 * This is what user 'manisandro' suggested 8 years ago, when warning about
 * using the private __libc_dlsym():
 * https://github.com/apitrace/apitrace/issues/258
 *
 * The maintainer of the repo didn't heed the warning back then, it came back
 * 8 years later and bit them.
 *
 * This is also what user "derhass" suggests:
 * https://stackoverflow.com/a/18825060
 * (See section "UPDATE FOR 2021/glibc-2.34").
 *
 * Given all the above, we obtain the real `dlsym()` as such:
 * real_dlsym=dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
 *
 * Since we have to explicitly use a version argument in dlvsym(), we also have
 * to define and export two versions of dlsym (hence the linker script.), one
 * for each distinct glibc symbol version.
 *
 */
static void *real_dlsym_225(void *handle, const char *symbol)
{
	typedef void *(dlsym_t)(void *, const char *);
	static dlsym_t *r_dlsym;
	char *err;


	if (!r_dlsym) {
		dlerror();
		r_dlsym = (dlsym_t*)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
		err = dlerror();
		if (err != NULL)
			log_fatal("%s", err);
	}

	return (*r_dlsym)(handle, symbol);
}

static void *real_dlsym_234(void *handle, const char *symbol)
{
	typedef void *(dlsym_t)(void *, const char *);
	static dlsym_t *r_dlsym;
	char *err;


	if (!r_dlsym) {
		dlerror();
		r_dlsym = (dlsym_t*)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
		err = dlerror();
		if (err != NULL)
			log_fatal("%s", err);
	}

	return (*r_dlsym)(handle, symbol);
}


/*
 * CUDA Runtime API uses dlopen()/dlsym() to obtain addresses of the Driver API
 * functions.
 *
 * [spoiler: from CUDA 11.3 onwards, it only uses dlsym() to get the address
 *  of cuGetProcAddress() and then uses the latter to obtain the addresses
 *  of all other Driver API functions/symbols.]
 *
 * When the user program calls dlsym() requesting a Driver API symbol, return
 * our interposed version.
 *
 * In all other cases, call the real dlsym() from glibc and pass on the
 * requested symbol string.
 */
static int called_from_libcuda(const void *caller)
{
	Dl_info info;

	if (libcuda_base == NULL || dladdr(caller, &info) == 0)
		return 0;
	return info.dli_fbase == libcuda_base;
}

void *dlsym_225(void *handle, const char *symbol)
{
	void *hook = hook_for_symbol(symbol);

	if (hook == NULL)
		return (real_dlsym_225(handle, symbol));

	log_debug("dlsym_225 for symbol %s requested", symbol);
	if (called_from_libcuda(__builtin_return_address(0)))
		return (real_dlsym_225(libcuda_handle, symbol));
	return hook;
}

void *dlsym_234(void *handle, const char *symbol)
{
	void *hook = hook_for_symbol(symbol);

	if (hook == NULL)
		return (real_dlsym_234(handle, symbol));

	log_debug("dlsym_234 for symbol %s requested", symbol);
	if (called_from_libcuda(__builtin_return_address(0)))
		return (real_dlsym_234(libcuda_handle, symbol));
	return hook;
}


/*
 * Older CUDA Runtime API (version <=11.2) does the following during internal
 * initialization (when the user program calls it for the first time):
 * 1. Calls dlopen("libcuda.so.1")
 * 2. Calls dlsym() for each function in the Driver API
 *
 * Newer CUDA Runtime API (version >=11.3) works like this:
 * 1. Calls dlopen("libcuda.so.1") and then dlsym("cuGetProcAddress")
 * 2. Calls cuGetProcAddress("cuGetProcAddress")
 *     1. If the pointer to "cuGetProcAddress" is NULL, it falls back to using
 *        dlsym() to get the Driver API function pointers
 *     2. If the pointer to "cuGetProcAddress" is not NULL, it uses
 *        cuGetProcAddress to get the Driver API function pointers.
 *
 * Interpose both, to cover all cases.
 *
 * The driver picks the variant of a function from cudaVersion and flags, so
 * we return our version of the function it picked.
 */
static void *get_proc_address_hook(int cudaVersion, cuuint64_t flags)
{
	void *fn = NULL;

	if (real_cuGetProcAddress_v2 != NULL)
		real_cuGetProcAddress_v2("cuGetProcAddress", &fn, cudaVersion,
					 flags, NULL);
	if (fn != NULL && fn == (void *)real_cuGetProcAddress_v2)
		return (void *)(&cuGetProcAddress_v2);
	return (void *)(&cuGetProcAddress);
}

static CUresult hook_proc(const char *symbol, void **pfn, CUresult result)
{
	void *hook = NULL;

	if (result == CUDA_SUCCESS && *pfn != NULL)
		hook = hook_for_real(*pfn);
	if (hook == NULL)
		hook = hook_for_proc(symbol);
	if (hook != NULL) {
		*pfn = hook;
		return CUDA_SUCCESS;
	}
	if (result == CUDA_SUCCESS && *pfn != NULL &&
	    hook_for_symbol(symbol) != NULL)
		log_warn("Can't hook %s, its work may break or bypass nvshare",
			 symbol);
	return result;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
	cuuint64_t flags)
{
	log_debug("cuGetProcAddress for symbol %s requested with cudaVersion %d",
		  symbol, cudaVersion);
	/*
	* cuGetProcAddress() will be called before cuInit() in CUDA
	* Runtime API (version >=11.3), so cuGetProcAddress() should also
	* serve as an entrypoint.
	* Otherwise, real_cuGetProcAddress may be a NULL pointer
	* when it is called.
	*/
	true_or_exit(pthread_once(&init_libnvshare_done, initialize_libnvshare) == 0);
	true_or_exit(pthread_once(&init_done, initialize_client) == 0);
	CUresult result = CUDA_SUCCESS;

	if (real_cuGetProcAddress == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	if (strcmp(symbol, "cuGetProcAddress") == 0) {
		*pfn = get_proc_address_hook(cudaVersion, flags);
	} else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
		*pfn = (void *)(&cuGetProcAddress_v2);
	} else {
		result = real_cuGetProcAddress(symbol, pfn, cudaVersion, flags);
		result = hook_proc(symbol, pfn, result);
	}

	return result;
}


CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
	cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
{
	log_debug("cuGetProcAddress_v2 requested");
	/*
	* cuGetProcAddress_v2() will be called before cuInit() in CUDA
	* Runtime API (version >=12.0), so cuGetProcAddress_v2()
	* should also serve as an entrypoint.
	*
	* Otherwise, real_cuGetProcAddress_v2 may be a
	* NULL pointer when it is called.
	*/
	true_or_exit(pthread_once(&init_libnvshare_done, initialize_libnvshare) == 0);
	true_or_exit(pthread_once(&init_done, initialize_client) == 0);
	CUresult result = CUDA_SUCCESS;

	if (real_cuGetProcAddress_v2 == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	/* This covers our custom "if" conditions.
	 * If we end up calling the real cuGetProcAddress_v2,
	 * it will overwrite this value.
	 */
	if (symbolStatus != NULL)
		*symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;

	if (strcmp(symbol, "cuGetProcAddress") == 0) {
		*pfn = get_proc_address_hook(cudaVersion, flags);
	} else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
		*pfn = (void *)(&cuGetProcAddress_v2);
	} else {
		result = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion,
					flags, symbolStatus);
		result = hook_proc(symbol, pfn, result);
		if (result == CUDA_SUCCESS && *pfn != NULL && symbolStatus != NULL)
			*symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
	}

	return result;
}


CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
	log_debug("cuMemAlloc requested");
	static int got_max_mem_size = 0;
	size_t junk;
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemAllocManaged == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	if (got_max_mem_size == 0) {
		result = cuMemGetInfo(&nvshare_size_mem_allocatable, &junk);
		cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemGetInfo));
		got_max_mem_size = 1;
	}

	if ((sum_allocated + bytesize) > nvshare_size_mem_allocatable) {
		if (enable_single_oversub == 0) {
			return CUDA_ERROR_OUT_OF_MEMORY;
		} else {
			log_warn("Memory allocations exceeded physical GPU"
				 " memory capacity. This can cause extreme"
				 " performance degradation!");
		}
	}

	log_debug("cuMemAlloc requested %zu bytes", bytesize);
	result = real_cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemAllocManaged));
	log_debug("cuMemAllocManaged allocated %zu bytes at 0x%llx",
		bytesize, *dptr);
	if (result == CUDA_SUCCESS) {
		insert_cuda_allocation(*dptr, bytesize);
	}

	return result;
}


CUresult cuMemFree(CUdeviceptr dptr)
{
	log_debug("cuMemFree requested");
	CUresult result = CUDA_SUCCESS;


	if (real_cuMemFree == NULL) return CUDA_ERROR_NOT_INITIALIZED;
	result = real_cuMemFree(dptr);
	if (result == CUDA_SUCCESS) remove_cuda_allocation(dptr);

	return result;
}


CUresult cuMemGetInfo(size_t *free, size_t *total)
{
	log_debug("cuMemGetInfo requested");
	long long reserve_mib;
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemGetInfo == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	result = real_cuMemGetInfo(free, total);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemGetInfo));

	log_debug("real_cuMemGetInfo returned free=%.2f MiB, total=%.2f MiB",
		 toMiB(*free), toMiB(*total));

        /*
	 * Hide a static amount of GPU memory from the applications. CUDA uses
	 * this memory to store context information and it is not pageable.
	 *
	 * In practice, this amount of memory is not static and depends on
	 * the number of colocated applications. Each one has its own context,
	 * which eats away some physical, non-pageable GPU memory.
	 *
	 * The first application that runs theoretically has (TOTAL_GPU_MEM -
	 * CONTEXT_SIZE) memory available.
	 *
	 * CONTEXT_SIZE typically uses a few hundred MB and depends on the GPU
	 * model.
	 *
	 * cuBLAS and other CUDA libraries also eat away at this memory.
	 *
	 * When another app runs, this "working memory size" shrinks further
	 * and can lead to thrashing within the first application, even when
	 * it runs alone.
	 *
	 * We cannot shrink the memory allocations of a running app, and the
	 * app thinks all of its memory is physically backed, since it's
	 * programmed with cuMemAlloc semantics in mind.
	 *
	 * To avoid internal thrashing, we empirically choose a sane value for
	 * MEMINFO_RESERVE_MIB.
	 */
	if (use_original_meminfo == 0) {
		reserve_mib = (MEMINFO_RESERVE_MIB) MiB;
		*free = *total - (size_t) reserve_mib;
	}

	log_debug("nvshare's cuMemGetInfo returning free=%.2f MiB,"
		  " total=%.2f MiB", toMiB(*free), toMiB(*total));
	return result;
}

/*
 * A call to cuInit is an indicator that the present application is a CUDA
 * application and that we should bootstrap nvshare.
 */
CUresult cuInit(unsigned int flags)
{
	log_debug("cuInit requested");
	CUresult result = CUDA_SUCCESS;

	true_or_exit(pthread_once(&init_libnvshare_done, initialize_libnvshare) == 0);
	true_or_exit(pthread_once(&init_done, initialize_client) == 0);

	result = real_cuInit(flags);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuInit));

	return result;
}


static int stream_is_capturing(CUstream hStream)
{
	CUstreamCaptureStatus status;

	if (__atomic_load_n(&active_captures, __ATOMIC_ACQUIRE) == 0 ||
	    real_cuStreamIsCapturing == NULL)
		return 0;
	if (real_cuStreamIsCapturing(hStream, &status) != CUDA_SUCCESS)
		return 0;
	return status != CU_STREAM_CAPTURE_STATUS_NONE;
}


static void count_launch(void)
{
	CUresult result = CUDA_SUCCESS;

	true_or_exit(pthread_mutex_lock(&kcount_mutex) == 0);


	/*
	 * Dynamic kernel submissing rate control.
	 *
	 * Some applications like to submit a huge amount of kernels in a short
	 * period of time.
	 *
	 * For nvshare, this means that they would still have pending kernels
	 * on the GPU when asked to relinquish the GPU lock.
	 *
	 * Since we sync the CUDA context before releasing the lock, this
	 * would mean we would hold the lock for much longer than TQ seconds,
	 * as that sync could possible take a very long time.
	 *
	 * To alleviate this source of unfairness, try to keep the completion
	 * time of submitted kernels to within 5 seconds, while simultaneously
	 * trying to maintain a good throughput rate for smaller kernels.
	 */
	kern_since_sync++;
	true_or_exit(pthread_mutex_lock(&capture_mutex) == 0);
	if (kern_since_sync >= pending_kernel_window && active_captures == 0) {
		struct timespec cuda_cuda_sync_start_time = {0, 0};
		struct timespec cuda_sync_complete_time = {0, 0};
		struct timespec cuda_sync_duration = {0, 0};
		true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_cuda_sync_start_time) == 0);
		result = real_cuCtxSynchronize();
		cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuCtxSynchronize));
		true_or_exit(clock_gettime(CLOCK_MONOTONIC, &cuda_sync_complete_time) == 0);
		timespecsub(&cuda_sync_complete_time, &cuda_cuda_sync_start_time, &cuda_sync_duration);

		/*
		 * Possibly a series of huge kernels. We cannot risk to
		 * simply fall back to previous window. Fall back to
		 * the initial window of 1.
		 */
		if (cuda_sync_duration.tv_sec >= KERN_SYNC_DURATION_BIG)
			pending_kernel_window = 1;

		/*
		 * Intermediate situation, don't be too harsh. Rein the
		 * rate in.
		 */
		else if (cuda_sync_duration.tv_sec >= KERN_SYNC_WINDOW_STEPDOWN_THRESH)
			pending_kernel_window = max(pending_kernel_window/2, 1);

		/*
		 * Max window size is simply a heuristic.
		 */
		else pending_kernel_window = min(pending_kernel_window*2,
				                 KERN_SYNC_WINDOW_MAX);

		log_debug("Pending Kernel Window is %d.", pending_kernel_window);
		kern_since_sync = 0;
	}
	true_or_exit(pthread_mutex_unlock(&capture_mutex) == 0);

	true_or_exit(pthread_mutex_unlock(&kcount_mutex) == 0);
}


static CUresult launch_kernel(cuLaunchKernel_func real, const char *name,
	CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
	unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
	unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
	void **kernelParams, void **extra, CUstream stream)
{
	CUresult result = CUDA_SUCCESS;
	int captured;

	log_debug("%s requested", name);

	/* Return immediately if not initialized */
	if (real == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	captured = stream_is_capturing(stream);
	if (!captured) continue_with_lock();
	result = real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
		blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
	cuda_driver_check_error(result, name);
	if (!captured) count_launch();

	return result;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
	unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
	unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
	void **extra)
{
	return launch_kernel(real_cuLaunchKernel, "cuLaunchKernel", f,
		gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
		sharedMemBytes, hStream, kernelParams, extra, hStream);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
	unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
	unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
	void **extra)
{
	return launch_kernel(real_cuLaunchKernel_ptsz, "cuLaunchKernel_ptsz", f,
		gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
		sharedMemBytes, hStream, kernelParams, extra,
		hStream ? hStream : CU_STREAM_PER_THREAD);
}


static CUresult launch_kernel_ex(cuLaunchKernelEx_func real, const char *name,
	const CUlaunchConfig *config, CUfunction f, void **kernelParams,
	void **extra, CUstream stream)
{
	CUresult result = CUDA_SUCCESS;
	int captured;

	log_debug("%s requested", name);
	if (real == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	captured = stream_is_capturing(stream);
	if (!captured) continue_with_lock();
	result = real(config, f, kernelParams, extra);
	cuda_driver_check_error(result, name);
	if (!captured) count_launch();

	return result;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
	void **kernelParams, void **extra)
{
	return launch_kernel_ex(real_cuLaunchKernelEx, "cuLaunchKernelEx",
		config, f, kernelParams, extra, config ? config->hStream : NULL);
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f,
	void **kernelParams, void **extra)
{
	CUstream hStream = config ? config->hStream : NULL;

	return launch_kernel_ex(real_cuLaunchKernelEx_ptsz,
		"cuLaunchKernelEx_ptsz", config, f, kernelParams, extra,
		hStream ? hStream : CU_STREAM_PER_THREAD);
}


static CUresult launch_cooperative_kernel(cuLaunchCooperativeKernel_func real,
	const char *name, CUfunction f, unsigned int gridDimX,
	unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
	unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
	CUstream stream)
{
	CUresult result = CUDA_SUCCESS;
	int captured;

	log_debug("%s requested", name);
	if (real == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	captured = stream_is_capturing(stream);
	if (!captured) continue_with_lock();
	result = real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
		blockDimZ, sharedMemBytes, hStream, kernelParams);
	cuda_driver_check_error(result, name);
	if (!captured) count_launch();

	return result;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
	unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
	unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
{
	return launch_cooperative_kernel(real_cuLaunchCooperativeKernel,
		"cuLaunchCooperativeKernel", f, gridDimX, gridDimY, gridDimZ,
		blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
		kernelParams, hStream);
}

CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX,
	unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX,
	unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams)
{
	return launch_cooperative_kernel(real_cuLaunchCooperativeKernel_ptsz,
		"cuLaunchCooperativeKernel_ptsz", f, gridDimX, gridDimY,
		gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes,
		hStream, kernelParams, hStream ? hStream : CU_STREAM_PER_THREAD);
}


static CUresult launch_graph(cuGraphLaunch_func real, const char *name,
	CUgraphExec hGraphExec, CUstream hStream, CUstream stream)
{
	CUresult result = CUDA_SUCCESS;
	int captured;

	log_debug("%s requested", name);
	if (real == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	captured = stream_is_capturing(stream);
	if (!captured) continue_with_lock();
	result = real(hGraphExec, hStream);
	cuda_driver_check_error(result, name);
	if (!captured) count_launch();

	return result;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
{
	return launch_graph(real_cuGraphLaunch, "cuGraphLaunch", hGraphExec,
		hStream, hStream);
}

CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream)
{
	return launch_graph(real_cuGraphLaunch_ptsz, "cuGraphLaunch_ptsz",
		hGraphExec, hStream, hStream ? hStream : CU_STREAM_PER_THREAD);
}


#define BEGIN_CAPTURE(name, params, args)                                  \
CUresult name params                                                       \
{                                                                          \
	CUresult result = CUDA_SUCCESS;                                    \
                                                                           \
	log_debug(#name " requested");                                     \
	if (real_##name == NULL) return CUDA_ERROR_NOT_INITIALIZED;        \
                                                                           \
	true_or_exit(pthread_mutex_lock(&capture_mutex) == 0);             \
	result = real_##name args;                                         \
	if (result == CUDA_SUCCESS)                                        \
		__atomic_add_fetch(&active_captures, 1, __ATOMIC_RELEASE); \
	true_or_exit(pthread_mutex_unlock(&capture_mutex) == 0);           \
                                                                           \
	return result;                                                     \
}

BEGIN_CAPTURE(cuStreamBeginCapture, (CUstream hStream), (hStream))
BEGIN_CAPTURE(cuStreamBeginCapture_ptsz, (CUstream hStream), (hStream))
BEGIN_CAPTURE(cuStreamBeginCapture_v2,
	(CUstream hStream, CUstreamCaptureMode mode), (hStream, mode))
BEGIN_CAPTURE(cuStreamBeginCapture_v2_ptsz,
	(CUstream hStream, CUstreamCaptureMode mode), (hStream, mode))
BEGIN_CAPTURE(cuStreamBeginCaptureToGraph,
	(CUstream hStream, CUgraph hGraph, const CUgraphNode *dependencies,
	 const CUgraphEdgeData *dependencyData, size_t numDependencies,
	 CUstreamCaptureMode mode),
	(hStream, hGraph, dependencies, dependencyData, numDependencies, mode))
BEGIN_CAPTURE(cuStreamBeginCaptureToGraph_ptsz,
	(CUstream hStream, CUgraph hGraph, const CUgraphNode *dependencies,
	 const CUgraphEdgeData *dependencyData, size_t numDependencies,
	 CUstreamCaptureMode mode),
	(hStream, hGraph, dependencies, dependencyData, numDependencies, mode))


static CUresult end_capture(cuStreamEndCapture_func real, const char *name,
	CUstream hStream, CUgraph *phGraph, CUstream stream)
{
	CUstreamCaptureStatus before, after;
	CUresult result = CUDA_SUCCESS;
	int ended;

	log_debug("%s requested", name);
	if (real == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	true_or_exit(pthread_mutex_lock(&capture_mutex) == 0);
	if (real_cuStreamIsCapturing == NULL) {
		result = real(hStream, phGraph);
		ended = (result == CUDA_SUCCESS ||
			 result == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
	} else {
		if (real_cuStreamIsCapturing(stream, &before) != CUDA_SUCCESS)
			before = CU_STREAM_CAPTURE_STATUS_NONE;
		result = real(hStream, phGraph);
		if (real_cuStreamIsCapturing(stream, &after) != CUDA_SUCCESS)
			after = CU_STREAM_CAPTURE_STATUS_NONE;
		ended = (before != CU_STREAM_CAPTURE_STATUS_NONE &&
			 after == CU_STREAM_CAPTURE_STATUS_NONE);
	}
	if (ended && active_captures > 0) {
		__atomic_sub_fetch(&active_captures, 1, __ATOMIC_RELEASE);
		true_or_exit(pthread_cond_broadcast(&capture_cv) == 0);
	}
	true_or_exit(pthread_mutex_unlock(&capture_mutex) == 0);

	return result;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph)
{
	return end_capture(real_cuStreamEndCapture, "cuStreamEndCapture",
		hStream, phGraph, hStream);
}

CUresult cuStreamEndCapture_ptsz(CUstream hStream, CUgraph *phGraph)
{
	return end_capture(real_cuStreamEndCapture_ptsz,
		"cuStreamEndCapture_ptsz", hStream, phGraph,
		hStream ? hStream : CU_STREAM_PER_THREAD);
}


/*
 * Memory copy functions can affect the resident pages on GPU, so we must
 * block them as well when the client doesn't have the GPU lock.
 */
#define GATED_COPY(name, params, args)                                \
CUresult name params                                                  \
{                                                                     \
	CUresult result = CUDA_SUCCESS;                               \
                                                                      \
	log_debug(#name " requested %zu bytes", ByteCount);           \
                                                                      \
	/* Return immediately if not initialized */                   \
	if (real_##name == NULL) return CUDA_ERROR_NOT_INITIALIZED;   \
                                                                      \
	continue_with_lock();                                         \
	result = real_##name args;                                    \
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(name));    \
                                                                      \
	return result;                                                \
}

#define COPY_PARAMS (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
#define COPY_ARGS (dst, src, ByteCount)
#define COPY_ASYNC_PARAMS \
	(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
#define COPY_ASYNC_ARGS (dst, src, ByteCount, hStream)
#define DTOH_PARAMS (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
#define DTOH_ARGS (dstHost, srcDevice, ByteCount)
#define DTOH_ASYNC_PARAMS \
	(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
#define DTOH_ASYNC_ARGS (dstHost, srcDevice, ByteCount, hStream)
#define HTOD_PARAMS \
	(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
#define HTOD_ARGS (dstDevice, srcHost, ByteCount)
#define HTOD_ASYNC_PARAMS \
	(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, \
	 CUstream hStream)
#define HTOD_ASYNC_ARGS (dstDevice, srcHost, ByteCount, hStream)
#define DTOD_PARAMS \
	(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
#define DTOD_ARGS (dstDevice, srcDevice, ByteCount)
#define DTOD_ASYNC_PARAMS \
	(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, \
	 CUstream hStream)
#define DTOD_ASYNC_ARGS (dstDevice, srcDevice, ByteCount, hStream)

GATED_COPY(cuMemcpy, COPY_PARAMS, COPY_ARGS)
GATED_COPY(cuMemcpyAsync, COPY_ASYNC_PARAMS, COPY_ASYNC_ARGS)
GATED_COPY(cuMemcpyDtoH, DTOH_PARAMS, DTOH_ARGS)
GATED_COPY(cuMemcpyDtoHAsync, DTOH_ASYNC_PARAMS, DTOH_ASYNC_ARGS)
GATED_COPY(cuMemcpyHtoD, HTOD_PARAMS, HTOD_ARGS)
GATED_COPY(cuMemcpyHtoDAsync, HTOD_ASYNC_PARAMS, HTOD_ASYNC_ARGS)
GATED_COPY(cuMemcpyDtoD, DTOD_PARAMS, DTOD_ARGS)
GATED_COPY(cuMemcpyDtoDAsync, DTOD_ASYNC_PARAMS, DTOD_ASYNC_ARGS)

GATED_COPY(cuMemcpy_ptds, COPY_PARAMS, COPY_ARGS)
GATED_COPY(cuMemcpyAsync_ptsz, COPY_ASYNC_PARAMS, COPY_ASYNC_ARGS)
GATED_COPY(cuMemcpyDtoH_v2_ptds, DTOH_PARAMS, DTOH_ARGS)
GATED_COPY(cuMemcpyDtoHAsync_v2_ptsz, DTOH_ASYNC_PARAMS, DTOH_ASYNC_ARGS)
GATED_COPY(cuMemcpyHtoD_v2_ptds, HTOD_PARAMS, HTOD_ARGS)
GATED_COPY(cuMemcpyHtoDAsync_v2_ptsz, HTOD_ASYNC_PARAMS, HTOD_ASYNC_ARGS)
GATED_COPY(cuMemcpyDtoD_v2_ptds, DTOD_PARAMS, DTOD_ARGS)
GATED_COPY(cuMemcpyDtoDAsync_v2_ptsz, DTOD_ASYNC_PARAMS, DTOD_ASYNC_ARGS)

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize,
	CUstream hStream __attribute__((unused)))
{
	log_debug("cuMemAllocAsync requested");
	static int got_max_mem_size = 0;
	size_t junk;
	CUresult result = CUDA_SUCCESS;


	/* Return immediately if not initialized */
	if (real_cuMemAllocManaged == NULL) return CUDA_ERROR_NOT_INITIALIZED;

	if (got_max_mem_size == 0) {
		result = cuMemGetInfo(&nvshare_size_mem_allocatable, &junk);
		cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemGetInfo));
		got_max_mem_size = 1;
	}

	if ((sum_allocated + bytesize) > nvshare_size_mem_allocatable) {
		if (enable_single_oversub == 0) {
			return CUDA_ERROR_OUT_OF_MEMORY;
		} else {
			log_warn("Memory allocations exceeded physical GPU"
				 " memory capacity. This can cause extreme"
				 " performance degradation!");
		}
	}

	log_debug("cuMemAllocAsync requested %zu bytes", bytesize);
	result = real_cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
	cuda_driver_check_error(result, CUDA_SYMBOL_STRING(cuMemAllocManaged));
	log_debug("cuMemAllocManaged allocated %zu bytes at 0x%llx",
		bytesize, *dptr);
	if (result == CUDA_SUCCESS) {
		insert_cuda_allocation(*dptr, bytesize);
	}

	return result;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr,
	CUstream hStream __attribute__((unused)))
{
	log_debug("cuMemFreeAsync requested");
	CUresult result = CUDA_SUCCESS;

	if (real_cuMemFree == NULL) return CUDA_ERROR_NOT_INITIALIZED;
	result = real_cuMemFree(dptr);
	if (result == CUDA_SUCCESS) remove_cuda_allocation(dptr);

	return result;
}


struct cuda_symbol {
	const char *name;
	void **real;
	int required;
	void *hook;
	const char *proc;
};

#define REAL(fn) ((void **)&real_##fn)

static const struct cuda_symbol cuda_symbols[] = {
	{"cuCtxSynchronize", REAL(cuCtxSynchronize), 1, NULL, NULL},
	{"cuMemAllocManaged", REAL(cuMemAllocManaged), 1, NULL, NULL},
	{"cuGetErrorString", REAL(cuGetErrorString), 1, NULL, NULL},
	{"cuGetErrorName", REAL(cuGetErrorName), 1, NULL, NULL},
	{"cuCtxSetCurrent", REAL(cuCtxSetCurrent), 1, NULL, NULL},
	{"cuCtxGetCurrent", REAL(cuCtxGetCurrent), 1, NULL, NULL},
	{"cuDeviceGet", REAL(cuDeviceGet), 0, NULL, NULL},
	{"cuDeviceTotalMem_v2", REAL(cuDeviceTotalMem), 0, NULL, NULL},
	{"cuStreamIsCapturing", REAL(cuStreamIsCapturing), 0, NULL, NULL},

	{"cuInit", REAL(cuInit), 1, cuInit, "cuInit"},
	{"cuGetProcAddress", REAL(cuGetProcAddress), 0, cuGetProcAddress, NULL},
	{"cuGetProcAddress_v2", REAL(cuGetProcAddress_v2), 0, cuGetProcAddress_v2, NULL},
	{"cuMemAlloc_v2", NULL, 0, cuMemAlloc, "cuMemAlloc"},
	{"cuMemFree_v2", REAL(cuMemFree), 1, cuMemFree, "cuMemFree"},
	{"cuMemGetInfo_v2", REAL(cuMemGetInfo), 1, cuMemGetInfo, "cuMemGetInfo"},
	{"cuMemAllocAsync", REAL(cuMemAllocAsync), 1, cuMemAllocAsync, "cuMemAllocAsync"},
	{"cuMemFreeAsync", REAL(cuMemFreeAsync), 1, cuMemFreeAsync, "cuMemFreeAsync"},

	{"cuLaunchKernel", REAL(cuLaunchKernel), 1, cuLaunchKernel, "cuLaunchKernel"},
	{"cuMemcpy", REAL(cuMemcpy), 1, cuMemcpy, "cuMemcpy"},
	{"cuMemcpyAsync", REAL(cuMemcpyAsync), 1, cuMemcpyAsync, "cuMemcpyAsync"},
	{"cuMemcpyDtoH_v2", REAL(cuMemcpyDtoH), 1, cuMemcpyDtoH, "cuMemcpyDtoH"},
	{"cuMemcpyDtoHAsync_v2", REAL(cuMemcpyDtoHAsync), 1, cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync"},
	{"cuMemcpyHtoD_v2", REAL(cuMemcpyHtoD), 1, cuMemcpyHtoD, "cuMemcpyHtoD"},
	{"cuMemcpyHtoDAsync_v2", REAL(cuMemcpyHtoDAsync), 1, cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync"},
	{"cuMemcpyDtoD_v2", REAL(cuMemcpyDtoD), 1, cuMemcpyDtoD, "cuMemcpyDtoD"},
	{"cuMemcpyDtoDAsync_v2", REAL(cuMemcpyDtoDAsync), 1, cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync"},

	{"cuLaunchKernel_ptsz", REAL(cuLaunchKernel_ptsz), 0, cuLaunchKernel_ptsz, NULL},
	{"cuMemcpy_ptds", REAL(cuMemcpy_ptds), 0, cuMemcpy_ptds, NULL},
	{"cuMemcpyAsync_ptsz", REAL(cuMemcpyAsync_ptsz), 0, cuMemcpyAsync_ptsz, NULL},
	{"cuMemcpyDtoH_v2_ptds", REAL(cuMemcpyDtoH_v2_ptds), 0, cuMemcpyDtoH_v2_ptds, NULL},
	{"cuMemcpyDtoHAsync_v2_ptsz", REAL(cuMemcpyDtoHAsync_v2_ptsz), 0, cuMemcpyDtoHAsync_v2_ptsz, NULL},
	{"cuMemcpyHtoD_v2_ptds", REAL(cuMemcpyHtoD_v2_ptds), 0, cuMemcpyHtoD_v2_ptds, NULL},
	{"cuMemcpyHtoDAsync_v2_ptsz", REAL(cuMemcpyHtoDAsync_v2_ptsz), 0, cuMemcpyHtoDAsync_v2_ptsz, NULL},
	{"cuMemcpyDtoD_v2_ptds", REAL(cuMemcpyDtoD_v2_ptds), 0, cuMemcpyDtoD_v2_ptds, NULL},
	{"cuMemcpyDtoDAsync_v2_ptsz", REAL(cuMemcpyDtoDAsync_v2_ptsz), 0, cuMemcpyDtoDAsync_v2_ptsz, NULL},

	{"cuLaunchKernelEx", REAL(cuLaunchKernelEx), 0, cuLaunchKernelEx, NULL},
	{"cuLaunchKernelEx_ptsz", REAL(cuLaunchKernelEx_ptsz), 0, cuLaunchKernelEx_ptsz, NULL},
	{"cuLaunchCooperativeKernel", REAL(cuLaunchCooperativeKernel), 0, cuLaunchCooperativeKernel, NULL},
	{"cuLaunchCooperativeKernel_ptsz", REAL(cuLaunchCooperativeKernel_ptsz), 0, cuLaunchCooperativeKernel_ptsz, NULL},
	{"cuGraphLaunch", REAL(cuGraphLaunch), 0, cuGraphLaunch, NULL},
	{"cuGraphLaunch_ptsz", REAL(cuGraphLaunch_ptsz), 0, cuGraphLaunch_ptsz, NULL},
	{"cuStreamBeginCapture", REAL(cuStreamBeginCapture), 0, cuStreamBeginCapture, NULL},
	{"cuStreamBeginCapture_ptsz", REAL(cuStreamBeginCapture_ptsz), 0, cuStreamBeginCapture_ptsz, NULL},
	{"cuStreamBeginCapture_v2", REAL(cuStreamBeginCapture_v2), 0, cuStreamBeginCapture_v2, NULL},
	{"cuStreamBeginCapture_v2_ptsz", REAL(cuStreamBeginCapture_v2_ptsz), 0, cuStreamBeginCapture_v2_ptsz, NULL},
	{"cuStreamBeginCaptureToGraph", REAL(cuStreamBeginCaptureToGraph), 0, cuStreamBeginCaptureToGraph, NULL},
	{"cuStreamBeginCaptureToGraph_ptsz", REAL(cuStreamBeginCaptureToGraph_ptsz), 0, cuStreamBeginCaptureToGraph_ptsz, NULL},
	{"cuStreamEndCapture", REAL(cuStreamEndCapture), 0, cuStreamEndCapture, NULL},
	{"cuStreamEndCapture_ptsz", REAL(cuStreamEndCapture_ptsz), 0, cuStreamEndCapture_ptsz, NULL},
};

#define CUDA_SYMBOLS_END \
	(cuda_symbols + sizeof(cuda_symbols) / sizeof(cuda_symbols[0]))

static void load_cuda_symbols(void *cuda_handle)
{
	const struct cuda_symbol *sym;
	void *addr;
	char *error;

	for (sym = cuda_symbols; sym < CUDA_SYMBOLS_END; sym++) {
		if (sym->real == NULL) continue;
		/*
		 * For dlsym(), a return value of NULL does not necessarily
		 * indicate an error. Therefore, we must:
		 *  1. clear the previous error state by calling dlerror()
		 *  2. call dlsym()
		 *  3. call dlerror()
		 * If the value which dlerror() returns is not NULL, an error
		 * occured.
		 */
		dlerror();
		addr = real_dlsym_225(cuda_handle, sym->name);
		error = dlerror();
		if (error != NULL) {
			if (sym->required) log_fatal("%s", error);
			/*
			 * This symbol may not be used. Older drivers lack the
			 * newer functions.
			 */
			log_debug("%s", error);
			addr = NULL;
		}
		memcpy(sym->real, &addr, sizeof(addr));
	}
}

static void *hook_for_symbol(const char *symbol)
{
	const struct cuda_symbol *sym;
	void *real;

	if (strncmp(symbol, "cu", 2) != 0) return NULL;
	for (sym = cuda_symbols; sym < CUDA_SYMBOLS_END; sym++) {
		if (sym->hook == NULL || strcmp(symbol, sym->name) != 0)
			continue;
		if (libcuda_base != NULL && sym->real != NULL) {
			memcpy(&real, sym->real, sizeof(real));
			if (real == NULL) return NULL;
		}
		return sym->hook;
	}
	return NULL;
}

static void *hook_for_proc(const char *symbol)
{
	const struct cuda_symbol *sym;

	for (sym = cuda_symbols; sym < CUDA_SYMBOLS_END; sym++)
		if (sym->proc != NULL && strcmp(symbol, sym->proc) == 0)
			return sym->hook;
	return NULL;
}

static void *hook_for_real(void *fn)
{
	const struct cuda_symbol *sym;
	void *real;

	for (sym = cuda_symbols; sym < CUDA_SYMBOLS_END; sym++) {
		if (sym->hook == NULL || sym->real == NULL) continue;
		memcpy(&real, sym->real, sizeof(real));
		if (real == fn) return sym->hook;
	}
	return NULL;
}


__asm__(".symver dlsym_225, dlsym@@GLIBC_2.2.5");
__asm__(".symver dlsym_234, dlsym@GLIBC_2.34");

