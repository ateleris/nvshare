/*
 * A CUDA application without a GPU, for tests of libnvshare. It runs against
 * the fake libcuda and carries out the steps given as arguments, in order:
 *
 *   launch:NAME[:COUNT]  call the launch or copy function NAME, COUNT times
 *   loop:SECONDS:NAME    call NAME every 10 ms for SECONDS
 *   proc:NAME:FLAGS:VERSION
 *                        look NAME up like the CUDA runtime does and call
 *                        it: find cuGetProcAddress() with dlsym(), ask it
 *                        for the cuGetProcAddress() of VERSION (from 12000
 *                        one that also reports a status), and use that one
 *   dlsym:NAME           look NAME up in libcuda with dlsym() and call it
 *   begin, end           begin and end a capture on its main stream
 *   stream:main|other    launch on the main stream (the default) or another
 *   end-joined           try to end the capture from a stream that joined it
 *   alloc:MIB, free      allocate GPU memory, free the last allocation
 *   sleep:SECONDS
 *   syncs, last          print how often the driver synchronized and which
 *                        launch or copy function it ran last
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define STREAM ((void *)0x100)
#define OTHER_STREAM ((void *)0x200)

typedef int (*generic_fn)(void *, void *, void *, void *, void *, void *,
			  void *, void *, void *, void *, void *);

struct launch_config {
	unsigned int grid[3];
	unsigned int block[3];
	unsigned int shared_mem;
	void *stream;
	void *attrs;
	unsigned int num_attrs;
};

int cuInit(unsigned int flags);
int cuMemAlloc_v2(unsigned long long *dptr, size_t bytesize);
int cuMemFree_v2(unsigned long long dptr);
int cuLaunchKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz,
		   unsigned int bx, unsigned int by, unsigned int bz,
		   unsigned int shared_mem, void *stream, void **params,
		   void **extra);
int cuLaunchKernel_ptsz(void *f, unsigned int gx, unsigned int gy,
			unsigned int gz, unsigned int bx, unsigned int by,
			unsigned int bz, unsigned int shared_mem, void *stream,
			void **params, void **extra);
int cuLaunchKernelEx(const struct launch_config *config, void *f,
		     void **params, void **extra);
int cuLaunchKernelEx_ptsz(const struct launch_config *config, void *f,
			  void **params, void **extra);
int cuLaunchCooperativeKernel(void *f, unsigned int gx, unsigned int gy,
			      unsigned int gz, unsigned int bx, unsigned int by,
			      unsigned int bz, unsigned int shared_mem,
			      void *stream, void **params);
int cuLaunchCooperativeKernel_ptsz(void *f, unsigned int gx, unsigned int gy,
				   unsigned int gz, unsigned int bx,
				   unsigned int by, unsigned int bz,
				   unsigned int shared_mem, void *stream,
				   void **params);
int cuGraphLaunch(void *exec, void *stream);
int cuGraphLaunch_ptsz(void *exec, void *stream);
int cuMemcpy(unsigned long long dst, unsigned long long src, size_t n);
int cuMemcpy_ptds(unsigned long long dst, unsigned long long src, size_t n);
int cuMemcpyAsync(unsigned long long dst, unsigned long long src, size_t n,
		  void *stream);
int cuMemcpyAsync_ptsz(unsigned long long dst, unsigned long long src,
		       size_t n, void *stream);
int cuMemcpyDtoH_v2(void *dst, unsigned long long src, size_t n);
int cuMemcpyDtoH_v2_ptds(void *dst, unsigned long long src, size_t n);
int cuMemcpyDtoHAsync_v2(void *dst, unsigned long long src, size_t n,
			 void *stream);
int cuMemcpyDtoHAsync_v2_ptsz(void *dst, unsigned long long src, size_t n,
			      void *stream);
int cuMemcpyHtoD_v2(unsigned long long dst, const void *src, size_t n);
int cuMemcpyHtoD_v2_ptds(unsigned long long dst, const void *src, size_t n);
int cuMemcpyHtoDAsync_v2(unsigned long long dst, const void *src, size_t n,
			 void *stream);
int cuMemcpyHtoDAsync_v2_ptsz(unsigned long long dst, const void *src,
			      size_t n, void *stream);
int cuMemcpyDtoD_v2(unsigned long long dst, unsigned long long src, size_t n);
int cuMemcpyDtoD_v2_ptds(unsigned long long dst, unsigned long long src,
			 size_t n);
int cuMemcpyDtoDAsync_v2(unsigned long long dst, unsigned long long src,
			 size_t n, void *stream);
int cuMemcpyDtoDAsync_v2_ptsz(unsigned long long dst, unsigned long long src,
			      size_t n, void *stream);
int cuStreamBeginCapture_v2(void *stream, int mode);
int cuStreamEndCapture(void *stream, void **graph);
const char *fake_last_call(void);
int fake_sync_count(void);

static void *stream = STREAM;

static int launch(const char *name)
{
	struct launch_config config = { .stream = stream };

	if (!strcmp(name, "cuLaunchKernel"))
		return cuLaunchKernel(NULL, 1, 1, 1, 1, 1, 1, 0, stream, NULL, NULL);
	if (!strcmp(name, "cuLaunchKernel_ptsz"))
		return cuLaunchKernel_ptsz(NULL, 1, 1, 1, 1, 1, 1, 0, stream, NULL, NULL);
	if (!strcmp(name, "cuLaunchKernelEx"))
		return cuLaunchKernelEx(&config, NULL, NULL, NULL);
	if (!strcmp(name, "cuLaunchKernelEx_ptsz"))
		return cuLaunchKernelEx_ptsz(&config, NULL, NULL, NULL);
	if (!strcmp(name, "cuLaunchCooperativeKernel"))
		return cuLaunchCooperativeKernel(NULL, 1, 1, 1, 1, 1, 1, 0, stream, NULL);
	if (!strcmp(name, "cuLaunchCooperativeKernel_ptsz"))
		return cuLaunchCooperativeKernel_ptsz(NULL, 1, 1, 1, 1, 1, 1, 0, stream, NULL);
	if (!strcmp(name, "cuGraphLaunch"))
		return cuGraphLaunch(NULL, stream);
	if (!strcmp(name, "cuGraphLaunch_ptsz"))
		return cuGraphLaunch_ptsz(NULL, stream);
	if (!strcmp(name, "cuMemcpy"))
		return cuMemcpy(0, 0, 1);
	if (!strcmp(name, "cuMemcpy_ptds"))
		return cuMemcpy_ptds(0, 0, 1);
	if (!strcmp(name, "cuMemcpyAsync"))
		return cuMemcpyAsync(0, 0, 1, stream);
	if (!strcmp(name, "cuMemcpyAsync_ptsz"))
		return cuMemcpyAsync_ptsz(0, 0, 1, stream);
	if (!strcmp(name, "cuMemcpyDtoH_v2"))
		return cuMemcpyDtoH_v2(NULL, 0, 1);
	if (!strcmp(name, "cuMemcpyDtoH_v2_ptds"))
		return cuMemcpyDtoH_v2_ptds(NULL, 0, 1);
	if (!strcmp(name, "cuMemcpyDtoHAsync_v2"))
		return cuMemcpyDtoHAsync_v2(NULL, 0, 1, stream);
	if (!strcmp(name, "cuMemcpyDtoHAsync_v2_ptsz"))
		return cuMemcpyDtoHAsync_v2_ptsz(NULL, 0, 1, stream);
	if (!strcmp(name, "cuMemcpyHtoD_v2"))
		return cuMemcpyHtoD_v2(0, NULL, 1);
	if (!strcmp(name, "cuMemcpyHtoD_v2_ptds"))
		return cuMemcpyHtoD_v2_ptds(0, NULL, 1);
	if (!strcmp(name, "cuMemcpyHtoDAsync_v2"))
		return cuMemcpyHtoDAsync_v2(0, NULL, 1, stream);
	if (!strcmp(name, "cuMemcpyHtoDAsync_v2_ptsz"))
		return cuMemcpyHtoDAsync_v2_ptsz(0, NULL, 1, stream);
	if (!strcmp(name, "cuMemcpyDtoD_v2"))
		return cuMemcpyDtoD_v2(0, 0, 1);
	if (!strcmp(name, "cuMemcpyDtoD_v2_ptds"))
		return cuMemcpyDtoD_v2_ptds(0, 0, 1);
	if (!strcmp(name, "cuMemcpyDtoDAsync_v2"))
		return cuMemcpyDtoDAsync_v2(0, 0, 1, stream);
	if (!strcmp(name, "cuMemcpyDtoDAsync_v2_ptsz"))
		return cuMemcpyDtoDAsync_v2_ptsz(0, 0, 1, stream);
	fprintf(stderr, "unknown function %s\n", name);
	exit(2);
}

static double now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void call(void *fn)
{
	printf("called: %d\n", ((generic_fn)fn)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

static void proc(const char *name, unsigned long long flags, int version)
{
	typedef int (*proc_v1)(const char *, void **, int, unsigned long long);
	typedef int (*proc_v2)(const char *, void **, int, unsigned long long, int *);
	void *libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
	void *get = NULL, *fn = NULL;
	int result, status = -1;
	proc_v1 first = (proc_v1)dlsym(libcuda, "cuGetProcAddress");

	if (first == NULL || first("cuGetProcAddress", &get, version, flags) != 0 ||
	    get == NULL) {
		printf("no cuGetProcAddress\n");
		exit(1);
	}
	if (version >= 12000)
		result = ((proc_v2)get)(name, &fn, version, flags, &status);
	else result = ((proc_v1)get)(name, &fn, version, flags);
	printf("result: %d\nstatus: %d\n", result, status);
	if (result != 0 || fn == NULL) {
		printf("function: none\n");
		return;
	}
	call(fn);
}

int main(int argc, char **argv)
{
	unsigned long long dptr = 0;
	void *graph, *fn;
	char *op, *arg, *arg2, *arg3;
	double end;
	int i, n;

	setvbuf(stdout, NULL, _IOLBF, 0);
	if (cuInit(0) != 0) {
		printf("cuInit failed\n");
		return 1;
	}

	for (i = 1; i < argc; i++) {
		op = strtok(argv[i], ":");
		arg = strtok(NULL, ":");
		arg2 = strtok(NULL, ":");
		arg3 = strtok(NULL, ":");

		if (!strcmp(op, "launch") && arg) {
			for (n = arg2 ? atoi(arg2) : 1; n > 0; n--)
				launch(arg);
		} else if (!strcmp(op, "loop") && arg && arg2) {
			for (end = now() + atof(arg); now() < end; usleep(10000))
				launch(arg2);
		} else if (!strcmp(op, "proc") && arg && arg2 && arg3) {
			proc(arg, strtoull(arg2, NULL, 0), atoi(arg3));
		} else if (!strcmp(op, "dlsym") && arg) {
			fn = dlsym(dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD), arg);
			if (fn == NULL) printf("function: none\n");
			else call(fn);
		} else if (!strcmp(op, "stream") && arg) {
			stream = strcmp(arg, "other") ? STREAM : OTHER_STREAM;
		} else if (!strcmp(op, "begin")) {
			printf("begin: %d\n", cuStreamBeginCapture_v2(STREAM, 0));
		} else if (!strcmp(op, "end")) {
			printf("end: %d\n", cuStreamEndCapture(STREAM, &graph));
		} else if (!strcmp(op, "end-joined")) {
			printf("end-joined: %d\n",
			       cuStreamEndCapture((char *)STREAM + 1, &graph));
		} else if (!strcmp(op, "alloc") && arg) {
			printf("alloc: %d\n",
			       cuMemAlloc_v2(&dptr, (size_t)atol(arg) << 20));
		} else if (!strcmp(op, "free")) {
			printf("free: %d\n", cuMemFree_v2(dptr));
		} else if (!strcmp(op, "sleep") && arg) {
			usleep((useconds_t)(atof(arg) * 1e6));
		} else if (!strcmp(op, "syncs")) {
			printf("syncs: %d\n", fake_sync_count());
		} else if (!strcmp(op, "last")) {
			printf("last: %s\n", fake_last_call());
		} else {
			fprintf(stderr, "bad step %s\n", argv[i]);
			return 2;
		}
	}
	printf("done\n");
	return 0;
}
