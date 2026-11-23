#include <cuda_runtime.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

constexpr uint32_t INIT_MULT = 2654435761u;
constexpr uint32_t LCG_MULT = 1664525u;
constexpr uint32_t LCG_PLUS = 1013904223u;
constexpr int ROUNDS = 16;
constexpr int THREADS = 256;
constexpr size_t CHUNK = (size_t)16 << 20;
constexpr double REPORT_SECONDS = 5.0;
constexpr double MIN_WAIT_SECONDS = 1.0;
constexpr double WAIT_FACTOR = 5.0;

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            printf("FAIL: %s: %s\n", #call, cudaGetErrorString(err_));    \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static volatile sig_atomic_t stop_requested = 0;

static void on_signal(int)
{
    stop_requested = 1;
}

__global__ void init_kernel(uint32_t *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (uint32_t)i * INIT_MULT;
}

__global__ void step_kernel(uint32_t *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t v = buf[i];
    for (int r = 0; r < ROUNDS; r++) v = v * LCG_MULT + LCG_PLUS;
    buf[i] = v;
}

__global__ void verify_kernel(const uint32_t *buf, size_t n, uint32_t mult,
                              uint32_t plus, unsigned int *bad)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && buf[i] != (uint32_t)i * INIT_MULT * mult + plus)
        atomicAdd(bad, 1u);
}

static void lcg_skip(uint64_t steps, uint32_t *mult_out, uint32_t *plus_out)
{
    uint32_t mult = LCG_MULT, plus = LCG_PLUS, acc_mult = 1, acc_plus = 0;
    while (steps > 0) {
        if (steps & 1) {
            acc_mult *= mult;
            acc_plus = acc_plus * mult + plus;
        }
        plus = (mult + 1) * plus;
        mult *= mult;
        steps >>= 1;
    }
    *mult_out = acc_mult;
    *plus_out = acc_plus;
}

static bool nvshare_loaded()
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
        if (line.find("libnvshare") != std::string::npos) return true;
    return false;
}

static double seconds_between(std::chrono::steady_clock::time_point a,
                              std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

static void usage(const char *prog)
{
    printf("usage: %s [--mem-mib N] [--seconds N]\n"
           "  --mem-mib N   GPU memory to allocate and keep in use (default 1024)\n"
           "  --seconds N   how long to run the workload (default 60)\n",
           prog);
}

int main(int argc, char **argv)
{
    long mem_mib = 1024, seconds = 60;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mem-mib") && i + 1 < argc) {
            mem_mib = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            seconds = atol(argv[++i]);
        } else {
            usage(argv[0]);
            return strcmp(argv[i], "--help") ? 2 : 0;
        }
    }
    if (mem_mib <= 0 || seconds <= 0) {
        usage(argv[0]);
        return 2;
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    size_t n = (size_t)mem_mib * 1024 * 1024 / sizeof(uint32_t);
    unsigned int blocks = (unsigned int)((n + THREADS - 1) / THREADS);

    cudaDeviceProp prop;
    size_t free_bytes, total_bytes;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("gpu: %s, %zu MiB total, %zu MiB free\n", prop.name,
           total_bytes >> 20, free_bytes >> 20);
    printf("libnvshare loaded: %s\n",
           nvshare_loaded() ? "yes" : "NO (running without nvshare)");
    printf("config: %ld MiB, %ld s\n", mem_mib, seconds);

    uint32_t *buf;
    unsigned int *bad_dev;
    unsigned int bad = 0;
    CHECK(cudaMalloc(&buf, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&bad_dev, sizeof(bad)));

    auto start = std::chrono::steady_clock::now();
    init_kernel<<<blocks, THREADS>>>(buf, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    auto loop_start = std::chrono::steady_clock::now();
    printf("[%6.1fs] initialised %ld MiB\n",
           seconds_between(start, loop_start), mem_mib);

    auto last_report = loop_start;
    uint64_t passes = 0, window_passes = 0, launches = 0, waits = 0;
    double fastest = 0, waited = 0, longest = 0;

    while (!stop_requested &&
           seconds_between(loop_start, std::chrono::steady_clock::now()) < seconds) {
        for (size_t off = 0; off < n; off += CHUNK) {
            size_t len = n - off < CHUNK ? n - off : CHUNK;
            unsigned int chunk_blocks = (unsigned int)((len + THREADS - 1) / THREADS);

            auto t0 = std::chrono::steady_clock::now();
            step_kernel<<<chunk_blocks, THREADS>>>(buf + off, len);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();

            double d = seconds_between(t0, t1);
            launches++;
            if (launches == 1 || d < fastest) fastest = d;
            if (d >= MIN_WAIT_SECONDS && d >= WAIT_FACTOR * fastest) {
                waits++;
                waited += d;
                if (d > longest) longest = d;
                printf("[%6.1fs] waited %.1f s for the gpu\n",
                       seconds_between(start, t1), d);
            }
        }
        passes++;
        window_passes++;

        auto now = std::chrono::steady_clock::now();
        if (seconds_between(last_report, now) >= REPORT_SECONDS) {
            printf("[%6.1fs] %llu passes (%.1f/s)\n",
                   seconds_between(start, now), (unsigned long long)passes,
                   window_passes / seconds_between(last_report, now));
            last_report = now;
            window_passes = 0;
        }
    }

    uint32_t mult, plus;
    lcg_skip(passes * ROUNDS, &mult, &plus);
    CHECK(cudaMemcpy(bad_dev, &bad, sizeof(bad), cudaMemcpyHostToDevice));
    verify_kernel<<<blocks, THREADS>>>(buf, n, mult, plus, bad_dev);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&bad, bad_dev, sizeof(bad), cudaMemcpyDeviceToHost));

    double elapsed = seconds_between(start, std::chrono::steady_clock::now());
    printf("passes over the memory: %llu in %.1f s\n",
           (unsigned long long)passes, elapsed);
    printf("waited for the gpu: %.1f s in %llu waits (longest %.1f s)\n",
           waited, (unsigned long long)waits, longest);
    if (bad != 0) {
        printf("data check: %u of %zu values wrong\n", bad, n);
        printf("FAIL\n");
        return 1;
    }
    printf("data check: all %zu values correct\n", n);
    printf("PASS\n");
    return 0;
}
