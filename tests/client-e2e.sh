#!/bin/bash
#
# End-to-end test of libnvshare without a GPU: the real library and scheduler,
# with a fake libcuda underneath and a scripted application on top. Needs gcc
# and make, no root.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
SOCK_DIR=$(mktemp -d /tmp/nvshare-XXXXXX)
SCHED_PID=
RUN=0
failed=0
export NVSHARE_SOCK_DIR=$SOCK_DIR

cleanup() {
    [ -n "$SCHED_PID" ] && kill "$SCHED_PID" 2> /dev/null
    wait 2> /dev/null
    rm -rf "$WORK" "$SOCK_DIR"
}
trap cleanup EXIT

check() {
    local desc=$1
    shift
    if "$@"; then
        echo "ok: $desc"
    else
        echo "FAIL: $desc"
        failed=1
    fi
}

start_scheduler() {
    NVSHARE_DEBUG=1 NVSHARE_TQ=${1:-30} "$ROOT/src/nvshare-scheduler" > "$WORK/scheduler.log" 2>&1 &
    SCHED_PID=$!
    for _ in $(seq 50); do
        grep -q "listening on" "$WORK/scheduler.log" && return 0
        sleep 0.1
    done
    echo "FAIL: scheduler did not start"
    cat "$WORK/scheduler.log"
    exit 1
}

stop_scheduler() {
    kill "$SCHED_PID"
    wait "$SCHED_PID"
    check "the scheduler stopped cleanly" [ $? -eq 0 ]
    SCHED_PID=
}

app() {
    RUN=$((RUN + 1))
    OUT=$WORK/app-$RUN.out
    ERR=$WORK/app-$RUN.err
    NVSHARE_DEBUG=1 LD_LIBRARY_PATH=$WORK LD_PRELOAD=$ROOT/src/libnvshare.so \
        timeout 60 "$WORK/app" "$@" > "$OUT" 2> "$ERR"
}

sched_count() {
    grep -c -- "$1" "$WORK/scheduler.log" || true
}

says() {
    grep -qx -- "$1" "$OUT"
}

logs() {
    grep -q -- "$1" "$ERR"
}

make -C "$ROOT/src" libnvshare.so nvshare-scheduler nvsharectl > "$WORK/build.log" 2>&1 || {
    cat "$WORK/build.log"
    exit 1
}
gcc -Wall -Wextra -Werror -shared -fPIC -Wl,-soname,libcuda.so.1 \
    -o "$WORK/libcuda.so.1" "$ROOT/tests/client/fake-libcuda.c" -ldl
ln -s libcuda.so.1 "$WORK/libcuda.so"
gcc -Wall -Wextra -Werror -o "$WORK/app" "$ROOT/tests/client/app.c" -L"$WORK" -lcuda -ldl

echo "== every way to run work on the GPU asks for the lock and reaches the driver"
start_scheduler
n=0
for fn in cuLaunchKernel cuLaunchKernel_ptsz cuLaunchKernelEx cuLaunchKernelEx_ptsz \
    cuLaunchCooperativeKernel cuLaunchCooperativeKernel_ptsz cuGraphLaunch cuGraphLaunch_ptsz \
    cuMemcpy cuMemcpy_ptds cuMemcpyAsync cuMemcpyAsync_ptsz \
    cuMemcpyDtoH_v2 cuMemcpyDtoH_v2_ptds cuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync_v2_ptsz \
    cuMemcpyHtoD_v2 cuMemcpyHtoD_v2_ptds cuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync_v2_ptsz \
    cuMemcpyDtoD_v2 cuMemcpyDtoD_v2_ptds cuMemcpyDtoDAsync_v2 cuMemcpyDtoDAsync_v2_ptsz; do
    n=$((n + 1))
    app "launch:$fn" last
    check "$fn waits for the lock" [ "$(sched_count 'Received REQ_LOCK')" -eq "$n" ]
    check "$fn runs in the driver" says "last: $fn"
done

echo "== a function looked up like the CUDA runtime does is the variant the driver picked"
app proc:cuLaunchKernel:0:12000 last
check "the legacy default stream gets cuLaunchKernel" says "last: cuLaunchKernel"
app proc:cuLaunchKernel:2:12000 last
check "the per-thread default stream gets cuLaunchKernel_ptsz" says "last: cuLaunchKernel_ptsz"
app proc:cuMemcpyDtoH:2:12000 last
check "the per-thread default stream gets cuMemcpyDtoH_v2_ptds" says "last: cuMemcpyDtoH_v2_ptds"
app proc:cuLaunchKernelEx:2:12000 last
check "the per-thread default stream gets cuLaunchKernelEx_ptsz" says "last: cuLaunchKernelEx_ptsz"
app proc:cuGraphLaunch:0:11040 last
check "CUDA 11 gets cuGraphLaunch" says "last: cuGraphLaunch"
app dlsym:cuLaunchKernel_ptsz last
check "dlsym() finds cuLaunchKernel_ptsz" says "last: cuLaunchKernel_ptsz"
check "all of them waited for the lock" [ "$(sched_count 'Received REQ_LOCK')" -eq $((n + 6)) ]

echo "== a missing function is reported the way the driver reports it"
app proc:cuNoSuchFunction:0:12000
check "CUDA 12: no error, no function, status not found" \
    [ "$(tr '\n' ' ' < "$OUT")" = "result: 0 status: 1 function: none done " ]
app proc:cuNoSuchFunction:0:11040
check "CUDA 11: an error" says "result: 500"

echo "== a variant the driver picks that libnvshare does not know"
FAKE_CUDA_UNKNOWN_PROCS=1 app proc:cuLaunchKernel:0:12000 last
check "a kernel launch falls back to cuLaunchKernel" says "last: cuLaunchKernel"
FAKE_CUDA_UNKNOWN_PROCS=1 app proc:cuGraphLaunch:0:12000 last
check "a graph launch is passed through" says "last: unknown"
check "with a warning" logs "Can't hook cuGraphLaunch"
FAKE_CUDA_FAIL_PROCS=1 app proc:cuLaunchKernel:0:11040 last
check "a failed lookup of a kernel launch falls back to cuLaunchKernel" says "last: cuLaunchKernel"

echo "== a driver whose cuInit() looks itself up with dlsym() does not deadlock"
FAKE_CUDA_REENTRANT_INIT=1 app last
check "the application finished" says "done"

echo "== a failing call is logged when the driver has no error strings"
FAKE_CUDA_FAIL_CALLS=1 FAKE_CUDA_NO_ERROR_STRINGS=1 app launch:cuLaunchKernel
check "the application finished" says "done"
check "the error is logged as unknown" logs "cuLaunchKernel returned unknown: unknown"
stop_scheduler

echo "== launches into a stream capture do not synchronize and need no lock"
start_scheduler
app begin launch:cuLaunchKernel:50 launch:cuGraphLaunch:50 syncs end
check "the capture began" says "begin: 0"
check "the driver was not synchronized" says "syncs: 0"
check "the capture is intact" says "end: 0"
check "the scheduler was not asked for the lock" [ "$(sched_count 'Received REQ_LOCK')" -eq 0 ]

echo "== work on another stream does not synchronize while a capture is open"
app begin stream:other launch:cuLaunchKernel:50 syncs end
check "the driver was not synchronized" says "syncs: 0"
check "the capture is intact" says "end: 0"
check "the work waited for the lock" [ "$(sched_count 'Received REQ_LOCK')" -eq 1 ]
app begin stream:other launch:cuLaunchKernel:50 end launch:cuLaunchKernel syncs
check "it synchronizes again once the capture has ended" says "syncs: 1"

echo "== a capture that could not be ended from a joined stream is still open"
app begin end-joined launch:cuLaunchKernel:50 end
check "ending it from the joined stream failed" says "end-joined: 903"
check "the capture is intact" says "end: 0"
stop_scheduler

echo "== the lock is given back only after an open capture has ended"
NVSHARE_PARALLEL=0 start_scheduler 1
app launch:cuLaunchKernel begin sleep:3 end sleep:1 &
A=$!
A_OUT=$WORK/app-$((RUN + 1)).out
sleep 0.5
app loop:1:cuLaunchKernel
wait "$A"
check "the scheduler asked for the lock back" [ "$(sched_count 'Sent DROP_LOCK')" -ge 1 ]
check "the capture is intact" grep -qx "end: 0" "$A_OUT"
check "the waiting application got the lock" says "done"
stop_scheduler

echo "== a capture that stays open does not keep the lock from the others"
NVSHARE_PARALLEL=0 start_scheduler 1
app launch:cuLaunchKernel begin sleep:9 end &
A=$!
A_OUT=$WORK/app-$((RUN + 1)).out
A_ERR=$WORK/app-$((RUN + 1)).err
sleep 0.5
start=$SECONDS
app launch:cuLaunchKernel
check "the waiting application got the lock while the capture was open" [ $((SECONDS - start)) -lt 8 ]
wait "$A"
check "the lock was given back without synchronizing" grep -q "A stream capture is still open" "$A_ERR"
check "the capture is intact" grep -qx "end: 0" "$A_OUT"
stop_scheduler

echo "== an application that is capturing keeps the lock although it launches nothing"
start_scheduler
app launch:cuLaunchKernel begin sleep:7 end
check "the capture is intact" says "end: 0"
check "the lock was not given back" [ "$(sched_count 'Received LOCK_RELEASED')" -eq 0 ]
stop_scheduler

echo "== an idle application gives the lock back"
start_scheduler
app launch:cuLaunchKernel sleep:7
check "without NVML" [ "$(sched_count 'Received LOCK_RELEASED')" -eq 1 ]
gcc -Wall -Wextra -Werror -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 \
    -o "$WORK/libnvidia-ml.so.1" "$ROOT/tests/client/fake-nvml.c"
app launch:cuLaunchKernel sleep:7 syncs
check "on an idle GPU" [ "$(sched_count 'Received LOCK_RELEASED')" -eq 2 ]
check "without synchronizing more than once" says "syncs: 1"
FAKE_NVML_UTIL=90 app launch:cuLaunchKernel sleep:7
check "on a GPU that others keep busy" [ "$(sched_count 'Received LOCK_RELEASED')" -eq 3 ]
FAKE_NVML_UTIL=90 FAKE_CUDA_SYNC_MS=300 app launch:cuLaunchKernel sleep:7
check "but not while its own work is running" [ "$(sched_count 'Received LOCK_RELEASED')" -eq 3 ]
stop_scheduler

echo "== the scheduler learns how much memory the applications have allocated"
start_scheduler 1
app alloc:1000 launch:cuLaunchKernel
check "an allocation is reported in steps of 64 MiB" logs "Sent MEM_USAGE (1024 MiB)"
app alloc:3000 free sleep:7
check "growth is reported at once" logs "Sent MEM_USAGE (3008 MiB)"
check "freed memory is reported once the application is idle" logs "Sent MEM_USAGE (0 MiB)"
app alloc:1000 loop:2:cuLaunchKernel &
A=$!
app alloc:1000 loop:2:cuLaunchKernel
wait "$A"
check "two applications that fit on the GPU together run in parallel" \
    [ "$(sched_count 'Sent DROP_LOCK')" -eq 0 ]
app alloc:3000 loop:3:cuLaunchKernel &
A=$!
app alloc:3000 loop:3:cuLaunchKernel
wait "$A"
check "two applications that do not fit are told so" \
    [ "$(sched_count 'does not fit (9088 of 8192 MiB)')" -eq 1 ]
check "and take turns" [ "$(sched_count 'Sent DROP_LOCK')" -ge 1 ]
FAKE_CUDA_NO_DEVICE=1 app launch:cuLaunchKernel
check "an application that cannot tell the size of the GPU says so" \
    logs "Could not determine the size of the GPU memory"
check "and so does the scheduler" [ "$(sched_count 'did not report the size of the GPU memory')" -eq 1 ]
stop_scheduler

if [ "$failed" -ne 0 ]; then
    for f in "$WORK"/scheduler.log "$WORK"/app-*.err; do
        echo "::group::$(basename "$f")"
        cat "$f"
        echo "::endgroup::"
    done
    exit 1
fi
echo "PASS: libnvshare gates every kind of GPU work, survives stream captures and reports its memory"
