#!/bin/bash
set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 BIN_DIR"
    echo "BIN_DIR must contain nvshare-scheduler, nvsharectl, libnvshare.so and gpu-test"
    exit 2
fi

BIN=$(cd "$1" && pwd)
MEM_MIB=${MEM_MIB:-2048}
DURATION=${DURATION:-40}
PARALLEL_DURATION=${PARALLEL_DURATION:-20}
TQ=${TQ:-5}
RESERVE_MIB=1536

LOGS=$(mktemp -d)
SOCK=/var/run/nvshare/scheduler.sock
SCHED_PID=
failed=0

fail() {
    echo "FAIL: $*"
    failed=1
}

show_logs() {
    for f in "$LOGS"/*.log; do
        echo "::group::$(basename "$f")"
        cat "$f"
        echo "::endgroup::"
    done
}

cleanup() {
    [ -n "$SCHED_PID" ] && kill "$SCHED_PID" 2>/dev/null
    wait 2>/dev/null
    rm -rf "$LOGS"
}
trap cleanup EXIT

wait_for_log() {
    for _ in $(seq 50); do
        grep -q "$1" "$LOGS/scheduler.log" && return 0
        kill -0 "$SCHED_PID" 2>/dev/null || return 1
        sleep 0.1
    done
    return 1
}

run() {
    local mem=$1 seconds=$2 n log id registered rc

    rm -f "$SOCK" "$LOGS"/*.log
    "$BIN/nvshare-scheduler" > "$LOGS/scheduler.log" 2>&1 &
    SCHED_PID=$!
    if ! wait_for_log "listening on"; then
        fail "scheduler did not start"
        show_logs
        exit 1
    fi

    "$BIN/nvsharectl" -T "$TQ" > "$LOGS/nvsharectl.log" 2>&1
    if ! wait_for_log "New TQ = $TQ"; then
        fail "scheduler did not accept time quantum $TQ"
        show_logs
        exit 1
    fi

    echo "running 2 workloads: $mem MiB each, $seconds s, time quantum $TQ s"
    for n in 1 2; do
        LD_PRELOAD="$BIN/libnvshare.so" "$BIN/gpu-test" \
            --mem-mib "$mem" --seconds "$seconds" > "$LOGS/workload-$n.log" 2>&1 &
        PIDS[n]=$!
    done
    for n in 1 2; do
        wait "${PIDS[n]}"
        RC[n]=$?
    done

    kill "$SCHED_PID"
    wait "$SCHED_PID"
    rc=$?
    SCHED_PID=

    show_logs

    for n in 1 2; do
        log="$LOGS/workload-$n.log"
        [ "${RC[n]}" -eq 0 ] || fail "workload $n exited with ${RC[n]}"
        grep -q '^PASS$' "$log" || fail "workload $n did not report PASS"
        grep -q '^libnvshare loaded: yes' "$log" || fail "workload $n ran without libnvshare"
        WAITS[n]=$(sed -n 's/^waited for the gpu: .* in \([0-9]*\) waits.*/\1/p' "$log")
    done

    while read -r id; do
        grep -q "Sent LOCK_OK to client $id" "$LOGS/scheduler.log" || fail "client $id never got the lock"
    done < <(sed -n 's/.*Registered client \([0-9a-f]*\) .*/\1/p' "$LOGS/scheduler.log")
    registered=$(grep -c "Registered client" "$LOGS/scheduler.log")
    drops=$(grep -c "Sent DROP_LOCK" "$LOGS/scheduler.log")
    turns=$(grep -c "one client at a time" "$LOGS/scheduler.log")
    [ "$registered" -eq 2 ] || fail "scheduler registered $registered clients, expected 2"
    [ "$rc" -eq 0 ] || fail "scheduler exited with $rc"
    if grep -q "did not report the size of the GPU memory" "$LOGS/scheduler.log"; then
        fail "a workload did not report the size of the GPU memory"
    fi
}

nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
BIG_MEM_MIB=${BIG_MEM_MIB:-$((total / 2 - 1280))}
if [ $((2 * (MEM_MIB + RESERVE_MIB + 64))) -gt "$total" ]; then
    echo "FAIL: two workloads of $MEM_MIB MiB do not fit a gpu of $total MiB, set MEM_MIB"
    exit 1
fi
if [ "$free" -lt $((2 * BIG_MEM_MIB + 512)) ]; then
    echo "FAIL: gpu has only $free MiB free, need $((2 * BIG_MEM_MIB + 512)) (in use by other processes?)"
    exit 1
fi

echo "== two workloads that fit in GPU memory together run in parallel"
run "$MEM_MIB" "$PARALLEL_DURATION"
for n in 1 2; do
    [ "${WAITS[n]:-1}" -eq 0 ] || fail "workload $n waited for the gpu"
done
[ "$drops" -eq 0 ] || fail "scheduler sent DROP_LOCK $drops times"
[ "$turns" -eq 0 ] || fail "scheduler made the workloads take turns"

echo "== two workloads that do not fit in GPU memory together take turns"
run "$BIG_MEM_MIB" "$DURATION"
for n in 1 2; do
    [ "${WAITS[n]:-0}" -ge 1 ] || fail "workload $n never waited for the gpu"
done
[ "$drops" -ge 1 ] || fail "scheduler never sent DROP_LOCK"
[ "$turns" -ge 1 ] || fail "scheduler let the workloads run in parallel"

if [ "$failed" -ne 0 ]; then
    exit 1
fi
echo "PASS: the workloads ran in parallel while they fit and shared the gpu when they did not ($drops lock handovers)"
