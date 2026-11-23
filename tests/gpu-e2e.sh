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
TQ=${TQ:-5}

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

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

rm -f "$SOCK"
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

echo "running 2 workloads: $MEM_MIB MiB each, $DURATION s, time quantum $TQ s"
for n in 1 2; do
    LD_PRELOAD="$BIN/libnvshare.so" "$BIN/gpu-test" \
        --mem-mib "$MEM_MIB" --seconds "$DURATION" > "$LOGS/workload-$n.log" 2>&1 &
    PIDS[n]=$!
done
for n in 1 2; do
    wait "${PIDS[n]}"
    RC[n]=$?
done

kill "$SCHED_PID"
wait "$SCHED_PID"
SCHED_RC=$?
SCHED_PID=

show_logs

for n in 1 2; do
    log="$LOGS/workload-$n.log"
    [ "${RC[n]}" -eq 0 ] || fail "workload $n exited with ${RC[n]}"
    grep -q '^PASS$' "$log" || fail "workload $n did not report PASS"
    grep -q '^libnvshare loaded: yes' "$log" || fail "workload $n ran without libnvshare"
    waits=$(sed -n 's/^waited for the gpu: .* in \([0-9]*\) waits.*/\1/p' "$log")
    [ "${waits:-0}" -ge 1 ] || fail "workload $n never waited for the gpu"
done

registered=$(grep -c "Registered client" "$LOGS/scheduler.log")
drops=$(grep -c "Sent DROP_LOCK" "$LOGS/scheduler.log")
[ "$registered" -eq 2 ] || fail "scheduler registered $registered clients, expected 2"
[ "$drops" -ge 1 ] || fail "scheduler never sent DROP_LOCK"
[ "$SCHED_RC" -eq 0 ] || fail "scheduler exited with $SCHED_RC"

if [ "$failed" -ne 0 ]; then
    exit 1
fi
echo "PASS: both workloads shared the gpu ($drops lock handovers)"
