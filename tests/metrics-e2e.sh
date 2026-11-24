#!/bin/bash
#
# End-to-end test of the metrics path without a GPU: the real scheduler and
# device plugin, driven by fake clients. The scheduler is built with the
# address and undefined-behaviour sanitizers. Needs root (it uses
# /var/run/nvshare and /var/lib/kubelet/device-plugins), gcc, make, go and
# curl; run it in a container, e.g.
#   docker run --rm -v "$PWD":/src -w /src golang:1.27 tests/metrics-e2e.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
NODE=e2e-node
PIDS=()
export GOFLAGS=-buildvcs=false

show_logs() {
    for f in "$WORK"/*.log "$WORK"/metrics.txt; do
        [ -f "$f" ] || continue
        echo "::group::$(basename "$f")"
        cat "$f"
        echo "::endgroup::"
    done
}

cleanup() {
    [ ${#PIDS[@]} -eq 0 ] || kill "${PIDS[@]}" 2> /dev/null || true
    wait 2> /dev/null || true
    rm -rf "$WORK"
}
trap 'echo "FAIL: line $LINENO"; show_logs' ERR
trap cleanup EXIT

pod() {
    echo "nvshare_pod_gpu_$1{gpu_uuid=\"GPU-e2e\",namespace=\"$2\",node=\"$NODE\",pod=\"$3\"${4:+,reason=\"$4\"}}"
}

sched() {
    echo "nvshare_scheduler_$1{node=\"$NODE\"}"
}

rejected() {
    echo "nvshare_device_plugin_metrics_rejected_total{node=\"$NODE\",reason=\"$1\"}"
}

scrape() {
    curl -fsS http://localhost:8080/metrics > "$WORK/metrics.txt"
}

value() {
    awk -v series="$1" '$1 == series { print $2; found = 1 }
        END { if (!found) print "missing" }' "$WORK/metrics.txt"
}

expect_eq() {
    local got
    got=$(value "$1")
    if [ "$got" != "$2" ]; then
        echo "FAIL: $1 is $got, expected $2"
        return 1
    fi
    echo "ok: $1 = $got"
}

expect_between() {
    local got
    got=$(value "$1")
    if ! awk -v v="$got" -v lo="$2" -v hi="$3" 'BEGIN { exit !(v ~ /^[0-9.e+-]+$/ && v >= lo && v <= hi) }'; then
        echo "FAIL: $1 is $got, expected between $2 and $3"
        return 1
    fi
    echo "ok: $1 = $got"
}

start_device_plugin() {
    NVIDIA_VISIBLE_DEVICES=GPU-e2e NVSHARE_VIRTUAL_DEVICES=2 NODE_NAME=$NODE \
        "$WORK/nvshare-device-plugin" >> "$WORK/device-plugin.log" 2>&1 &
    PLUGIN_PID=$!
    PIDS+=("$PLUGIN_PID")
    for _ in $(seq 50); do
        scrape 2> /dev/null && return 0
        sleep 0.2
    done
    echo "FAIL: device plugin metrics endpoint did not come up"
    return 1
}

make -C "$ROOT/src" nvsharectl > /dev/null
gcc -fsanitize=address,undefined -fno-sanitize-recover=undefined -g -O1 -std=gnu99 \
    -Wall -Wextra -Werror -I"$ROOT/src" "$ROOT/src/scheduler.c" "$ROOT/src/metrics.c" \
    "$ROOT/src/pod_identity.c" "$ROOT/src/comm.c" "$ROOT/src/common.c" -o "$WORK/nvshare-scheduler" -lpthread
gcc -Wall -Wextra -Werror -I"$ROOT/src" "$ROOT/tests/fake-client.c" \
    "$ROOT/src/comm.o" "$ROOT/src/common.o" -o "$WORK/fake-client"
(cd "$ROOT/kubernetes/device-plugin" && go build -o "$WORK/nvshare-device-plugin" .)

mkdir -p /var/run/nvshare /var/lib/kubelet/device-plugins

ASAN_OPTIONS=detect_leaks=0 NVSHARE_TQ=7 "$WORK/nvshare-scheduler" > "$WORK/scheduler.log" 2>&1 &
SCHED_PID=$!
PIDS+=("$SCHED_PID")
start_device_plugin
for _ in $(seq 50); do
    grep -q "listening on" "$WORK/scheduler.log" && break
    sleep 0.1
done
grep -q "TQ = 7" "$WORK/scheduler.log"
grep -q "Parallel GPU use = ON" "$WORK/scheduler.log"
"$ROOT/src/nvsharectl" -T 2 > "$WORK/nvsharectl.log" 2>&1

echo "== pod-a holds the lock, pod-b queues behind it"
"$WORK/fake-client" team-a pod-a 30 &
A=$!
sleep 0.5
"$WORK/fake-client" team-b pod-b 1 &
B=$!
sleep 0.5
scrape
expect_eq "$(pod lock_holding team-a pod-a)" 1
expect_eq "$(pod lock_waiting team-a pod-a)" 0
expect_eq "$(pod lock_holding team-b pod-b)" 0
expect_eq "$(pod lock_waiting team-b pod-b)" 1
expect_eq "$(pod clients team-b pod-b)" 1
expect_eq "$(sched lock_queue_length)" 1
expect_eq "$(sched clients)" 2
expect_eq "$(sched time_quantum_seconds)" 2
expect_eq "$(sched enabled)" 1
expect_eq "$(sched parallel)" 0

echo "== pod-a is preempted after the time quantum, pod-b releases on its own, pod-c exits while holding"
wait "$A" "$B"
"$WORK/fake-client" team-c pod-c 1 exit
sleep 0.5
scrape
expect_between "$(pod lock_held_seconds_total team-a pod-a)" 1.8 2.6
expect_eq "$(pod lock_wait_seconds_total team-a pod-a)" 0
expect_eq "$(pod locks_acquired_total team-a pod-a)" 1
expect_eq "$(pod lock_releases_total team-a pod-a quantum)" 1
expect_eq "$(pod lock_releases_total team-a pod-a idle)" 0
expect_eq "$(pod lock_releases_total team-a pod-a exit)" 0

expect_between "$(pod lock_wait_seconds_total team-b pod-b)" 1.2 2.2
expect_between "$(pod lock_held_seconds_total team-b pod-b)" 0.9 1.6
expect_eq "$(pod locks_acquired_total team-b pod-b)" 1
expect_eq "$(pod lock_releases_total team-b pod-b idle)" 1
expect_eq "$(pod lock_releases_total team-b pod-b quantum)" 0

expect_between "$(pod lock_held_seconds_total team-c pod-c)" 0.9 1.6
expect_eq "$(pod lock_releases_total team-c pod-c exit)" 1
expect_eq "$(pod lock_holding team-c pod-c)" 0
expect_eq "$(pod clients team-c pod-c)" 0

expect_eq "$(sched lock_queue_length)" 0
expect_eq "$(sched clients)" 0
expect_eq "$(sched metrics_dropped_total)" 0
expect_eq "$(rejected version)" 0
expect_eq "$(rejected malformed)" 0

echo "== pod-d keeps the lock past the time quantum while nobody waits, and gives it up as soon as pod-e asks"
"$WORK/fake-client" team-d pod-d 30 &
D=$!
sleep 3
scrape
expect_eq "$(pod lock_holding team-d pod-d)" 1
expect_eq "$(pod lock_releases_total team-d pod-d quantum)" 0
"$WORK/fake-client" team-e pod-e 1 &
E=$!
wait "$D" "$E"
sleep 0.5
scrape
expect_between "$(pod lock_held_seconds_total team-d pod-d)" 2.9 3.9
expect_eq "$(pod locks_acquired_total team-d pod-d)" 1
expect_eq "$(pod lock_releases_total team-d pod-d quantum)" 1
expect_between "$(pod lock_wait_seconds_total team-e pod-e)" 0 0.5
expect_between "$(pod lock_held_seconds_total team-e pod-e)" 0.9 1.6
expect_eq "$(pod lock_releases_total team-e pod-e idle)" 1

echo "== pod-h and pod-i fit in GPU memory together and both hold the lock, past the time quantum"
GPU_MIB=16000 MEM_MIB=4000 "$WORK/fake-client" team-h pod-h 30 &
H=$!
sleep 0.3
GPU_MIB=16000 MEM_MIB=4000 "$WORK/fake-client" team-i pod-i 30 &
I=$!
sleep 2.7
scrape
expect_eq "$(sched parallel)" 1
expect_eq "$(sched lock_queue_length)" 0
expect_eq "$(pod lock_holding team-h pod-h)" 1
expect_eq "$(pod lock_holding team-i pod-i)" 1
expect_between "$(pod lock_wait_seconds_total team-i pod-i)" 0 0.1
expect_eq "$(pod memory_allocated_bytes team-i pod-i)" 4.194304e+09
expect_eq "$(pod lock_releases_total team-h pod-h quantum)" 0

echo "== pod-j does not fit next to them: pod-i gives the lock back, and so does pod-h, whose time quantum is over"
GPU_MIB=16000 MEM_MIB=10000 "$WORK/fake-client" team-j pod-j 1 &
J=$!
wait "$H" "$I" "$J"
sleep 0.5
scrape
expect_eq "$(pod lock_releases_total team-i pod-i memory)" 1
expect_eq "$(pod lock_releases_total team-i pod-i quantum)" 0
expect_eq "$(pod lock_releases_total team-h pod-h quantum)" 1
expect_eq "$(pod lock_releases_total team-h pod-h memory)" 0
expect_between "$(pod lock_wait_seconds_total team-j pod-j)" 0 0.5
expect_eq "$(pod lock_releases_total team-j pod-j idle)" 1
expect_eq "$(sched clients)" 0

echo "== pod-m and pod-n do not fit in GPU memory together and take turns"
GPU_MIB=16000 MEM_MIB=4000 "$WORK/fake-client" team-m pod-m 30 &
M=$!
sleep 0.5
GPU_MIB=16000 MEM_MIB=10000 "$WORK/fake-client" team-n pod-n 1 &
N=$!
sleep 0.5
scrape
expect_eq "$(sched parallel)" 0
expect_eq "$(pod lock_holding team-m pod-m)" 1
expect_eq "$(pod lock_holding team-n pod-n)" 0
expect_eq "$(pod lock_waiting team-n pod-n)" 1
wait "$M" "$N"
sleep 0.5
scrape
expect_eq "$(pod lock_releases_total team-m pod-m quantum)" 1
expect_between "$(pod lock_wait_seconds_total team-n pod-n)" 1.2 2.2
expect_eq "$(sched parallel)" 1
expect_eq "$(sched clients)" 0

echo "== with parallel GPU use turned off, pods that fit together take turns until it is turned on again"
"$ROOT/src/nvsharectl" -P off >> "$WORK/nvsharectl.log" 2>&1
GPU_MIB=16000 MEM_MIB=4000 "$WORK/fake-client" team-k pod-k 3 &
K=$!
sleep 0.3
GPU_MIB=16000 MEM_MIB=4000 "$WORK/fake-client" team-l pod-l 1 &
L=$!
sleep 0.4
scrape
expect_eq "$(sched parallel)" 0
expect_eq "$(pod lock_holding team-k pod-k)" 1
expect_eq "$(pod lock_waiting team-l pod-l)" 1
"$ROOT/src/nvsharectl" -P on >> "$WORK/nvsharectl.log" 2>&1
sleep 0.4
scrape
expect_eq "$(sched parallel)" 1
expect_eq "$(pod lock_holding team-k pod-k)" 1
expect_eq "$(pod lock_holding team-l pod-l)" 1
wait "$K" "$L"

echo "== only the owner can write to the metrics socket"
mode=$(stat -c %a /var/run/nvshare/metrics.sock)
if [ "$mode" != 600 ]; then
    echo "FAIL: metrics.sock has mode $mode, expected 600"
    exit 1
fi
curl -fsS http://localhost:8080/healthz

echo "== a restarted device plugin gets the same totals back from the scheduler"
held_a=$(value "$(pod lock_held_seconds_total team-a pod-a)")
wait_b=$(value "$(pod lock_wait_seconds_total team-b pod-b)")
kill "$PLUGIN_PID"
wait "$PLUGIN_PID" 2> /dev/null || true
start_device_plugin
for _ in $(seq 100); do
    scrape
    [ "$(value "$(pod lock_held_seconds_total team-a pod-a)")" != missing ] && break
    sleep 0.2
done
expect_eq "$(pod lock_held_seconds_total team-a pod-a)" "$held_a"
expect_eq "$(pod lock_wait_seconds_total team-b pod-b)" "$wait_b"
expect_eq "$(pod lock_releases_total team-c pod-c exit)" 1

echo "== the scheduler survives a waiting client that dies just before it is given the lock"
"$WORK/fake-client" team-f pod-f 1 &
F=$!
sleep 0.3
"$WORK/fake-client" team-g pod-g 1 &
G=$!
sleep 0.3
kill -STOP "$SCHED_PID"
wait "$F"
kill -KILL "$G"
wait "$G" 2> /dev/null || true
kill -CONT "$SCHED_PID"
sleep 0.5
kill -0 "$SCHED_PID"
scrape
expect_eq "$(pod clients team-g pod-g)" 0
expect_eq "$(sched clients)" 0
expect_eq "$(sched lock_queue_length)" 0

echo "== the scheduler stops cleanly and the sanitizers found nothing"
kill "$SCHED_PID"
wait "$SCHED_PID"
if grep -E "Sanitizer|runtime error" "$WORK/scheduler.log"; then
    echo "FAIL: the sanitizers reported a problem in the scheduler"
    show_logs
    exit 1
fi

echo "PASS: metrics match what the clients did and survive a device plugin restart"
