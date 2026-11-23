# nvshare metrics

nvshare exports Prometheus metrics about who holds the GPU, for how long, and who is waiting for it.

## How it works

The scheduler serializes GPU access with a lock, so it knows exactly which pod holds the GPU at any moment. It keeps cumulative totals per pod and sends them over a Unix datagram socket (`/var/run/nvshare/metrics.sock`) to the device plugin on the same node, which serves them on `:8080/metrics`.

The scheduler sends a pod's totals on every change and re-sends everything every 10 seconds. The messages carry state, not events: a lost message or a restarted device plugin is repaired by the next refresh, and no time is lost.

## Metrics

Per pod, with labels `namespace`, `pod`, `gpu_device`, `node`:

| Metric | Type | Meaning |
|---|---|---|
| `nvshare_pod_gpu_lock_held_seconds_total` | counter | Time the pod held the GPU lock. The basis for billing. |
| `nvshare_pod_gpu_lock_wait_seconds_total` | counter | Time the pod spent queued for the lock. |
| `nvshare_pod_gpu_locks_acquired_total` | counter | Number of times the pod got the lock. |
| `nvshare_pod_gpu_lock_releases_total` | counter | Number of times the pod gave up the lock, by `reason`. |
| `nvshare_pod_gpu_lock_holding` | gauge | 1 while the pod holds the lock. |
| `nvshare_pod_gpu_lock_waiting` | gauge | Number of the pod's processes queued for the lock. |
| `nvshare_pod_gpu_clients` | gauge | Number of the pod's processes connected to the scheduler. |

Release reasons:

| `reason` | Meaning |
|---|---|
| `quantum` | The time quantum expired and the scheduler asked for the lock back. |
| `idle` | The pod released the lock on its own because it had no GPU work. |
| `exit` | The process went away while holding the lock. |
| `off` | The scheduler was turned off or stopped. |

Per node, with label `node`:

| Metric | Type | Meaning |
|---|---|---|
| `nvshare_scheduler_enabled` | gauge | 1 if the scheduler is on, 0 if it was turned off with `nvsharectl`. |
| `nvshare_scheduler_time_quantum_seconds` | gauge | Current time quantum. |
| `nvshare_scheduler_clients` | gauge | Processes connected to the scheduler. |
| `nvshare_scheduler_lock_queue_length` | gauge | Processes queued for the lock right now. |
| `nvshare_scheduler_metrics_dropped_total` | counter | Messages the scheduler could not deliver to the device plugin. |
| `nvshare_scheduler_last_update_timestamp_seconds` | gauge | When the device plugin last heard from the scheduler. |

## What to know when using them

- **Lock time is exclusive.** Only one pod holds the lock at a time, so the held time of all pods on a node adds up to at most one GPU-second per second.
- **Counters include the hold in progress** and advance at least every 10 seconds.
- **Counters restart when the scheduler restarts**, not when the device plugin restarts. `increase()` and `rate()` handle the reset.
- **A pod's series disappear** about five minutes after its last process disconnected.
- **Nothing is counted while the scheduler is off.** With `nvshare_scheduler_enabled == 0` all pods use the GPU freely and no lock is held.
- **Pod identity is self-reported.** The client library reads the pod name from `HOSTNAME` and the namespace from the service account mount. Both are `none` outside Kubernetes, the namespace is `none` without a mounted service account token, and a pod can report a false identity.
- **`gpu_device` is always `nvidia0`.** nvshare manages one GPU per node; use `node` to tell GPUs apart.

## Scraping

`kubernetes/manifests/metrics-service.yaml` creates a headless Service `nvshare-metrics` in `nvshare-system` for port 8080.

The metrics carry their own `namespace` and `pod` labels. If your scrape configuration also attaches `namespace` and `pod` labels for the scraped target, Prometheus renames these to `exported_namespace` and `exported_pod`. Set `honor_labels: true` (`honorLabels: true` in a ServiceMonitor) for this job to keep them.

## Example queries

```promql
# Share of the GPU each pod used (0..1)
rate(nvshare_pod_gpu_lock_held_seconds_total[5m])

# GPU-hours per namespace over 30 days
sum by (namespace) (increase(nvshare_pod_gpu_lock_held_seconds_total[30d])) / 3600

# How busy each GPU is (0..1)
sum by (node) (rate(nvshare_pod_gpu_lock_held_seconds_total[5m]))

# Average wait before a pod gets the GPU
rate(nvshare_pod_gpu_lock_wait_seconds_total[5m]) / rate(nvshare_pod_gpu_locks_acquired_total[5m])

# Share of its time a pod spends waiting instead of computing
rate(nvshare_pod_gpu_lock_wait_seconds_total[5m])
  / (rate(nvshare_pod_gpu_lock_wait_seconds_total[5m]) + rate(nvshare_pod_gpu_lock_held_seconds_total[5m]))

# Preemptions per minute
rate(nvshare_pod_gpu_lock_releases_total{reason="quantum"}[5m]) * 60

# Scheduler silent for more than a minute
time() - nvshare_scheduler_last_update_timestamp_seconds > 60
```

## Upgrading

The scheduler and the device plugin must speak the same protocol version (`METRICS_PROTOCOL_VERSION` in `src/metrics.h`, `protocolVersion` in `kubernetes/device-plugin/metrics.go`). A mismatch is safe but exports no nvshare metrics, and the device plugin logs `unsupported protocol version`. Roll out both DaemonSets together.

## Tests

- `tests/metrics-e2e.sh` runs the real scheduler and device plugin with scripted fake clients and checks the exported values, including a device plugin restart. It needs no GPU.
- `go test` in `kubernetes/device-plugin` checks the wire format against files generated from the C structs (`testdata/gen_wire_golden.c`) and the exported metrics.
