# nvshare metrics

nvshare exports Prometheus metrics about who holds the GPU, for how long, and who is waiting for it.

## How it works

The scheduler serializes GPU access with a lock, so it knows exactly which pod holds the GPU at any moment. It keeps cumulative totals per pod and sends them over a Unix datagram socket (`/var/run/nvshare/metrics.sock`) to the device plugin on the same node, which serves them on `:8080/metrics`. Set `NVSHARE_METRICS_PORT` on the device plugin to use another port.

The scheduler sends a pod's totals on every change and re-sends everything every 10 seconds. The messages carry state, not events: a lost message or a restarted device plugin is repaired by the next refresh, and no time is lost.

## Metrics

Per pod, with labels `namespace`, `pod`, `gpu_uuid`, `node`:

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

From the device plugin itself, with labels `node`, `reason`:

| Metric | Type | Meaning |
|---|---|---|
| `nvshare_device_plugin_metrics_rejected_total` | counter | Scheduler messages the device plugin ignored. `reason="version"` means the scheduler and the device plugin speak different protocol versions, `reason="malformed"` covers everything else. |

## What to know when using them

- **Lock time is exclusive.** Only one pod holds the lock at a time, so the held time of all pods on a node adds up to at most one GPU-second per second.
- **Counters include the hold in progress** and advance at least every 10 seconds.
- **Counters restart when the scheduler restarts**, not when the device plugin restarts. `increase()` and `rate()` handle the reset.
- **A pod's series disappear** about five minutes after its last process disconnected.
- **Nothing is counted while the scheduler is off.** With `nvshare_scheduler_enabled == 0` all pods use the GPU freely and no lock is held.
- **`gpu_uuid` is the UUID of the physical GPU**, as long as the NVIDIA device plugin identifies GPUs by UUID (its default). It is then the same value the NVIDIA DCGM exporter puts in its `UUID` label.

## Pod identity

The client library reports a pod name (from `HOSTNAME`) and a namespace (from the service account mount), but a pod controls both. The scheduler therefore does not trust them on Kubernetes. When a client registers, the scheduler:

1. asks the kernel for the process at the other end of the socket (`SO_PEERCRED`),
2. reads the pod UID from that process's cgroup (`/proc/<pid>/cgroup`),
3. finds the pod's namespace and name in the kubelet's log directory (`/var/log/pods/<namespace>_<name>_<uid>`).

This needs `hostPID: true` and a read-only mount of `/var/log/pods` on the scheduler, both set in `kubernetes/manifests/scheduler.yaml`. What ends up in the labels:

| Situation | `namespace` / `pod` |
|---|---|
| Client runs in a pod | The pod's real namespace and name. |
| Client runs in a pod, but `/var/log/pods` has no entry for it (not mounted) | `unknown` / the pod UID. |
| Client does not run in a pod (local use without Kubernetes) | What the client reported, `none` / `none` by default. |
| Scheduler is not in the host PID namespace | What the client reported. **Not verified**: a pod can report a false identity. The scheduler logs a warning once. |

A client whose process is gone before the scheduler could inspect it is rejected.

The metrics socket (`/var/run/nvshare/metrics.sock`) has mode 0600, so the scheduler and the device plugin must run as the same user (root in the provided manifests), and no other user on the node can inject metrics.

## Scraping

`kubernetes/manifests/metrics-service.yaml` creates a headless Service `nvshare-metrics` in `nvshare-system` for port 8080.

The metrics carry their own `namespace` and `pod` labels. If your scrape configuration also attaches `namespace` and `pod` labels for the scraped target, Prometheus renames these to `exported_namespace` and `exported_pod`. Set `honor_labels: true` for this job to keep them.

With the Prometheus Operator, apply `kubernetes/manifests/metrics-servicemonitor.yaml`. It scrapes the Service with `honorLabels: true`.

The device plugin also answers `/healthz` on the same port, which the DaemonSet uses for its probes.

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

# Scheduler and device plugin run different versions
increase(nvshare_device_plugin_metrics_rejected_total{reason="version"}[5m]) > 0

# GPU memory in use next to the pod that holds the lock (with the DCGM exporter)
DCGM_FI_DEV_FB_USED
  * on (UUID) group_right
    label_replace(nvshare_pod_gpu_lock_holding == 1, "UUID", "$1", "gpu_uuid", "(.+)")
```

## Upgrading

The scheduler and the device plugin must speak the same protocol version (`METRICS_PROTOCOL_VERSION` in `src/metrics.h`, `protocolVersion` in `kubernetes/device-plugin/metrics.go`). A mismatch is safe but exports no pod or scheduler metrics: the device plugin logs `unsupported protocol version` and counts the messages in `nvshare_device_plugin_metrics_rejected_total{reason="version"}`.

The manifests pin all images to one release tag (`scheduler-vX.Y.Z`, `device-plugin-vX.Y.Z`, `libnvshare-vX.Y.Z`). Change all three together and roll out both DaemonSets together. Avoid the `-latest` tags: with `imagePullPolicy: IfNotPresent` every node keeps whatever it pulled first.

## Tests

- `tests/metrics-e2e.sh` runs the real scheduler and device plugin with scripted fake clients and checks the exported values, including a device plugin restart. It needs no GPU.
- `tests/kind-e2e.sh` deploys the manifests to a kind cluster and checks that a client claiming to be another pod is reported as the pod it runs in.
- `go test` in `kubernetes/device-plugin` checks the wire format against files generated from the C structs (`testdata/gen_wire_golden.c`) and the exported metrics.
