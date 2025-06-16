# NVshare Pod-Level GPU Metrics Implementation

This implementation adds per-pod GPU metrics to nvshare, enabling accurate billing per tenant application in Kubernetes.

## Architecture Overview

The implementation consists of three main components:

1. **Scheduler Enhancement**: Modified `nvshare-scheduler` tracks GPU sessions per pod
2. **Metrics Collector**: New Go-based metrics system in the device plugin
3. **Unix Socket Communication**: Scheduler communicates GPU events to metrics collector

## Exported Metrics

The following Prometheus metrics are now available:

### Primary Metrics

```prometheus
# GPU utilization percentage per pod
nvshare_pod_gpu_utilization_percent{namespace="tenant-1-subscription-10", pod="service-ollama-59969bb7b8-r4qpb", container="container", gpu_device="nvidia0", node="worker-node-1"}

# GPU memory usage in bytes per pod  
nvshare_pod_gpu_memory_used_bytes{namespace="tenant-1-subscription-10", pod="service-ollama-59969bb7b8-r4qpb", container="container", gpu_device="nvidia0", node="worker-node-1"}

# Whether pod has active GPU session (1=active, 0=inactive)
nvshare_pod_gpu_session_active{namespace="tenant-1-subscription-10", pod="service-ollama-59969bb7b8-r4qpb", container="container", gpu_device="nvidia0", node="worker-node-1"}
```

## Implementation Details

### 1. Pod Context Tracking

The scheduler now tracks active GPU sessions with pod metadata:

```c
struct pod_gpu_session {
    char namespace[POD_NAMESPACE_LEN_MAX];
    char pod_name[POD_NAME_LEN_MAX];
    char container[128];
    char device_id[MAX_DEVICE_ID_LEN];
    int process_id;
    uint64_t client_id;
    time_t start_time;
    time_t last_active;
};
```

### 2. Metrics Communication

- **Protocol**: Unix datagram socket (`/var/run/nvshare/metrics.sock`)
- **Events**: Session start, session end, session update
- **Message Format**: Binary struct with pod metadata

### 3. Data Collection

- **Utilization**: Parsed from `/proc/driver/nvidia/gpus/*/information`
- **Memory**: Extracted from process memory stats and GPU memory usage
- **Update Frequency**: Every 30 seconds

## File Changes

### New Files

- `src/metrics.h` - Metrics system header
- `src/metrics.c` - Metrics implementation
- `src/nvshare_types.h` - Shared type definitions
- `kubernetes/device-plugin/metrics.go` - Go metrics collector
- `kubernetes/manifests/metrics-service.yaml` - Kubernetes service

### Modified Files

- `src/scheduler.c` - Added metrics tracking hooks
- `src/Makefile` - Added metrics.o build target
- `kubernetes/device-plugin/main.go` - Integrated metrics server
- `kubernetes/device-plugin/go.mod` - Added Prometheus dependency
- `kubernetes/manifests/device-plugin.yaml` - Added metrics port and NODE_NAME env var

## Deployment

### 1. Build Updated Components

```bash
# Build scheduler
cd src
make clean && make nvshare-scheduler

# Build device plugin
cd ../kubernetes/device-plugin
go build .
```

### 2. Update Container Images

Build and push updated container images with the new binaries.

### 3. Deploy Manifests

```bash
kubectl apply -f kubernetes/manifests/device-plugin.yaml
kubectl apply -f kubernetes/manifests/metrics-service.yaml
```

### 4. Configure Prometheus

Add service discovery or manual scrape config:

```yaml
- job_name: 'nvshare-metrics'
  kubernetes_sd_configs:
    - role: endpoints
      namespaces:
        names:
          - nvshare-system
  relabel_configs:
    - source_labels: [__meta_kubernetes_service_name]
      action: keep
      regex: nvshare-metrics
```

## Testing

Run the included test script:

```bash
./test-metrics.sh
```

Expected output includes:
- Successful component builds
- Metrics server startup on port 8080
- Unix socket creation at `/var/run/nvshare/metrics.sock`
- Metrics endpoint responding at `http://localhost:8080/metrics`

## Integration with Bazar

Once deployed, the metrics will be available for Prometheus scraping:

```promql
# Query GPU utilization by tenant
nvshare_pod_gpu_utilization_percent{namespace=~"tenant-.*-subscription-.*"}

# Query GPU memory usage by tenant  
nvshare_pod_gpu_memory_used_bytes{namespace=~"tenant-.*-subscription-.*"}

# Query active GPU sessions by tenant
nvshare_pod_gpu_session_active{namespace=~"tenant-.*-subscription-.*"}
```

## Limitations and Notes

1. **GPU Metrics Source**: Currently uses fallback values; integration with NVIDIA DCGM recommended for production
2. **Process Tracking**: Uses simplified process ID mapping; could be enhanced with cgroup analysis
3. **Error Handling**: Metrics collection is non-blocking; failures won't affect GPU scheduling
4. **Memory Usage**: Minimal overhead; session tracking uses small in-memory structures

## Future Enhancements

1. **DCGM Integration**: Direct integration with NVIDIA Data Center GPU Manager
2. **Process-Level Tracking**: Enhanced process-to-pod mapping via cgroups
3. **Historical Data**: Optional persistence of session history
4. **Custom Metrics**: Configurable additional metrics per deployment

## Troubleshooting

### Common Issues

1. **Metrics not appearing**: Check metrics socket permissions and device plugin logs
2. **Incorrect pod attribution**: Verify NODE_NAME environment variable is set
3. **Missing utilization data**: Check access to `/proc/driver/nvidia/` directory

### Debug Commands

```bash
# Check metrics socket
ls -la /var/run/nvshare/metrics.sock

# Test metrics endpoint
curl http://localhost:8080/metrics | grep nvshare_pod

# Check device plugin logs
kubectl logs -n nvshare-system -l name=nvshare-device-plugin

# Check scheduler logs  
journalctl -u nvshare-scheduler
```