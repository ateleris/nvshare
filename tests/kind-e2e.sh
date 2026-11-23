#!/bin/bash
set -euo pipefail

CLUSTER=nvshare-e2e
IMAGE=nvshare
NS=nvshare-system
ROOT=$(cd "$(dirname "$0")/.." && pwd)
KUBECONFIG=$(mktemp)
export KUBECONFIG

diagnose() {
    echo "FAIL: line $1"
    kubectl get nodes -o wide || true
    kubectl get pods -A -o wide || true
    kubectl -n "$NS" describe pods || true
    kubectl -n "$NS" logs -l name=nvshare-scheduler --all-containers --tail=100 || true
    kubectl -n "$NS" logs -l name=nvshare-device-plugin --all-containers --tail=100 || true
    kubectl describe pod nvshare-e2e || true
}

cleanup() {
    kind delete cluster --name "$CLUSTER" > /dev/null 2>&1 || true
    rm -f "$KUBECONFIG"
}
trap 'diagnose $LINENO' ERR
trap cleanup EXIT

cd "$ROOT"
docker build -q -f Dockerfile.scheduler -t "$IMAGE:scheduler-e2e" .
docker build -q -f Dockerfile.device_plugin -t "$IMAGE:device-plugin-e2e" .
docker build -q -f Dockerfile.libnvshare -t "$IMAGE:libnvshare-e2e" .
docker build -q -f tests/dockerfiles/Dockerfile.fakeclient -t "$IMAGE:fake-client-e2e" .

kind create cluster --name "$CLUSTER" --wait 120s
kind load docker-image --name "$CLUSTER" \
    "$IMAGE:scheduler-e2e" "$IMAGE:device-plugin-e2e" "$IMAGE:libnvshare-e2e" \
    "$IMAGE:fake-client-e2e"

use_local_images() {
    sed "s|ghcr.io/ateleris/nvshare:\([a-z-]*\)-v[0-9][0-9.]*|$IMAGE:\1-e2e|" "$1"
}

kubectl apply -f kubernetes/manifests/nvshare-system.yaml
kubectl apply -f kubernetes/manifests/nvshare-system-quotas.yaml
kubectl apply -f kubernetes/manifests/metrics-service.yaml
use_local_images kubernetes/manifests/scheduler.yaml | kubectl apply -f -
use_local_images kubernetes/manifests/device-plugin.yaml \
    | kubectl patch --local -f - --type=json -o yaml -p '[
        {"op": "remove", "path": "/spec/template/spec/containers/1/resources/limits/nvidia.com~1gpu"},
        {"op": "add", "path": "/spec/template/spec/containers/1/env/-",
         "value": {"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-e2e"}}]' \
    | kubectl apply -f -

kubectl -n "$NS" rollout status ds/nvshare-scheduler --timeout=180s
kubectl -n "$NS" rollout status ds/nvshare-device-plugin --timeout=180s

gpus=
for _ in $(seq 60); do
    gpus=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvshare\.com/gpu}')
    [ "$gpus" = 10 ] && break
    sleep 2
done
echo "node advertises nvshare.com/gpu: ${gpus:-none}"
[ "$gpus" = 10 ]

# shellcheck disable=SC2016
kubectl run nvshare-e2e --restart=Never --image="$IMAGE:scheduler-e2e" \
    --override-type=strategic \
    --overrides='{"apiVersion": "v1", "spec": {"containers": [{"name": "nvshare-e2e",
        "resources": {"limits": {"nvshare.com/gpu": "1"}}}]}}' \
    --command -- sh -c '
        echo "LD_PRELOAD=$LD_PRELOAD"
        echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
        test -f /usr/lib/nvshare/libnvshare.so
        test -S /var/run/nvshare/scheduler.sock
        nvsharectl -T 31'
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/nvshare-e2e --timeout=120s

pod_log=$(kubectl logs nvshare-e2e)
echo "$pod_log"
grep -q '^LD_PRELOAD=/usr/lib/nvshare/libnvshare.so$' <<< "$pod_log"
grep -q '^NVIDIA_VISIBLE_DEVICES=GPU-e2e$' <<< "$pod_log"

kubectl -n "$NS" logs -l name=nvshare-scheduler --tail=-1 | grep -q 'New TQ = 31'

plugin=$(kubectl -n "$NS" get pods -l name=nvshare-device-plugin -o jsonpath='{.items[0].metadata.name}')
kubectl -n "$NS" logs "$plugin" -c nvshare-device-plugin | grep -q 'Metrics listener ready'
metrics() {
    kubectl get --raw "/api/v1/namespaces/$NS/pods/$plugin:8080/proxy/metrics"
}
metrics | grep -q '^promhttp_metric_handler_requests_total'

# A client that claims to be another pod is reported as the pod it runs in.
kubectl run nvshare-identity --restart=Never --image="$IMAGE:fake-client-e2e" \
    --override-type=strategic \
    --overrides='{"apiVersion": "v1", "spec": {"containers": [{"name": "nvshare-identity",
        "resources": {"limits": {"nvshare.com/gpu": "1"}}}]}}' \
    -- other-namespace other-pod 1
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/nvshare-identity --timeout=120s

acquired=
for _ in $(seq 30); do
    acquired=$(metrics | grep '^nvshare_pod_gpu_locks_acquired_total' || true)
    [ -n "$acquired" ] && break
    sleep 1
done
echo "$acquired"
grep -q 'gpu_uuid="GPU-e2e",namespace="default",.*pod="nvshare-identity"} 1$' <<< "$acquired"
if metrics | grep -E 'other-namespace|other-pod'; then
    echo "FAIL: the scheduler trusted the pod identity the client reported"
    exit 1
fi

echo "PASS: device plugin registered, a pod got a virtual gpu and reached the scheduler under its real identity"
