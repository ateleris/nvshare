package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	metricsSocketPath = "/var/run/nvshare/metrics.sock"

	msgLockAcquired int32 = 100
	msgLockReleased int32 = 101
)

type activeHold struct {
	Namespace string
	PodName   string
	Container string
	DeviceID  string
	Node      string
	StartTime time.Time
}

type MetricsCollector struct {
	mutex       sync.Mutex
	activeHolds map[uint64]*activeHold
	metricsPort int

	lockHeldSeconds *prometheus.CounterVec
	locksAcquired   *prometheus.CounterVec
}

func NewMetricsCollector(port int) *MetricsCollector {
	mc := &MetricsCollector{
		activeHolds: make(map[uint64]*activeHold),
		metricsPort: port,

		lockHeldSeconds: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "nvshare_pod_gpu_lock_held_seconds_total",
				Help: "Total seconds each pod has held the exclusive nvshare GPU lock.",
			},
			[]string{"namespace", "pod", "container", "gpu_device", "node"},
		),

		locksAcquired: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "nvshare_pod_gpu_locks_acquired_total",
				Help: "Total number of times each pod has acquired the nvshare GPU lock.",
			},
			[]string{"namespace", "pod", "container", "gpu_device", "node"},
		),
	}

	prometheus.MustRegister(mc.lockHeldSeconds)
	prometheus.MustRegister(mc.locksAcquired)

	return mc
}

func nodeName() string {
	n := os.Getenv("NODE_NAME")
	if n == "" {
		return "unknown"
	}
	return n
}

func (mc *MetricsCollector) StartMetricsServer() error {
	http.Handle("/metrics", promhttp.Handler())
	go func() {
		log.Printf("Starting metrics server on :%d", mc.metricsPort)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", mc.metricsPort), nil); err != nil {
			log.Printf("Metrics server failed: %v", err)
		}
	}()
	go mc.startMetricsListener()
	return nil
}

type metricsMessage struct {
	Type      int32
	Namespace [254]byte
	PodName   [254]byte
	Container [128]byte
	DeviceID  [32]byte
	ProcessID int32
	ClientID  uint64
	Timestamp int64
}

func (mc *MetricsCollector) startMetricsListener() {
	os.Remove(metricsSocketPath)

	addr, err := net.ResolveUnixAddr("unixgram", metricsSocketPath)
	if err != nil {
		log.Printf("ERROR: resolve unix addr: %v", err)
		return
	}

	conn, err := net.ListenUnixgram("unixgram", addr)
	if err != nil {
		log.Printf("ERROR: listen unixgram: %v", err)
		return
	}
	defer conn.Close()

	if err := os.Chmod(metricsSocketPath, 0666); err != nil {
		log.Printf("WARN: chmod %s: %v", metricsSocketPath, err)
	}

	log.Printf("Metrics listener ready on %s", metricsSocketPath)

	buf := make([]byte, 2048)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			log.Printf("ERROR: read metrics socket: %v", err)
			continue
		}
		if n < int(unsafe.Sizeof(metricsMessage{})) {
			log.Printf("WARN: short metrics message: %d bytes", n)
			continue
		}
		msg := (*metricsMessage)(unsafe.Pointer(&buf[0]))
		mc.handleMessage(msg)
	}
}

func trimZero(b []byte) string {
	return strings.TrimRight(string(b), "\x00")
}

func (mc *MetricsCollector) handleMessage(msg *metricsMessage) {
	namespace := trimZero(msg.Namespace[:])
	pod := trimZero(msg.PodName[:])
	container := trimZero(msg.Container[:])
	device := trimZero(msg.DeviceID[:])

	if namespace == "" || pod == "" {
		log.Printf("WARN: metrics message missing namespace/pod")
		return
	}

	switch msg.Type {
	case msgLockAcquired:
		mc.handleAcquire(msg.ClientID, namespace, pod, container, device)
	case msgLockReleased:
		mc.handleRelease(msg.ClientID, namespace, pod, container, device, msg.Timestamp)
	default:
		log.Printf("WARN: unknown metrics message type %d", msg.Type)
	}
}

func (mc *MetricsCollector) handleAcquire(clientID uint64, namespace, pod, container, device string) {
	n := nodeName()
	mc.mutex.Lock()
	mc.activeHolds[clientID] = &activeHold{
		Namespace: namespace,
		PodName:   pod,
		Container: container,
		DeviceID:  device,
		Node:      n,
		StartTime: time.Now(),
	}
	mc.mutex.Unlock()
	mc.locksAcquired.WithLabelValues(namespace, pod, container, device, n).Inc()
	log.Printf("acquire %s/%s (client=%x)", namespace, pod, clientID)
}

func (mc *MetricsCollector) handleRelease(clientID uint64, namespace, pod, container, device string, durationSec int64) {
	n := nodeName()

	mc.mutex.Lock()
	hold, had := mc.activeHolds[clientID]
	if had {
		delete(mc.activeHolds, clientID)
	}
	mc.mutex.Unlock()

	// Prefer the duration computed by the scheduler (authoritative; survives
	// device-plugin restarts mid-hold). Fall back to local elapsed time.
	seconds := float64(durationSec)
	if seconds <= 0 && had {
		seconds = time.Since(hold.StartTime).Seconds()
	}
	if seconds <= 0 {
		log.Printf("release %s/%s: zero duration, skipping", namespace, pod)
		return
	}

	mc.lockHeldSeconds.WithLabelValues(namespace, pod, container, device, n).Add(seconds)
	log.Printf("release %s/%s +%.2fs (client=%x)", namespace, pod, seconds, clientID)
}
