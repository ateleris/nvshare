package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	metricsSocketPath = "/var/run/nvshare/metrics.sock"

	// Wire protocol, kept in lockstep with src/metrics.h.
	protocolVersion uint32 = 1

	msgLockAcquired uint32 = 200
	msgLockReleased uint32 = 201

	// Fixed on-wire field widths. Must match src/comm.h + src/metrics.h.
	nsFieldLen     = 254
	podFieldLen    = 254
	deviceFieldLen = 32

	// version(4) + type(4) + ns(254) + pod(254) + device(32) + client_id(8) + duration_ms(8)
	metricsMsgSize = 4 + 4 + nsFieldLen + podFieldLen + deviceFieldLen + 8 + 8
)

type metricsMessage struct {
	Version    uint32
	Type       uint32
	Namespace  string
	PodName    string
	DeviceID   string
	ClientID   uint64
	DurationMS int64
}

type activeHold struct {
	Namespace string
	PodName   string
	DeviceID  string
	Node      string
	StartTime time.Time
}

type MetricsCollector struct {
	mutex       sync.Mutex
	activeHolds map[uint64]*activeHold
	metricsPort int

	// Labels: namespace, pod, gpu_device, node.
	//
	// `pod` is high-cardinality by design -- per-pod billing is the point.
	// Operators deploying in clusters with many short-lived pods sharing a
	// GPU should lower their scrape frequency or use relabel rules to
	// aggregate series before ingestion.
	lockHeldSeconds *prometheus.CounterVec
	locksAcquired   *prometheus.CounterVec
}

func NewMetricsCollector(port int) *MetricsCollector {
	labels := []string{"namespace", "pod", "gpu_device", "node"}

	mc := &MetricsCollector{
		activeHolds: make(map[uint64]*activeHold),
		metricsPort: port,

		lockHeldSeconds: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "nvshare_pod_gpu_lock_held_seconds_total",
				Help: "Total seconds each pod has held the exclusive nvshare GPU lock.",
			},
			labels,
		),

		locksAcquired: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "nvshare_pod_gpu_locks_acquired_total",
				Help: "Total number of times each pod has acquired the nvshare GPU lock.",
			},
			labels,
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

// parseMetricsMessage decodes the on-wire metrics message emitted by the
// scheduler (src/metrics.c). Fields are read explicitly in little-endian
// order so Go's natural struct alignment cannot diverge from C's packed
// layout.
func parseMetricsMessage(buf []byte) (*metricsMessage, error) {
	if len(buf) < metricsMsgSize {
		return nil, fmt.Errorf("short message: %d < %d bytes", len(buf), metricsMsgSize)
	}

	m := &metricsMessage{}
	off := 0

	m.Version = binary.LittleEndian.Uint32(buf[off:])
	off += 4
	if m.Version != protocolVersion {
		return nil, fmt.Errorf("unsupported protocol version: %d (expected %d)", m.Version, protocolVersion)
	}

	m.Type = binary.LittleEndian.Uint32(buf[off:])
	off += 4

	m.Namespace = trimZero(buf[off : off+nsFieldLen])
	off += nsFieldLen
	m.PodName = trimZero(buf[off : off+podFieldLen])
	off += podFieldLen
	m.DeviceID = trimZero(buf[off : off+deviceFieldLen])
	off += deviceFieldLen

	m.ClientID = binary.LittleEndian.Uint64(buf[off:])
	off += 8
	m.DurationMS = int64(binary.LittleEndian.Uint64(buf[off:]))

	return m, nil
}

func trimZero(b []byte) string {
	before, _, _ := bytes.Cut(b, []byte{0})
	return string(before)
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
			time.Sleep(100 * time.Millisecond)
			continue
		}
		msg, perr := parseMetricsMessage(buf[:n])
		if perr != nil {
			log.Printf("WARN: %v", perr)
			continue
		}
		mc.handleMessage(msg)
	}
}

func (mc *MetricsCollector) handleMessage(msg *metricsMessage) {
	if msg.Namespace == "" || msg.PodName == "" {
		log.Printf("WARN: metrics message missing namespace/pod")
		return
	}

	switch msg.Type {
	case msgLockAcquired:
		mc.handleAcquire(msg.ClientID, msg.Namespace, msg.PodName, msg.DeviceID)
	case msgLockReleased:
		mc.handleRelease(msg.ClientID, msg.Namespace, msg.PodName, msg.DeviceID, msg.DurationMS)
	default:
		log.Printf("WARN: unknown metrics message type %d", msg.Type)
	}
}

func (mc *MetricsCollector) handleAcquire(clientID uint64, namespace, pod, device string) {
	n := nodeName()
	mc.mutex.Lock()
	mc.activeHolds[clientID] = &activeHold{
		Namespace: namespace,
		PodName:   pod,
		DeviceID:  device,
		Node:      n,
		StartTime: time.Now(),
	}
	mc.mutex.Unlock()
	mc.locksAcquired.WithLabelValues(namespace, pod, device, n).Inc()
	log.Printf("acquire %s/%s (client=%x)", namespace, pod, clientID)
}

func (mc *MetricsCollector) handleRelease(clientID uint64, namespace, pod, device string, durationMS int64) {
	n := nodeName()

	mc.mutex.Lock()
	hold, had := mc.activeHolds[clientID]
	if had {
		delete(mc.activeHolds, clientID)
	}
	mc.mutex.Unlock()

	// Prefer the duration computed by the scheduler (authoritative; survives
	// device-plugin restarts mid-hold). Fall back to local elapsed time.
	seconds := float64(durationMS) / 1000.0
	if seconds <= 0 && had {
		seconds = time.Since(hold.StartTime).Seconds()
	}
	if seconds <= 0 {
		log.Printf("release %s/%s: zero duration, skipping", namespace, pod)
		return
	}

	mc.lockHeldSeconds.WithLabelValues(namespace, pod, device, n).Add(seconds)
	log.Printf("release %s/%s +%.3fs (client=%x)", namespace, pod, seconds, clientID)
}
