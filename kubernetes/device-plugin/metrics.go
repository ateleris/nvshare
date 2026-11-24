package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	metricsSocketPath = "/var/run/nvshare/metrics.sock"

	// Wire protocol, kept in lockstep with src/metrics.h.
	protocolVersion uint32 = 3

	msgPodStats   uint32 = 300
	msgSchedStats uint32 = 301

	nameFieldLen   = 254
	releaseReasons = 5

	// version(4) + type(4) + ns(254) + pod(254) + clients(4) + holding(4) +
	// waiting(4) + held_ms(8) + wait_ms(8) + acquisitions(8) + releases(5*8) +
	// mem_mib(8)
	podStatsSize = 4 + 4 + 2*nameFieldLen + 3*4 + 3*8 + releaseReasons*8 + 8

	// version(4) + type(4) + scheduler_on(4) + tq_sec(4) + clients(4) +
	// queue_length(4) + dropped_messages(8) + parallel(4)
	schedStatsSize = 6*4 + 8 + 4

	// The scheduler re-sends everything every 10 seconds. State that has
	// not been refreshed for this long is no longer exported.
	staleAfter = 60 * time.Second

	rejectedVersion   = 0
	rejectedMalformed = 1
	rejectReasons     = 2
)

var releaseReasonNames = [releaseReasons]string{"quantum", "idle", "exit", "off", "memory"}

var rejectReasonNames = [rejectReasons]string{"version", "malformed"}

var errProtocolVersion = errors.New("unsupported protocol version")

type podStats struct {
	Namespace    string
	Pod          string
	Clients      uint32
	Holding      uint32
	Waiting      uint32
	HeldMS       uint64
	WaitMS       uint64
	Acquisitions uint64
	Releases     [releaseReasons]uint64
	MemoryMiB    uint64
}

type schedStats struct {
	SchedulerOn     uint32
	TQSeconds       uint32
	Clients         uint32
	QueueLength     uint32
	DroppedMessages uint64
	Parallel        uint32
}

type podKey struct {
	namespace string
	pod       string
}

type podEntry struct {
	stats podStats
	seen  time.Time
}

// MetricsCollector exports the state the scheduler sends over the metrics
// socket. The scheduler owns all counters and sends cumulative values, so
// nothing is lost when a message is dropped or this process restarts.
type MetricsCollector struct {
	mutex       sync.Mutex
	pods        map[podKey]podEntry
	sched       schedStats
	schedSeen   time.Time
	rejected    [rejectReasons]uint64
	lastWarning time.Time

	metricsPort int
	node        string
	gpuUUID     string
	now         func() time.Time

	// Labels: namespace, pod, gpu_uuid, node.
	//
	// `pod` is high-cardinality by design -- per-pod billing is the point.
	lockHeldSeconds *prometheus.Desc
	lockWaitSeconds *prometheus.Desc
	locksAcquired   *prometheus.Desc
	lockReleases    *prometheus.Desc
	lockHolding     *prometheus.Desc
	lockWaiting     *prometheus.Desc
	podClients      *prometheus.Desc
	podMemory       *prometheus.Desc

	// Labels: node.
	schedEnabled     *prometheus.Desc
	schedParallel    *prometheus.Desc
	schedTimeQuantum *prometheus.Desc
	schedClients     *prometheus.Desc
	schedQueueLength *prometheus.Desc
	schedDropped     *prometheus.Desc
	schedLastUpdate  *prometheus.Desc

	// Labels: node, reason.
	messagesRejected *prometheus.Desc
}

func newMetricsCollector(port int, node, gpuUUID string, now func() time.Time) *MetricsCollector {
	podLabels := []string{"namespace", "pod", "gpu_uuid", "node"}
	nodeLabels := []string{"node"}

	return &MetricsCollector{
		pods:        make(map[podKey]podEntry),
		metricsPort: port,
		node:        node,
		gpuUUID:     gpuUUID,
		now:         now,

		lockHeldSeconds: prometheus.NewDesc(
			"nvshare_pod_gpu_lock_held_seconds_total",
			"Total seconds each pod has held the nvshare GPU lock.",
			podLabels, nil),
		lockWaitSeconds: prometheus.NewDesc(
			"nvshare_pod_gpu_lock_wait_seconds_total",
			"Total seconds each pod has spent queued for the nvshare GPU lock.",
			podLabels, nil),
		locksAcquired: prometheus.NewDesc(
			"nvshare_pod_gpu_locks_acquired_total",
			"Total number of times each pod has acquired the nvshare GPU lock.",
			podLabels, nil),
		lockReleases: prometheus.NewDesc(
			"nvshare_pod_gpu_lock_releases_total",
			"Total number of times each pod has given up the nvshare GPU lock, by reason: quantum (time quantum expired while another process was waiting), idle (released on its own), exit (process went away), off (scheduler turned off or stopped), memory (parallel GPU use ended).",
			append(podLabels, "reason"), nil),
		lockHolding: prometheus.NewDesc(
			"nvshare_pod_gpu_lock_holding",
			"1 if the pod currently holds the nvshare GPU lock.",
			podLabels, nil),
		lockWaiting: prometheus.NewDesc(
			"nvshare_pod_gpu_lock_waiting",
			"Number of the pod's processes currently queued for the nvshare GPU lock.",
			podLabels, nil),
		podClients: prometheus.NewDesc(
			"nvshare_pod_gpu_clients",
			"Number of the pod's processes connected to the nvshare scheduler.",
			podLabels, nil),
		podMemory: prometheus.NewDesc(
			"nvshare_pod_gpu_memory_allocated_bytes",
			"GPU memory the pod's processes have allocated, as they reported it to the nvshare scheduler.",
			podLabels, nil),

		schedEnabled: prometheus.NewDesc(
			"nvshare_scheduler_enabled",
			"1 if the nvshare scheduler is on (GPU access is serialized), 0 if it was turned off.",
			nodeLabels, nil),
		schedParallel: prometheus.NewDesc(
			"nvshare_scheduler_parallel",
			"1 if the GPU memory of all connected processes fits on the GPU together, so they may hold the nvshare GPU lock at the same time.",
			nodeLabels, nil),
		schedTimeQuantum: prometheus.NewDesc(
			"nvshare_scheduler_time_quantum_seconds",
			"Time quantum after which the lock holder is asked to release the GPU if another process is waiting for it.",
			nodeLabels, nil),
		schedClients: prometheus.NewDesc(
			"nvshare_scheduler_clients",
			"Number of processes connected to the nvshare scheduler.",
			nodeLabels, nil),
		schedQueueLength: prometheus.NewDesc(
			"nvshare_scheduler_lock_queue_length",
			"Number of processes currently queued for the nvshare GPU lock.",
			nodeLabels, nil),
		schedDropped: prometheus.NewDesc(
			"nvshare_scheduler_metrics_dropped_total",
			"Total number of metrics messages the scheduler could not deliver.",
			nodeLabels, nil),
		schedLastUpdate: prometheus.NewDesc(
			"nvshare_scheduler_last_update_timestamp_seconds",
			"Unix time of the last message received from the nvshare scheduler.",
			nodeLabels, nil),

		messagesRejected: prometheus.NewDesc(
			"nvshare_device_plugin_metrics_rejected_total",
			"Total number of scheduler messages the device plugin ignored, by reason: version (scheduler and device plugin speak different protocol versions), malformed.",
			append(nodeLabels, "reason"), nil),
	}
}

func NewMetricsCollector(port int, gpuUUID string) *MetricsCollector {
	mc := newMetricsCollector(port, nodeName(), gpuUUID, time.Now)
	prometheus.MustRegister(mc)
	return mc
}

func nodeName() string {
	n := os.Getenv("NODE_NAME")
	if n == "" {
		return "unknown"
	}
	return n
}

func (mc *MetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, desc := range []*prometheus.Desc{
		mc.lockHeldSeconds, mc.lockWaitSeconds, mc.locksAcquired, mc.lockReleases,
		mc.lockHolding, mc.lockWaiting, mc.podClients, mc.podMemory,
		mc.schedEnabled, mc.schedParallel, mc.schedTimeQuantum, mc.schedClients, mc.schedQueueLength,
		mc.schedDropped, mc.schedLastUpdate, mc.messagesRejected,
	} {
		ch <- desc
	}
}

func (mc *MetricsCollector) Collect(ch chan<- prometheus.Metric) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	now := mc.now()

	for key, entry := range mc.pods {
		if now.Sub(entry.seen) > staleAfter {
			delete(mc.pods, key)
			log.Printf("pod %s/%s no longer reported by the scheduler", key.namespace, key.pod)
			continue
		}

		s := entry.stats
		labels := []string{s.Namespace, s.Pod, mc.gpuUUID, mc.node}
		counter := func(desc *prometheus.Desc, v float64, extra ...string) {
			ch <- prometheus.MustNewConstMetric(desc, prometheus.CounterValue, v, append(labels, extra...)...)
		}
		gauge := func(desc *prometheus.Desc, v float64) {
			ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, v, labels...)
		}

		counter(mc.lockHeldSeconds, float64(s.HeldMS)/1000)
		counter(mc.lockWaitSeconds, float64(s.WaitMS)/1000)
		counter(mc.locksAcquired, float64(s.Acquisitions))
		for i, reason := range releaseReasonNames {
			counter(mc.lockReleases, float64(s.Releases[i]), reason)
		}
		gauge(mc.lockHolding, float64(s.Holding))
		gauge(mc.lockWaiting, float64(s.Waiting))
		gauge(mc.podClients, float64(s.Clients))
		gauge(mc.podMemory, float64(s.MemoryMiB)*(1<<20))
	}

	for i, reason := range rejectReasonNames {
		ch <- prometheus.MustNewConstMetric(mc.messagesRejected, prometheus.CounterValue,
			float64(mc.rejected[i]), mc.node, reason)
	}

	if mc.schedSeen.IsZero() {
		return
	}
	ch <- prometheus.MustNewConstMetric(mc.schedLastUpdate, prometheus.GaugeValue,
		float64(mc.schedSeen.UnixNano())/1e9, mc.node)
	if now.Sub(mc.schedSeen) > staleAfter {
		return
	}
	ch <- prometheus.MustNewConstMetric(mc.schedEnabled, prometheus.GaugeValue, float64(mc.sched.SchedulerOn), mc.node)
	ch <- prometheus.MustNewConstMetric(mc.schedParallel, prometheus.GaugeValue, float64(mc.sched.Parallel), mc.node)
	ch <- prometheus.MustNewConstMetric(mc.schedTimeQuantum, prometheus.GaugeValue, float64(mc.sched.TQSeconds), mc.node)
	ch <- prometheus.MustNewConstMetric(mc.schedClients, prometheus.GaugeValue, float64(mc.sched.Clients), mc.node)
	ch <- prometheus.MustNewConstMetric(mc.schedQueueLength, prometheus.GaugeValue, float64(mc.sched.QueueLength), mc.node)
	ch <- prometheus.MustNewConstMetric(mc.schedDropped, prometheus.CounterValue, float64(mc.sched.DroppedMessages), mc.node)
}

// StartMetricsServer binds the HTTP endpoint and the metrics socket and
// returns an error if either is unavailable, so a broken metrics path is
// visible at startup instead of silently exporting nothing.
func (mc *MetricsCollector) StartMetricsServer() error {
	httpListener, err := net.Listen("tcp", fmt.Sprintf(":%d", mc.metricsPort))
	if err != nil {
		return fmt.Errorf("listen on metrics port: %w", err)
	}

	os.Remove(metricsSocketPath)
	conn, err := net.ListenUnixgram("unixgram", &net.UnixAddr{Name: metricsSocketPath, Net: "unixgram"})
	if err != nil {
		httpListener.Close()
		return fmt.Errorf("listen on %s: %w", metricsSocketPath, err)
	}
	if err := os.Chmod(metricsSocketPath, 0600); err != nil {
		log.Printf("WARN: chmod %s: %v", metricsSocketPath, err)
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	server := &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}
	go func() {
		log.Printf("Starting metrics server on :%d", mc.metricsPort)
		log.Fatalf("Metrics server failed: %v", server.Serve(httpListener))
	}()

	log.Printf("Metrics listener ready on %s", metricsSocketPath)
	go mc.receive(conn)
	return nil
}

func (mc *MetricsCollector) receive(conn *net.UnixConn) {
	buf := make([]byte, 2048)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			log.Printf("ERROR: read metrics socket: %v", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		mc.handleDatagram(buf[:n])
	}
}

func (mc *MetricsCollector) handleDatagram(buf []byte) {
	msg, err := parseMetricsMessage(buf)

	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	now := mc.now()
	if err != nil {
		if errors.Is(err, errProtocolVersion) {
			mc.rejected[rejectedVersion]++
		} else {
			mc.rejected[rejectedMalformed]++
		}
		// An old scheduler keeps sending; do not flood the log.
		if now.Sub(mc.lastWarning) > time.Minute {
			log.Printf("WARN: ignoring metrics message: %v", err)
			mc.lastWarning = now
		}
		return
	}

	switch m := msg.(type) {
	case *podStats:
		key := podKey{m.Namespace, m.Pod}
		if _, known := mc.pods[key]; !known {
			log.Printf("pod %s/%s reported by the scheduler", m.Namespace, m.Pod)
		}
		mc.pods[key] = podEntry{stats: *m, seen: now}
	case *schedStats:
		mc.sched = *m
		mc.schedSeen = now
	}
}

// parseMetricsMessage decodes a message emitted by the scheduler
// (src/metrics.c). Fields are read explicitly in little-endian order so
// Go's natural struct alignment cannot diverge from C's packed layout.
func parseMetricsMessage(buf []byte) (any, error) {
	if len(buf) < 8 {
		return nil, fmt.Errorf("short message: %d bytes", len(buf))
	}

	le := binary.LittleEndian
	if version := le.Uint32(buf); version != protocolVersion {
		return nil, fmt.Errorf("%w %d (expected %d)", errProtocolVersion, version, protocolVersion)
	}

	switch msgType := le.Uint32(buf[4:]); msgType {
	case msgPodStats:
		if len(buf) != podStatsSize {
			return nil, fmt.Errorf("pod stats message is %d bytes, expected %d", len(buf), podStatsSize)
		}
		m := &podStats{}
		off := 8
		m.Namespace = trimZero(buf[off : off+nameFieldLen])
		off += nameFieldLen
		m.Pod = trimZero(buf[off : off+nameFieldLen])
		off += nameFieldLen
		if m.Namespace == "" || m.Pod == "" {
			return nil, fmt.Errorf("pod stats message without namespace or pod")
		}
		if !utf8.ValidString(m.Namespace) || !utf8.ValidString(m.Pod) {
			return nil, fmt.Errorf("pod stats message with a namespace or pod that is not valid UTF-8")
		}
		m.Clients = le.Uint32(buf[off:])
		m.Holding = le.Uint32(buf[off+4:])
		m.Waiting = le.Uint32(buf[off+8:])
		off += 12
		m.HeldMS = le.Uint64(buf[off:])
		m.WaitMS = le.Uint64(buf[off+8:])
		m.Acquisitions = le.Uint64(buf[off+16:])
		off += 24
		for i := range m.Releases {
			m.Releases[i] = le.Uint64(buf[off+8*i:])
		}
		m.MemoryMiB = le.Uint64(buf[off+8*releaseReasons:])
		return m, nil

	case msgSchedStats:
		if len(buf) != schedStatsSize {
			return nil, fmt.Errorf("scheduler stats message is %d bytes, expected %d", len(buf), schedStatsSize)
		}
		return &schedStats{
			SchedulerOn:     le.Uint32(buf[8:]),
			TQSeconds:       le.Uint32(buf[12:]),
			Clients:         le.Uint32(buf[16:]),
			QueueLength:     le.Uint32(buf[20:]),
			DroppedMessages: le.Uint64(buf[24:]),
			Parallel:        le.Uint32(buf[32:]),
		}, nil

	default:
		return nil, fmt.Errorf("unknown message type %d", msgType)
	}
}

func trimZero(b []byte) string {
	before, _, _ := bytes.Cut(b, []byte{0})
	return string(before)
}
