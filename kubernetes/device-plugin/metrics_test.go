package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

// buildPodStats and buildSchedStats construct byte buffers that match the
// on-wire layout emitted by src/metrics.c (packed, little-endian).
func buildPodStats(version uint32, s podStats) []byte {
	buf := make([]byte, podStatsSize)
	le := binary.LittleEndian
	le.PutUint32(buf[0:], version)
	le.PutUint32(buf[4:], msgPodStats)
	off := 8
	copy(buf[off:off+nameFieldLen], s.Namespace)
	off += nameFieldLen
	copy(buf[off:off+nameFieldLen], s.Pod)
	off += nameFieldLen
	le.PutUint32(buf[off:], s.Clients)
	le.PutUint32(buf[off+4:], s.Holding)
	le.PutUint32(buf[off+8:], s.Waiting)
	off += 12
	le.PutUint64(buf[off:], s.HeldMS)
	le.PutUint64(buf[off+8:], s.WaitMS)
	le.PutUint64(buf[off+16:], s.Acquisitions)
	off += 24
	for i, v := range s.Releases {
		le.PutUint64(buf[off+8*i:], v)
	}
	return buf
}

func buildSchedStats(version uint32, s schedStats) []byte {
	buf := make([]byte, schedStatsSize)
	le := binary.LittleEndian
	le.PutUint32(buf[0:], version)
	le.PutUint32(buf[4:], msgSchedStats)
	le.PutUint32(buf[8:], s.SchedulerOn)
	le.PutUint32(buf[12:], s.TQSeconds)
	le.PutUint32(buf[16:], s.Clients)
	le.PutUint32(buf[20:], s.QueueLength)
	le.PutUint64(buf[24:], s.DroppedMessages)
	return buf
}

// The golden files are byte-for-byte dumps of the C structs in src/metrics.h
// (see testdata/gen_wire_golden.c). If C changes a field, its size or the
// order, either the golden no longer parses to these values or the Go
// builder no longer reproduces it, and the tests fail until both sides are
// brought back in sync.
var goldenPodStats = podStats{
	Namespace:    "default",
	Pod:          "pod-abc",
	Clients:      2,
	Holding:      1,
	Waiting:      1,
	HeldMS:       0x1122334455667788,
	WaitMS:       12345,
	Acquisitions: 7,
	Releases:     [releaseReasons]uint64{4, 3, 2, 1},
}

var goldenSchedStats = schedStats{
	SchedulerOn:     1,
	TQSeconds:       30,
	Clients:         5,
	QueueLength:     3,
	DroppedMessages: 0x0102030405060708,
}

func TestParsePodStats_AgainstCGolden(t *testing.T) {
	golden, err := os.ReadFile("testdata/pod_stats_golden.bin")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	msg, err := parseMetricsMessage(golden)
	if err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	got, ok := msg.(*podStats)
	if !ok {
		t.Fatalf("golden parsed as %T, want *podStats", msg)
	}
	if *got != goldenPodStats {
		t.Errorf("golden: got %+v, want %+v", *got, goldenPodStats)
	}
	if !bytes.Equal(buildPodStats(protocolVersion, goldenPodStats), golden) {
		t.Errorf("buildPodStats output differs from C golden; wire format has drifted")
	}
}

func TestParseSchedStats_AgainstCGolden(t *testing.T) {
	golden, err := os.ReadFile("testdata/sched_stats_golden.bin")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	msg, err := parseMetricsMessage(golden)
	if err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	got, ok := msg.(*schedStats)
	if !ok {
		t.Fatalf("golden parsed as %T, want *schedStats", msg)
	}
	if *got != goldenSchedStats {
		t.Errorf("golden: got %+v, want %+v", *got, goldenSchedStats)
	}
	if !bytes.Equal(buildSchedStats(protocolVersion, goldenSchedStats), golden) {
		t.Errorf("buildSchedStats output differs from C golden; wire format has drifted")
	}
}

func TestParseMetricsMessage_Rejects(t *testing.T) {
	unknownType := buildSchedStats(protocolVersion, goldenSchedStats)
	binary.LittleEndian.PutUint32(unknownType[4:], 999)

	cases := map[string][]byte{
		"empty":              {},
		"shorter than head":  make([]byte, 7),
		"old version":        buildPodStats(protocolVersion-1, goldenPodStats),
		"newer version":      buildSchedStats(protocolVersion+1, goldenSchedStats),
		"truncated pod":      buildPodStats(protocolVersion, goldenPodStats)[:podStatsSize-1],
		"oversized sched":    append(buildSchedStats(protocolVersion, goldenSchedStats), 0),
		"unknown type":       unknownType,
		"no namespace":       buildPodStats(protocolVersion, podStats{Pod: "p"}),
		"no pod":             buildPodStats(protocolVersion, podStats{Namespace: "ns"}),
		"namespace not utf8": buildPodStats(protocolVersion, podStats{Namespace: "ns\xff", Pod: "p"}),
		"pod not utf8":       buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "p\xff\xfe"}),
	}
	for name, buf := range cases {
		if _, err := parseMetricsMessage(buf); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

type fakeClock struct {
	t time.Time
}

func (c *fakeClock) now() time.Time {
	return c.t
}

func newTestCollector() (*MetricsCollector, *prometheus.Registry, *fakeClock) {
	clock := &fakeClock{t: time.Unix(1700000000, 0)}
	mc := newMetricsCollector(0, "node-1", "GPU-test", clock.now)
	reg := prometheus.NewRegistry()
	reg.MustRegister(mc)
	return mc, reg, clock
}

func TestCollector_ExportsSchedulerState(t *testing.T) {
	mc, reg, _ := newTestCollector()

	mc.handleDatagram(buildPodStats(protocolVersion, podStats{
		Namespace:    "team-a",
		Pod:          "train-1",
		Clients:      2,
		Holding:      1,
		Waiting:      1,
		HeldMS:       90500,
		WaitMS:       30250,
		Acquisitions: 4,
		Releases:     [releaseReasons]uint64{3, 1, 0, 0},
	}))
	mc.handleDatagram(buildSchedStats(protocolVersion, schedStats{
		SchedulerOn:     1,
		TQSeconds:       30,
		Clients:         3,
		QueueLength:     2,
		DroppedMessages: 5,
	}))

	want := `
# HELP nvshare_device_plugin_metrics_rejected_total Total number of scheduler messages the device plugin ignored, by reason: version (scheduler and device plugin speak different protocol versions), malformed.
# TYPE nvshare_device_plugin_metrics_rejected_total counter
nvshare_device_plugin_metrics_rejected_total{node="node-1",reason="malformed"} 0
nvshare_device_plugin_metrics_rejected_total{node="node-1",reason="version"} 0
# HELP nvshare_pod_gpu_clients Number of the pod's processes connected to the nvshare scheduler.
# TYPE nvshare_pod_gpu_clients gauge
nvshare_pod_gpu_clients{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1"} 2
# HELP nvshare_pod_gpu_lock_held_seconds_total Total seconds each pod has held the exclusive nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_held_seconds_total counter
nvshare_pod_gpu_lock_held_seconds_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1"} 90.5
# HELP nvshare_pod_gpu_lock_holding 1 if the pod currently holds the nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_holding gauge
nvshare_pod_gpu_lock_holding{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1"} 1
# HELP nvshare_pod_gpu_lock_releases_total Total number of times each pod has given up the nvshare GPU lock, by reason: quantum (time quantum expired), idle (released on its own), exit (process went away), off (scheduler turned off or stopped).
# TYPE nvshare_pod_gpu_lock_releases_total counter
nvshare_pod_gpu_lock_releases_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1",reason="exit"} 0
nvshare_pod_gpu_lock_releases_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1",reason="idle"} 1
nvshare_pod_gpu_lock_releases_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1",reason="off"} 0
nvshare_pod_gpu_lock_releases_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1",reason="quantum"} 3
# HELP nvshare_pod_gpu_lock_wait_seconds_total Total seconds each pod has spent queued for the nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_wait_seconds_total counter
nvshare_pod_gpu_lock_wait_seconds_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1"} 30.25
# HELP nvshare_pod_gpu_lock_waiting Number of the pod's processes currently queued for the nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_waiting gauge
nvshare_pod_gpu_lock_waiting{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1"} 1
# HELP nvshare_pod_gpu_locks_acquired_total Total number of times each pod has acquired the nvshare GPU lock.
# TYPE nvshare_pod_gpu_locks_acquired_total counter
nvshare_pod_gpu_locks_acquired_total{gpu_uuid="GPU-test",namespace="team-a",node="node-1",pod="train-1"} 4
# HELP nvshare_scheduler_clients Number of processes connected to the nvshare scheduler.
# TYPE nvshare_scheduler_clients gauge
nvshare_scheduler_clients{node="node-1"} 3
# HELP nvshare_scheduler_enabled 1 if the nvshare scheduler is on (GPU access is serialized), 0 if it was turned off.
# TYPE nvshare_scheduler_enabled gauge
nvshare_scheduler_enabled{node="node-1"} 1
# HELP nvshare_scheduler_last_update_timestamp_seconds Unix time of the last message received from the nvshare scheduler.
# TYPE nvshare_scheduler_last_update_timestamp_seconds gauge
nvshare_scheduler_last_update_timestamp_seconds{node="node-1"} 1.7e+09
# HELP nvshare_scheduler_lock_queue_length Number of processes currently queued for the nvshare GPU lock.
# TYPE nvshare_scheduler_lock_queue_length gauge
nvshare_scheduler_lock_queue_length{node="node-1"} 2
# HELP nvshare_scheduler_metrics_dropped_total Total number of metrics messages the scheduler could not deliver.
# TYPE nvshare_scheduler_metrics_dropped_total counter
nvshare_scheduler_metrics_dropped_total{node="node-1"} 5
# HELP nvshare_scheduler_time_quantum_seconds Time quantum after which the lock holder is asked to release the GPU.
# TYPE nvshare_scheduler_time_quantum_seconds gauge
nvshare_scheduler_time_quantum_seconds{node="node-1"} 30
`
	if err := testutil.GatherAndCompare(reg, strings.NewReader(want)); err != nil {
		t.Error(err)
	}
}

// The scheduler sends cumulative totals, so a newer message must replace
// the previous one rather than be added to it.
func TestCollector_LaterMessageReplacesEarlier(t *testing.T) {
	mc, reg, _ := newTestCollector()

	mc.handleDatagram(buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "p", HeldMS: 10000, Acquisitions: 1}))
	mc.handleDatagram(buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "p", HeldMS: 25000, Acquisitions: 2}))

	want := `
# HELP nvshare_pod_gpu_lock_held_seconds_total Total seconds each pod has held the exclusive nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_held_seconds_total counter
nvshare_pod_gpu_lock_held_seconds_total{gpu_uuid="GPU-test",namespace="ns",node="node-1",pod="p"} 25
# HELP nvshare_pod_gpu_locks_acquired_total Total number of times each pod has acquired the nvshare GPU lock.
# TYPE nvshare_pod_gpu_locks_acquired_total counter
nvshare_pod_gpu_locks_acquired_total{gpu_uuid="GPU-test",namespace="ns",node="node-1",pod="p"} 2
`
	err := testutil.GatherAndCompare(reg, strings.NewReader(want),
		"nvshare_pod_gpu_lock_held_seconds_total", "nvshare_pod_gpu_locks_acquired_total")
	if err != nil {
		t.Error(err)
	}
}

func TestCollector_CountsInvalidMessages(t *testing.T) {
	mc, reg, _ := newTestCollector()

	mc.handleDatagram(buildPodStats(protocolVersion-1, podStats{Namespace: "ns", Pod: "p", HeldMS: 10000}))
	mc.handleDatagram(buildSchedStats(protocolVersion+1, goldenSchedStats))
	mc.handleDatagram([]byte("garbage"))
	mc.handleDatagram(buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "bad\xff", HeldMS: 10000}))
	mc.handleDatagram(buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "good", HeldMS: 10000}))

	want := `
# HELP nvshare_device_plugin_metrics_rejected_total Total number of scheduler messages the device plugin ignored, by reason: version (scheduler and device plugin speak different protocol versions), malformed.
# TYPE nvshare_device_plugin_metrics_rejected_total counter
nvshare_device_plugin_metrics_rejected_total{node="node-1",reason="malformed"} 2
nvshare_device_plugin_metrics_rejected_total{node="node-1",reason="version"} 2
# HELP nvshare_pod_gpu_lock_held_seconds_total Total seconds each pod has held the exclusive nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_held_seconds_total counter
nvshare_pod_gpu_lock_held_seconds_total{gpu_uuid="GPU-test",namespace="ns",node="node-1",pod="good"} 10
`
	err := testutil.GatherAndCompare(reg, strings.NewReader(want),
		"nvshare_device_plugin_metrics_rejected_total",
		"nvshare_pod_gpu_lock_held_seconds_total",
		"nvshare_scheduler_enabled")
	if err != nil {
		t.Error(err)
	}
}

// State the scheduler stopped refreshing must disappear, so finished pods
// do not stay exported forever and a dead scheduler is visible.
func TestCollector_DropsStaleState(t *testing.T) {
	mc, reg, clock := newTestCollector()

	mc.handleDatagram(buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "old", HeldMS: 1000}))
	mc.handleDatagram(buildSchedStats(protocolVersion, schedStats{SchedulerOn: 1, TQSeconds: 30}))

	clock.t = clock.t.Add(staleAfter - time.Second)
	mc.handleDatagram(buildPodStats(protocolVersion, podStats{Namespace: "ns", Pod: "fresh", HeldMS: 2000}))

	clock.t = clock.t.Add(2 * time.Second)

	want := `
# HELP nvshare_pod_gpu_lock_held_seconds_total Total seconds each pod has held the exclusive nvshare GPU lock.
# TYPE nvshare_pod_gpu_lock_held_seconds_total counter
nvshare_pod_gpu_lock_held_seconds_total{gpu_uuid="GPU-test",namespace="ns",node="node-1",pod="fresh"} 2
# HELP nvshare_scheduler_last_update_timestamp_seconds Unix time of the last message received from the nvshare scheduler.
# TYPE nvshare_scheduler_last_update_timestamp_seconds gauge
nvshare_scheduler_last_update_timestamp_seconds{node="node-1"} 1.7e+09
`
	err := testutil.GatherAndCompare(reg, strings.NewReader(want),
		"nvshare_pod_gpu_lock_held_seconds_total",
		"nvshare_scheduler_last_update_timestamp_seconds",
		"nvshare_scheduler_enabled")
	if err != nil {
		t.Error(err)
	}
}
