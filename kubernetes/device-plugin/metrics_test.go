package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

// buildWireMessage constructs a byte buffer that matches the on-wire
// layout emitted by src/metrics.c (packed, little-endian).
func buildWireMessage(version, msgType uint32, namespace, pod, device string, clientID uint64, durationMS int64) []byte {
	buf := make([]byte, metricsMsgSize)
	off := 0
	binary.LittleEndian.PutUint32(buf[off:], version)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], msgType)
	off += 4
	copy(buf[off:off+nsFieldLen], namespace)
	off += nsFieldLen
	copy(buf[off:off+podFieldLen], pod)
	off += podFieldLen
	copy(buf[off:off+deviceFieldLen], device)
	off += deviceFieldLen
	binary.LittleEndian.PutUint64(buf[off:], clientID)
	off += 8
	binary.LittleEndian.PutUint64(buf[off:], uint64(durationMS))
	return buf
}

func TestParseMetricsMessage_Release(t *testing.T) {
	buf := buildWireMessage(protocolVersion, msgLockReleased, "default", "pod-abc", "nvidia0", 0x1122334455667788, 12345)

	msg, err := parseMetricsMessage(buf)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if msg.Version != protocolVersion {
		t.Errorf("version: got %d, want %d", msg.Version, protocolVersion)
	}
	if msg.Type != msgLockReleased {
		t.Errorf("type: got %d, want %d", msg.Type, msgLockReleased)
	}
	if msg.Namespace != "default" {
		t.Errorf("namespace: got %q, want %q", msg.Namespace, "default")
	}
	if msg.PodName != "pod-abc" {
		t.Errorf("pod: got %q, want %q", msg.PodName, "pod-abc")
	}
	if msg.DeviceID != "nvidia0" {
		t.Errorf("device: got %q, want %q", msg.DeviceID, "nvidia0")
	}
	if msg.ClientID != 0x1122334455667788 {
		t.Errorf("client_id: got %x, want %x", msg.ClientID, uint64(0x1122334455667788))
	}
	if msg.DurationMS != 12345 {
		t.Errorf("duration_ms: got %d, want 12345", msg.DurationMS)
	}
}

func TestParseMetricsMessage_RejectsShort(t *testing.T) {
	buf := make([]byte, metricsMsgSize-1)
	if _, err := parseMetricsMessage(buf); err == nil {
		t.Fatal("expected error on short buffer, got nil")
	}
}

func TestParseMetricsMessage_RejectsVersionMismatch(t *testing.T) {
	buf := buildWireMessage(protocolVersion+1, msgLockAcquired, "ns", "pod", "nvidia0", 1, 0)
	if _, err := parseMetricsMessage(buf); err == nil {
		t.Fatal("expected error on version mismatch, got nil")
	}
}

// TestParseMetricsMessage_AgainstCGolden loads testdata/wire_golden.bin --
// a byte-for-byte dump of one metrics_message produced by compiling
// src/metrics.h and writing the struct to stdout (see the generator in
// testdata/gen_wire_golden.c). This is the real cross-language regression
// check: if C changes POD_NAMESPACE_LEN_MAX, field order, or endianness,
// either the golden no longer parses or buildWireMessage no longer matches
// it -- and the test fails until Go and C are brought back in sync.
func TestParseMetricsMessage_AgainstCGolden(t *testing.T) {
	golden, err := os.ReadFile("testdata/wire_golden.bin")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if len(golden) != metricsMsgSize {
		t.Fatalf("golden size: got %d, want %d -- regenerate testdata/wire_golden.bin", len(golden), metricsMsgSize)
	}

	msg, err := parseMetricsMessage(golden)
	if err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if msg.Version != protocolVersion || msg.Type != msgLockReleased {
		t.Errorf("golden header: got version=%d type=%d, want version=%d type=%d",
			msg.Version, msg.Type, protocolVersion, msgLockReleased)
	}
	if msg.Namespace != "default" || msg.PodName != "pod-abc" || msg.DeviceID != "nvidia0" {
		t.Errorf("golden strings: got ns=%q pod=%q device=%q",
			msg.Namespace, msg.PodName, msg.DeviceID)
	}
	if msg.ClientID != 0x1122334455667788 {
		t.Errorf("golden client_id: got %x", msg.ClientID)
	}
	if msg.DurationMS != 12345 {
		t.Errorf("golden duration_ms: got %d", msg.DurationMS)
	}

	// Byte-for-byte: Go's buildWireMessage with the same inputs must
	// produce an identical layout. This catches field-order or alignment
	// drift that parseMetricsMessage alone might tolerate (e.g., if a new
	// field were added symmetrically on both read paths but not on write).
	ours := buildWireMessage(protocolVersion, msgLockReleased, "default", "pod-abc", "nvidia0", 0x1122334455667788, 12345)
	if !bytes.Equal(ours, golden) {
		t.Fatalf("buildWireMessage output differs from C golden; wire format has drifted")
	}
}
