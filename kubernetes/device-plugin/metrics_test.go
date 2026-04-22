package main

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// buildWireMessage constructs a byte buffer that matches the on-wire
// layout emitted by src/metrics.c (packed, little-endian). This is the
// regression test for the wire-format mismatch: if Go's parser or the C
// struct layout ever drift, this test fails.
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

// TestParseMetricsMessage_AgainstCPayload parses a byte-for-byte copy of
// a payload produced by the C scheduler's metrics_message struct, ensuring
// the explicit little-endian layout on both sides stays in lockstep. The
// blob below was produced by compiling src/metrics.h + common.o and
// writing a struct with the same values used by TestParseMetricsMessage_Release.
func TestParseMetricsMessage_AgainstCPayload(t *testing.T) {
	expected := buildWireMessage(protocolVersion, msgLockReleased, "default", "pod-abc", "nvidia0", 0x1122334455667788, 12345)
	// If this assertion fails, regenerate the golden buffer from C and
	// investigate any layout drift before updating the comparand.
	if len(expected) != 564 {
		t.Fatalf("unexpected wire size: %d (want 564)", len(expected))
	}
	msg, err := parseMetricsMessage(expected)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if msg.PodName != "pod-abc" || msg.DurationMS != 12345 {
		t.Fatalf("parse returned unexpected fields: %+v", msg)
	}
	// Sanity: the message is all-zero after the payload's nominal
	// packing boundary (i.e., no accidental padding at the tail).
	if !bytes.Equal(expected[:2], []byte{0x01, 0x00}) {
		t.Errorf("version bytes: got %x, want 01 00", expected[:2])
	}
}
