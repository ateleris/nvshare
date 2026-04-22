/*
 * Generator for testdata/wire_golden.bin. Produces one canonical
 * metrics_message (LOCK_RELEASED, known fields) byte-for-byte as the
 * scheduler would emit it. The Go test parses this file as a cross-
 * language regression check: if either side's layout drifts, the Go
 * test fails.
 *
 * Regenerate:
 *   make -C ../../../src metrics.o common.o
 *   gcc -I../../../src gen_wire_golden.c ../../../src/common.o \
 *       -o /tmp/gen_wire_golden && /tmp/gen_wire_golden > wire_golden.bin
 *
 * Any regeneration must be committed together with the Go test change
 * that accepts the new layout.
 */
#include <stdio.h>
#include <stddef.h>
#include "common.h"
#include "metrics.h"

int main(void) {
	struct metrics_message m;
	memset(&m, 0, sizeof(m));
	m.version = METRICS_PROTOCOL_VERSION;
	m.type = METRICS_LOCK_RELEASED;
	strlcpy(m.namespace, "default", sizeof(m.namespace));
	strlcpy(m.pod_name, "pod-abc", sizeof(m.pod_name));
	strlcpy(m.device_id, "nvidia0", sizeof(m.device_id));
	m.client_id = 0x1122334455667788ULL;
	m.duration_ms = 12345;
	fwrite(&m, sizeof(m), 1, stdout);
	return 0;
}
