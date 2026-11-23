/*
 * Generator for the golden files in this directory. Produces one canonical
 * message of each type byte-for-byte as the scheduler would emit it. The Go
 * test parses these files as a cross-language regression check: if either
 * side's layout drifts, the Go test fails.
 *
 * Regenerate:
 *   make -C ../../../src common.o
 *   gcc -I../../../src gen_wire_golden.c ../../../src/common.o -o /tmp/gen_wire_golden
 *   /tmp/gen_wire_golden pod > pod_stats_golden.bin
 *   /tmp/gen_wire_golden sched > sched_stats_golden.bin
 *
 * Any regeneration must be committed together with the Go test change
 * that accepts the new layout.
 */
#include <stdio.h>
#include <string.h>
#include "common.h"
#include "metrics.h"

int main(int argc, char **argv) {
	if (argc == 2 && strcmp(argv[1], "pod") == 0) {
		struct metrics_pod_stats m;
		memset(&m, 0, sizeof(m));
		m.version = METRICS_PROTOCOL_VERSION;
		m.type = METRICS_POD_STATS;
		strlcpy(m.namespace, "default", sizeof(m.namespace));
		strlcpy(m.pod_name, "pod-abc", sizeof(m.pod_name));
		m.clients = 2;
		m.holding = 1;
		m.waiting = 1;
		m.held_ms = 0x1122334455667788ULL;
		m.wait_ms = 12345;
		m.acquisitions = 7;
		m.releases[METRICS_RELEASE_QUANTUM] = 4;
		m.releases[METRICS_RELEASE_IDLE] = 3;
		m.releases[METRICS_RELEASE_EXIT] = 2;
		m.releases[METRICS_RELEASE_OFF] = 1;
		fwrite(&m, sizeof(m), 1, stdout);
		return 0;
	}
	if (argc == 2 && strcmp(argv[1], "sched") == 0) {
		struct metrics_sched_stats m;
		memset(&m, 0, sizeof(m));
		m.version = METRICS_PROTOCOL_VERSION;
		m.type = METRICS_SCHED_STATS;
		m.scheduler_on = 1;
		m.tq_sec = 30;
		m.clients = 5;
		m.queue_length = 3;
		m.dropped_messages = 0x0102030405060708ULL;
		fwrite(&m, sizeof(m), 1, stdout);
		return 0;
	}
	fprintf(stderr, "usage: %s pod|sched\n", argv[0]);
	return 2;
}
