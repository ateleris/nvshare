/*
 * A scheduler client without a GPU, for tests. It registers as the given
 * pod, requests the lock once and holds it for HOLD_SECONDS, or until the
 * scheduler asks for it back with DROP_LOCK. With "exit" it goes away while
 * still holding the lock instead of releasing it. GPU_MIB and MEM_MIB in the
 * environment are reported as the size of the GPU memory and as the memory
 * it has allocated.
 */
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "comm.h"
#include "common.h"

static void send_msg(int sock, struct message *msg, enum message_type type)
{
	msg->type = type;
	true_or_exit(write_whole(sock, msg, sizeof(*msg)) == sizeof(*msg));
}

static enum message_type recv_msg(int sock)
{
	struct message msg;

	true_or_exit(nvshare_receive_block(sock, &msg, sizeof(msg)) == sizeof(msg));
	return msg.type;
}

int main(int argc, char **argv)
{
	char path[NVSHARE_SOCK_PATH_MAX];
	struct message msg;
	struct pollfd pfd;
	int sock, hold_ms, exit_holding;
	const char *gpu_mib = getenv("GPU_MIB"), *mem_mib = getenv("MEM_MIB");

	if (argc < 4 || argc > 5 || (argc == 5 && strcmp(argv[4], "exit") != 0)) {
		fprintf(stderr, "usage: %s NAMESPACE POD HOLD_SECONDS [exit]\n", argv[0]);
		return 2;
	}
	hold_ms = (int)(atof(argv[3]) * 1000);
	exit_holding = (argc == 5);

	true_or_exit(nvshare_get_scheduler_path(path) == 0);
	true_or_exit(nvshare_connect(&sock, path) == 0);

	memset(&msg, 0, sizeof(msg));
	strlcpy(msg.pod_namespace, argv[1], sizeof(msg.pod_namespace));
	strlcpy(msg.pod_name, argv[2], sizeof(msg.pod_name));

	if (gpu_mib != NULL) strlcpy(msg.data, gpu_mib, sizeof(msg.data));
	send_msg(sock, &msg, REGISTER);
	true_or_exit(recv_msg(sock) == SCHED_ON);
	if (mem_mib != NULL) {
		strlcpy(msg.data, mem_mib, sizeof(msg.data));
		send_msg(sock, &msg, MEM_USAGE);
	}
	memset(msg.data, 0, sizeof(msg.data));
	send_msg(sock, &msg, REQ_LOCK);
	true_or_exit(recv_msg(sock) == LOCK_OK);

	pfd.fd = sock;
	pfd.events = POLLIN;
	if (poll(&pfd, 1, hold_ms) > 0)
		true_or_exit(recv_msg(sock) == DROP_LOCK);
	else if (exit_holding)
		return 0;

	send_msg(sock, &msg, LOCK_RELEASED);
	return 0;
}
