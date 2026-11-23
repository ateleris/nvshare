/*
 * A scheduler client without a GPU, for tests. It registers as the given
 * pod, requests the lock once and holds it for HOLD_SECONDS, or until the
 * scheduler asks for it back with DROP_LOCK. With "exit" it goes away while
 * still holding the lock instead of releasing it.
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

	send_msg(sock, &msg, REGISTER);
	true_or_exit(recv_msg(sock) == SCHED_ON);
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
