#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>

#include "metrics.h"
#include "common.h"
#include "utlist.h"
#include "comm.h"

enum client_state {
	CLIENT_IDLE,
	CLIENT_WAITING,
	CLIENT_HOLDING,
};

struct metrics_pod {
	char namespace[POD_NAMESPACE_LEN_MAX];
	char pod_name[POD_NAME_LEN_MAX];
	uint32_t clients;
	uint64_t held_ms;
	uint64_t wait_ms;
	uint64_t acquisitions;
	uint64_t releases[METRICS_RELEASE_REASONS];
	struct timespec empty_since;
	struct metrics_pod *next;
};

struct metrics_client {
	uint64_t id;
	struct metrics_pod *pod;
	enum client_state state;
	int drop_sent;
	enum metrics_release_reason drop_reason;
	uint64_t mem_mib;
	struct timespec since;
	struct metrics_client *next;
};

/*
 * All state here is protected by the scheduler's global_mutex (every caller
 * into this module -- scheduler.c -- holds it). No local mutex is required.
 */
static int socket_fd = -1;
static struct metrics_pod *pods = NULL;
static struct metrics_client *tracked = NULL;
static int sched_on;
static int sched_tq;
static int sched_parallel;
static uint64_t dropped_messages;

static int try_connect_socket(void)
{
	int fd;
	struct sockaddr_un addr;
	const char *dir = nvshare_sock_dir();

	fd = socket(AF_LOCAL, SOCK_DGRAM, 0);
	if (fd < 0) {
		log_warn("METRICS: Failed to create socket: %s", strerror(errno));
		return -1;
	}

	int flags = fcntl(fd, F_GETFL, 0);
	if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
		log_warn("METRICS: Failed to set socket non-blocking: %s", strerror(errno));
		close(fd);
		return -1;
	}

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_LOCAL;
	snprintf(addr.sun_path, sizeof(addr.sun_path), "%s%smetrics.sock", dir,
		 dir[strlen(dir) - 1] == '/' ? "" : "/");

	if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		log_debug("METRICS: Cannot connect to %s: %s", addr.sun_path, strerror(errno));
		close(fd);
		return -1;
	}

	socket_fd = fd;
	log_info("METRICS: Connected to device plugin at %s", addr.sun_path);
	return 0;
}

static void close_socket(void)
{
	if (socket_fd >= 0) {
		close(socket_fd);
		socket_fd = -1;
	}
}

static int try_send(const void *msg, size_t len)
{
	ssize_t sent;

	if (socket_fd < 0 && try_connect_socket() < 0)
		return -1;

	sent = send(socket_fd, msg, len, MSG_DONTWAIT | MSG_NOSIGNAL);
	if (sent < 0) {
		/* Receiver is busy. Keep the connection, the next message repairs it. */
		if (errno == EAGAIN || errno == EWOULDBLOCK)
			return -1;

		/* Connection broke (the device plugin restarted). Reconnect, retry once. */
		log_debug("METRICS: send failed (%s), reconnecting", strerror(errno));
		close_socket();
		if (try_connect_socket() < 0)
			return -1;

		sent = send(socket_fd, msg, len, MSG_DONTWAIT | MSG_NOSIGNAL);
		if (sent < 0) {
			close_socket();
			return -1;
		}
	}

	return (sent == (ssize_t)len) ? 0 : -1;
}

static void send_message(const void *msg, size_t len)
{
	if (try_send(msg, len) < 0)
		dropped_messages++;
}

static struct timespec now_ts(void)
{
	struct timespec ts;

	true_or_exit(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
	return ts;
}

static uint64_t elapsed_ms(const struct timespec *start, const struct timespec *end)
{
	int64_t ms = (int64_t)(end->tv_sec - start->tv_sec) * 1000;

	ms += ((int64_t)end->tv_nsec - (int64_t)start->tv_nsec) / 1000000;
	return (ms < 0) ? 0 : (uint64_t)ms;
}

static struct metrics_client *find_client(uint64_t id)
{
	struct metrics_client *c;

	LL_FOREACH(tracked, c) {
		if (c->id == id)
			return c;
	}
	return NULL;
}

static struct metrics_pod *find_pod(const char *namespace, const char *pod_name)
{
	struct metrics_pod *p;

	LL_FOREACH(pods, p) {
		if (strcmp(p->namespace, namespace) == 0 &&
		    strcmp(p->pod_name, pod_name) == 0)
			return p;
	}
	return NULL;
}

static void publish_pod(const struct metrics_pod *pod, const struct timespec *now)
{
	struct metrics_pod_stats msg;
	struct metrics_client *c;

	memset(&msg, 0, sizeof(msg));
	msg.version = METRICS_PROTOCOL_VERSION;
	msg.type = METRICS_POD_STATS;
	strlcpy(msg.namespace, pod->namespace, sizeof(msg.namespace));
	strlcpy(msg.pod_name, pod->pod_name, sizeof(msg.pod_name));
	msg.clients = pod->clients;
	msg.held_ms = pod->held_ms;
	msg.wait_ms = pod->wait_ms;
	msg.acquisitions = pod->acquisitions;
	memcpy(msg.releases, pod->releases, sizeof(msg.releases));

	LL_FOREACH(tracked, c) {
		if (c->pod != pod)
			continue;
		msg.mem_mib += c->mem_mib;
		if (c->state == CLIENT_HOLDING) {
			msg.holding = 1;
			msg.held_ms += elapsed_ms(&c->since, now);
		} else if (c->state == CLIENT_WAITING) {
			msg.waiting++;
			msg.wait_ms += elapsed_ms(&c->since, now);
		}
	}

	send_message(&msg, sizeof(msg));
}

static void publish_sched(void)
{
	struct metrics_sched_stats msg;
	struct metrics_client *c;

	memset(&msg, 0, sizeof(msg));
	msg.version = METRICS_PROTOCOL_VERSION;
	msg.type = METRICS_SCHED_STATS;
	msg.scheduler_on = sched_on ? 1 : 0;
	msg.tq_sec = (uint32_t)sched_tq;
	msg.dropped_messages = dropped_messages;
	msg.parallel = sched_parallel ? 1 : 0;

	LL_FOREACH(tracked, c) {
		msg.clients++;
		if (c->state == CLIENT_WAITING)
			msg.queue_length++;
	}

	send_message(&msg, sizeof(msg));
}

static void publish_client(const struct metrics_client *c, const struct timespec *now)
{
	publish_pod(c->pod, now);
	publish_sched();
}

/* Close whatever the client is doing and account for it in its pod. */
static void leave_state(struct metrics_client *c, const struct timespec *now,
			enum metrics_release_reason reason)
{
	if (c->state == CLIENT_HOLDING) {
		c->pod->held_ms += elapsed_ms(&c->since, now);
		c->pod->releases[reason]++;
	} else if (c->state == CLIENT_WAITING) {
		c->pod->wait_ms += elapsed_ms(&c->since, now);
	}
	c->state = CLIENT_IDLE;
	c->drop_sent = 0;
}

void metrics_init(int scheduler_on, int tq, int parallel)
{
	sched_on = scheduler_on;
	sched_tq = tq;
	sched_parallel = parallel;
	if (try_connect_socket() < 0)
		log_info("METRICS: Device plugin socket not yet available; will keep retrying");
}

void metrics_client_registered(const struct nvshare_client *client)
{
	struct metrics_client *c;
	struct metrics_pod *pod;
	struct timespec now = now_ts();

	if (find_client(client->id) != NULL)
		return;

	pod = find_pod(client->pod_namespace, client->pod_name);
	if (pod == NULL) {
		true_or_exit(pod = calloc(1, sizeof(*pod)));
		strlcpy(pod->namespace, client->pod_namespace, sizeof(pod->namespace));
		strlcpy(pod->pod_name, client->pod_name, sizeof(pod->pod_name));
		LL_APPEND(pods, pod);
	}
	pod->clients++;

	true_or_exit(c = calloc(1, sizeof(*c)));
	c->id = client->id;
	c->pod = pod;
	c->state = CLIENT_IDLE;
	LL_APPEND(tracked, c);

	publish_client(c, &now);
}

void metrics_client_removed(const struct nvshare_client *client)
{
	struct metrics_client *c = find_client(client->id);
	struct metrics_pod *pod;
	struct timespec now = now_ts();

	/* Clients that never registered (nvsharectl) are not tracked. */
	if (c == NULL)
		return;

	leave_state(c, &now, METRICS_RELEASE_EXIT);
	pod = c->pod;
	pod->clients--;
	if (pod->clients == 0)
		pod->empty_since = now;

	LL_DELETE(tracked, c);
	free(c);

	publish_pod(pod, &now);
	publish_sched();
}

void metrics_lock_requested(const struct nvshare_client *client)
{
	struct metrics_client *c = find_client(client->id);
	struct timespec now = now_ts();

	if (c == NULL || c->state != CLIENT_IDLE)
		return;

	c->state = CLIENT_WAITING;
	c->since = now;
	publish_client(c, &now);
}

void metrics_lock_acquired(const struct nvshare_client *client)
{
	struct metrics_client *c = find_client(client->id);
	struct timespec now = now_ts();

	if (c == NULL || c->state == CLIENT_HOLDING)
		return;

	if (c->state == CLIENT_WAITING)
		c->pod->wait_ms += elapsed_ms(&c->since, &now);
	c->state = CLIENT_HOLDING;
	c->since = now;
	c->drop_sent = 0;
	c->pod->acquisitions++;
	publish_client(c, &now);
}

void metrics_client_memory(const struct nvshare_client *client)
{
	struct metrics_client *c = find_client(client->id);
	struct timespec now = now_ts();

	if (c == NULL || c->mem_mib == client->mem_mib)
		return;

	c->mem_mib = client->mem_mib;
	publish_pod(c->pod, &now);
}

void metrics_drop_sent(const struct nvshare_client *client,
		       enum metrics_release_reason reason)
{
	struct metrics_client *c = find_client(client->id);

	if (c != NULL && c->state == CLIENT_HOLDING) {
		c->drop_sent = 1;
		c->drop_reason = reason;
	}
}

void metrics_lock_released(const struct nvshare_client *client)
{
	struct metrics_client *c = find_client(client->id);
	struct timespec now = now_ts();

	if (c == NULL || c->state == CLIENT_IDLE)
		return;

	leave_state(c, &now, c->drop_sent ? c->drop_reason : METRICS_RELEASE_IDLE);
	publish_client(c, &now);
}

/*
 * When the scheduler is turned off every client runs freely and the
 * requests list is discarded, so close all holds and waits.
 */
void metrics_scheduler_state(int scheduler_on, int tq, int parallel)
{
	struct metrics_client *c;
	struct metrics_pod *pod;
	struct timespec now = now_ts();

	if (sched_on && !scheduler_on) {
		LL_FOREACH(tracked, c)
			leave_state(c, &now, METRICS_RELEASE_OFF);
		LL_FOREACH(pods, pod)
			publish_pod(pod, &now);
	}

	sched_on = scheduler_on;
	sched_tq = tq;
	sched_parallel = parallel;
	publish_sched();
}

void metrics_tick(void)
{
	struct metrics_pod *pod, *tmp;
	struct timespec now = now_ts();

	LL_FOREACH_SAFE(pods, pod, tmp) {
		if (pod->clients == 0 &&
		    elapsed_ms(&pod->empty_since, &now) >
		    (uint64_t)METRICS_POD_RETENTION_SEC * 1000) {
			LL_DELETE(pods, pod);
			free(pod);
			continue;
		}
		publish_pod(pod, &now);
	}
	publish_sched();
}

void metrics_cleanup(void)
{
	struct metrics_client *c;
	struct metrics_pod *pod;
	struct timespec now = now_ts();

	LL_FOREACH(tracked, c)
		leave_state(c, &now, METRICS_RELEASE_OFF);
	LL_FOREACH(pods, pod)
		publish_pod(pod, &now);
	publish_sched();
	close_socket();
}
