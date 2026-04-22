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

/*
 * All state here is protected by the scheduler's global_mutex (every caller
 * into this module -- scheduler.c -- holds it). No local mutex is required.
 */
static int socket_fd = -1;
static struct pod_gpu_session *active_sessions = NULL;

static int try_connect_socket(void) {
	int fd;
	struct sockaddr_un addr;

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
	strlcpy(addr.sun_path, METRICS_SOCKET_PATH, sizeof(addr.sun_path));

	if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		log_debug("METRICS: Cannot connect to %s: %s", METRICS_SOCKET_PATH, strerror(errno));
		close(fd);
		return -1;
	}

	socket_fd = fd;
	log_info("METRICS: Connected to device plugin at %s", METRICS_SOCKET_PATH);
	return 0;
}

static void close_socket(void) {
	if (socket_fd >= 0) {
		close(socket_fd);
		socket_fd = -1;
	}
}

int metrics_init(void) {
	log_info("Initializing metrics system...");
	if (try_connect_socket() < 0) {
		log_warn("METRICS: Device plugin socket not yet available; will retry on first event");
	}
	return 0;
}

static int send_metrics_message(const struct metrics_message *msg) {
	ssize_t sent;

	if (socket_fd < 0 && try_connect_socket() < 0) {
		return -1;
	}

	sent = send(socket_fd, msg, sizeof(*msg), MSG_DONTWAIT | MSG_NOSIGNAL);
	if (sent < 0) {
		if (errno == EAGAIN || errno == EWOULDBLOCK) {
			/* Transient, don't tear down a working connection. */
			return -1;
		}

		/* Connection broke (EPIPE, ECONNREFUSED, ENOTCONN, etc.).
		 * Tear down and attempt reconnect + one retry. */
		log_warn("METRICS: send failed (%s), reconnecting", strerror(errno));
		close_socket();
		if (try_connect_socket() < 0) return -1;

		sent = send(socket_fd, msg, sizeof(*msg), MSG_DONTWAIT | MSG_NOSIGNAL);
		if (sent < 0) {
			log_warn("METRICS: send retry failed: %s", strerror(errno));
			close_socket();
			return -1;
		}
	}

	if (sent != (ssize_t)sizeof(*msg)) {
		log_warn("METRICS: partial send (%zd/%zu)", sent, sizeof(*msg));
		return -1;
	}

	return 0;
}

static struct pod_gpu_session *find_session(uint64_t client_id, const char *device_id) {
	struct pod_gpu_session *s;
	LL_FOREACH(active_sessions, s) {
		if (s->client_id == client_id && strcmp(s->device_id, device_id) == 0)
			return s;
	}
	return NULL;
}

static int64_t elapsed_ms(const struct timespec *start, const struct timespec *now) {
	int64_t ms = (int64_t)(now->tv_sec - start->tv_sec) * 1000;
	ms += ((int64_t)now->tv_nsec - (int64_t)start->tv_nsec) / 1000000;
	if (ms < 0) ms = 0;
	return ms;
}

static void fill_message(struct metrics_message *msg,
			 uint32_t type,
			 const char *ns, const char *pod,
			 const char *device_id,
			 uint64_t client_id,
			 int64_t duration_ms) {
	memset(msg, 0, sizeof(*msg));
	msg->version = METRICS_PROTOCOL_VERSION;
	msg->type = type;
	strlcpy(msg->namespace, ns, sizeof(msg->namespace));
	strlcpy(msg->pod_name, pod, sizeof(msg->pod_name));
	strlcpy(msg->device_id, device_id, sizeof(msg->device_id));
	msg->client_id = client_id;
	msg->duration_ms = duration_ms;
}

int metrics_lock_acquired(const struct nvshare_client *client, const char *device_id) {
	struct pod_gpu_session *session;
	struct metrics_message msg;
	struct timespec now;

	if (!client || !device_id || device_id[0] == '\0') return -1;
	if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;

	session = find_session(client->id, device_id);
	if (session != NULL) {
		/* Already holding lock for this device. Refresh start time;
		 * the previous hold's duration is lost, but the alternative
		 * (double-counting) is worse. */
		session->start_time = now;
		log_debug("METRICS: refresh acquire for client %lx", client->id);
		return 0;
	}

	session = malloc(sizeof(*session));
	if (!session) {
		log_warn("METRICS: failed to allocate session");
		return -1;
	}

	strlcpy(session->namespace, client->pod_namespace, sizeof(session->namespace));
	strlcpy(session->pod_name, client->pod_name, sizeof(session->pod_name));
	strlcpy(session->device_id, device_id, sizeof(session->device_id));
	session->client_id = client->id;
	session->start_time = now;
	LL_APPEND(active_sessions, session);

	fill_message(&msg, METRICS_LOCK_ACQUIRED,
		     client->pod_namespace, client->pod_name,
		     device_id, client->id, 0);
	send_metrics_message(&msg);
	log_info("METRICS: lock acquired by %s/%s", client->pod_namespace, client->pod_name);
	return 0;
}

/*
 * Emit a LOCK_RELEASED for the given session. Caller must LL_DELETE the
 * session from active_sessions before calling; this function does not
 * touch the list and does not free the session.
 */
static void emit_release_for_session(struct pod_gpu_session *session,
				      const struct timespec *now) {
	struct metrics_message msg;
	int64_t duration_ms = elapsed_ms(&session->start_time, now);

	fill_message(&msg, METRICS_LOCK_RELEASED,
		     session->namespace, session->pod_name,
		     session->device_id, session->client_id, duration_ms);
	send_metrics_message(&msg);
	log_info("METRICS: lock released by %s/%s after %lld ms",
		 session->namespace, session->pod_name, (long long)duration_ms);
}

int metrics_lock_released(const struct nvshare_client *client, const char *device_id) {
	struct pod_gpu_session *session;
	struct timespec now;

	if (!client || !device_id || device_id[0] == '\0') return -1;
	if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;

	session = find_session(client->id, device_id);
	if (!session) {
		log_debug("METRICS: release without active session for client %lx", client->id);
		return 0;
	}

	LL_DELETE(active_sessions, session);
	emit_release_for_session(session, &now);
	free(session);
	return 0;
}

/*
 * Emit LOCK_RELEASED for every currently-active session and drop them.
 * Used when the scheduler transitions to SCHED_OFF (lock holder is about
 * to be invalidated) and on graceful shutdown (SIGTERM).
 */
void metrics_flush_all(void) {
	struct pod_gpu_session *session, *tmp;
	struct timespec now;

	if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return;

	LL_FOREACH_SAFE(active_sessions, session, tmp) {
		LL_DELETE(active_sessions, session);
		emit_release_for_session(session, &now);
		free(session);
	}
}

void metrics_cleanup(void) {
	metrics_flush_all();
	close_socket();
	log_info("Metrics system cleaned up");
}
