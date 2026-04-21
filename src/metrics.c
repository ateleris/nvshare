#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>

#include "metrics.h"
#include "common.h"
#include "utlist.h"
#include "comm.h"

static int socket_fd = -1;
static struct pod_gpu_session *active_sessions = NULL;
static pthread_mutex_t metrics_mutex = PTHREAD_MUTEX_INITIALIZER;

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
		log_debug("METRICS: send dropped, no connection");
		return -1;
	}

	sent = send(socket_fd, msg, sizeof(*msg), MSG_DONTWAIT);
	if (sent < 0) {
		if (errno == EAGAIN || errno == EWOULDBLOCK) {
			log_debug("METRICS: socket busy, message dropped");
			return -1;
		}

		/* Connection broke (EPIPE, ECONNREFUSED, ENOTCONN, etc.).
		 * Tear down and attempt reconnect + one retry. */
		log_warn("METRICS: send failed (%s), reconnecting", strerror(errno));
		close_socket();
		if (try_connect_socket() < 0) return -1;

		sent = send(socket_fd, msg, sizeof(*msg), MSG_DONTWAIT);
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

	log_debug("METRICS: sent type=%d for %s/%s", msg->type, msg->namespace, msg->pod_name);
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

int metrics_lock_acquired(const struct nvshare_client *client, const char *device_id) {
	struct pod_gpu_session *session;
	struct metrics_message msg;
	time_t now = time(NULL);

	if (!client || !device_id || device_id[0] == '\0') return -1;

	pthread_mutex_lock(&metrics_mutex);

	session = find_session(client->id, device_id);
	if (session != NULL) {
		/* Already holding lock for this device.  This can happen if the
		 * scheduler is restarted without the device plugin seeing the
		 * corresponding release; just refresh the start time. */
		session->start_time = now;
		session->last_active = now;
		pthread_mutex_unlock(&metrics_mutex);
		log_debug("METRICS: refresh acquire for client %lx", client->id);
		return 0;
	}

	session = malloc(sizeof(*session));
	if (!session) {
		pthread_mutex_unlock(&metrics_mutex);
		log_warn("METRICS: failed to allocate session");
		return -1;
	}

	strlcpy(session->namespace, client->pod_namespace, sizeof(session->namespace));
	strlcpy(session->pod_name, client->pod_name, sizeof(session->pod_name));
	strlcpy(session->container, "container", sizeof(session->container));
	strlcpy(session->device_id, device_id, sizeof(session->device_id));
	session->process_id = 0;
	session->client_id = client->id;
	session->start_time = now;
	session->last_active = now;
	LL_APPEND(active_sessions, session);

	pthread_mutex_unlock(&metrics_mutex);

	memset(&msg, 0, sizeof(msg));
	msg.type = (int32_t)METRICS_LOCK_ACQUIRED;
	strlcpy(msg.namespace, client->pod_namespace, sizeof(msg.namespace));
	strlcpy(msg.pod_name, client->pod_name, sizeof(msg.pod_name));
	strlcpy(msg.container, "container", sizeof(msg.container));
	strlcpy(msg.device_id, device_id, sizeof(msg.device_id));
	msg.process_id = 0;
	msg.client_id = client->id;
	msg.timestamp = (int64_t)now;

	send_metrics_message(&msg);
	log_info("METRICS: lock acquired by %s/%s", client->pod_namespace, client->pod_name);
	return 0;
}

int metrics_lock_released(const struct nvshare_client *client, const char *device_id) {
	struct pod_gpu_session *session;
	struct metrics_message msg;
	time_t now = time(NULL);
	int64_t duration_sec;

	if (!client || !device_id || device_id[0] == '\0') return -1;

	pthread_mutex_lock(&metrics_mutex);

	session = find_session(client->id, device_id);
	if (!session) {
		pthread_mutex_unlock(&metrics_mutex);
		log_debug("METRICS: release without active session for client %lx", client->id);
		return 0;
	}

	duration_sec = (int64_t)now - (int64_t)session->start_time;
	if (duration_sec < 0) duration_sec = 0;

	memset(&msg, 0, sizeof(msg));
	msg.type = (int32_t)METRICS_LOCK_RELEASED;
	strlcpy(msg.namespace, session->namespace, sizeof(msg.namespace));
	strlcpy(msg.pod_name, session->pod_name, sizeof(msg.pod_name));
	strlcpy(msg.container, session->container, sizeof(msg.container));
	strlcpy(msg.device_id, session->device_id, sizeof(msg.device_id));
	msg.process_id = session->process_id;
	msg.client_id = session->client_id;
	msg.timestamp = duration_sec;

	LL_DELETE(active_sessions, session);
	free(session);
	pthread_mutex_unlock(&metrics_mutex);

	send_metrics_message(&msg);
	log_info("METRICS: lock released by %s/%s after %lld s",
		  msg.namespace, msg.pod_name, (long long)duration_sec);
	return 0;
}

void metrics_cleanup(void) {
	struct pod_gpu_session *session, *tmp;

	pthread_mutex_lock(&metrics_mutex);
	LL_FOREACH_SAFE(active_sessions, session, tmp) {
		LL_DELETE(active_sessions, session);
		free(session);
	}
	pthread_mutex_unlock(&metrics_mutex);

	close_socket();
	log_info("Metrics system cleaned up");
}
