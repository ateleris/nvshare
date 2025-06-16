#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>

#include "metrics.h"
#include "common.h"
#include "utlist.h"
#include "comm.h"

static int metrics_socket = -1;
static struct pod_gpu_session *active_sessions = NULL;
static pthread_mutex_t metrics_mutex = PTHREAD_MUTEX_INITIALIZER;

int metrics_init(void) {
	struct sockaddr_un addr;
	int flags;
	
	if (metrics_socket >= 0) {
		return 0;
	}
	
	metrics_socket = socket(AF_UNIX, SOCK_DGRAM, 0);
	if (metrics_socket < 0) {
		log_warn("Failed to create metrics socket: %s", strerror(errno));
		return -1;
	}
	
	flags = fcntl(metrics_socket, F_GETFL, 0);
	if (flags < 0 || fcntl(metrics_socket, F_SETFL, flags | O_NONBLOCK) < 0) {
		log_warn("Failed to set metrics socket to non-blocking: %s", strerror(errno));
		close(metrics_socket);
		metrics_socket = -1;
		return -1;
	}
	
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strlcpy(addr.sun_path, METRICS_SOCKET_PATH, sizeof(addr.sun_path));
	
	if (connect(metrics_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		log_debug("Metrics socket not available, metrics disabled: %s", strerror(errno));
		close(metrics_socket);
		metrics_socket = -1;
		return -1;
	}
	
	log_info("Metrics system initialized");
	return 0;
}

static int send_metrics_message(const struct metrics_message *msg) {
	if (metrics_socket < 0) {
		return -1;
	}
	
	ssize_t sent = send(metrics_socket, msg, sizeof(*msg), MSG_DONTWAIT);
	if (sent < 0) {
		if (errno != EAGAIN && errno != EWOULDBLOCK) {
			log_debug("Failed to send metrics message: %s", strerror(errno));
		}
		return -1;
	}
	
	if (sent != sizeof(*msg)) {
		log_debug("Partial metrics message sent");
		return -1;
	}
	
	return 0;
}

static char* extract_device_id_from_uuid(const char* uuid) {
	static char device_id[MAX_DEVICE_ID_LEN];
	
	if (strncmp(uuid, "GPU-", 4) == 0) {
		snprintf(device_id, sizeof(device_id), "nvidia0");
	} else {
		strlcpy(device_id, "nvidia0", sizeof(device_id));
	}
	
	return device_id;
}

static int get_process_id_from_client(const struct nvshare_client *client) {
	char proc_path[256];
	char comm_path[256];
	FILE *fp;
	int pid = -1;
	
	snprintf(proc_path, sizeof(proc_path), "/proc/net/unix");
	fp = fopen(proc_path, "r");
	if (!fp) {
		return -1;
	}
	
	char line[512];
	while (fgets(line, sizeof(line), fp)) {
		char *token = strtok(line, " \t");
		if (!token) continue;
		
		for (int i = 0; i < 6 && token; i++) {
			token = strtok(NULL, " \t");
		}
		
		if (token) {
			int sock_inode = atoi(token);
			
			char search_path[256];
			snprintf(search_path, sizeof(search_path), "/proc/*/fd/*");
			
			break;
		}
	}
	
	fclose(fp);
	
	return getpid();
}

int metrics_register_session(const struct nvshare_client *client, const char *device_id, int process_id) {
	struct metrics_message msg;
	struct pod_gpu_session *session;
	
	if (!client || metrics_socket < 0) {
		return -1;
	}
	
	if (process_id <= 0) {
		process_id = get_process_id_from_client(client);
	}
	
	pthread_mutex_lock(&metrics_mutex);
	
	LL_FOREACH(active_sessions, session) {
		if (session->client_id == client->id && 
		    strcmp(session->device_id, device_id) == 0) {
			pthread_mutex_unlock(&metrics_mutex);
			return 0;
		}
	}
	
	session = malloc(sizeof(*session));
	if (!session) {
		pthread_mutex_unlock(&metrics_mutex);
		return -1;
	}
	
	strlcpy(session->namespace, client->pod_namespace, sizeof(session->namespace));
	strlcpy(session->pod_name, client->pod_name, sizeof(session->pod_name));
	strlcpy(session->container, "container", sizeof(session->container));
	strlcpy(session->device_id, device_id, sizeof(session->device_id));
	session->process_id = process_id;
	session->client_id = client->id;
	session->start_time = time(NULL);
	session->last_active = session->start_time;
	
	LL_APPEND(active_sessions, session);
	
	pthread_mutex_unlock(&metrics_mutex);
	
	memset(&msg, 0, sizeof(msg));
	msg.type = METRICS_SESSION_START;
	strlcpy(msg.namespace, client->pod_namespace, sizeof(msg.namespace));
	strlcpy(msg.pod_name, client->pod_name, sizeof(msg.pod_name));
	strlcpy(msg.container, "container", sizeof(msg.container));
	strlcpy(msg.device_id, device_id, sizeof(msg.device_id));
	msg.process_id = process_id;
	msg.client_id = client->id;
	msg.timestamp = session->start_time;
	
	if (send_metrics_message(&msg) == 0) {
		log_debug("Registered GPU session for pod %s/%s on device %s", 
			  client->pod_namespace, client->pod_name, device_id);
	}
	
	return 0;
}

int metrics_unregister_session(const struct nvshare_client *client, const char *device_id) {
	struct metrics_message msg;
	struct pod_gpu_session *session, *tmp;
	
	if (!client || metrics_socket < 0) {
		return -1;
	}
	
	pthread_mutex_lock(&metrics_mutex);
	
	LL_FOREACH_SAFE(active_sessions, session, tmp) {
		if (session->client_id == client->id && 
		    strcmp(session->device_id, device_id) == 0) {
			
			LL_DELETE(active_sessions, session);
			
			memset(&msg, 0, sizeof(msg));
			msg.type = METRICS_SESSION_END;
			strlcpy(msg.namespace, session->namespace, sizeof(msg.namespace));
			strlcpy(msg.pod_name, session->pod_name, sizeof(msg.pod_name));
			strlcpy(msg.container, session->container, sizeof(msg.container));
			strlcpy(msg.device_id, session->device_id, sizeof(msg.device_id));
			msg.process_id = session->process_id;
			msg.client_id = session->client_id;
			msg.timestamp = time(NULL);
			
			free(session);
			
			pthread_mutex_unlock(&metrics_mutex);
			
			if (send_metrics_message(&msg) == 0) {
				log_debug("Unregistered GPU session for pod %s/%s on device %s",
					  msg.namespace, msg.pod_name, device_id);
			}
			
			return 0;
		}
	}
	
	pthread_mutex_unlock(&metrics_mutex);
	return -1;
}

int metrics_update_session(const struct nvshare_client *client, const char *device_id) {
	struct pod_gpu_session *session;
	
	if (!client || metrics_socket < 0) {
		return -1;
	}
	
	pthread_mutex_lock(&metrics_mutex);
	
	LL_FOREACH(active_sessions, session) {
		if (session->client_id == client->id && 
		    strcmp(session->device_id, device_id) == 0) {
			session->last_active = time(NULL);
			break;
		}
	}
	
	pthread_mutex_unlock(&metrics_mutex);
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
	
	if (metrics_socket >= 0) {
		close(metrics_socket);
		metrics_socket = -1;
	}
	
	log_info("Metrics system cleaned up");
}