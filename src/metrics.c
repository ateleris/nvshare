#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int metrics_init(void) {
	log_info("Initializing metrics system...");
	
	socket_fd = socket(AF_LOCAL, SOCK_DGRAM, 0);
	if (socket_fd < 0) {
		log_warn("METRICS ERROR: Failed to create socket: %s", strerror(errno));
		return -1;
	}
	log_debug("Created metrics socket fd=%d", socket_fd);
	
	int flags;
	flags = fcntl(socket_fd, F_GETFL, 0);
	if (flags < 0 || fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
		log_warn("METRICS ERROR: Failed to set socket non-blocking: %s", strerror(errno));
		close(socket_fd);
		socket_fd = -1;
		return -1;
	}
	log_debug("Set metrics socket to non-blocking mode");
	
	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_LOCAL;
	strlcpy(addr.sun_path, METRICS_SOCKET_PATH, sizeof(addr.sun_path));
	log_debug("Connecting to metrics socket: %s", METRICS_SOCKET_PATH);
	
	if (connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		log_warn("METRICS WARNING: Cannot connect to device plugin socket: %s", strerror(errno));
		log_warn("METRICS WARNING: Metrics will be disabled until device plugin is available");
		close(socket_fd);
		socket_fd = -1;
		return -1;
	}
	
	log_info("METRICS SUCCESS: Connected to device plugin at %s", METRICS_SOCKET_PATH);
	return 0;
}

static int send_metrics_message(const struct metrics_message *msg) {
	if (socket_fd < 0) {
		log_debug("METRICS ERROR: Cannot send message - socket not initialized");
		return -1;
	}
	
	log_debug("METRICS: Sending message type %d for pod %s/%s", msg->type, msg->namespace, msg->pod_name);
	
	ssize_t sent = send(socket_fd, msg, sizeof(*msg), MSG_DONTWAIT);
	if (sent < 0) {
		if (errno != EAGAIN && errno != EWOULDBLOCK) {
			log_warn("METRICS ERROR: Failed to send message: %s", strerror(errno));
		} else {
			log_debug("METRICS: Socket busy, message dropped");
		}
		return -1;
	}
	
	if (sent != sizeof(*msg)) {
		log_warn("METRICS ERROR: Partial message sent (%zd/%zu bytes)", sent, sizeof(*msg));
		return -1;
	}
	
	log_info("METRICS SUCCESS: Sent %zd bytes to device plugin", sent);
	return 0;
}

/*
static char* extract_device_id_from_uuid(const char* uuid) {
	static char device_id[MAX_DEVICE_ID_LEN];
	
	if (strncmp(uuid, "GPU-", 4) == 0) {
		snprintf(device_id, sizeof(device_id), "nvidia0");
	} else {
		strlcpy(device_id, "nvidia0", sizeof(device_id));
	}
	
	return device_id;
}
*/

static int get_process_id_from_client(const struct nvshare_client *client) {
	
	//FILE* fd = NULL;
	//
	//fd = fopen("/proc/net/unix", "r");
	//if (!fd) {
	//	return -1;
	//}
	//
	//char line[512];
	//while (fgets(line, sizeof(line), fd)) {
	//	char *token = strtok(line, " \t");
	//	if (!token) continue;
	//	
	//	for (int i = 0; i < 6 && token; i++) {
	//		token = strtok(NULL, " \t");
	//	}
	//	
	//	if (token) {
	//		int sock_inode = atoi(token);
	//		
	//		char search_path[256];
	//		snprintf(search_path, sizeof(search_path), "/proc/*/fd/*");
	//		
	//		break;
	//	}
	//}
	//
	//fclose(fd);
	
	return getpid();
}

int metrics_register_session(const struct nvshare_client *client, const char *device_id, int process_id) {
	if (!client) {
		log_warn("METRICS ERROR: Cannot register session - client is NULL");
		return -1;
	}
	
	if (socket_fd < 0) {
		log_debug("METRICS WARNING: Cannot register session - socket not connected");
		return -1;
	}

	if (device_id == 0x00) {
		log_warn("METRICS ERROR: Cannot register session - device id is empty string");
		return -1;
	}

	log_info("METRICS: Registering GPU session for pod %s/%s on device %s", 
		 client->pod_namespace, client->pod_name, device_id);
	
	if (process_id <= 0) {
		process_id = get_process_id_from_client(client);
		log_debug("METRICS: Resolved process_id to %d", process_id);
	}
	
	pthread_mutex_lock(&metrics_mutex);
	
	struct pod_gpu_session* session;
	LL_FOREACH(active_sessions, session) {
		if (session->client_id == client->id && 
		    strcmp(session->device_id, device_id) == 0) {
			log_debug("METRICS: Session already exists for client %lx", client->id);
			pthread_mutex_unlock(&metrics_mutex);
			return 0;
		}
	}
	
	session = malloc(sizeof(*session));
	if (!session) {
		log_warn("METRICS ERROR: Failed to allocate session memory");
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
	log_debug("METRICS: Added session to active list (client_id=%lx)", client->id);
	
	pthread_mutex_unlock(&metrics_mutex);
	
	struct metrics_message msg;
	memset(&msg, 0, sizeof(msg));
	msg.type = (int32_t)METRICS_SESSION_START;
	strlcpy(msg.namespace, client->pod_namespace, sizeof(msg.namespace));
	strlcpy(msg.pod_name, client->pod_name, sizeof(msg.pod_name));
	strlcpy(msg.container, "container", sizeof(msg.container));
	strlcpy(msg.device_id, device_id, sizeof(msg.device_id));
	msg.process_id = process_id;
	msg.client_id = client->id;
	msg.timestamp = session->start_time;
	
	log_debug("METRICS: Prepared message for %s/%s", msg.namespace, msg.pod_name);
	
	if (send_metrics_message(&msg) == 0) {
		log_info("METRICS SUCCESS: Registered GPU session for pod %s/%s on device %s", 
			  client->pod_namespace, client->pod_name, device_id);
	} else {
		log_warn("METRICS ERROR: Failed to send registration message");
	}
	
	return 0;
}

int metrics_unregister_session(const struct nvshare_client *client, const char *device_id) {
	struct metrics_message msg;
	struct pod_gpu_session *session, *tmp;
	
	if (!client || socket_fd < 0) {
		return -1;
	}

	if (device_id == 0x00) {
		log_warn("METRICS ERROR: Cannot register session - device id is empty string");
		return -1;
	}
	
	pthread_mutex_lock(&metrics_mutex);
	
	LL_FOREACH_SAFE(active_sessions, session, tmp) {
		if (session->client_id == client->id && 
		    strcmp(session->device_id, device_id) == 0) {
			
			LL_DELETE(active_sessions, session);
			
			memset(&msg, 0, sizeof(msg));
			msg.type = (int32_t)METRICS_SESSION_END;
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
	
	if (!client || socket_fd < 0) {
		return -1;
	}

	if (device_id == 0x00) {
		log_warn("METRICS ERROR: Cannot register session - device id is empty string");
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
	
	if (socket_fd >= 0) {
		close(socket_fd);
		socket_fd = -1;
	}
	
	log_info("Metrics system cleaned up");
}