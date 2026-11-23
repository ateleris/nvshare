#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "common.h"
#include "pod_identity.h"

#define POD_LOG_DIR "/var/log/pods"

#define POD_UID_LEN_MIN 32
#define POD_UID_LEN_MAX 36

#ifndef SO_PEERPIDFD
#define SO_PEERPIDFD 77
#endif

static int pod_uid_from_cgroup(const char *line, char *uid)
{
	const char *p = line;
	size_t len;

	while ((p = strstr(p, "pod")) != NULL) {
		p += 3;
		len = 0;
		while (isxdigit((unsigned char)p[len]) || p[len] == '-' || p[len] == '_')
			len++;
		if (len < POD_UID_LEN_MIN || len > POD_UID_LEN_MAX)
			continue;
		for (size_t i = 0; i < len; i++)
			uid[i] = (p[i] == '_') ? '-' : p[i];
		uid[len] = '\0';
		return 0;
	}
	return -1;
}

static int pod_uid_of_pid(pid_t pid, char *uid)
{
	char path[64];
	char line[PATH_MAX + 256];
	FILE *fp;
	int found = 0;

	snprintf(path, sizeof(path), "/proc/%ld/cgroup", (long)pid);
	fp = fopen(path, "r");
	if (fp == NULL)
		return -1;
	while (!found && fgets(line, sizeof(line), fp) != NULL) {
		if (pod_uid_from_cgroup(line, uid) == 0)
			found = 1;
	}
	fclose(fp);
	return found;
}

static int pod_names_from_uid(const char *uid, char *namespace,
			      size_t namespace_len, char *name, size_t name_len)
{
	DIR *dir;
	struct dirent *entry;
	const char *first, *last;
	int found = -1;

	dir = opendir(POD_LOG_DIR);
	if (dir == NULL)
		return -1;
	while ((entry = readdir(dir)) != NULL) {
		first = strchr(entry->d_name, '_');
		last = strrchr(entry->d_name, '_');
		if (first == NULL || first == last || strcmp(last + 1, uid) != 0)
			continue;
		if (first == entry->d_name || last == first + 1)
			continue;
		if ((size_t)(first - entry->d_name) >= namespace_len ||
		    (size_t)(last - first - 1) >= name_len)
			continue;
		memcpy(namespace, entry->d_name, first - entry->d_name);
		namespace[first - entry->d_name] = '\0';
		memcpy(name, first + 1, last - first - 1);
		name[last - first - 1] = '\0';
		found = 0;
		break;
	}
	closedir(dir);
	return found;
}

static int peer_exited(int fd)
{
	struct pollfd pfd;
	socklen_t len = sizeof(pfd.fd);
	int exited;

	if (getsockopt(fd, SOL_SOCKET, SO_PEERPIDFD, &pfd.fd, &len) != 0)
		return errno == ESRCH;
	pfd.events = POLLIN;
	exited = RETRY_INTR(poll(&pfd, 1, 0)) != 0;
	close(pfd.fd);
	return exited;
}

int pod_identity_of_peer(int fd, char *namespace, size_t namespace_len,
			 char *name, size_t name_len)
{
	static int warned;
	struct ucred cred;
	socklen_t cred_len = sizeof(cred);
	char uid[POD_UID_LEN_MAX + 1];
	int in_pod;

	if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &cred_len) != 0 ||
	    cred.pid <= 0) {
		if (!warned) {
			log_warn("Cannot see the processes of clients, so pod"
				 " identities are self-reported and not verified."
				 " Run the scheduler in the host PID namespace"
				 " (hostPID: true) to verify them.");
			warned = 1;
		}
		return 0;
	}

	in_pod = pod_uid_of_pid(cred.pid, uid);
	if (in_pod < 0 || peer_exited(fd)) {
		log_debug("Client process %ld is gone, cannot tell which pod it"
			  " runs in", (long)cred.pid);
		return -1;
	}
	if (in_pod == 0)
		return 0;

	if (pod_names_from_uid(uid, namespace, namespace_len, name, name_len) != 0) {
		log_warn("No directory for pod %s in %s, reporting the pod by"
			 " its UID", uid, POD_LOG_DIR);
		strlcpy(namespace, "unknown", namespace_len);
		strlcpy(name, uid, name_len);
	}
	return 1;
}
