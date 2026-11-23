package main

import (
	"context"
	"reflect"
	"testing"

	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

func setGlobals(t *testing.T, uuid string, devices int, useMounts bool) {
	oldUUID, oldDevices, oldMounts := UUID, NvshareVirtualDevices, nvidiaRuntimeUseMounts
	t.Cleanup(func() {
		UUID, NvshareVirtualDevices, nvidiaRuntimeUseMounts = oldUUID, oldDevices, oldMounts
	})
	UUID, NvshareVirtualDevices, nvidiaRuntimeUseMounts = uuid, devices, useMounts
}

func allocate(t *testing.T, ids ...string) (*pluginapi.AllocateResponse, error) {
	t.Helper()
	return NewNvshareDevicePlugin().Allocate(context.Background(), &pluginapi.AllocateRequest{
		ContainerRequests: []*pluginapi.ContainerAllocateRequest{{DevicesIds: ids}},
	})
}

func TestGetDevices(t *testing.T) {
	setGlobals(t, "GPU-abc", 3, false)

	var ids []string
	for _, d := range getDevices() {
		if d.Health != pluginapi.Healthy {
			t.Errorf("device %s is %s, want %s", d.ID, d.Health, pluginapi.Healthy)
		}
		ids = append(ids, d.ID)
	}
	want := []string{"GPU-abc__1", "GPU-abc__2", "GPU-abc__3"}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("got devices %v, want %v", ids, want)
	}
}

func TestAllocate_EnvExposure(t *testing.T) {
	setGlobals(t, "GPU-abc", 2, false)

	resp, err := allocate(t, "GPU-abc__2")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ContainerResponses) != 1 {
		t.Fatalf("got %d container responses, want 1", len(resp.ContainerResponses))
	}
	got := resp.ContainerResponses[0]

	wantEnvs := map[string]string{
		"LD_PRELOAD":        LibNvshareContainerPath,
		NvidiaDevicesEnvVar: "GPU-abc",
	}
	if !reflect.DeepEqual(got.Envs, wantEnvs) {
		t.Errorf("got envs %v, want %v", got.Envs, wantEnvs)
	}

	wantMounts := []*pluginapi.Mount{
		{HostPath: LibNvshareHostPath, ContainerPath: LibNvshareContainerPath, ReadOnly: true},
		{HostPath: SocketHostPath, ContainerPath: SocketContainerPath, ReadOnly: true},
	}
	if !reflect.DeepEqual(got.Mounts, wantMounts) {
		t.Errorf("got mounts %v, want %v", got.Mounts, wantMounts)
	}
}

func TestAllocate_MountExposure(t *testing.T) {
	setGlobals(t, "GPU-abc", 2, true)

	resp, err := allocate(t, "GPU-abc__1")
	if err != nil {
		t.Fatal(err)
	}
	got := resp.ContainerResponses[0]

	if got.Envs[NvidiaDevicesEnvVar] != NvidiaExposeMountDir {
		t.Errorf("%s is %q, want %q", NvidiaDevicesEnvVar, got.Envs[NvidiaDevicesEnvVar], NvidiaExposeMountDir)
	}
	want := &pluginapi.Mount{
		HostPath:      NvidiaExposeMountHostPath,
		ContainerPath: NvidiaExposeMountDir + "/GPU-abc",
	}
	if len(got.Mounts) != 3 || !reflect.DeepEqual(got.Mounts[2], want) {
		t.Errorf("got mounts %v, want a third mount %v", got.Mounts, want)
	}
}

func TestAllocate_UnknownDevice(t *testing.T) {
	setGlobals(t, "GPU-abc", 2, false)

	for _, id := range []string{"GPU-abc__3", "GPU-other__1", ""} {
		if _, err := allocate(t, id); err == nil {
			t.Errorf("allocating %q succeeded, want an error", id)
		}
	}
}
