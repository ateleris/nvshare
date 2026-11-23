/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2023, Georgios Alexopoulos
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	resourceName = "nvshare.com/gpu"
	serverSock   = pluginapi.DevicePluginPath + "nvshare-device-plugin.sock"

	registerTimeout = 5 * time.Second
)

type NvshareDevicePlugin struct {
	pluginapi.UnimplementedDevicePluginServer

	devs   []*pluginapi.Device
	socket string

	stop   chan interface{}
	health chan *pluginapi.Device

	server *grpc.Server
}

func NewNvshareDevicePlugin() *NvshareDevicePlugin {
	return &NvshareDevicePlugin{
		devs:   getDevices(),
		socket: serverSock,

		stop:   make(chan interface{}),
		health: make(chan *pluginapi.Device),
	}
}

func (m *NvshareDevicePlugin) initialize() {
	m.server = grpc.NewServer([]grpc.ServerOption{}...)
	m.health = make(chan *pluginapi.Device)
	m.stop = make(chan interface{})
}

func (m *NvshareDevicePlugin) cleanup() {
	close(m.stop)
	m.server = nil
	m.health = nil
	m.stop = nil
}

/*
 * Starts the gRPC server, registers the device plugin with the Kubelet.
 */
func (m *NvshareDevicePlugin) Start() error {
	m.initialize()

	err := m.Serve()
	if err != nil {
		log.Printf("Could not start device plugin for '%s': %s", resourceName, err)
		m.cleanup()
		return err
	}
	log.Printf("Starting to serve '%s' on %s", resourceName, m.socket)

	err = m.Register()
	if err != nil {
		log.Printf("Could not register device plugin: %s", err)
		m.Stop()
		return err
	}
	log.Printf("Registered device plugin for '%s' with Kubelet", resourceName)

	return nil
}

/* Stop the gRPC server and clean up the UNIX socket file */
func (m *NvshareDevicePlugin) Stop() error {
	if (m == nil) || (m.server == nil) {
		return nil
	}
	log.Printf("Stopping to serve '%s' on %s\n", resourceName, m.socket)
	m.server.Stop()
	err := os.Remove(m.socket)
	if (err != nil) && (!os.IsNotExist(err)) {
		return err
	}
	m.cleanup()
	return nil
}

/* Starts the gRPC server which serves incoming requests from kubelet */
func (m *NvshareDevicePlugin) Serve() error {
	os.Remove(m.socket)
	sock, err := net.Listen("unix", m.socket)
	if err != nil {
		return err
	}

	pluginapi.RegisterDevicePluginServer(m.server, m)

	server := m.server
	go func() {
		log.Printf("Starting gRPC server for '%s'", resourceName)
		if err := server.Serve(sock); err != nil {
			log.Fatalf("gRPC server for '%s' failed: %v", resourceName, err)
		}
	}()

	return nil
}

/* Registers the device plugin for resourceName with kubelet */
func (m *NvshareDevicePlugin) Register() error {
	conn, err := grpc.NewClient("unix://"+pluginapi.KubeletSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	reqt := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     path.Base(m.socket),
		ResourceName: resourceName,
		Options: &pluginapi.DevicePluginOptions{
			GetPreferredAllocationAvailable: false,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), registerTimeout)
	defer cancel()
	_, err = client.Register(ctx, reqt)
	return err
}

func (m *NvshareDevicePlugin) GetDevicePluginOptions(context.Context, *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	options := &pluginapi.DevicePluginOptions{
		PreStartRequired:                false,
		GetPreferredAllocationAvailable: false,
	}
	return options, nil
}

/*
 * Reports available devices to kubelet and (theoretically) updates that list
 * according to their health status.
 *
 * We don't monitor health for Nvshare devices at the moment, we consider them
 * all to be healthy.
 *
 * If the underlying GPU goes unhealthy, NVIDIA's device
 * plugin will detect it and fail the (Nvshare device plugin) Pod.
 *
 * For device health handling see also the official device plugin proposal:
 * https://github.com/kubernetes/community/blob/c4466d9fbfa6645410083e37560810a9aa000267/contributors/design-proposals/resource-management/device-plugin.md#healthcheck-and-failure-recovery
 */
func (m *NvshareDevicePlugin) ListAndWatch(e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	if err := s.Send(&pluginapi.ListAndWatchResponse{Devices: m.devs}); err != nil {
		return err
	}
	log.Printf("Sent ListAndWatchResponse with DeviceIDs")
	select {
	case <-m.stop:
	case <-s.Context().Done():
	}
	return nil
}

/*
 * Kubelet calls this method when it wants to run containers in a Pod that
 * has requested an Nvshare GPU.
 */
func (m *NvshareDevicePlugin) Allocate(ctx context.Context, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	responses := pluginapi.AllocateResponse{}
	for _, req := range reqs.ContainerRequests {
		for _, id := range req.DevicesIds {
			log.Printf("Received Allocate request for %s", id)
			if !m.deviceExists(id) {
				return nil, fmt.Errorf("invalid allocation request for '%s' - unknown device: %s", resourceName, id)
			}
		}

		response := pluginapi.ContainerAllocateResponse{}

		envsMap := make(map[string]string)
		envsMap["LD_PRELOAD"] = LibNvshareContainerPath
		if !nvidiaRuntimeUseMounts {
			envsMap[NvidiaDevicesEnvVar] = UUID
		} else {
			envsMap[NvidiaDevicesEnvVar] = NvidiaExposeMountDir
		}

		response.Envs = envsMap

		/* Add libnvshare to the Mounts section of the ContainerResponse */
		var mounts []*pluginapi.Mount
		/* Mount libnvshare */
		mount := &pluginapi.Mount{
			HostPath:      LibNvshareHostPath,
			ContainerPath: LibNvshareContainerPath,
			ReadOnly:      true,
		}
		mounts = append(mounts, mount)
		/* Mount scheduler socket */
		mount = &pluginapi.Mount{
			HostPath:      SocketHostPath,
			ContainerPath: SocketContainerPath,
			ReadOnly:      true,
		}
		mounts = append(mounts, mount)
		/*
		 * If the method for requesting GPUs from the underlying NVIDIA
		 * container runtime is through Volume Mounts, set symbolic /dev/null
		 * mount for GPU exposure
		 */
		if nvidiaRuntimeUseMounts {
			mount = &pluginapi.Mount{
				HostPath:      NvidiaExposeMountHostPath,
				ContainerPath: filepath.Join(NvidiaExposeMountDir, UUID),
			}
			mounts = append(mounts, mount)
		}

		response.Mounts = mounts
		responses.ContainerResponses = append(responses.ContainerResponses, &response)
	}

	return &responses, nil
}

/* GetPreferredAllocation is unimplemented for Nvshare device plugin */
func (m *NvshareDevicePlugin) GetPreferredAllocation(ctx context.Context, r *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	response := &pluginapi.PreferredAllocationResponse{}
	return response, nil
}

/* PreStartContainer is unimplemented for Nvshare device plugin */
func (m *NvshareDevicePlugin) PreStartContainer(context.Context, *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	response := &pluginapi.PreStartContainerResponse{}
	return response, nil
}

func (m *NvshareDevicePlugin) deviceExists(id string) bool {
	for _, d := range m.devs {
		if d.ID == id {
			return true
		}
	}
	return false
}
