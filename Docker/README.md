# Docker

## Building

`docker build`: The base command
`-t`: Tag which is simply a user name for the build instance.
`fireship/demo:1.0`: The name of the build instance with its version number.
`.`: The path to the build instance

`docker build -t fireship/demo:1.0 .`

## Running

`docker run`: The base command to run the build
`-p`: Port forwarding simply exposes a port on our local machine to get to the port in running instance: `Local:Container`.
`[docker build id]`: The build id of the build instance.

`docker run -p 5000:8080 b95264fgdfg`

## Volumes

This is a presistent volume of data, that can be accessible by specific containers or all the containers, and it will withstand even if the docker container stops running.

`docker volume create [volume-name]`

`docker run --mount source=[volume-name], target=[directory: ex: '/example']`

## Docker Compose

`docker-compose up`: Start
`docker-compose down`: Stop

# History and Motivation

## Motivation

Docker came along and now setting up our development and deployment environment, and its entire sequence of events into a singular system and can quickly be used and setup.

## What is a Container?

A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application.

We can run multiple instances of these Docker containers.

Container Image: Has all of the dependencies in Docker. (Class)
Container: Actual Instance of that Container Image. (Instance)

## Open Container Initiative (OCI)

Open governance structure for the express purpose of creating open industry standards around container formats and runtimes.

- Runtime Specification
- Image Specification
- Distribution Specification

## Evolution of Virtualization

### Bare Metal

Before Containers of VMs, we just had the hardware it self (also known as Bare Metal), and we would install any OS we would want and we would install any libraries and then run applications on top of it.

- Hellish Dependency Conflicts (When running multiple applications)
- Low utilization efficiency
- Large blast radius
- Slow start up and Shut down Speed
- Very slow provisioning and Decommissioning

### Virtualization

This was the next level abstraction for [Bare Metal](#bare-metal), what it does is pretty much take out the requirement to manage Physical Hardware, Hardware OS, and utilize a Hypervisor to management the virtual machines capacity, in which the virtual machines would seem like they have the exact specific configured RAM, CPU Core, etc... but it is configured by the Hypervisor, and then we can have multiple virtual machines running where it would resolve a lot of the issues that was there within the [Bare Metal](#bare-metal) Era.

- No dependency conflicts
- Better utilization efficacy
- Small blast radius
- Faster startup and shutdown (mins)
- Faster provisioning and Decommissioning

### Containers

The container can be put on top of either a Virtual or Physical Machine, and then the container it self, has its own application and libraries. Similar to the hypervisor in Virtualization, It knows how to take a container and then run it in as a container, the Containers have their own way of running Linux, it is sharing the kernel with the host machine. We're using certain features of the Linux Kernal to have isolation between the containers.

- No dependency conflicts
- Better utilization efficiency
- Small blast radius
- Faster startup and shutdown
- Even faster provisioning and Decommissioning
- Lightweight enough to use in development

#### Types of Containers

- Desktop Container Platforms
  - Docker
  - Podman
- Container Runtimes
  - ContainerD
  - Cri-o

## Current

Virtual Machines + Containers + Orchestrators are combined and used together at the moment, so we have a structure in which containers are inside VM's which are running on top of the bare metal hardware, and to manage these complex setup's is why orchestration tools are needed such as Kubernetes.

# Building Container Images
docker build -t [name] .
