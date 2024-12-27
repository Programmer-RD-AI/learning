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
