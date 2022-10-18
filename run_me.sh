#!/bin/bash

xhost local:root > /dev/null
export XAUTH_FILE=$(xauth info | head -n1 | tr -s ' ' | cut -d' ' -f3)
export ARCH=$(uname -p)

# to run frigate run
docker-compose -f docker-compose-run.yml up
