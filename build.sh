#!/bin/bash

commit_id="$(git rev-parse HEAD)"
registry="docker.io"
repository="thexjx/transformers-online-inference"
tag="${commit_id}-amd64"

function build() {
    image_name="${registry}/${repository}:${tag}"
    echo image_name: "${image_name}"
    docker build --no-cache --platform linux/amd64 -t "${image_name}" -f Dockerfile .
    docker push "${image_name}"
}

build