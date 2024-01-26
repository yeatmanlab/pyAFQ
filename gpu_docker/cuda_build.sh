#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Error: At least one argument is required."
    exit 1
fi

export AFQ_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

sed "s|PLACEHOLDER_FOR_PACKAGE_PATH|${AFQ_PATH}|g" ${AFQ_PATH}/gpu_docker/cuda_track_template.def > ${AFQ_PATH}/gpu_docker/_temp.def

apptainer build ${1} ${AFQ_PATH}/gpu_docker/_temp.def

rm ${AFQ_PATH}/gpu_docker/_temp.def

