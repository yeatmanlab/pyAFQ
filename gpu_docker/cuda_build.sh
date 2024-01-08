if [ $# -lt 1 ]; then
    echo "Error: At least one argument is required."
    exit 1
fi

export AFQ_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export CUDA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

apptainer build ${1} ${CUDA_DIR}/cuda_track.def

