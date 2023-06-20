COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT

NVIDIAVERSION=${3}
export NVIDIAVERSION
NO_TAG="ghcr.io/${2}/pyafq_gpu_cuda_${4}"
TAG="${NO_TAG}:${COMMIT}"
TAG2="${NO_TAG}:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
TAG2="$(echo "${TAG2}" | tr -d '[:space:]')"
echo $TAG
docker build -t $TAG -t $TAG2 --build-arg COMMIT --build-arg NVIDIAVERSION ./gpu_docker
