COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT

NO_TAG="ghcr.io/${2}/pyafq_gpu_cuda_11"
TAG="${NO_TAG}:${COMMIT}"
TAG2="${NO_TAG}:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
TAG2="$(echo "${TAG2}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG -t $TAG2 --build-arg COMMIT ./gpu_docker

NVIDIAVERSION="12.0.1-devel-ubuntu20.04"
export NVIDIAVERSION
NO_TAG="ghcr.io/${2}/pyafq_gpu_cuda_12"
TAG="${NO_TAG}:${COMMIT}"
TAG2="${NO_TAG}:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
TAG2="$(echo "${TAG2}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG -t $TAG2 --build-arg COMMIT --build-arg NVIDIAVERSION ./gpu_docker
