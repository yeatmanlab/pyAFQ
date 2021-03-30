COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT
TAG="ghcr.io/${2}/pyafq:${COMMIT}"
TAG2="ghcr.io/${2}/pyafq:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG -t $TAG2 --build-arg COMMIT ./pyafq_docker
docker push --all-tags ghcr.io/${2}/pyafq
