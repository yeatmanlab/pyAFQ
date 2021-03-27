COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT
TAG="ghcr.io/${2}/afqsi:${COMMIT}"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG --build-arg COMMIT ./afqsi_docker
docker push $TAG