COMMIT=${1}
export COMMIT
TAG="${2}/pyafq:${COMMIT}"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG ./pyafq_docker
docker push $TAG