COMMIT=${1}
export COMMIT
TAG="${2}/pyafq:${COMMIT}"
TAG="$(echo -e "${TAG}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG .
docker push $TAG