COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT
TAG="ghcr.io/${2}/afqsi:${COMMIT}"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG --build-arg COMMIT ./afqsi_docker
docker run --rm -it $TAG pytest /usr/local/lib/python3.7/site-packages/AFQ/tests/test_api.py::test_AFQ_data_waypoint
docker push $TAG