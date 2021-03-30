COMMIT=${1}
COMMIT="$(echo "${COMMIT}" | tr -d '[:space:]')"
export COMMIT
TAG="ghcr.io/${2}/afqsi:${COMMIT}"
TAG2="ghcr.io/${2}/afqsi:latest"
TAG="$(echo "${TAG}" | tr -d '[:space:]')"
echo $TAG
docker build --no-cache -t $TAG -t $TAG2 --build-arg COMMIT ./afqsi_docker
docker run --rm -i --entrypoint /usr/local/miniconda/bin/pytest $TAG /usr/local/miniconda/lib/python3.7/site-packages/AFQ/tests/test_api.py::test_AFQ_data_waypoint
docker push --all-tags ghcr.io/${2}/afqsi