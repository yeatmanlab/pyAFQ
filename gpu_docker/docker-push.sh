NO_TAG="ghcr.io/${1}/pyafq_gpu"
NO_TAG="$(echo "${NO_TAG}" | tr -d '[:space:]')"

docker push --all-tags $NO_TAG
