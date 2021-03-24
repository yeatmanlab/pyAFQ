export COMMIT=$1
export FREESURFER_LICENSE=$3
docker build --no-cache -t $2/pyafq:$1 .
docker push $2/pyafq:$1