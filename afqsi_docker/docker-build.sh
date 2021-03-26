export COMMIT=$1
docker build --no-cache -t $2/pyafq:$1 .
docker push $2/afqsi:$1