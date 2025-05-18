GPU_INDEX="$1" 
SCENES="$2"
TAGS="$3"
DATA_DIR=/data/voyager/$SCENES
PROJECT_DIR=/data/zliu/voyager 
IMAGE_NAME=voyager 
IMAGE_VERSION=h3dgs 
CONTAINER=voyager_${SCENES}_${TAGS}_${GPU_INDEX}

docker run -it --rm --detach-keys="ctrl-x" \
    --ulimit memlock=-1:-1 --shm-size=16g \
    --gpus device=${GPU_INDEX} \
    -v ${DATA_DIR}:/workspace/data \
    -v ${PROJECT_DIR}:/workspace/code \
    --name ${CONTAINER} \
    --network host \
    ${IMAGE_NAME}:${IMAGE_VERSION} bash 

# docker commit ${CONTAINER} IMAGE_NAME:${IMAGE_NEXT_VERSION} 
