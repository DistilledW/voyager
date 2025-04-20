DATA_DIR=/data/zliu/small_city 
PROJECT_DIR=/data/zliu/voyager 
IMAGE_NAME=voyager 
IMAGE_VERSION=h3dgs 
GPU_INDEX=2 
CONTAINER=voyager_${GPU_INDEX} 

docker run -it --rm \
    --ulimit memlock=-1:-1 \
    --shm-size=16g \
    --gpus device=${GPU_INDEX} \
    --network host \
    -v ${DATA_DIR}:/workspace/data -v ${PROJECT_DIR}:/workspace/code \
    --name ${CONTAINER} \
    ${IMAGE_NAME}:${IMAGE_VERSION} bash 

# docker commit ${CONTAINER} IMAGE_NAME:${IMAGE_NEXT_VERSION} 
# First try:
# nvcc --version
# conda env list
# conda activate 3dgs
# python -c "import torch; print(torch.cuda.is_available())"

# sudo nvidia-smi -i 3 -pl 100 # 300
# sudo nvidia-smi -i 3 -lgc 1300,1300
# sudo nvidia-smi -rgc 
