PROJECT_DIR=/data/zliu/webGS 
DATA_DIR=/data/zliu/docker_3dgs/small_city
IMAGE_NAME=webgs
IMAGE_VERSION=v0
IMAGE_NEXT_VERSION=v1
GPU_INDEX=1
CONTAINER=render_2

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
# pip install submodules/gaussianhierarchy
# pip install submodules/hierarchy-rasterizer
# pip install submodules/simple-knn
