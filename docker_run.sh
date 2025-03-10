PROJECT_DIR=/home/zliu/project/webGS
DATA_DIR=/data/zliu/docker_3dgs
IMAGE_NAME=webgs
IMAGE_VERSION=v0
IMAGE_NEXT_VERSION=v1
CONTAINER=webGS

docker run -it --rm \
    --ulimit memlock=-1:-1 \
    --shm-size=16g \
    --gpus all \
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
