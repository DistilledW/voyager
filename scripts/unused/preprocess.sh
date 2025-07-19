#!/bin/bash
# conda activate check 
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH"
    exit 1 
else 
    ENV_NAME=$(basename "$CONDA_PREFIX") 
    if [ "$ENV_NAME" != "3dgs" ]; then 
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate "3dgs" 
    fi 
fi 

DATASET=$1 
if [ "$DATASET" = "360" ]; then # 4946 * 3286 
    SCENES=("garden") # "bicycle" "stump" "room" "counter" "kitchen" "bonsai" "flowers" "treehill"
    resolution=4 
elif [ "$DATASET" = "MegaNeRF" ]; then # 4608 * 3456 
    SCENES=("building" "rubble") 
    resolution=4 
    for scene in "${SCENES[@]}" 
    do 
        ln -s /workspace/data/${scene}/train/images /workspace/data/${scene}/images 
    done 
elif [ "${DATASET}" = "small_city" ]; then # 1024 * 690 
    SCENES=(".") 
    resolution=1 
elif [ "$DATASET" = "UrbanScene3D" ]; then # 5472 * 3648 
    SCENES=("Residence") #  "sciArt"
    resolution=4 
    for scene in "${SCENES[@]}" 
    do 
        ln -s /workspace/data/${scene}/train/images /workspace/data/${scene}/images 
    done 
elif [ "$DATASET" = "Tanks_Templates" ]; then # 960 * 545 
    SCENES=("train" "truck") 
    resolution=1 
elif [ "$DATASET" = "DeepBlending" ]; then # 1328 * 864
    SCENES=("DrJohnson" "Playroom") 
    resolution=1 
elif [ "$DATASET" = "campus" ]; then # 1435 * 1077
    SCENES=("campus_part_0" "campus_part_1") 
    resolution=1 
else 
    SCENES=("DrJohnson" "Playroom") 
    resolution=1 
fi 

cd /workspace/code 
SKIP_COLMAP="False" 
DELETE_NOT_USED="False" 
if [ ! "$SKIP_COLMAP" = "True" ]; then 
    for scene in "${SCENES[@]}" 
    do 
        DATASET_DIR="/workspace/data/${scene}" 
        IMAGES_DIR="${DATASET_DIR}/images"
        python preprocess/generate_colmap.py    --project_dir ${DATASET_DIR} --images_dir ${IMAGES_DIR}  --small_scale --use_exhaustive_matcher
        python preprocess/generate_chunks.py    --project_dir ${DATASET_DIR} --small_scale 
        python preprocess/generate_depth.py     --project_dir ${DATASET_DIR} 
        if [ "$DELETE_NOT_USED" = "True" ]; then 
            rm -rf ${DATASET_DIR}/camera_calibration/raw_chunks
            rm -rf ${DATASET_DIR}/camera_calibration/unrectified
            rm -rf ${DATASET_DIR}/camera_calibration/rectified/sparse 
            rm -rf ${DATASET_DIR}/camera_calibration/rectified/stereo 
            rm -rf ${DATASET_DIR}/camera_calibration/rectified/run-colmap-geometric.sh 
            rm -rf ${DATASET_DIR}/camera_calibration/rectified/run-colmap-photometric.sh 
            rm -rf ${DATASET_DIR}/camera_calibration/colmap_mapper.log 
        fi 
    done 
fi 