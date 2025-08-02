#!/bin/bash 
DATASET=$1 
SETS=$2

if [ "$DATASET" = "360" ]; then # 4946 * 3286 
    # "bicycle"
    if [ "$SETS" = "0" ]; then
        SCENES=("bonsai" "counter" "garden") 
    elif [ "$SETS" = "1" ]; then
        SCENES=("kitchen" "room") 
    else 
        SCENES=("stump" "treehill") 
    fi 
    # SCENES=("bonsai" "counter" "garden" "kitchen" "room" "stump" "treehill") #  "flowers" 
    resolution=4 
elif [ "$DATASET" = "MegaNeRF" ]; then # 4608 * 3456 
    SCENES=("building" "rubble") 
    resolution=4 
elif [ "${DATASET}" = "small_city" ]; then # 1024 * 690 
    SCENES=(".") 
    resolution=1 
elif [ "$DATASET" = "UrbanScene3D" ]; then # 5472 * 3648 
    SCENES=("Residence") #  "sciArt"
    resolution=4 
elif [ "$DATASET" = "Tanks_Templates" ]; then # 960 * 545 
    SCENES=("train" "truck") 
    resolution=1 
elif [ "$DATASET" = "DeepBlending" ]; then # 1328 * 864
    SCENES=("DrJohnson" "Playroom") 
    resolution=1 
elif [ "$DATASET" = "campus" ]; then # 1435 * 1077 
    SCENES=("campus") 
    resolution=1 
    # SCENES=("campus_part_0" "campus_part_1") 
else 
    DATASET="" 
    SCENES=("") 
fi 

# conda activate check 
CONDA_NAME="3dgs" 
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH"
    exit 1 
else 
    ENV_NAME=$(basename "$CONDA_PREFIX") 
    if [ "$ENV_NAME" != "${CONDA_NAME}" ]; then 
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate "${CONDA_NAME}"
    fi 
fi 
TRAIN="/workspace/code/train"
cd ${TRAIN} 
# compiling hierarchy generator and merger before train 
FILE="submodules/gaussianhierarchy/build/GaussianHierarchyCreator"
if [ ! -e "$FILE" ]; then 
    cd submodules/gaussianhierarchy 
    cmake . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --config Release
    cd ../..
fi 
# pip install submodules 
SOURCES=("${TRAIN}/submodules/hierarchy-rasterizer" "${TRAIN}/submodules/simple-knn" "${TRAIN}/submodules/gaussianhierarchy")
TARGETS=("diff_gaussian_rasterization" "simple_knn" "gaussian_hierarchy") 
REBUILD=1 
for i in "${!TARGETS[@]}"; do 
    SOURCE="${SOURCES[$i]}"
    PACKAGE="${TARGETS[$i]}"
    if [ "${REBUILD}" -eq 1 ] || ! pip show "$PACKAGE" > /dev/null 2>&1; then 
        pip install "$SOURCE"
    fi 
done 

for scene in "${SCENES[@]}" 
do  
    DATASET="/workspace/data/${scene}"
    ALIGNED=${DATASET}/camera_calibration/aligned 
    RECTIFIED=${DATASET}/camera_calibration/rectified 
    CHUNKS=${DATASET}/camera_calibration/chunks 
    OUTPUT=${DATASET}/h_3dgs 
    python /workspace/code/train/scripts/full_train.py \
        --project_dir ${DATASET} \
        --colmap_dir ${ALIGNED} \
        --images_dir ${RECTIFIED}/images \
        --chunks_dir ${CHUNKS} \
        --output_dir ${OUTPUT} \
        --skip_if_exists --not_depths \
        --extra_training_args "--resolution ${resolution}" 
        # --depths_dir ${RECTIFIED}/depths \ 
        # --masks_dir ${RECTIFIED}/masks \
done 