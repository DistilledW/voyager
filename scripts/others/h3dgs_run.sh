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
# cp vgg* 
VGG_SOURCE_DIR="/workspace/code/dataset/vgg"
VGG_DESTINATION="/root/.cache/torch/hub/checkpoints/" 
VGG_FILES=("vgg.pth" "vgg16-397923af.pth") 
mkdir -p "${VGG_SOURCE_DIR}"
mkdir -p "${VGG_DESTINATION}"
for vgg_file in "${VGG_FILES[@]}"; do 
    if [ -e "${VGG_SOURCE_DIR}/${vgg_file}" ] && [ ! -e "${VGG_DESTINATION}/${vgg_file}" ]; then
        cp "${VGG_SOURCE_DIR}/${vgg_file}" "${VGG_DESTINATION}/${vgg_file}" 
    fi 
done 

DATASET=$1 
SEQ=$2
SETS=$3
ALPHA_MASKS_ARGS=()
if [ "$DATASET" = "360" ]; then # 4946 * 3286 
    # if [ "$SETS" = "0" ]; then
    #     SCENES=("bicycle" "bonsai" "counter" "garden") 
    # elif [ "$SETS" = "1" ]; then
    #     SCENES=("kitchen" "room" "stump" "treehill") 
    # else 
    #     SCENES=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump" "treehill") #  "flowers" 
    # fi 
    SCENES=("garden" "counter") 
    resolution=4 
elif [ "$DATASET" = "MegaNeRF" ]; then # 4608 * 3456 
    if [ "$SETS" = "0" ]; then
        SCENES=("building") 
    elif [ "$SETS" = "1" ]; then
        SCENES=("rubble") 
    else 
        SCENES=("building" "rubble") #
    fi 
    resolution=4 
    
    for scene in "${SCENES[@]}" 
    do 
        ln -s /workspace/data/${scene}/train/images /workspace/data/${scene}/images 
    done 
elif [ "${DATASET}" = "small_city" ]; then # 1024 * 690 
    SCENES=(".") 
    resolution=1 
    ALPHA_MASKS_ARGS=(
        --alpha_masks /workspace/data/camera_calibration/rectified/masks 
    )
elif [ "$DATASET" = "UrbanScene3D" ]; then # 5472 * 3648  
    if [ "$SETS" = "0" ]; then
        SCENES=("sciArt") 
    elif [ "$SETS" = "1" ]; then
        SCENES=("Residence") 
    else 
        SCENES=("sciArt" "Residence") #
    fi 
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
    # if [ "$SETS" = "0" ]; then
    #     SCENES=("campus")
    # elif [ "$SETS" = "1" ]; then
    #     SCENES=("campus_part_0")
    # elif [ "$SETS" = "2" ]; then
    #     SCENES=("campus_part_1")
    # else 
    #     SCENES=("campus_part_0" "campus_part_1") 
    # fi 
    SCENES=("campus")
    # SCENES=("campus_part_0" "campus_part_1") 
    resolution=1 
else 
    SCENES=("DrJohnson" "Playroom") 
    resolution=1 
fi 

SOURCES=("/workspace/code/submodules/hierarchy-rasterizer" "/workspace/code/submodules/gaussianhierarchy" "/workspace/code/submodules/simple-knn")
TARGETS=("diff_gaussian_rasterization" "gaussian_hierarchy" "simple_knn") 
REBUILD=0 
for i in "${!TARGETS[@]}"; do 
    SOURCE="${SOURCES[$i]}"
    PACKAGE="${TARGETS[$i]}"
    if [ "${REBUILD}" -eq 1 ] || ! pip show "$PACKAGE" > /dev/null 2>&1; then 
        pip install "$SOURCE"
    fi
done 

CODE="/workspace/code/h3dgs_render"
cd ${CODE} 
Taus=(3.0 15.0) # 0.0 3.0 6.0 15.0
PYTHON_FILE="render_hierarchy.py" 
for scene in "${SCENES[@]}"; 
do 
    DATASET_PATH="/workspace/data/${scene}"
    ALIGNED="${DATASET_PATH}/camera_calibration/aligned"
    RECTIFIED="${DATASET_PATH}/camera_calibration/rectified"
    MODLE_PATH="${DATASET_PATH}/h_3dgs/scaffold"
    MERGED_HIER="${DATASET_PATH}/h_3dgs/merged.hier"
    CAMERA_DIR="${ALIGNED}/sparse/0/tests/$SEQ"
    mkdir -p "${CAMERA_DIR}" 
    ARGS=( 
        --source_path "${ALIGNED}"
        --model_path "${MODLE_PATH}"
        --scaffold_file "${MODLE_PATH}/point_cloud/iteration_30000"
        --hierarchy "${MERGED_HIER}" 
        --images "${DATASET_PATH}/images" 
        # --alpha_masks "${RECTIFIED}/masks" 
        --resolution "${resolution}" 
        --eval
    ) 
    # LOGS_DIR="/workspace/code/dataset/logs/h3dgs_render/${DATASET}/${scene}/performance"
    # for tau in "${Taus[@]}"; 
    # do 
    #     python ${PYTHON_FILE} \
    #         --log_file "${LOGS_DIR}/${tau}.txt" \
    #         --taus ${tau} \
    #         --per --test "$seq.txt" \
    #         "${ARGS[@]}" 
    # done 
    LOGS_DIR="/workspace/code/dataset/logs/h3dgs_render/${DATASET}/${scene}/performance/$SEQ"
    mkdir -p "${LOGS_DIR}" 
    for tau in "${Taus[@]}"; 
    do 
        python ${PYTHON_FILE} --taus ${tau} \
            --log_file "${LOGS_DIR}/${tau}.txt" \
            --res_file "${LOGS_DIR}/res_${tau}.txt" \
            --cameras_dir "${CAMERA_DIR}" --test "$SEQ.txt" \
            "${ARGS[@]}" "${ALPHA_MASKS_ARGS[@]}" --out_dir /workspace/code/dataset/$tau
    done 
    # LOGS_DIR="/workspace/code/dataset/logs/h3dgs_render/${DATASET}/${scene}/accuracy" --path 
    # RENDERS="/workspace/code/dataset/logs/h3dgs_render/${DATASET}/${scene}/renders" 
    # mkdir -p "${LOGS_DIR}" 
    # for tau in "${Taus[@]}"; 
    # do 
    #     mkdir -p "${RENDERS}/${tau}" 
    #     python ${PYTHON_FILE} \
    #         --log_file "${LOGS_DIR}/${tau}.txt" \
    #         --taus ${tau} \
    #         "${ARGS[@]}" --train_test_exp --test "$SEQ.txt" 
    #         # --out_dir "${RENDERS}/${tau}" \
    # done 
done 
