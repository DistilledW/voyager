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

TYPE=$1 # cloud or client
DATASET=$2 
scene=$3 
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
    SCENES=("sciArt") # "Residence" 
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

PROJECT="/workspace/code"
DATASET="/workspace/data/${scene}"
ALIGNED="${DATASET}/camera_calibration/aligned"
RECTIFIED="${DATASET}/camera_calibration/rectified"
MODLE_PATH="${DATASET}/h_3dgs/scaffold"
MERGED_HIER="${DATASET}/h_3dgs/merged.hier"
LOGS_DIR="${PROJECT}/dataset/logs/$TYPE/$DATASET/$scene/" 
# LOGS_DIR="${PROJECT}/dataset/logs/$TYPE/${DATASET}/${scene}/accuracy" 
mkdir -p "${LOGS_DIR}" 
if [ "$TYPE" = "cloud" ]; then 
    SOURCES=("${PROJECT}/submodules/flashTreeTraversal/" "${PROJECT}/submodules/simple-knn/") 
    TARGETS=("flash_tree_traversal" "simple_knn") 
    TT_MODE=$4 # Fused = 0, Default = 1, LayerWise = 2 
    ARGS=( 
        --source_path "${ALIGNED}"
        --model_path "${MODLE_PATH}"
        --scaffold_file "${MODLE_PATH}/point_cloud/iteration_30000"
        --hierarchy "${MERGED_HIER}" 
        --log_file "${LOGS_DIR}/${TT_MODE}" 
        --tt_mode "${TT_MODE}" 
        --client 1 
    ) 
elif [ "$TYPE" = "client" ]; then 
    VIEWPOINT_PATH="${PROJECT}/dataset/viewpoints_all.txt" 
    SOURCES=("${PROJECT}/submodules/flashTreeTraversal/" "${PROJECT}/submodules/fast_hier/")
    TARGETS=("flash_tree_traversal" "fast_hier") 
    TAU=$4 
    RENDERS="${LOGS_DIR}/renders" 
    mkdir -p "${RENDERS}/${TAU}" 
    mkdir -p "$LOGS_DIR/vq/"
    ARGS=( 
        --images "${RECTIFIED}/images" 
        --alpha_masks "${RECTIFIED}/masks" 
        --viewpointFilePath "${VIEWPOINT_PATH}" 
        --log_file "$LOGS_DIR/vq/test_tt.txt" 
        --out_dir "${RENDERS}/${TAU}" 
        --resolution "${resolution}" 
        --tau "${TAU}" 
        --eval 
        --train_test_exp 
    ) 
else 
    echo "Select type: cloud or client" 
    exit 0 
fi 
REBUILD=1 
for i in "${!TARGETS[@]}"; do 
    SOURCE="${SOURCES[$i]}"
    PACKAGE="${TARGETS[$i]}"
    if [ "${REBUILD}" -eq 1 ] || ! pip show "$PACKAGE" > /dev/null 2>&1; then 
        pip install "$SOURCE"
    fi 
done 
SHARED_ARGS=( 
    --ip "127.0.0.1" 
    --port 60000 
    --frustum_culling 
    --local 
) 
CODE="$PROJECT/$TYPE" 
PYTHON_FILE="$TYPE.py" 
cd ${CODE} 
python ${PYTHON_FILE} "${SHARED_ARGS[@]}" "${ARGS[@]}" 

# bash cloud_client.sh cloud small_city . 0 
# bash cloud_client.sh client small_city . 6.0 