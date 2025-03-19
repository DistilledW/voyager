DATASET=/workspace/data 
CODE=/workspace/code
ALIGNED=${DATASET}/camera_calibration/aligned 
# CHUNKS=${DATASET}/camera_calibration/chunks 
OUTPUT=${DATASET}/output 
RECTIFIED=../rectified 
LOG_DIR=${CODE}/dataset/logs/cloud  
DATE="240317"
# conda activate check 
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH"
    exit 1 
fi 
# (conda init problem)
ENV_NAME=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$ENV_NAME" != "3dgs" ]; then
    # source activate 3dgs 
    echo "Conda environment is not activated."
    echo "$ENV_NAME"
    exit 1
fi 

# compiling hierarchy generator and merger before train 
FILE="submodules/gaussianhierarchy/build/GaussianHierarchyCreator"
if [ ! -e "$FILE" ]; then
    cd submodules/gaussianhierarchy
    cmake . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --config Release
    cd ../..
else
    echo "GaussianHierarchy has been built."
fi

# pip install submodules
TARGETS=("diff_gaussian_rasterization" "simple_knn" "gaussian_hierarchy")
SOURCES=("submodules/hierarchy-rasterizer" "submodules/simple-knn" "submodules/gaussianhierarchy")
for i in "${!TARGETS[@]}"; do
    PACKAGE="${TARGETS[$i]}"
    SOURCE="${SOURCES[$i]}"

    if ! pip show "$PACKAGE" > /dev/null 2>&1; then
        echo "$PACKAGE is not installed, wait seconds to install..."
        pip install "$SOURCE"
    else 
        echo "$PACKAGE is already installed"
    fi
done 

# cp vgg* 
VGG_SOURCE_DIR="${CODE}/dataset/vgg"
VGG_DESTINATION="/root/.cache/torch/hub/checkpoints" 
VGG_FILES=("vgg.pth" "vgg16-397923af.pth") 
mkdir -p "${VGG_SOURCE_DIR}"
mkdir -p "${VGG_DESTINATION}"
for vgg_file in "${VGG_FILES[@]}"; do 
    if [ ! -e "${VGG_SOURCE_DIR}/${vgg_file}" ]; then 
        wget -O "${VGG_SOURCE_DIR}/${vgg_file}" http://10.147.18.182/${vgg_file} 
    fi 
    if [ -e "${VGG_SOURCE_DIR}/${vgg_file}" ] && [ ! -e "${VGG_DESTINATION}/${vgg_file}" ]; then
        cp "${VGG_SOURCE_DIR}/${vgg_file}" "${VGG_DESTINATION}/${vgg_file}" 
    fi 
done 

python server.py \
    --source_path ${ALIGNED} \
    --model_path ${OUTPUT}/scaffold \
    --hierarchy ${OUTPUT}/merged.hier \
    --images ${RECTIFIED}/images \
    --alpha_masks ${RECTIFIED}/masks \
    --scaffold_file ${OUTPUT}/scaffold/point_cloud/iteration_30000 \
    --eval # > "${LOG_DIR}/${DATE}_log.txt"
    # --frustum_culling \

# compute-sanitizer --launch-timeout 5000 --target-processes all 