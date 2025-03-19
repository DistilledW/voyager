DATASET=/workspace/code/dataset
VIEWPOINT_PATH=${DATASET}/viewpoints.txt 
LOGS_DIR=${DATASET}/logs/client 
RENDER_DIR=${DATASET}/renders 
IP="127.0.0.1"
PORT=50000 

RECTIFIED=/workspace/data/camera_calibration/rectified # Evaluation pictures 

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
VGG_SOURCE_DIR="${DATASET}/vgg"
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

python client.py \
    --tau 6.0 \
    --ip ${IP} \
    --port ${PORT} \
    --render_dir ${RENDER_DIR} \
    --save_log \
    --logs_dir ${LOGS_DIR} \
    --viewpointFilePath ${VIEWPOINT_PATH} \
    --images ${RECTIFIED}/images \
    --alpha_masks ${RECTIFIED}/masks \
    --eval 
    # --frustum_culling \

# compute-sanitizer --launch-timeout 5000 --target-processes all 