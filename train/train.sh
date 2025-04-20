DATASET=/workspace/data
ALIGNED=${DATASET}/camera_calibration/aligned
RECTIFIED=${DATASET}/camera_calibration/rectified
CHUNKS=${DATASET}/camera_calibration/chunks
OUTPUT=${DATASET}/h_output
LOG_DIR=/workspace/code/dataset/logs

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

clear
python scripts/full_train.py \
    --project_dir ${DATASET} \
    --colmap_dir ${ALIGNED} \
    --images_dir ${RECTIFIED}/images \
    --masks_dir ${RECTIFIED}/masks \
    --chunks_dir ${CHUNKS} \
    --output_dir ${OUTPUT} \
    --skip_if_exists > ${LOG_DIR}/campus_train_3.log
