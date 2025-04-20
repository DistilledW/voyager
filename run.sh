#!/bin/bash 
source "$1" # /path/to/config.cfg 
cd ${CODE} 

# conda activate check 
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
for i in "${!TARGETS[@]}"; do 
    PACKAGE="${TARGETS[$i]}"
    SOURCE="${SOURCES[$i]}"
    if [ "${REBUILD}" -eq 1 ] || ! pip show "$PACKAGE" > /dev/null 2>&1; then 
        pip install "$SOURCE"
    fi
done 

# cp vgg* 
VGG_SOURCE_DIR="/worksapce/code/dataset/vgg"
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

if [ -n "${Taus}" ]; then 
    # 运行一组程序
    for tau in "${Taus[@]}"; do 
        mkdir -p "${LOGS_DIR}"
        mkdir -p "${RENDERS}/${tau}" 
        python ${PYTHON_FILE} \
            --log_file "${LOGS_DIR}/${tau}.txt" \
            --render_dir "${RENDERS}/${tau}" \
            "${ARGS[@]}" 
    done 
else 
    # 运行单个程序 
    python ${PYTHON_FILE} "${ARGS[@]}" 
fi 
