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
COLMAP_PROJECT="/workspace/data"
DATABASE_PATH="${COLMAP_PROJECT}/database.db"
SPARSE_DIR="${COLMAP_PROJECT}/sparse"
IMAGES_DIR="${COLMAP_PROJECT}/camera_calibration/rectified/images" 

cd /workspace/code

mkdir -p ${SPARSE_DIR} 
colmap feature_extractor \
    --database_path "${DATABASE_PATH}" \
    --image_path "${IMAGES_DIR}" \
    --ImageReader.default_focal_length_factor 0.5 \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1

python "/workspace/code/preprocess/make_colmap_custom_matcher.py" \
    --image_path "${IMAGES_DIR}" \
    --output_path "${COLMAP_PROJECT}/matching.txt" 

colmap matches_importer \
    --database_path "${DATABASE_PATH}" \
    --match_list_path "${COLMAP_PROJECT}/matching.txt" 

colmap mapper \
    --database_path "${DATABASE_PATH}" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR"
