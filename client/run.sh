DATASET=../dataset
RENDER_DIR=${DATASET}/renders
RECTIFIED=${DATASET}/camera_calibration/rectified 
VIEWPOINTFILEPATH=${DATASET}/viewpoints.txt
LOGS_DIR=${DATASET}/logs
TEST_DATA_DIR=${DATASET}
IP="127.0.0.1"
PORT=50000 

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

ulimit -n 65535 
python client.py \
    --tau 6.0 \
    --ip ${IP} \
    --port ${PORT} \
    --with_culling \
    --render_dir ${RENDER_DIR} \
    --save_log \
    --logs_dir ${LOGS_DIR} \
    --test_data_dir ${TEST_DATA_DIR}\
    --viewpointFilePath ${VIEWPOINTFILEPATH} \
    --eval 

# compute-sanitizer --launch-timeout 5000 --target-processes all 
# cd /mnt/c/Users/win11/Desktop/webGS/client 