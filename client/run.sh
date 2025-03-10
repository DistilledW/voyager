PROJECT_DIR=../dataset
RENDER_DIR=${PROJECT_DIR}/renders
RECTIFIED=${PROJECT_DIR}/camera_calibration/rectified 
VIEWPOINTFILEPATH=${PROJECT_DIR}/viewpoints.txt
LOGS_DIR=${PROJECT_DIR}/logs
TEST_DATA_DIR=${PROJECT_DIR}/test_data
IP="10.147.18.182"
PORT=50000 

# pip install submodules/gaussianhierarchy
# pip install submodules/hierarchy-rasterizer
# pip install submodules/simple-knn
ulimit -n 65535 
CUDA_LAUNCH_BLOCKING=1 python3 client.py \
    --tau 6.0 \
    --ip ${IP} \
    --port ${PORT} \
    --with_culling True \
    --out_dir ${RENDER_DIR} \
    --save_log True \
    --logs_dir ${LOGS_DIR} \
    --test_data_dir ${TEST_DATA_DIR}\
    --viewpointFilePath ${VIEWPOINTFILEPATH} \
    --eval 
# compute-sanitizer --launch-timeout 5000 --target-processes all 
