#!/bin/bash 
DATASET_PATH="/data/voyager/small_city"
ALIGNED="${DATASET_PATH}/camera_calibration/aligned"
RECTIFIED="${DATASET_PATH}/camera_calibration/rectified"
MODLE_PATH="${DATASET_PATH}/h_3dgs/scaffold"
MERGED_HIER="${DATASET_PATH}/h_3dgs/merged.hier"

LOGS_DIR="../dataset/logs/cloud" 
mkdir -p "${LOGS_DIR}" 

ARGS=( 
    --source_path "${ALIGNED}"
    --model_path "${MODLE_PATH}"
    --scaffold_file "${MODLE_PATH}/point_cloud/iteration_30000"
    --hierarchy "${MERGED_HIER}" 
    --log_file "${LOGS_DIR}" 
    --client 1 
) 

python cloud.py --ip "127.0.0.1" --port 40000 "${ARGS[@]}" 

# bash cloud_client.sh cloud small_city . 0 
