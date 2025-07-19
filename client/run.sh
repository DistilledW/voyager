#!/bin/bash 
LOGS_DIR="../dataset/logs/client" 
mkdir -p "${LOGS_DIR}" 
RECTIFIED="/data/voyager/small_city/camera_calibration/rectified"

RENDERS="../dataset/logs/renders"
mkdir -p "${RENDERS}/${TAU}" 
ARGS=( 
    --images "${RECTIFIED}/images" 
    --alpha_masks "${RECTIFIED}/masks" 
    --log_file "$LOGS_DIR" 
    # --out_dir "${RENDERS}/${TAU}" 
    --resolution 1 
    --n_frames 60 
    --tau 3.0 
    --eval --train_test_exp 
) 

python client.py --ip "127.0.0.1" --port 40000 "${ARGS[@]}" 

# bash cloud_client.sh client small_city . 15.0 32 