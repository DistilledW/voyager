#!/bin/bash 
DATASET_PATH="/data/voyager/small_city"
ALIGNED="${DATASET_PATH}/camera_calibration/aligned"
RECTIFIED="${DATASET_PATH}/camera_calibration/rectified"
MODLE_PATH="${DATASET_PATH}/h_3dgs/scaffold"
MERGED_HIER="${DATASET_PATH}/h_3dgs/merged.hier"

ARGS=( 
    --source_path "${ALIGNED}"
    --model_path "${MODLE_PATH}"
    --scaffold_file "${MODLE_PATH}/point_cloud/iteration_30000"
    --hierarchy "${MERGED_HIER}" 
    --images "${RECTIFIED}/images" 
    --alpha_masks "${RECTIFIED}/masks" 
    --resolution 1 
    --eval --train_test_exp 
) 
CURRENT=2 
Taus=(0.0) 
if [ "$CURRENT" -eq "0" ]; then # Get Performance while cameras 
    LOGS_DIR="../dataset/logs/voyagerL/performance"
    mkdir -p "${LOGS_DIR}" 
    for tau in "${Taus[@]}"; 
    do 
        python "render_hierarchy.py" \
            --log_file "${LOGS_DIR}/${tau}.txt" \
            --taus ${tau} \
            --per --test "00.txt" \
            "${ARGS[@]}" 
    done 
elif [ "$CURRENT" -eq "1" ]; then # Get Performance while camera path 
    LOGS_DIR="../dataset/logs/voyagerL/camera_path_log"
    mkdir -p "${LOGS_DIR}/renders" 
    CAMERA_DIR="${ALIGNED}/sparse/0/tests/00"
    mkdir -p "${CAMERA_DIR}" 
    for tau in "${Taus[@]}"; 
    do 
        python "render_hierarchy.py" --taus ${tau} \
            --out_dir  "${LOGS_DIR}/renders" \
            --log_file "${LOGS_DIR}/${tau}.txt" \
            --res_file "${LOGS_DIR}/res_${tau}.txt" \
            --path --cameras_dir "${CAMERA_DIR}" --test "00.txt" \
            "${ARGS[@]}" 
    done 
elif [ "$CURRENT" -eq "2" ]; then # Get Accuracy 
    LOGS_DIR="../dataset/logs/voyagerL/accuracy" 
    RENDERS="../dataset/logs/voyagerL/accuracy/renders" 
    for tau in "${Taus[@]}"; 
    do 
        mkdir -p "${RENDERS}/${tau}" 
        python "render_hierarchy.py" \
            --log_file "${LOGS_DIR}/${tau}.txt" \
            --taus ${tau} \
            --out_dir "${RENDERS}/${tau}" --test "05.txt" \
            "${ARGS[@]}" # --compress 
    done 
else
    echo "Error Input"
fi 