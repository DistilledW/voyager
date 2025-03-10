export DATASET=/workspace/data/small_city
export ALIGNED=/workspace/data/small_city/camera_calibration/aligned
export CHUNKS=/workspace/data/small_city/camera_calibration/chunks
export RECTIFIED=/workspace/data/small_city/camera_calibration/rectified

python scripts/full_train.py --project_dir ${DATASET} --colmap_dir ${ALIGNED} --images_dir ${RECTIFIED}/images --depths_dir ${RECTIFIED}/depths --masks_dir ${RECTIFIED}/masks --chunks_dir ${CHUNKS} --output_dir ${DATASET}/output