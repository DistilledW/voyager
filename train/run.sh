DATASET=/workspace/data/small_city
ALIGNED=/workspace/data/small_city/camera_calibration/aligned
CHUNKS=/workspace/data/small_city/camera_calibration/chunks
RECTIFIED=../rectified
OUTPUT=/workspace/data/small_city/h_output
RENDER_DIR=/workspace/data/small_city/renders/241226_01

python render_hierarchy.py --source_path ${ALIGNED} --model_path ${OUTPUT}/scaffold --hierarchy ${OUTPUT}/merged.hier --out_dir ${RENDER_DIR} \
    --images ${RECTIFIED}/images --alpha_masks ${RECTIFIED}/masks --scaffold_file ${OUTPUT}/scaffold/point_cloud/iteration_30000 --eval 
