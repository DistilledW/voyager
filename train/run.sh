DATASET=/workspace/data/small_city 
ALIGNED=${DATASET}/camera_calibration/aligned 
CHUNKS=${DATASET}/camera_calibration/chunks 
OUTPUT=${DATASET}/output 
RENDER_DIR=${DATASET}/renders/241226_01 
RECTIFIED=../rectified 

python render_hierarchy.py \
    --source_path ${ALIGNED} \
    --model_path ${OUTPUT}/scaffold \
    --hierarchy ${OUTPUT}/merged.hier \
    --out_dir ${RENDER_DIR} \
    --images ${RECTIFIED}/images \
    --alpha_masks ${RECTIFIED}/masks \
    --scaffold_file ${OUTPUT}/scaffold/point_cloud/iteration_30000 \
    --eval 
