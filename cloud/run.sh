DATASET=/workspace/data/small_city 
ALIGNED=${DATASET}/camera_calibration/aligned 
CHUNKS=${DATASET}/camera_calibration/chunks 
OUTPUT=${DATASET}/output 
RECTIFIED=../rectified 

python server.py \
    --source_path ${ALIGNED} \
    --model_path ${OUTPUT}/scaffold \
    --hierarchy ${OUTPUT}/merged.hier \
    --images ${RECTIFIED}/images \
    --alpha_masks ${RECTIFIED}/masks \
    --scaffold_file ${OUTPUT}/scaffold/point_cloud/iteration_30000 \
    --eval 

# compute-sanitizer --launch-timeout 5000 --target-processes all 