DATASET=/workspace/data/small_city

ALIGNED=${DATASET}/camera_calibration/aligned
# CHUNKS=/workspace/data/small_city/camera_calibration/chunks
OUTPUT=${DATASET}/h_output
RENDER_DIR=${DATASET}/h_renders/25020821_01
RECTIFIED=../rectified
TEST_DIR=/workspace/code/render

# pip install submodules/gaussianhierarchy
python server.py --source_path ${ALIGNED} --model_path ${OUTPUT}/scaffold --hierarchy ${OUTPUT}/merged.hier \
    --images ${RECTIFIED}/images --alpha_masks ${RECTIFIED}/masks --scaffold_file ${OUTPUT}/scaffold/point_cloud/iteration_30000 --eval 
# compute-sanitizer --launch-timeout 5000 --target-processes all 

# torch.Size([18654425, 3]) 
# 0.0 11435316
