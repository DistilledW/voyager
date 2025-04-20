import os
filename = "test.txt"

with open(filename, "w") as fout:
    directory="/workspace/data/camera_calibration/rectified/images"
    file_names = os.listdir(directory)
    file_names = sorted(file_names)
    for idx, file_name in enumerate(file_names):
        if idx >= 100:
            break 
        # if idx % 8 == 0:
        fout.write(f"{file_name}\n") 
