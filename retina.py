import subprocess
import os
import cv2
import numpy as np
from typing import Dict
import json
import random

# define the range for center of retina
x_range = [50, 280] 
y_range = [60, 180]

sigma = 0.5

# get the list of image filenames in the folder
folder_path = "/home/qw3971/clevr2/image_generation/new/"  
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#output_file_path = ("/home/qw3971/clevr2/image_generation/combined_file.json")
#with open(output_file_path, 'r') as outfile:
    #data = json.load(outfile)

# initialize empty dictionary to hold the image data
image_data: Dict[str, Dict[str, int]] = {}

# loop over the image files
for image_file in image_files:
    print(image_file)
    # randomly generate center of retina to place filter
    #xc = data[image_file][0]
    #yc = data[image_file][1]
    xc = 160
    yc = 120


    cmd = f"python3 retina_transform.py {os.path.join(folder_path, image_file)} {xc} {yc} {sigma}"

    # run retina_transform and get the transformed image
    result = subprocess.run(cmd, shell=True, capture_output=True)

    if result.returncode != 0:
        print(f"Error running retina_transform on {image_file}: {result.stderr}")
        continue

    # record coordinates used for each image
    #image_data[image_file] = {'xc': xc, 'yc': yc}

# write the image data to a JSON file
# with open("/home/qw3971/clevr2/image_generation/retina_rotated.json", "w") as f: 
    #data = json.dump(f, image_data)