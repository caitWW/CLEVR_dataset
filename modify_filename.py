import os

# path to folder containing images
folder_path = '/home/qw3971/clevr2/image_generation/rotated_new'
path = '/home/qw3971/clevr2/image_generation/rotated_new2_revised'


# loop through each file in the folder
for filename in os.listdir(folder_path):
    if '_rotated.png' in filename:
        # ff the file does not need to be changed, just report it
        print(f"Skipping {filename}, does not require renaming")
        
    else:
       # create the new filename by replacing '_rotated.png' with '.png'
        new_filename = filename.replace('.png', '_rotated.png')
        # full path for old and new filenames
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {filename} to {new_filename}")

       


