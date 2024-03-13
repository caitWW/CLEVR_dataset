import json

# path to your JSON files
file_path1 = '/home/qw3971/clevr2/image_generation/camera_rotations.json'
file_path2 = '/home/qw3971/clevr2/image_generation/camera_rotations2.json'

# reading the first file
with open(file_path1, 'r') as file:
    data1 = json.load(file)

print(len(data1))

# reading the second file
with open(file_path2, 'r') as file:
    data2 = json.load(file)

'''

print(len(data2))

# Function to modify the file names
def modify_filenames(data, offset):
    modified_data = {}
    for key, value in data.items():
        # Split the file name into parts
        parts = key.split('_')
        # Modify the numeric part
        number_part = int(parts[2]) + offset
        # Reform the file name with new number
        new_key = f"{parts[0]}_{parts[1]}_{str(number_part).zfill(6)}_{parts[3]}"
        # Add to the modified dictionary
        modified_data[new_key] = value
    return modified_data

# Modify the data2 with an offset of 3500
modified_data2 = modify_filenames(data2, 3500)
print(len(modified_data2))
'''

combined_data = {**data1, **data2}
print(len(combined_data))

combined_file_path = '/home/qw3971/clevr2/image_generation/combined_file2.json'

# writing the combined data
with open(combined_file_path, 'w') as file:
    json.dump(combined_data, file)