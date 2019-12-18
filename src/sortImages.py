#A quick script to sort the images from the Editied Pictures directory into
#Training, Testing and Validation
import os
import sys
import shutil

root_dir="C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\PicturesEdited"
train_dir="C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\Training"
test_dir="C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\Testing"
noise_dir="C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\Noise_Results"

def sort_img(imgName, imgPath):
	if "OUT" in imgName:
		shutil.copy(imgPath, noise_dir)
	elif "GLOVES=N" in imgName and "PERSON=FENTON" in imgName:
		shutil.copy(imgPath, test_dir)
	else:
		shutil.copy(imgPath, train_dir)

if __name__ == '__main__':
	all_file_names = [name for root, dirs, files in os.walk(root_dir) for name in files]
	all_files_with_root = [os.path.join(root, name) for root, dirs, files in os.walk(root_dir) for name in files]

	for img in range(0, len(all_file_names)):
		sort_img(all_file_names[img], all_files_with_root[img])

