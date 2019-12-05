from __future__ import print_function, division
import sys
import os, os.path
from os import walk
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import cv2
from sklearn.preprocessing import LabelEncoder
import statistics
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class HandGestureDataset(Dataset):
	"""Hand Gesture dataset."""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			root_dir (string): path to the images
			transform (callable, optional): transform to be applied on a sample
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.all_file_names = [name for root, dirs, files in os.walk(root_dir) for name in files]
		self.all_files_with_root = [os.path.join(root, name) for root, dirs, files in os.walk(root_dir) for name in files]
		

		#self._init_dataset()


	def __len__(self):
		return len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.all_files_with_root[idx]
		image = io.imread(img_name)
		sample = {'image': image, 'name': img_name}

		if self.transform:
			sample = self.transform(sample)

		return sample

	#def init_dataset(self):



	def getNumberFromName(self, idx):
		"""This is dependent on our dataset. As we use the format NUM=#_etc for
		file names, we can parse the # out. This assumes that the max number of
		digits is 2. After all, who is trying to count 100 fingers?
		Returns the number of fingers in the picture
		"""
		file = self.all_file_names[idx]
		number = file[4]
		if file[5].isdigit(): 
			number += file[5]
		return int(number)

	def getGlovesFromName(self, idx):
		""" Returns true the glove is on, else false
		"""
		file = self.all_file_names[idx]
		gloveIndex = file.find("GLOVES") + 7
		if file[sleeveIndex] == 'Y':
			return True
		return False

	def getSleevesFromName(self, idx):
		"""returns true if the sleeves are on, else false
		"""
		file = self.all_file_names[idx]
		sleeveIndex = file.find("SLEEVES") + 8
		if file[sleeveIndex] == 'Y':
			return True
		return False

	def getBackgroundFromName(self, idx):
		file = self.all_file_names[idx]
		backIndex = file.find("BACK") + 5
		if file[backIndex] == 'O':
			return("OUT")
		return("IN")

	def getPersonFromName(self, idx):
		file = self.all_file_names[idx]
		personIndex = file.find("PERSON") + 7
		person = ""
		while(file[personIndex] != "_"):
			person += file[personIndex]
			personIndex += 1

	def getOrientationFromName(self, idx):
		file = self.all_file_names[idx]
		orientIndex = fiile.find("ORIENT") + 7
		if file[orientIndex] == 'F':
			return "FRONT"
		return "BACK"

	def getHandFromName(self, idx):
		file = self.all_file_names[idx]
		handIndex = file.find("HAND") + 5
		if file[handIndex] == "R":
			return "RIGHT"
		return "LEFT"
		

class Resize(object):
	""" Rescale the image in a sample to a given size.

	Args:
		output_size: Desired output size, tuple or int (if int, make a square)

	"""
	def __init__(self, output_size):
		assert(isinstance(output_size, (int, tuple)))
		self.output_size = output_size

	def __call__(self, sample):
		image = sample['image']
		height, width = image.shape[:2]
		if isinstance(self.output_size, int):
			if height > width:
				new_h = self.output_size * height / width, 
				new_w = self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * width / height
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		new_img = transform.resize(image, (new_h, new_w))

		return {'image': new_img, 'name': sample['name']}


class GaussianFilter(object):
	""" Applies Gaussian smoothing to reduce image noise. Involves performing
	a 2-D Weierstrass transform which lowers the high-frequency parts of the
	image

	kernel_size (tuple, positive and odd): The height and width of the kernel
	sigmaX (int): The standard deviation of the kernel size in direction X
	sigmaY (int): The standard deviation of the kernel size in direction Y
	borderType (int): Probably unimportant

	"""
	def __call__(self, sample, kernel_size=(5,5), sigmaX=5, sigmaY=0, borderType = cv2.BORDER_DEFAULT):
		assert(kernel_size[0] % 2 != 0 and kernel_size[1] %2 != 0)
		image = sample['image']
		blur = cv2.GaussianBlur(image, kernel_size, sigmaX, sigmaY, borderType)
		return {'image': blur, 'name': sample['name']}

#Does not work on my images :(
class MedianFilter(object):
	""" Applies a median filter to an image. 
	NEED TO FIND MORE DETAIL???
	"""
	def __call__(self, sample, kernel_size=3):
		image = sample['image']
		blur = cv2.medianBlur(image, kernel_size)
		return {'image': blur, 'name': sample['name']}


#Only works on 8u and 32f images
class BilateralFilter(object):
	#See http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
	#Good for preserving edges

	"""
	Arguments are the image, pixel neighborhood, and sigmas of the color space

	"""
	def __call__(self, sample, neighborhood=5, sigmaA=75, sigmaB=75):
		image = sample['image']
		blur = cv2.bilateralFilter(image, neighborhood, sigmaA, sigmaB)
		return {'image': blur, 'name': sample['name']}

class colorSegment(object):
	"""Color segmetnation based on euclidean difference from the average 
	of the image.
	"""
	def __call__(self, sample):
		image = sample['image']
		average = self._averageColor(image)

		...

	def _averageColor(self, image):
		blue, green, red = np.uint8(cv2.mean(image))
		
		...


class ToTensor(object):
	"""converts ndarrays in sample to Tensors."""
	
	def __call__(self, sample):
		image = sample['image']

		#Swap color axis because:
		#numpy image: H * W * C and
		#torch image: C * H * W
		image = image.transpose(2, 0, 1)
		return {'image': torch.from_numpy(image), 'name': sample['name']}

#Notes: Neet to call MedianFilter and/or BilateralFilter before Resize

#Testing
initTime = time.time()

root_dir="C:\\Users\\Noah\\Documents\\CISL\\PicturesEdited"
# images = []
# all_images = [cv2.imread(os.path.join(root, name)) for root, dirs, files in os.walk(root_dir) for name in files]
# print(all_images[0][0][0])
# all_xs = []
# all_ys = []
# all_color = []
# for image in all_images:
# 	for row in image:
# 		for column in row:
# 			all_xs.append(column[0])
# 			all_ys.append(column[1])
# 			all_color.append(column[2])
# print(statistics.stdev(all_xs))
# x_mean = statistics.mean(all_xs)
# y_mean = statistics.mean(all_y)
# c_mean = statistics.mean(all_color)
# x_std = statistics.stdev(all_xs)
# y_std = statistics.stdev(all_ys)
# c_std = statistics.stdev(all_color)
# print(x_mean, y_mean, c_mean, x_std, y_std, c_std)

# finalTime = time.time()
# diffTime = finalTime - initTime

# print("Final time: %.5f seconds"%diffTime)
# writeFile = open("AvgandStd.txt", "w+")
# writeFile.write("Mean for x: {x_mean}\n Stdev for x: {x_std}\n Mean for y {y_mean}\n Stdev for y: {y_std}\n Mean for color: {c_mean}\n Stdev  for color: {c_std}")
# writeFile.close()
# wr

# sys.exit()
hand_dataset = HandGestureDataset(root_dir="C:\\Users\\Noah\\Documents\\CISL\\PicturesEdited")
print(hand_dataset[0]['image'])
scale = Resize(256)
gaussian = GaussianFilter()
median = MedianFilter()
bilateral = BilateralFilter()
tensor = ToTensor()
composed = transforms.Compose([GaussianFilter(), Resize(256), ToTensor])




data_set = HandGestureDataset(root_dir="C:\\Users\\Noah\\Documents\\CISL\\PicturesEdited\\", transform=composed)
dataset_loader = torch.utils.data.DataLoader(data_set, batch_size=4, shuffle=True, num_workers=4)

# fig  = plt.figure
# sample = hand_dataset[3]

# for i, trans in enumerate([scale, gaussian, bilateral, composed]):
# 	trans_sample = trans(sample)

# 	ax = plt.subplot(1, 4, i+1)
# 	plt.tight_layout()
# 	ax.set_title(type(trans).__name__)
# 	plt.imshow(trans_sample['image'])

# plt.show()