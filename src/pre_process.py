import sys
import os
sys.path.append(os.path.abspath('..\\modules'))
import HandGestureDataset
from HandGestureDataset import HandGestureDataset, Resize, Cluster, ToTensor, GaussianFilter
import NeuralNet
from NeuralNet import ConNet
from torchvision import transforms, utils, datasets
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import cv2
import matplotlib.pyplot as plt


def stuf():
	hand_dataset = HandGestureDataset(root_dir="C:\\Users\\Noah\\Documents\\CISL\\PicturesEdited")
	#print(hand_dataset[0]['image'])
	scale = Resize(256)
	gaussian = GaussianFilter()
	tensor = ToTensor()
	cluster = Cluster()
	composed = transforms.Compose([Cluster(), GaussianFilter(), Resize(256)])

	data_set = HandGestureDataset(root_dir="C:\\Users\\Noah\\Documents\\CISL\\PicturesEdited\\", transform=composed)
	dataset_loader = torch.utils.data.DataLoader(data_set, batch_size=4, shuffle=True, num_workers=4)

	out_set = HandGestureDataset(root_dir="C:\\Users\\Noah\\Documents\\CISL\\Noise_Results", )
	fig  = plt.figure
	sample = hand_dataset[5]
	print("for")
	for i, trans in enumerate([cluster, gaussian]):
		print("ah")
		trans_sample = trans(sample)
		print("sampled")

		ax = plt.subplot(1, 3, i+1)
		plt.tight_layout()
		ax.set_title(type(trans).__name__)
		plt.imshow(trans_sample['image'])

	plt.show()
stuf()