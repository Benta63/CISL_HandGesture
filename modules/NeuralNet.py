#Handles the neural networks. We are building a convolution Neural Network using PyTorch
from __future__ import print_function, division
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, SGD
import time
from torch.autograd import Variable
from helper import Helper
import HandGestureDataset
from HandGestureDataset import *

#A Convolutional Neural network for three-channel (color) images
class ConNet(nn.Module):

	def __init__(self):
		super(ConNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels= 32, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
		self.dropout1 = nn.Dropout2d(0.25)
		self.dropout2 = nn.Dropout2d(0.35)
		self.fc1 = nn.Linear(921600, 120) #The image is sized to 244X244
		self.fc2 = nn.Linear(120, 6)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x - self.dropout1(x)

		x = torch.flatten(x, 1)
		x = self.fc1(x)

		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)

		output = F.log_softmax(x, dim=1)
		return output

	def save(self, path):
		torch.save(self.state_dict(), path)

def train(model, optimizer, data_path, epoch, transforms, previous_model=None, 
		 batch=1, shuffle=True, workers=1, seed=None, log_interval=10):
	
	"""
	model: A ConNet object
	optimizer: A pytorch optimizer from optim class
	data_path: the path to the images that this model will train on (String)
	epoch: The number of epochs we run for (int)
	transform: The different transformations we should perform on the images (List)
	previous_model: An optional argument that is a path to a saved Network that we can load (String)
	batch: An optional argument for the batch size (int)
	shuffle: An optional argument to shuffle the data_loader (Boolean)
	workers: An optional argument if we have multiple CPUs (int)
	seed: An optional argument that sets the random seed for this run (int)
	log_interval: An argument argument which sets the number of batches to wait before logging training status

	"""

	#Set the seed?

	try:
		#Here we try to access the previous_model, if it exists
		if previous_model:
			model = Helper.load(previous_model)
	except:
		pass
	#Now let's set up the dataloader
	dataset = HandGestureDataset(root_dir=data_path, transform=transforms)
	train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, 
				shuffle=shuffle, num_workers=workers)

	model = model.double() #Always pays to make sure
	model.train()
	for batch_idx, data in enumerate(train_loader):
		inputs = data['image']
		target = data['name']
		optimizer.zero_grad()
		inputs = inputs.double()
		output = model(inputs)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(
				 epoch, batch_idx * len(data), len(train_loader.dataset), 
				 100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_path, transforms,previous_model=None,batch=1, shuffle=False, workers=1):
	"""
	model: A ConNet object
	data_path: the path to the images that this model will test with (String)
	transform: The different transformations we should perform on the images (List)
	previous_model: An optional argument that is a path to a saved Network that we can load (String)
	batch: An optional argument for the batch size (int)
	shuffle: An optional argument to shuffle the data_loader (Boolean)
	workers: An optional argument if we have multiple CPUs (int)
	

	"""

	try:
		if previous_model:
			model = Helper.load(previous_model)
	except:
		pass
	testset = HandGestureDataset(root_dir=data_path, transform=transforms)
	test_loader =torch.utils.data.DataLoader(testset, batch_size=batch, 
		 shuffle=shuffle, num_workers=workers)
	model = model.double()
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data in test_loader:
			inputs = data['image']
			target = data['name']
			inputs = inputs.double()
			inputs = inputs.view(1, -1)
			output = model(inputs)
			test_loss += F.nll_loss(output, target, reduction='sum').item()
			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)

	print("\n Test set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
		 test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



	

if __name__=='__main__':
	train = pd.read_csv('training.txt', sep=",", header=None)
	testing = pd.read_csv('testing.txt', sep=",", header=None)

	#Let's make some datasets now 
	train_dir = "C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\modules\\Training"
	test_dir = "C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\modules\\Testing"

	scale = Resize(256)
	gaussian = GaussianFilter()
	median = MedianFilter()
	bilateral = BilateralFilter()
	tensor = ToTensor()
	#I need to work on binary image pre-processing
	composed = transforms.Compose([GaussianFilter(), Resize(256), ToTensor()])


	train_dataset = HandGestureDataset(root_dir=train_dir, transform=composed)
	test_dataset = HandGestureDataset(root_dir=test_dir, transform=composed)

	print(train_dataset[4]['image'].shape)
	print(train_dataset[29]['image'].shape)


	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)



	model = ConNet()
	optimize = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	criterion = CrossEntropyLoss()

	model = model.double()

	for epoch in range(5):
		run_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			inputs = data['image']
			labels = data['name']
			optimize.zero_grad()

			outputs = model(inputs.double())
			loss = criterion(outputs, labels)
			loss.backward()
			optimize.step()

			#Statistics
			run_loss += loss.item()
			if i % 2000 == 1999: 
				print('[%d, %5d] loss %.3f' % (epoch + 1, i + 1, run_loss / 2000))
				run_loss = 0.0

	print ("doneso")

	model.save("C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\models\\trained.pth")	


	#We're going to test using the testing data and then train on it as well

	correct = 0
	total = 0

	with torch.no_grad():
		for data in test_loader:
			images = data['image']
			labels = data['name']
			outputs = model(images.double())
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print("Accurracy is %d %%" % (100 * correct / total))



#Let's