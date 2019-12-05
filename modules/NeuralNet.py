#Handles the neural networks. We are building a convolution Neural Network using PyTorch
from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable

#A Convolutional Neural network for three-channel (color) images
class ConNet(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		#Let's keep it small at first
		# Making a 2D convolution layer
		self.cnn_layers = Sequential(
			Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(4),
			ReLU(inplace=True),
			MaxPool2d(kernel_size=2, stride=2),
			#Another layer
			Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
			BatchNorm2d(4)
			ReLU(inplace=True),
			MaxPool2d(kernel_size=2, stride=2),

		)
		self.linear_layers = Sequential(
			Linear(4*7*7, 10)
		)
		# self.con1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
		# self.pool = nn.MaxPool2d(2, 2)
		# self.con2 = nn.Conv2d(6, 16, 5)
		# self.fc1 = nn.Linear(16*5*5, 120)
		# self.fc2 = nn.Linear(120, 84)
		# self.fc3 = nn.Linear(84, 10)

		#Number of times to loop over the dataset
		self.epochs = 2

	def forward(self, x):
		x = self.cnn_layers(x)
		x = x.view(x.size(0), -1)
		x = self.linear_layers(x)
		return x

	def loss(self):
		...

	def save(self, path):
		torch.save(self.state_dict(), path)

	def train(self, epoch=self.epochs):
		model.train()
		tr_loss = 0
		#Need to split data into training and validation. Where??
		

	def run(self):
		for epoch in range(self.epochs):
			...


model = ConNet()
optim = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()
print(model)

