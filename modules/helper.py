# Helper functions of for now unspecified class
import NeuralNet

import torch
import torch.optim as optim
from torch.optim import Adam, SGD


class Helper():

	def __init__():
		pass

	#Loads a pytorch NN given a path (String) Returns a Neural Net torch object
	def load(path):
		model = ConNet()
		model.load_state_dict(torch.load(path))
		model.eval()
		return model

	def checkpoint(path, model=None, optimizer=None, epoch=None, loss=None):
		torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
			'loss': loss
			#Add
			})


	#Loads a pytorch NN for resuming training
	def load_and_resume(path):
		model = ConNet()
		optimize = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
		check = torch.load(path)
		model.load_state_dict(check['model_state_dict'])
		optimize.load_state_dict(check['optimizer_state_dict'])
		epoch = check['epoch']
		loss = checkpoint['loss']

		
		