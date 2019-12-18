import sys
import os
sys.path.append(os.path.abspath('..\\modules'))
import HandGestureDataset
from HandGestureDataset import HandGestureDataset, Resize, Cluster, ToTensor
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

train_path = "C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\Training"
test_path = "C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\Testing"
noise_path = "C:\\Users\\Noah\\Documents\\CISL\\CISL_HandGesture\\Noise_Results"
def main():
	model = ConNet()
	composed = transforms.Compose([Cluster(), Resize(244), ToTensor()])
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum =0.09)
	scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
	classes = ('1', '2', '3', '4', '5')
	epochs = 10
	savefile = ""
	for epoch in range(1, epochs + 1):
		if savefile != "":
			NeuralNet.train(model, optimizer, train_path, epoch, composed,workers=2, previous_model=savefile)
		else:
			NeuralNet.train(model, optimizer, train_path, epoch, composed,workers=2)

		NeuralNet.test(model, test_path, composed)
		scheduler.step()
		savefile = "models\\cnn_step" +str(epoch) + ".pt"
		model.save(savefile)
	model.save("cnn_pre_added.pt")
	
	#Here we incorporate the testing data into the model to validate on the outside data
	for epoch in range(1, epochs + 1):
		NeuralNet.train(model, optimizer, test_path, epochs, composed)
		scheduler.step()
	module.save("cnn_fully_trained.pt")
	#Here we test the accuracy on the outside data
	NeuralNet.test(model, noise_path, composed)



if __name__=='__main__':
	main()