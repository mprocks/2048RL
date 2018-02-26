import torch
import torch.nn as nn
import torch.nn.functional as F

# 10 Layer CNN

class Model(nn.Module):

	def __init__(self, boardsize = 16, hiddensize = 32, outsize = 4):
		
		super(Model, self).__init__()
		layers = []
		fc_layers = []

		layers.append(nn.Conv2d(1, hiddensize, kernel_size=2, padding=1))
		layers.append(nn.ReLU())
		for i in range(4):
			layers.append(nn.Conv2d(hiddensize, hiddensize, kernel_size=2, padding=1))
			layers.append(nn.ReLU())
		for i in range(4):
			layers.append(nn.Conv2d(hiddensize, hiddensize, kernel_size=3, padding=1))
			layers.append(nn.ReLU())
		# for i in range(4):
		# 	layers.append(nn.Conv2d(hiddensize, hiddensize, kernel_size=4, padding=1))
		# 	layers.append(nn.ReLU())

		fc_layers.append(nn.Linear(hiddensize*9*9, hiddensize))
		fc_layers.append(nn.ReLU())
		fc_layers.append(nn.Linear(hiddensize, outsize))

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)

		#initialisation
		for module in self.modules():
			if isinstance(module, nn.Linear):
				module.weight.data.normal_(0, 0.05)
			if isinstance(module, nn.Conv2d):
				module.weight.data.normal_(0, 0.05)

	def forward(self, x):
		# print(x.size())
		x = self.layers(x)
		# print(x.size())
		x = x.view(-1)
		x = self.fc_layers(x)
		# print(x.size())
		return x		