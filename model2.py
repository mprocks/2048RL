import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

	def __init__(self, boardsize = 16, hiddensize = 32, outsize = 4):
		
		super(Model, self).__init__()
		layers = []

		layers.append(nn.Linear(boardsize, hiddensize))
		layers.append(nn.ReLU())
		layers.append(nn.Linear(hiddensize, hiddensize))
		layers.append(nn.ReLU())
		layers.append(nn.Linear(hiddensize, outsize))

		self.layers = nn.Sequential(*layers)

		#initialisation
		for module in self.modules():
			if isinstance(module, nn.Linear):
				module.weight.data.normal_(0, 0.05)

	def forward(self, x):
		x = self.layers(x)
		return x		