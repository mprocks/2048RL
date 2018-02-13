import model1
import random
import torch
from torch.autograd import Variable
import numpy as np

class bot:
	def __init__(self):
		self.net = model1.Model().cuda()
		# print("NET = ", self.net)
		self.randomization = 0.1
		self.movestrings = ["w", "s", "a", "d"]

	def move(self, board, score):
		board = np.array(board)
		board = np.log2(board+0.5).astype(int)
		x = board.flatten()
		x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
		x = Variable(x)
		print("x = ", x)
		out = self.net.forward(x).cuda()
		sorted, indices = torch.sort(out, descending = True)
		movelist = indices.data.cpu().numpy()
		final_move = np.random.choice(movelist, 1, p=[1-self.randomization] + (len(movelist)-1)*[self.randomization/(len(movelist)-1)])[0]
		# returnmoves = 
		return final_move