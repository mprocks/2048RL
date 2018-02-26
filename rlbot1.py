import random
import torch
import copy
from torch.autograd import Variable
import numpy as np

class bot:
	def __init__(self, net, randomization):
		self.net = net
		# print("NET = ", self.net)
		self.randomization = randomization
		self.movestrings = ["w", "s", "a", "d"]

	def canMove(self, board, move):

		if move == 0:
			BOARD_SIZE = 4
			for i in range(0, BOARD_SIZE):
				for j in range(1, BOARD_SIZE):
					if board[i][j-1] == -1 and board[i][j] > -1:
						return True
					elif (board[i][j-1] == board[i][j]) and board[i][j-1] != -1:
						return True
			return False

		elif move == 1:
			BOARD_SIZE = 4
			for i in range(0, BOARD_SIZE):
				for j in range(0, BOARD_SIZE-1):
					if board[i][j+1] == -1 and board[i][j] > -1:
						return True
					elif (board[i][j+1] == board[i][j]) and board[i][j+1] != -1:
						return True
			return False

		elif move == 2:
			BOARD_SIZE = 4
			for i in range(1, BOARD_SIZE):
				for j in range(0, BOARD_SIZE):
					if board[i-1][j] == -1 and board[i][j] > -1:
						return True
					elif (board[i-1][j] == board[i][j]) and board[i-1][j] != -1:
						return True
			return False

		elif move == 3:
			BOARD_SIZE = 4
			for i in range(0, BOARD_SIZE-1):
				for j in range(0, BOARD_SIZE):
					if board[i+1][j] == -1 and board[i][j] > -1:
						return True
					elif (board[i+1][j] == board[i][j]) and board[i+1][j] != -1:
						return True
			return False

	def move(self, board, score, ):
		board = np.array(board)
		board = np.log2(board+0.5).astype(int)
		x = board.flatten()
		x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
		x = x.view(1,1,4,4) # add this for convnets
		x = Variable(x)
		# print("x = ", x)
		out = self.net.forward(x).cuda()
		sorted, indices = torch.sort(out, descending = True)
		movelist = indices.data.cpu().numpy().tolist()
		temp = copy.deepcopy(movelist)
		for i in temp:
			if self.canMove(board, i) == False:
				movelist.remove(i)
				
		movelist = np.asarray(movelist)
		# print("movelist :", movelist)
		if len(movelist) > 1:
			final_move = np.random.choice(movelist, 1, p=[1-self.randomization] + (len(movelist)-1)*[self.randomization/(len(movelist)-1)])[0]
		else:
			final_move = movelist[0]
		# print(final_move)
		return final_move