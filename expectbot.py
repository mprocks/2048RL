import random
import copy
import time 
import numpy as np

class bot:
	def __init__(self, depth):
		self.max_depth = depth
		self.loss_score = -10**7

		self.board_map1 = np.asarray([[10, 5, 2, 0],
									 [15, 10, 5, 2],
									 [30, 15, 10, 5],
									 [40, 30, 15, 10]
									])
		self.board_map2 = np.asarray([
									 [40, 30, 15, 10],
									 [30, 15, 10, 5],
									 [15, 10, 5, 2],
									 [10, 5, 2, 0]
									])
		self.board_map3 = np.asarray([ [0, 2, 5, 10],
										[2, 5, 10, 15],
										[5, 10, 15, 30],
										[10, 15, 30, 40]
									])
		self.board_map4 = np.asarray([ 
										[10, 15, 30, 40],
										[5, 10, 15, 30],
										[2, 5, 10, 15],
										[0, 2, 5, 10]
									])

		self.exp = [14**i for i in range(16)]

		self.board_map1 = self.board_map1/2
		self.board_map2 = self.board_map2/2
		self.board_map3 = self.board_map3/2
		self.board_map4 = self.board_map4/2
		self.scores = [0, 0, 0, 0]
		pass

	def canMove(self, board, move):

		if move == 0:
			BOARD_SIZE = 4
			for i in range(0, BOARD_SIZE):
				for j in range(1, BOARD_SIZE):
					if board[i][j-1] == 0 and board[i][j] > 0:
						return True
					elif (board[i][j-1] == board[i][j]) and board[i][j-1] != 0:
						return True
			return False

		elif move == 1:
			BOARD_SIZE = 4
			for i in range(0, BOARD_SIZE):
				for j in range(0, BOARD_SIZE-1):
					if board[i][j+1] == 0 and board[i][j] > 0:
						return True
					elif (board[i][j+1] == board[i][j]) and board[i][j+1] != 0:
						return True
			return False

		elif move == 2:
			BOARD_SIZE = 4
			for i in range(1, BOARD_SIZE):
				for j in range(0, BOARD_SIZE):
					if board[i-1][j] == 0 and board[i][j] > 0:
						return True
					elif (board[i-1][j] == board[i][j]) and board[i-1][j] != 0:
						return True
			return False

		elif move == 3:
			BOARD_SIZE = 4
			for i in range(0, BOARD_SIZE-1):
				for j in range(0, BOARD_SIZE):
					if board[i+1][j] == 0 and board[i][j] > 0:
						return True
					elif (board[i+1][j] == board[i][j]) and board[i+1][j] != 0:
						return True
			return False

	def rotateMatrixClockwise(self, tileMatrix, rotations):
		BOARD_SIZE = 4
		for r in range(rotations):
			for i in range(0, int(BOARD_SIZE/2)):
				for k in range(i, BOARD_SIZE- i - 1):
					temp1 = tileMatrix[i][k]
					temp2 = tileMatrix[BOARD_SIZE - 1 - k][i]
					temp3 = tileMatrix[BOARD_SIZE - 1 - i][BOARD_SIZE - 1 - k]
					temp4 = tileMatrix[k][BOARD_SIZE - 1 - i]

					tileMatrix[BOARD_SIZE - 1 - k][i] = temp1
					tileMatrix[BOARD_SIZE - 1 - i][BOARD_SIZE - 1 - k] = temp2
					tileMatrix[k][BOARD_SIZE - 1 - i] = temp3
					tileMatrix[i][k] = temp4

	def moveTiles(self, tileMatrix):
	# We want to work column by column shifting up each element in turn.
		BOARD_SIZE = 4
		for i in range(0, BOARD_SIZE): # Work through our 4 columns.
			for j in range(0, BOARD_SIZE - 1): # Now consider shifting up each element by checking top 3 elements if 0.
				while tileMatrix[i][j] == 0 and sum(tileMatrix[i][j:]) > 0: # If any element is 0 and there is a number to shift we want to shift up elements below.
					for k in range(j, BOARD_SIZE - 1): # Move up elements below.
						tileMatrix[i][k] = tileMatrix[i][k + 1] # Move up each element one.
					tileMatrix[i][BOARD_SIZE - 1] = 0

	def mergeTiles(self, tileMatrix):
		BOARD_SIZE = 4
		for i in range(0, BOARD_SIZE):
			for k in range(0, BOARD_SIZE - 1):
				if tileMatrix[i][k] == tileMatrix[i][k + 1] and tileMatrix[i][k] != 0:
					tileMatrix[i][k] = tileMatrix[i][k] * 2
					tileMatrix[i][k + 1] = 0
					# TOTAL_POINTS += tileMatrix[i][k]
					self.moveTiles(tileMatrix)

	def gethash(self, board):
		hashval = 0
		for i in range(4):
			for j in range(4):
				hashval += board[i][j]*self.exp[4*i+j]
		return hashval

	def calc_heuristics(self, board):
		# Implemented directly in function
		final = (16-np.count_nonzero(board))*10000
		board = board**2
		temp = board.ravel().dot(self.board_map1.ravel())
		temp = max(temp, board.ravel().dot(self.board_map2.ravel()))
		temp = max(temp, board.ravel().dot(self.board_map3.ravel()))
		temp = max(temp, board.ravel().dot(self.board_map4.ravel()))
		final += temp
		return final

	def expectimax(self, board, depth):
		# print ("TEST", depth)
		if depth == 0:
			# Calculating heuristics
			final = (16-np.count_nonzero(board))*10000
			board = board**2
			temp = board.ravel().dot(self.board_map1.ravel())
			temp = max(temp, board.ravel().dot(self.board_map2.ravel()))
			temp = max(temp, board.ravel().dot(self.board_map3.ravel()))
			temp = max(temp, board.ravel().dot(self.board_map4.ravel()))
			final += temp
			ret = (-1, final)
			# ret = (-1, self.calc_heuristics(board))
			# print("depth = ", depth, ret)
			return ret

		elif depth%2 == 1:
			alpha = self.loss_score*(depth**2)
			move = -1
			movemap = [0, 2, 1, 3]
			cnt = 0
			# movemap = [0, 1, 2, 3]
			r_board = np.copy(board)
			# r_board = board
			for i in range(4):
				if(self.canMove(board, movemap[i])):
					# temp_board = r_board
					temp_board = np.copy(r_board)
					self.moveTiles(temp_board)
					self.mergeTiles(temp_board)
					# self.rotateMatrixClockwise(temp_board, 4-i)
					temp_alpha = self.expectimax(temp_board, depth-1)[1]
					# print("TEMP+ALPHA!", temp_alpha, alpha)
					if(temp_alpha > alpha):
						move = movemap[i]
						# print("change!", move)
						alpha = temp_alpha
				r_board = np.rot90(r_board)
			ret = (move, alpha)
			# print("depth = ", depth, ret)
			return ret

		else:
			alpha = 0
			cntz = 16-np.count_nonzero(board)
			if(cntz < 8):
				for i in range(4):
					for j in range(4):
						if(board[i][j] == 0):
							temp_board = np.copy(board)
							temp_board[i][j] = 2

							temp_alpha = (1.0/cntz)*self.expectimax(temp_board, depth-1)[1]
							if(random.randint(1, 10) >= 0): # Change this to exchange time with accuracy
								# temp_board = board
								temp_board = np.copy(board)
								temp_board[i][j] = 4
								alpha += 0.9*temp_alpha + 0.1*(1.0/cntz)*self.expectimax(temp_board, depth-1)[1]
							else:
								alpha += temp_alpha
			else:
				partial = 0
				for i in random.sample(list(range(4)), 4):
					for j in random.sample(list(range(4)), 4):
						if(partial >= 6):
							break
						if(board[i][j] == 0):
							temp_board = np.copy(board)
							temp_board[i][j] = 2
							alpha += (1.0/6)*self.expectimax(temp_board, depth-1)[1]
							# alpha += 0.9*(1.0/cntz)*self.expectimax(temp_board, depth-1)[1]
							# temp_board = np.copy(board)
							# temp_board[i][j] = 4
							# alpha += 0.1*(1.0/cntz)*self.expectimax(temp_board, depth-1)[1]
							partial += 1
			ret = (-1, alpha)
			# print("depth = ", depth, ret)
			return ret

	def move(self, board, score):
		temp_board = copy.deepcopy(np.asarray(board))
		final_move = self.expectimax(temp_board, self.max_depth)
		# print(final_move)
		# time.sleep(0.05)

		# print(board)
		# return random.randint(0, 3)
		return final_move[0]