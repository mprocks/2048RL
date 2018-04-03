import random
import copy
import time 

class bot:
	def __init__(self, depth):
		self.max_depth = depth
		self.loss_score = -1000000
		self.board_map = [[10, 5, 2, 0],
					 [15, 10, 5, 2],
					 [30, 20, 10, 5],
					 [40, 30, 15, 10]
					]
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

	def calc_heuristics(self, board):
		final = sum(row.count(0) for row in board)*2000
		board_map = self.board_map
		scores = [0, 0, 0, 0]
		for i in range(4):
			for j in range(4):
				scores[0] += board[i][j]*board_map[i][j]
				scores[1] += board[i][j]*board_map[i][3-j]
				scores[2] += board[i][j]*board_map[3-i][j]
				scores[3] += board[i][j]*board_map[3-i][3-j]
		final += max(scores)
		return final

	def minimax(self, board, depth, alpha, beta):
		# print ("TEST", depth)
		if depth == 0:
			ret = (-1, self.calc_heuristics(board))
			# print("depth = ", depth, ret)
			return ret

		elif depth%2 == 1:
			val = -10**14
			move = -1
			movemap = [0, 2, 1, 3]
			cnt = 0
			# movemap = [0, 1, 2, 3]
			for i in range(4):
				if(self.canMove(board, movemap[i])):
					temp_board = copy.deepcopy(board)
					self.rotateMatrixClockwise(temp_board, i)
					self.moveTiles(temp_board)
					self.mergeTiles(temp_board)
					# self.rotateMatrixClockwise(temp_board, 4-i)
					temp_alpha = self.minimax(temp_board, depth-1, alpha, beta)[1]
					# print("TEMP+ALPHA!", temp_alpha, alpha)
					if(temp_alpha > val):
						move = movemap[i]
						# print("change!", move)
						val = temp_alpha
					alpha = max(alpha, val)
					if(beta <= alpha):
						break
			ret = (move, val)
			# print("depth = ", depth, ret)
			return ret

		else:
			val = 10**14
			for i in range(4):
				for j in range(4):
					if(board[i][j] == 0):
						temp_board = copy.deepcopy(board)
						temp_board[i][j] = 2
						val = min(val, self.minimax(temp_board, depth-1, alpha, beta)[1])
						beta = min(beta, val)
						if(beta <= alpha):
							break
						temp_board = copy.deepcopy(board)
						temp_board[i][j] = 4
						val = min(val, self.minimax(temp_board, depth-1, alpha, beta)[1])
						beta = min(beta, val)
						if(beta <= alpha):
							break
			ret = (-1, val)
			# print("depth = ", depth, ret)
			return ret

	def move(self, board, score):
		final_move = self.minimax(board, self.max_depth, -10**14, 10**14)
		# print(final_move)
		# time.sleep(0.05)

		# print(board)
		# return random.randint(0, 3)
		return final_move[0]