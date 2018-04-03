# 2048 Game written using the Pygame module
# 
# Lewis Deane
# 23/12/2014

import pygame, sys, time
from pygame.locals import *
from colours import *
from random import *
import copy
import randbot
import manual
import model1
import model2
import model3
import model4
import model5
import rlbot1
import rlbot2
import expectbot
import minimaxbot
import random as rnd
import torch
import numpy as np
import pickle

TOTAL_POINTS = 0
DEFAULT_SCORE = 2
BOARD_SIZE = 4
avg_score = 0
n_moves = 0
ind = 0

pygame.init()

SURFACE = pygame.display.set_mode((400, 500), 0, 32)
pygame.display.set_caption("2048")

myfont = pygame.font.SysFont("monospace", 25)
scorefont = pygame.font.SysFont("monospace", 50)

tileMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
undoMat = []

memory_size = 1000000
memory = [0 for i in range (memory_size)]
tile_count = {}
max_tile_count = {}

def add_to_memory(old_state, move, r, new_state):
	global ind
	memory[ind%memory_size] = [old_state, move, r, new_state]
	ind += 1
	# if(len(memory) >= memory_size):
	# 	memory[rnd.randint(0, 199)] = [old_state, move, r, new_state]
	# else:
	# 	memory.append([old_state, move, r, new_state])

def move():
	moves = ["w", "s", "a", "d"]
	return choice(moves)

def main(bot, fromLoaded = False):

	'''
		0 - up
		1 - down
		2 - left
		3 - right
	'''

	global TOTAL_POINTS
	global tileMatrix
	global avg_score
	global n_moves

	if not fromLoaded:
		placeRandomTile()
		placeRandomTile()

	printMatrix()

	while True:
		# for event in pygame.event.get():
		# 	if event.type == QUIT:
		# 		pygame.quit()
		# 		sys.exit()

		if checkIfCanGo() == True:
			old_state = copy.deepcopy(tileMatrix)
			old_score = TOTAL_POINTS
			n_moves += 1
			event = bot.move(tileMatrix, TOTAL_POINTS)
			# if event == "s":
				# if isArrow(event.key):
			rotations = getRotations(event)

			addToUndo()

			for i in range(0, rotations):
				rotateMatrixClockwise()

			if canMove():
				moveTiles()
				mergeTiles()
				placeRandomTile()
			else:
				print("Invalid")

			for j in range(0, (4 - rotations) % 4):
				rotateMatrixClockwise()
			if(TOTAL_POINTS - old_score >= 32):
				add_to_memory(old_state, event, (TOTAL_POINTS - old_score)/10, tileMatrix)
			else:
				add_to_memory(old_state, event, -1, tileMatrix)
			# add_to_memory(old_state, event, 0, tileMatrix)
			printMatrix()
		else:
			add_to_memory(old_state, event, -TOTAL_POINTS/100, -1)
			# add_to_memory(old_state, event, -1, -1)
			# add_to_memory(old_state, event, 0, -1)
			avg_score += TOTAL_POINTS
			printGameOver()
			reset(bot)
			break

			# if event.type == KEYDOWN:
			# 	global BOARD_SIZE

			# if event.key == pygame.K_r:
			# 	reset()

			# 	if 50 < event.key and 56 > event.key:
			# 		BOARD_SIZE = event.key - 48
			# 		reset()

			# 	if event.key == pygame.K_s:
			# 		saveGameState()
			# 	elif event.key == pygame.K_l:
			# 		loadGameState()
			# 	elif event.key == pygame.K_u:
			# 		undo()

		# pygame.display.update()

def printMatrix():
	pass
	# SURFACE.fill(BLACK)

	# global BOARD_SIZE
	# global TOTAL_POINTS

	# for i in range(0, BOARD_SIZE):
	# 	for j in range(0, BOARD_SIZE):
	# 		pygame.draw.rect(SURFACE, getColour(tileMatrix[i][j]), (i*(400/BOARD_SIZE), j*(400/BOARD_SIZE) + 100, 400/BOARD_SIZE, 400/BOARD_SIZE))
			
	# 		label = myfont.render(str(tileMatrix[i][j]), 1, (255,255,255))
	# 		label2 = scorefont.render("Score:" + str(TOTAL_POINTS), 1, (255, 255, 255))

	# 		SURFACE.blit(label, (i*(400/BOARD_SIZE) + 30, j*(400/BOARD_SIZE) + 130))
	# 		SURFACE.blit(label2, (10, 20))

def printGameOver():
	global TOTAL_POINTS

	SURFACE.fill(BLACK)

	label = scorefont.render("Game Over!", 1, (255,255,255))
	label2 = scorefont.render("Score:" + str(TOTAL_POINTS), 1, (255,255,255))
	label3 = myfont.render("Press r to restart!", 1, (255,255,255))

	SURFACE.blit(label, (50, 100))
	SURFACE.blit(label2, (50, 200))
	SURFACE.blit(label3, (50, 300))

def placeRandomTile():
	count = 0
	empty_tiles = []
	for i in range(0, BOARD_SIZE):
		for j in range(0, BOARD_SIZE):
			if tileMatrix[i][j] == 0:
				empty_tiles.append((i,j))

	tile = rnd.choice(empty_tiles)
	prob = rnd.randint(1, 10)

	if prob <= 9:
		tileMatrix[tile[0]][tile[1]] = 2
	else:
		tileMatrix[tile[0]][tile[1]] = 4

def floor(n):
	return int(n - (n % 1))

def moveTiles():
	# We want to work column by column shifting up each element in turn.
	for i in range(0, BOARD_SIZE): # Work through our 4 columns.
		for j in range(0, BOARD_SIZE - 1): # Now consider shifting up each element by checking top 3 elements if 0.
			while tileMatrix[i][j] == 0 and sum(tileMatrix[i][j:]) > 0: # If any element is 0 and there is a number to shift we want to shift up elements below.
				for k in range(j, BOARD_SIZE - 1): # Move up elements below.
					tileMatrix[i][k] = tileMatrix[i][k + 1] # Move up each element one.
				tileMatrix[i][BOARD_SIZE - 1] = 0

def mergeTiles():
	global TOTAL_POINTS

	for i in range(0, BOARD_SIZE):
		for k in range(0, BOARD_SIZE - 1):
				if tileMatrix[i][k] == tileMatrix[i][k + 1] and tileMatrix[i][k] != 0:
					tileMatrix[i][k] = tileMatrix[i][k] * 2
					tileMatrix[i][k + 1] = 0
					TOTAL_POINTS += tileMatrix[i][k]
					moveTiles()

def checkIfCanGo():
	# print("IN CHECK", tileMatrix)
	for i in range(0, BOARD_SIZE ** 2):
		if tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE] == 0:
			return True

	for i in range(0, BOARD_SIZE):
		for j in range(0, BOARD_SIZE - 1):
			if tileMatrix[i][j] == tileMatrix[i][j + 1]:
				return True
			elif tileMatrix[j][i] == tileMatrix[j + 1][i]:
				return True
	return False

def reset(bot):
	global TOTAL_POINTS
	global tileMatrix
	global n_moves
	global max_tile_count

	# f = open("log.txt", "a")

	# f.write(str(TOTAL_POINTS) + ", ")
	# f.write(str(max(max(tileMatrix))))
	# f.write("\n")

	print("Total Points: ", TOTAL_POINTS)
	print("Number of moves: ", n_moves)
	# print("Max Tile: ", max(max(tileMatrix)))

	TOTAL_POINTS = 0
	n_moves = 0
	SURFACE.fill(BLACK)
	max_tile = max(max(tileMatrix))

	for i in tileMatrix:
		for j in i:
			if j in tile_count.keys():
				# temp_dict[j] += 1
				tile_count[j] += 1
			else:
				max_tile_count[j] = 0
				tile_count[j] = 1

	max_tile_count[max_tile] += 1

	tileMatrix = [[0 for i in range(0, BOARD_SIZE)] for j in range(0, BOARD_SIZE)]

	# main(bot)

def canMove():
	# print("TITLE : ", tileMatrix)
	for i in range(0, BOARD_SIZE):
		for j in range(1, BOARD_SIZE):
			if tileMatrix[i][j-1] == 0 and tileMatrix[i][j] > 0:
				return True
			elif (tileMatrix[i][j-1] == tileMatrix[i][j]) and tileMatrix[i][j-1] != 0:
				return True

	return False

def saveGameState():
	f = open("savedata", "w")

	line1 = " ".join([str(tileMatrix[floor(x / BOARD_SIZE)][x % BOARD_SIZE]) for x in range(0, BOARD_SIZE**2)])
	
	f.write(line1 + "\n")
	f.write(str(BOARD_SIZE)  + "\n")
	f.write(str(TOTAL_POINTS))
	f.close()

def loadGameState():
	global TOTAL_POINTS
	global BOARD_SIZE
	global tileMatrix

	f = open("savedata", "r")

	mat = (f.readline()).split(' ', BOARD_SIZE ** 2)
	BOARD_SIZE = int(f.readline())
	TOTAL_POINTS = int(f.readline())

	for i in range(0, BOARD_SIZE ** 2):
		tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE] = int(mat[i])

	f.close()

	main(True)

def rotateMatrixClockwise():
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

def isArrow(k):
	return(k == pygame.K_UP or k == pygame.K_DOWN or k == pygame.K_LEFT or k == pygame.K_RIGHT)

def getRotations(k):
	if k == 0:
		return 0
	elif k == 1:
		return 2
	elif k == 2:
		return 1
	elif k == 3:
		return 3
		
def convertToLinearMatrix():
	mat = []

	for i in range(0, BOARD_SIZE ** 2):
		mat.append(tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE])

	mat.append(TOTAL_POINTS)

	return mat

def addToUndo():
	undoMat.append(convertToLinearMatrix())

def undo():
	if len(undoMat) > 0:
		mat = undoMat.pop()

		for i in range(0, BOARD_SIZE ** 2):
			tileMatrix[floor(i / BOARD_SIZE)][i % BOARD_SIZE] = mat[i]

		global TOTAL_POINTS
		TOTAL_POINTS = mat[BOARD_SIZE ** 2]

		printMatrix()

def mse_loss(input, target):
    return torch.sum((input - target)**2)

def train_bot(net, learning_rate):
	# rnd.shuffle(memory)
	# loss = mse_loss()
	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
	gamma = 0.9
	for i in reversed(memory):
		if(i == 0):
			continue
		old_state = np.asarray(i[0])
		optimizer.zero_grad()
		x1 = np.zeros([16, 4, 4])
		for dim in range(16):
			if dim == 0:
				x1[dim][np.where(old_state == -1)] = 1
			else:
				x1[dim][np.where(old_state == i)] = 1
		x1 = torch.from_numpy(x1).type(torch.FloatTensor).cuda()
		x1 = x1.view(1,16,4,4) # add this for convnets
		x1 = torch.autograd.Variable(x1)
		out1 = net.forward(x1).cuda()


		r = i[2]

		move = i[1]

		if(i[3] != -1):
			new_state = np.asarray(i[3])

			x2 = np.zeros([16, 4, 4])
			for i in range(16):
				if i == 0:
					x2[i][np.where(old_state == -1)] = 1
				else:
					x2[i][np.where(old_state == i)] = 1
			x2 = torch.from_numpy(x2).type(torch.FloatTensor).cuda()
			x2 = x2.view(1,16,4,4) # add this for convnets
			x2 = torch.autograd.Variable(x2)
			# x2 = new_state.flatten()
			# x2 = torch.from_numpy(x2).type(torch.FloatTensor).cuda()
			# x2 = x2.view(1,1,4,4)	# add this for convnets
			# x2 = torch.autograd.Variable(x2)

			out2 = net.forward(x2).cuda()
			val = r + gamma*torch.max(out2)

		else:
			val = r

		
		# print(r, out1.data)

		target1 = out1.clone()
		target1[move] = val

		target1 = target1.clamp(-10, 10)

		# print("target1 = ", target1.data)

		outloss = mse_loss(target1, out1)
		outloss.backward()
		optimizer.step()


# net = model1.Model().cuda()
# bot = randbot.bot()
# net = torch.load("epoch_lr-5_rnd0.2950.pt")
# max_score = 2500
# while 1:
# net = model5.Model().cuda()
# net.load_state_dict("good_initializationcnn2.pt")
# net = torch.load('final50.pt')
# net = torch.load('16-channelcnn(post0_1e-06)250.pt')
# params = [(0.2, 0.0001), (0.1, 0.00001), (0, 0.000001)]
# for param in params:
# param = (0, 0.0000001)
# print(param)
# bot = rlbot2.bot(net, param[0])
bot = expectbot.bot(3)
# bot = minimaxbot.bot(5)
ngames = 100000
# running_avg = [0 for i in range(ngames)]
for i in range(0, ngames):
	main(bot)
	# running_avg[i] = avg_score
	# print("average score: ", (avg_score - running_avg[max(i-10, 0)])/(min(i+1, 10)))
	print("average score: ", avg_score/(i+1))
	print("number of games: ", i+1)
	print("Tile Counts: ")
	for key, value in sorted(tile_count.items()):
		print(key, value)
	print("Tile Counts: ")
	print("Ind = ", ind)
	print("memory_size = ", memory_size)
	if(ind > memory_size):
		with open("dataset", 'wb') as fp:
			pickle.dump(memory, fp)
			break
	# for key, value in sorted(max_tile_count.items()):
	# 	print(key, value)

	# train_bot(net, param[1])
	# if i%200 == 0:
	# 	torch.save(net, '16-channelcnn(post' + str(param[0]) + '_' + str(param[1]) + ')' + str(i) + '.pt')
	# print(avg_score/ngames, max_score)
	# if(avg_score/ngames > max_score):
	# 	max_score = avg_score/ngames
	# 	print("writing ", max_score)
	# 	torch.save(net, 'good_initializationcnn2.pt')
	# avg_score = 0