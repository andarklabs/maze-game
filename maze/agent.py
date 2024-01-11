import torch as pt
import random as rd 
from env import Board
from model import Model, Trainer
import maze.params as PARAMS
from collections import deque as dq

""" Agent Class:

    This class controls our ball and its choices when traversing the maze.
    Everything from is actions to its state to its memory to when it learns 
    is determined here.

    * memory: 
    * radius:
    * epsilon:
    * beta: 
    * gameboard:    exit is an int tuple that represents our exit and 
                    walls is a 2d array with 1's as walls and 0's as 
                    possibles (exit is 0 and if the entrance is a break 
                    in the maze it is a 1 - there should only be 1 maze break) 
"""
class Agent:

    def __init__(self) -> None:
        self.memory = dq(maxlen = PARAMS.MAX_MEM)
        self.radius = PARAMS.RADIUS
        self.epsilon = PARAMS.EPSILON
        self.beta = PARAMS.BETA
        self.gameboard = Board()
        self.model = Model(8,256,9)
        self.trainer = Trainer(self.model, lr=PARAMS.ALPHA, gamma=PARAMS.GAMMA)

        return None

    def cycle(self, state) -> None:
        action, new_state = self.choose_and_move(state)
        reward, done = self.reward()
        self.remember(state, action, reward, new_state, done)
        self.train_stm(state, action, reward, new_state, done)
        self.gameboard.show()
        return None

    """ self.move():
    
    """
    def choose_and_move(self, state) -> None:
        moves = [-1,0,1]
        choice = [0, 0, 0, 0, 0, 0, 0, 0, 0] # output is size 9
        
        if rd.uniform(0,1) > self.epsilon: 
            state0 = pt.tensor(state, dtype=pt.float)
            qs = self.model(state0)
            move = pt.argmax(qs) # gets the index out from q values
            rem = move%3
            xmove = rem - 1
            ymove = (move-rem)/3 - 1 
        else: 
            xmove = rd.choice(moves)
            ymove = rd.choice(moves)

        new_state = (state[0] + xmove, state[1] + ymove)
        choice[move] = 1

        return choice, new_state

    """ self.remember():
    
    """
    def remember(self, state, action, reward, new_state, done) -> None:
        self.memory.append((state, action, reward, new_state, done))
        return None
    
    """ self.train_stm():

    """
    def train_stm(self, state, action, reward, new_state, done) -> None:
        self.trainer(state, action, reward, new_state, done)
        return None

    """ self.train_ltm():
    
    """
    def train_ltm(self) -> None:
        if len(self.memory) > PARAMS.BATCH_SIZE:
            mini_batch = rd.sample(self.memory, PARAMS.BATCH_SIZE) # list of tuples
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        return None

    """ self.reward():
    
    """
    def reward(self) -> int:
        # works only if all walls are labeled as 1 (and even entrance is labeled as 1)
        if self.gameboard.walls[self.state[0]][self.state[1]] == 1: 
            reward = -10
            game_over = True
        elif self.gameboard.exit == self.state:
            reward = 100
            game_over = True
        else:
            reward = 5 
            game_over = False

        return reward, game_over


