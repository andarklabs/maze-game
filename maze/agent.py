import torch as pt
import random as rd 
from env import Board
from model import Model, Trainer
import params as PARAMS
import utils
from collections import deque as dq

""" Agent Class:

    This class controls our ball and its choices when traversing the maze.
    Everything from is actions to its state to its memory to when it learns 
    is determined here.

    * memory: 
    * epsilon:
    * position:
    * gameboard:    exit is an int tuple that represents our exit and 
                    walls is a 2d array with 1's as walls and 0's as 
                    possibles (exit is 0 and if the entrance is a break 
                    in the maze it is a 1 - there should only be 1 maze break) 
"""
class Agent:

    def __init__(self) -> None:
        self.gameboard = Board()
        self.position = self.gameboard.initial_position
        self.state = self.get_state()
        self.moves = [(-1,0),(1,0),(0,-1),(0,1)]
        self.memory = dq(maxlen = PARAMS.MAX_MEM)
        self.epsilon = PARAMS.EPSILON_INIT
        self.device = "mps" if pt.backends.mps.is_available() else "cpu"
        self.model = Model(PARAMS.FLAT_SIZE,PARAMS.HIDDEN_SIZE,PARAMS.CHOICES, self.device)
        self.trainer = Trainer(self.model, lr=PARAMS.ALPHA, gamma=PARAMS.GAMMA)

        print(f"Using device: {self.device}")
        return None
    
    def get_state(self):
        state = self.gameboard.maze # state is all gameboard info
        state[self.position[0]][self.position[1]] = 2 # with the rat
        state[self.gameboard.exit[0]][self.gameboard.exit[1]] = 3 # and the exit
        return utils.flatten(state) # all flattened

    """ self.cycle():

    """
    def cycle(self, show = False) -> bool:
        state, action, new_state = self.choose_and_move()
        reward, done = self.reward()
        self.remember(state, action, reward, new_state, done)
        self.train_stm(state, action, reward, new_state, done)
        if show: self.gameboard.show()
        return done
    
    def reset(self) -> None:
        self.position = self.gameboard.initial_position
        return None

    """ self.move():
    
    """
    def choose_and_move(self) -> None:

        # prepare to pass state:
        state = self.state # state is all gameboard info

        if rd.uniform(0,1) > self.epsilon:  # we exploit:
            state0 = pt.tensor(state, dtype=pt.float)
            qs = self.model(state0)
            indx = pt.argmax(qs) # gets the index out from q values
        else:   # we explore:
            indx = rd.randint(0,PARAMS.CHOICES-1)

        move = self.moves[indx]

        # crate new_state:
        new_state = state
        # remove the old rat (from flattened new_state)
        new_state[self.position[0] * PARAMS.DIM + self.position[1]] = 1 
    
        # change the rat's position
        self.position = (self.position[0] + move[0], self.position[1] + move[1])
        
        # build the new rat
        new_state[self.position[0] * PARAMS.DIM + self.position[1]] = 2 

        # move the rat
        self.state = new_state

        # create choice array
        choice = [0, 0, 0, 0] # output is size 4
        choice[indx] = 1

        return state, choice, new_state

    """ self.remember():
    
    """
    def remember(self, state, action, reward, new_state, done) -> None:
        self.memory.append((state, action, reward, new_state, done))
        return None
    
    """ self.train_stm():

    """
    def train_stm(self, state, action, reward, new_state, done) -> None:
        self.trainer.train_step(state, action, reward, new_state, done)
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
        if self.gameboard.maze[self.position[0]][self.position[1]] == 0: # if we hit a wall
            reward = -10
            game_over = True
        elif self.gameboard.exit == self.position: # if we found the exit
            reward = 100
            game_over = True
        else: # we are in another part of the maze
            reward = 5 
            game_over = False

        return reward, game_over


def play():
    rat = Agent()

    # train
    for _ in range(PARAMS.GAMES):
        while rat.cycle():
            pass
        rat.train_ltm()
        rat.reset()
        rat.epsilon -= PARAMS.EPSILON_DEC

    # test
    while rat.cycle():
        rat.gameboard.show(rat.position)

if __name__ == "__main__":
    play()
    
