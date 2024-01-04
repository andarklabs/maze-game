import torch as pt
import random as rd 
import numpy as np 
import params as PARAMS
from collections import deque as dq

""" Agent Class:

    This class controls our ball and its choices when traversing the maze.
    Everything from is actions to its state to its memory to when it learns 
    is determined here.

    * memory: 
    * radius:
    * state: 
    * epsilon:
    * alpha:
    * beta: 
    * gamma:
"""
class Agent:

    def __init__(self) -> None:
        self.memory = dq(maxlen = PARAMS.MAX_MEM)
        self.radius = PARAMS.RADIUS
        self.state = PARAMS.INIT_STATE
        self.epsilon = PARAMS.EPSILON
        self.alpha = PARAMS.ALPHA
        self.beta = PARAMS.BETA
        self.gamma = PARAMS.GAMMA

        return None

    """ self.move():
    
    """
    def move(self) -> None:
        return None

    """ self.remember():
    
    """
    def remember(self) -> None:
        return None
    
    """ self.train_stm():

    """
    def train_stm(self) -> None:
        return None

    """ self.train_ltm():
    
    """
    def train_ltm(self) -> None:
        return None

    """ self.reward():
    
    """
    def reward(self) -> None:
        return None
