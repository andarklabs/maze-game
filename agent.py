import torch as pt
import random as rd 
import numpy as np 
import params as PARAMS


class Agent:

    def __init__(self) -> None:
        self.memory = ...
        self.state = ...
        self.epsilon = PARAMS.EPSILON
        self.alpha = PARAMS.ALPHA
        self.beta = ...
        self.gamma = ...
        pass

    def move(self):
        pass

    def remember(self):
        pass
    
    def train_stm(self):
        pass

    def train_ltm(self):
        pass

    def reward(self):
        pass
