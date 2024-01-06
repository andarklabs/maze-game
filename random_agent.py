import random
from collections import deque

import numpy as np
from matplotlib.pyplot import plot  # type: ignore

from env import BallAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def random_one_hot():
    # For the one-hot encoding
    one_hot = [0, 0, 0, 0, 0]
    random_index = random.randint(0, 4)
    one_hot[random_index] = 1  # Set the selected index to 1
    return one_hot


def random_number():
    # For the random number between 1 and 5
    random_number = random.randint(1, 5)
    return random_number


class RandomAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model: callable = random_one_hot
        # no need for a trainer
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        location = game.location
        point_l = Point(location.x - 5, location.y)
        point_r = Point(location.x + 5, location.y)
        point_u = Point(location.x, location.y - 5)
        point_d = Point(location.x, location.y + 5)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger left
            game._is_collision(point_l),
            # Danger up
            game._is_collision(point_u),
            # Danger right
            game._is_collision(point_r),
            # Danger down
            game._is_collision(point_d),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    # def train_long_memory(self):
    #     if len(self.memory) > BATCH_SIZE:
    #         mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    #     else:
    #         mini_sample = self.memory

    #     states, actions, rewards, next_states, dones = zip(*mini_sample)
    #     self.trainer.train_step(states, actions, rewards, next_states, dones)
    # for state, action, reward, nexrt_state, done in mini_sample:
    #    self.trainer.train_step(state, action, reward, next_state, done)

    # def train_short_memory(self, state, action, reward, next_state, done):
    #     self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games
        # final_move = [0,0,0]
        # if random.randint(0, 200) < self.epsilon:
        #     move = random.randint(0, 2)
        #     final_move[move] = 1
        # else:
        #     state0 = torch.tensor(state, dtype=torch.float)
        #     prediction = self.model(state0)
        #     move = torch.argmax(prediction).item()
        # final_move[move] = 1

        # just use random one hot encoding
        final_move = self.model()
        return final_move


def play():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = RandomAgent()
    game = BallAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        print(final_move)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    play()
