# ref: https://github.com/patrickloeber/snake-ai-pytorch
from collections import namedtuple
from enum import Enum

import numpy as np  # type: ignore
import pygame

pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 1
SPEED = 2


class BallAI:
    def __init__(self, w=1280, h=720, radius=2, file_path="assets/env_1280_720.png"):
        self.radius = radius
        self.boundary = self._interpret_boundary(file_path)
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Ball")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.location = Point(self.w / 2, self.h / 2)
        self.direction = Direction.RIGHT
        self.score = 0
        self.frame_iteration = 0

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 1000:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 5. return game over and score
        return reward, game_over, self.score

    def _is_collison(
        # self, player_pos: pygame.Vector2, boundary: set[Tuple[int, int]]
        self,
    ) -> bool:
        """Return True if the player is colliding with the boundary."""
        # handle issue of continuous values not being picked up
        if (int(self.location.x), int(self.location.y)) in self.boundary:
            return True
        return False

    def _update_ui(self, bg_path: str, screen: pygame.Surface):
        screen.fill("black")
        bg = pygame.image.load(bg_path)
        screen.blit(bg, (0, 0))
        pygame.display.flip()

    def _move(self, action):
        """
        i might adjust this wholesale st we just do
        pygame.draw.circle(surface=screen, color="red", center=player_pos, radius=2)
        collision detection should be done inside the agent.py vs the env.py
        bc we aren't playing snake anymore

        based on agent action, move the ball, return mouse eye view
        """
        # [straight, right, left]
        # i.e. [1, 0, 0] = no change, [0, 1, 0] = right turn, [0, 0, 1] = left turn
        # based on current direction, turn right or left or no change

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.location.x
        y = self.location.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.location = Point(x, y)
