# ref: https://github.com/patrickloeber/snake-ai-pytorch
from collections import namedtuple
from enum import Enum
from typing import Tuple, Union

import numpy as np  # type: ignore
import pygame

pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    NOOP = 5


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 10
SPEED = 2


class BallAI:
    def __init__(self, w=1280, h=720, radius=2, file_path="assets/env_1280_720.png"):
        self.radius = radius
        self.w, self.h = w, h
        self.bg_path = file_path  # background image path
        self.boundary = self._interpret_boundary(file_path)
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Ball")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.location = Point(self.w / 2, self.h / 2)
        self.direction = Direction.NOOP
        self.score = 0
        self.frame_iteration = 0

    def play_step(self, action):
        """player starts at the center of the screen, no movement."""
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
        if self._is_collision() or self.frame_iteration > 1000:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 5. return game over and score
        return reward, game_over, self.score

    def _is_collision(
        # self, player_pos: pygame.Vector2, boundary: set[Tuple[int, int]]
        self,
        pt: Union[Point, None] = None,
    ) -> bool:
        """Return True if the player is colliding with the boundary."""
        if pt is None:
            # handle issue of continuous values not being picked up
            if (int(self.location.x), int(self.location.y)) in self.boundary:
                return True
            return False
        else:
            if pt.x < 0 or pt.x > self.w or pt.y < 0 or pt.y > self.h:
                return True
            if (int(pt.x), int(pt.y)) in self.boundary:
                return True
            return False

    def _interpret_boundary(self, image_path: str) -> set[Tuple[int, int]]:
        """Interpret the boundary of the image as a list of points."""
        image = pygame.image.load(image_path)
        boundary = set()
        for i in range(image.get_width()):
            for j in range(image.get_height()):
                # if the pixel is not white
                if image.get_at((i, j)) != (255, 255, 255, 255):
                    boundary.add((i, j))
        return boundary

    def _update_ui(self):
        self.display.fill("black")
        bg = pygame.image.load(self.bg_path)
        self.display.blit(bg, (0, 0))
        pygame.draw.circle(
            surface=self.display, color="red", center=self.location, radius=2
        )
        pygame.display.flip()

    def _move(self, action):
        """
        based on agent action, update location,
        """

        # action == [r=1, l=0, u=0, d=0] face to the right
        if np.array_equal(action, [1, 0, 0, 0, 0]):
            self.direction = Direction.RIGHT
        elif np.array_equal(action, [0, 1, 0, 0, 0]):
            self.direction = Direction.LEFT
        elif np.array_equal(action, [0, 0, 1, 0, 0]):
            self.direction = Direction.UP
        elif np.array_equal(action, [0, 0, 0, 1, 0]):
            self.direction = Direction.DOWN
        else:
            # no action, stay still
            self.direction = Direction.NOOP

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
        else:
            x, y = x, y

        self.location = Point(x, y)
