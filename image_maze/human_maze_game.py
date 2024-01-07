# Example file showing a circle moving on screen
from typing import Tuple

import pygame

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt: float = 0.0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)


def interpret_boundary(image_path: str) -> set[Tuple[int, int]]:
    """Interpret the boundary of the image as a list of points."""
    image = pygame.image.load(image_path)
    boundary = set()
    for i in range(image.get_width()):
        for j in range(image.get_height()):
            # if the pixel is not white
            if image.get_at((i, j)) != (255, 255, 255, 255):
                boundary.add((i, j))
    return boundary


def _is_collison(player_pos: pygame.Vector2, boundary: set[Tuple[int, int]]) -> bool:
    """Return True if the player is colliding with the boundary."""
    # handle issue of continuous values not being picked up
    if (int(player_pos.x), int(player_pos.y)) in boundary:
        return True
    return False


def update_ui(bg_path: str):
    screen.fill("black")
    bg = pygame.image.load(bg_path)
    screen.blit(bg, (0, 0))


# read image for boundary
bg_path = "assets/env_1280_720.png"
boundary = interpret_boundary(bg_path)
while running:
    # poll for events
    # pygame.QUIT event means the user Xw to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    update_ui(bg_path=bg_path)

    pygame.draw.circle(surface=screen, color="red", center=player_pos, radius=2)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player_pos.y -= 300 * dt
    if keys[pygame.K_s]:
        player_pos.y += 300 * dt
    if keys[pygame.K_a]:
        player_pos.x -= 300 * dt
    if keys[pygame.K_d]:
        player_pos.x += 300 * dt

    # check for collision
    if _is_collison(player_pos, boundary):
        print("collision")
        running = False
    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
