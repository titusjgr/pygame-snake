from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
import pygame
import random
from snake import Snake


SCREEN_SIZE = 320, 320
MAP_SIZE = 32, 32  # This is the map for snake coordination.
SNAKE_SIZE = 10  # This indicates how big a block of the snake looks.
INITIAL_LENGTH = 4
BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)
FPS = 15


# Basic setup
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()

background = pygame.Surface(screen.get_size()).convert()

snake = Snake(background)

# Main loop
while snake.alive:
    # Reset screen
    background.fill(BLACK)

    for event in pygame.event.get():
        if event.type == QUIT or \
                (event.type == KEYDOWN and event.key == K_ESCAPE):
            snake.alive = False

        elif event.type == KEYDOWN and event.key in snake.key_to_direction:
            direction = snake.key_to_direction[event.key]
            if direction is not snake.invalid_direction:
                snake.direction = direction

    # Display score
    pygame.display.set_caption('Snake | Score:{}'.format(snake.score))

    # Update snake position
    snake.update()

    # Draw things
    snake.draw_body()
    snake.draw_food()

    # Update screen
    screen.blit(background, (0, 0))
    pygame.display.flip()

    # FPS
    clock.tick(FPS)


pygame.quit()
