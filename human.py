from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
import pygame
import random
from snake import Snake
from config import SCREEN_SIZE, MAP_SIZE, SNAKE_SIZE, INITIAL_LENGTH, FPS

INITIAL_LENGTH = 4
BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)


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
    pygame.display.set_caption(str(snake.score))

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

print("Your score is", snake.score)
pygame.quit()
