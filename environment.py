from pygame.locals import *
from snake import Snake
import pygame
import random
import numpy as np

SCREEN_SIZE = 600, 480
MAP_SIZE = 60, 48 # This is the map for snake coordination.
SNAKE_SIZE = 10 # This indicates how big a block of the snake looks.
INITIAL_LENGTH = 4
BLACK = (  0  ,  0  ,  0  )
WHITE = ( 255 , 255 , 255 )
GREEN = (  0  , 255 ,  0  )


class Environment():
    def __init__(self):
        # Basic setup
        self.snake = Snake(None) # Not gonna use background

    def positions_to_image(self):
        state = np.zeros(MAP_SIZE)
        for pos in self.snake.positions:
            state[pos] = -1
        state[self.snake.food_position] = 1
        return state

    def step(self, action):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                going = False

            elif event.type == KEYDOWN and event.key in snake.key_to_direction:
                direction = snake.key_to_direction[event.key]
                if direction is not self.snake.invalid_direction:
                    self.snake.direction = direction
                    
        # Update snake position
        reward = self.snake.update()

        next_state = self.positions_to_image()
        
        return next_state, reward, not self.snake.alive 

    pygame.quit()
