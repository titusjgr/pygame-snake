from snake import Snake
import pygame
import random
import numpy as np

SCREEN_SIZE = 320, 320
MAP_SIZE = 32, 32  # This is the map for snake coordination.
SNAKE_SIZE = 10  # This indicates how big a block of the snake looks.
INITIAL_LENGTH = 4
BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)


class Environment():
    def __init__(self):
        # Basic setup
        self.snake = Snake(None)  # Not gonna use background

    def step(self, action):
        '''A step of the environment

        Input one of the four directions: L, R, U, D.
        Then change the state and return the next state, the reward,
        and whether the episode ends.'''

        def positions_to_image(self):
            '''Turn position info into an image

            Turn position information into a image.
            This way we may use it as input to neural network'''

            state = np.zeros(MAP_SIZE)
            for pos in self.snake.positions:
                state[pos] = -1
            state[self.snake.food_position] = 1
            return state

        direction = action
        if direction is not self.snake.invalid_direction:
            self.snake.direction = direction

        # Update snake position
        reward = self.snake.update()

        next_state = positions_to_image(self)

        return next_state, reward, not self.snake.alive

    pygame.quit()
