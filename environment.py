from snake import Snake
from numpy import zeros, stack
import pygame
import random

SCREEN_SIZE = 320, 320
MAP_SIZE = 32, 32  # This is the map for snake coordination.
SNAKE_SIZE = 10  # This indicates how big a block of the snake looks.
INITIAL_LENGTH = 4
BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)
NUM_FRAMES = 4  # num of frames used as input state at a time


class Environment():
    def __init__(self):
        # Basic setup
        self.snake = Snake(None)  # Not gonna use background
        # Stack NUM_FRAMES as a state
        self.frames = [zeros(MAP_SIZE) for _ in range(NUM_FRAMES - 1)]

    def positions_to_image(self):
        '''Turn position info into an image

        Turn position information into a image.
        This way we may use it as input to neural network'''

        frame = zeros(MAP_SIZE)
        for pos in self.snake.positions:
            frame[pos] = -1
        frame[self.snake.food_position] = 1

        self.frames.append(frame)
        state = stack(self.frames, axis=-1)
        self.frames.pop(0)

        return state

    def step(self, action):
        '''A step of the environment

        Input one of the four directions: L, R, U, D.
        Then change the state and return the next state, the reward,
        and whether the episode ends.'''

        direction = action
        if direction is not self.snake.invalid_direction:
            self.snake.direction = direction

        # Update snake position
        reward = self.snake.update()

        next_state = self.positions_to_image()

        #      next state, reward, done
        return next_state, reward, not self.snake.alive

    pygame.quit()
