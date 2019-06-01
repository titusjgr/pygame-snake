from snake import Snake
import random

from numpy import stack, zeros

from config import NUM_FRAMES, SCREEN_SIZE, SNAKE_SIZE

BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)


class Environment():
    def __init__(self, background=None):
        # Basic setup
        self.snake = Snake(background)
        # Stack NUM_FRAMES as a state
        self.frames = [zeros(SCREEN_SIZE) for _ in range(NUM_FRAMES - 1)]

    def positions_to_image(self):
        '''Turn position info into an image

        Turn position information into a image.
        This way we may use it as input to neural network'''

        frame = zeros(SCREEN_SIZE)

        if self.snake.alive is False:
            self.snake.positions.pop(0)

        for x, y in self.snake.positions:
            frame[x * SNAKE_SIZE: (x + 1) * SNAKE_SIZE,
                  y * SNAKE_SIZE: (y + 1) * SNAKE_SIZE] = -1

        x, y = self.snake.food_position
        frame[x * SNAKE_SIZE: (x + 1) * SNAKE_SIZE,
              y * SNAKE_SIZE: (y + 1) * SNAKE_SIZE] = 1

        self.frames.append(frame)
        state = stack(self.frames, axis=-1)
        self.frames.pop(0)

        return state

    def step(self, action):
        '''A step of the environment

        Input one of the four directions: 1, 2, 3, 4 (L, R, U, D).
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
