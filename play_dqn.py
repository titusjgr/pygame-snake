import random

import pygame
from numpy import argmax
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from config import (FPS, INITIAL_LENGTH, MAP_SIZE, NUM_FRAMES, SCREEN_SIZE,
                    SNAKE_SIZE)
from environment import Environment

BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)


def build_q_network():
    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=(2, 2), activation='elu',
                     input_shape=(34, 34, NUM_FRAMES)))
    model.add(Conv2D(32, (2, 2), activation='elu'))
    model.add(Conv2D(32, (2, 2), activation='elu'))
    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(4))

    return model


parser = ArgumentParser(
    usage='python play_dqn.py checkpoint_filename',
    description='Train the DQN'
)
parser.add_argument(
    help='load weights from hdf5 file',
    dest='ckptfilename',
)
args = parser.pars

q_network = build_q_network()
q_network.load_weights('ckpt/')

# Basic setup
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()

background = pygame.Surface(screen.get_size()).convert()

env = Environment(background)

done = False
# Main loop
while not done:
    # Reset screen
    background.fill(BLACK)

    for event in pygame.event.get():
        if event.type == QUIT or \
                (event.type == KEYDOWN and event.key == K_ESCAPE):
            env.snake.alive = False

    # Display score
    pygame.display.set_caption('Snake | Score:{}'.format(env.snake.score))

    # Update snake position
    action = argmax()
    env.step()

    # Draw things
    env.snake.draw_body()
    env.snake.draw_food()

    # Update screen
    screen.blit(background, (0, 0))
    pygame.display.flip()

    # FPS
    clock.tick(FPS)


pygame.quit()