from argparse import ArgumentParser
from os import environ

import pygame
from numpy import argmax, newaxis
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT, K_SPACE
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from config import (FPS, INITIAL_LENGTH, MAP_SIZE, NUM_FRAMES, DISPLAY_SIZE,
                    INPUT_SHAPE)
from environment import Environment


BLACK = (0,  0,  0)
WHITE = (255, 255, 255)
GREEN = (0, 255,  0)


def build_q_network():
    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=(2, 2), activation='elu',
                     input_shape=INPUT_SHAPE))
    model.add(Conv2D(32, (2, 2), activation='elu'))
    model.add(Conv2D(32, (2, 2), activation='elu'))
    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(4))

    return model


parser = ArgumentParser(
    usage='python play_dqn.py checkpoint_filepath',
    description='Train the DQN with specified weights checkpoint\n \
        checkpoint_filepath should be the path relative to current file'
)
parser.add_argument(
    help='load weights from hdf5 file',
    dest='ckptfilepath',
)
args = parser.parse_args()

q_network = build_q_network()
q_network.load_weights(args.ckptfilepath)

# Basic setup
pygame.init()
screen = pygame.display.set_mode(DISPLAY_SIZE)
pygame.display.set_caption('Snake')
clock = pygame.time.Clock()

background = pygame.Surface(screen.get_size()).convert()

env = Environment(background)
state = env.positions_to_image()
done = False
# Main loop
while not done:
    # Reset screen
    background.fill(BLACK)

    for event in pygame.event.get():
        if event.type == QUIT or \
                (event.type == KEYDOWN and event.key == K_ESCAPE):
            env.snake.alive = False

        elif event.type == KEYDOWN and event.key == K_SPACE:
            pause = True
            while pause:
                for event in pygame.event.get():
                    if event.type == QUIT or \
                            (event.type == KEYDOWN and event.key == K_ESCAPE):
                        env.snake.alive = False
                        pause = False
                    elif event.type == KEYDOWN and event.key == K_SPACE:
                        pause = False
                clock.tick(FPS)

    # Display score
    pygame.display.set_caption(str(env.snake.score))

    # Update snake position
    action = argmax(q_network.predict(state[newaxis, :]))
    state, reward, done = env.step(action)

    # Draw things
    env.snake.draw_body()
    env.snake.draw_food()

    # Update screen
    screen.blit(background, (0, 0))
    pygame.display.flip()

    # FPS
    clock.tick(FPS)

print("Your score is", env.snake.score)
pygame.quit()
