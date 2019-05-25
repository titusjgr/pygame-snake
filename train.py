# Haven't implemented:
# 1. Loss function
# 2. Training loop

from pygame.locals import *
from random import sample, random, choice
from numpy import argmax, min
from environment import Environment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_LOWERING_STEPS = 10000000

NUM_ACTIONS = 4
NUM_FRAMES = 4  # num of frames used as input state at a time
BATCH_SIZE = 5
EPISODES = 10000
MEMORY_CAPACITY = 4000
ACTIONS = ('L', 'R', 'U', 'D')  # The set of available actions


def build_q_network():
    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=(2, 2), activation='elu',
                     input_shape=(32, 32, NUM_FRAMES)))
    model.add(Conv2D(32, (2, 2), activation='elu'))
    model.add(Conv2D(32, (2, 2), activation='elu'))
    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dense(4))

    return model


class ReplayMemory():
    '''Save transition info for experience replay'''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position_to_modify = 0

    def push(self, state, action, reward, next_state):
        '''Store transition imformation into the queue'''

        if len(self.memory) >= self.capacity:
            self.memory[self.position_to_modify] = (
                state, action, reward, next_state
            )
            self.position_to_modify += 1
            self.position_to_modify %= self.capacity
        else:
            self.memory.append([state, action, reward, next_state])

    def sample(self, batch_size):
        '''Sampe data from experience

        Returns a batch containing "batch_size" samples,
        where each sample consists of num_frames,
        and the size of the batch.'''

        batch_size = min((batch_size, len(self.memory)))
        transition_batch = sample(self.memory, batch_size)
        return transition_batch, batch_size


def epsilon_greedy(epsilon_start, epsilon_end, step, state):
    '''Select action using epsilon-greedy strategy

    Epsilon is lowered linearly from start to end'''

    epsilon = epsilon_start + \
        (epsilon_end - epsilon_start) * step / EPSILON_LOWERING_STEPS
    if random() < epsilon:
        action = choice(ACTIONS)
    else:
        action = argmax(main_q_network.predict(state))
    return action


def loss():
    pass


# Initialize something
env = Environment()
replaymemory = ReplayMemory(MEMORY_CAPACITY)
state = env.positions_to_image()
step = 0

target_q_network = build_q_network()
main_q_network = build_q_network()

for episode in range(EPISODES):
    done = False
    while not done:
        action = epsilon_greedy(EPSILON_START, EPSILON_END, step, state)
        next_state, reward, done = env.step(action)
        replaymemory.push(state, action, reward, next_state)

    step += 1
