from pygame.locals import *
from random import randrange
from environment import Environment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

NUM_ACTIONS = 4
NUM_FRAMES = 4  # num of frames used as input state at a time
BATCH_SIZE = 5


def build_q_network():
    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=(2, 2), activation='elu',
                     input_shape=(32, 32, 1)))
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

    def push(self, state, action, reward, next_shape):
        '''Store transition imformation into the queue'''

        if len(self.memory) >= self.capacity:
            self.memory[self.position_to_modify] = (
                state, action, reward, next_shape
            )
            self.position_to_modify += 1
            self.position_to_modify %= self.capacity
        else:
            self.memory.append([state, action, reward, next_shape])

    def sample(self, num_frames, batch_size):
        '''Sampe data from experience

        Returns a batch containing "batch_size" samples,
        where each sample consists of num_frames.'''
        indices = [randrange(num_frames - 1, len(self.memory))
                   for _ in range(batch_size)]
        transition_batch = []

        for index in indices:
            stacked_state = tuple(
                zip(*self.memory[index - num_frames + 1: index + 1]))[0]
            transition_batch.append([stacked_state] + self.memory[index][1:])

        return transition_batch


env = Environment()
