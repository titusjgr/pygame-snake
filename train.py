from pygame.locals import *
from environment import Environment
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

NUM_ACTIONS = 4
NUM_FRAMES = 4 # num of frames used as input state at a time



def build_q_network():
    model = Sequential()
    model.add(Conv2D(16))


env = Environment()
