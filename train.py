# Haven't implemented:
# 1. Loss function
# 2. Training loop

from random import random, sample, randint
from os import listdir

from numpy import argmax, min, array, newaxis
from pygame.locals import *
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from environment import Environment

EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_LOWERING_STEPS = 10000000
DISCOUNT_FACTOR = 0.97
COPY_STEPS = 100

NUM_ACTIONS = 4
NUM_FRAMES = 4  # num of frames used as input state at a time
BATCH_SIZE = 5
EPISODES = 1000
MEMORY_CAPACITY = 4000


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
    '''Save transition info for experience replay

    Storing format: state, action, reward, next_state'''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position_to_modify = 0

    def push(self, state, action, reward, next_state, done):
        '''Store transition information into the queue'''

        if len(self.memory) >= self.capacity:
            self.memory[self.position_to_modify] = (
                state, action, reward, next_state, done
            )
            self.position_to_modify += 1
            self.position_to_modify %= self.capacity
        else:
            self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''Sampe (state, action, reward, next_state, done) pairs from experience

        Returns a batch containing "batch_size" samples,
        where each sample consists of num_frames,
        and the size of the batch.'''

        batch_size = min((batch_size, len(self.memory)))
        transition_batch = sample(self.memory, batch_size)
        return transition_batch


def epsilon_greedy(epsilon_start, epsilon_end, step, state):
    '''Select action using epsilon-greedy strategy

    Epsilon is lowered linearly from start to end'''

    epsilon = epsilon_start + \
        (epsilon_end - epsilon_start) * step / EPSILON_LOWERING_STEPS
    if random() < epsilon:
        action = randint(0, 3)
    else:
        action = argmax(main_q_network.predict(state[newaxis, :]))
    return action


def update_parameters():
    transition_batch = replaymemory.sample(BATCH_SIZE)
    state_batch, _, _, next_state_batch, _ = map(
        array, zip(*transition_batch))

    next_action_batch = argmax(
        main_q_network.predict(next_state_batch), axis=-1)
    q_batch = target_q_network.predict(next_state_batch)

    # Target are initially the same as prediction
    # But some positions in target will be changed to:
    # r+γQ(s^',argmax_(a^' ) Q(s^',a^';θ);θ^- )
    target = main_q_network.predict(state_batch)
    for i, (_, action, reward, _, done) in enumerate(transition_batch):
        target[i][action] = reward
        if not done:
            target[i][action] += DISCOUNT_FACTOR * \
                q_batch[i][next_action_batch[i]]

    main_q_network.train_on_batch(state_batch, target)


replaymemory = ReplayMemory(MEMORY_CAPACITY)
step = 0

target_q_network = build_q_network()
main_q_network = build_q_network()

main_q_network.compile(optimizer='rmsprop', loss='mse')

for episode in range(EPISODES):
    env = Environment()
    state = env.positions_to_image()
    done = False
    game_steps = 0
    while not done:
        # Choose an action and transform to the next state
        action = epsilon_greedy(EPSILON_START, EPSILON_END, step, state)
        next_state, reward, done = env.step(action)
        replaymemory.push(state, action, reward, next_state, done)

        # Update parameters
        update_parameters()

        # Periodically copy weights to target network
        if step % COPY_STEPS == 0:
            target_q_network.set_weights(main_q_network.get_weights())

        step += 1
        game_steps += 1

    print('Score:', env.snake.score)
    checkpoint_filename = 'ckpt/num{}-steps{}-score{}.hdf5'.format(
        len(listdir('ckpt')), game_steps, env.snake.score)
    main_q_network.save_weights(checkpoint_filename)
