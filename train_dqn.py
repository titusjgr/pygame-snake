import time
from argparse import ArgumentParser
from os import listdir
from random import randint, random, sample

from matplotlib.pyplot import plot, savefig, show, subplot, title
from numpy import argmax, array, mean, min, newaxis
from pygame.locals import *
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from config import MAP_SIZE, NUM_FRAMES, DISCOUNT_FACTOR, INPUT_SHAPE
from environment import Environment

start_time = time.time()

parser = ArgumentParser(
    usage='python train_dqn.py [--epsilon start_eps end_eps eps_decay_steps] \
          [--episode num_of_episodes] \
          [-lw checkpoint_filepath] \
          [-lr learning_rate]',
    description='Train the DQN'
)
parser.add_argument(
    '--epsilon',
    nargs=3,
    metavar=('start_eps', 'end_eps', 'eps_decay_steps'),
    help='''specify intended epsilon value for epsilon-greedy
    the epsilon value will go from start_eps to end_eps
    in "eps_decay_steps" steps''',
    dest='epsilon',
    default=[1, 0.1, 1000000],
    type=float,
)
parser.add_argument(
    '--episodes',
    help='specify the number of episodes to run',
    dest='episodes',
    default=10000,
    type=int,
)
parser.add_argument(
    '-lw',
    help='load weights from hdf5 file',
    dest='ckptfilepath',
    default=None,
)
parser.add_argument(
    '-lr',
    help='learning rate',
    dest='learning_rate',
    default=0.001,
    type=float
)
args = parser.parse_args()

EPSILON_START, EPSILON_END, EPSILON_LOWERING_STEPS = args.epsilon
EPISODES = args.episodes
COPY_STEPS = 100
LEARNING_RATE = args.learning_rate
NUM_ACTIONS = 4
BATCH_SIZE = 64
MEMORY_CAPACITY = 4000
SAVE_EPISODES = 10000


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

    # return the loss
    return main_q_network.train_on_batch(state_batch, target)


replaymemory = ReplayMemory(MEMORY_CAPACITY)
step = 0


target_q_network = build_q_network()
main_q_network = build_q_network()

load_ckpt_filepath = args.ckptfilepath
if load_ckpt_filepath is not None:
    main_q_network.load_weights(load_ckpt_filepath)
target_q_network.set_weights(main_q_network.get_weights())

main_q_network.compile(
    optimizer=RMSprop(lr=LEARNING_RATE),
    loss='mse'
)

loss_history = []
q_history = []


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
        loss_history.append(update_parameters())

        # Periodically copy weights to target network
        if step % COPY_STEPS == 0:
            target_q_network.set_weights(main_q_network.get_weights())
            q_history.append(
                mean(target_q_network.predict(state[newaxis, :])))

        step += 1
        game_steps += 1

    if episode % SAVE_EPISODES == 0:
        print('Score:', env.snake.score, 'Episode:', episode)
        checkpoint_filepath = 'ckpt/num{}-steps{}-score{}.h5'.format(
            len(listdir('ckpt')), game_steps, env.snake.score)
        main_q_network.save_weights(checkpoint_filepath)

print('Time used:', time.time() - start_time)

subplot(211)
plot(loss_history)
title('loss')
subplot(212)
plot(q_history)
title('q value')
savefig('loss-plot/' + str(int(time.time())) +
        '-lr' + str(LEARNING_RATE) + '.png')
