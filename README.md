# Snake game

This is just a simple snake game.

## Install Dependencies

```
pip install numpy
pip install matplotlib
pip install pygame
pip install tensorflow==2.0.0a0
```

## Structure

Snake class is in `snake.py`.

Human-playable game is in `human.py`.

The environment for the network to interact with is `environment.py`.

The neural network training script is `train_dqn.py`.

Finally, to see DQN playing game, execute `play_dqn.py`.

## Usage

If you want to play the game on your own:
`python human.py`

Train the network:
```
python train_dqn.py [--epsilon start_eps end_eps eps_decay_steps]
      [--episode num_of_episodes]
      [-lw checkpoint_filepath]
      [-lr learning_rate]
```

Let you network play the game:
`python play_dqn.py checkpoint_filepath`
