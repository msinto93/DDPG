# Deep Deterministic Policy Gradients (DDPG)
A Tensorflow implementation of a [**Deep Deterministic Policy Gradient (DDPG) network**](https://arxiv.org/pdf/1509.02971.pdf) for continuous control.

![](https://image.ibb.co/i5uzQq/actor-critic.png)

Trained on [OpenAI Gym environments](https://gym.openai.com/envs).

This implementation has been successfully trained and tested on the [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/) and [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) environments. This code can however be run 'out of the box' on any environment with a low-dimensional state space and continuous action space.

**This currently holds the high score for the Pendulum-v0 environment on the [OpenAI leaderboard](https://github.com/openai/gym/wiki/Leaderboard#pendulum-v0)**

## Requirements
Note: Versions stated are the versions I used, however this will still likely work with other versions.

- Ubuntu 16.04 (Most (non-Atari) envs will also work on Windows)
- python 3.5
- [OpenAI Gym](https://github.com/openai/gym) 0.10.8 (See link for installation instructions + dependencies)
- [tensorflow-gpu](https://www.tensorflow.org/) 1.5.0
- [numpy](http://www.numpy.org/) 1.15.2
- [scipy](http://www.scipy.org/install.html) 1.1.0
- [opencv-python](http://opencv.org/) 3.4.0
- [imageio](http://imageio.github.io/) 2.4.1 (requires [pillow](https://python-pillow.org/))
- [inotify-tools](https://github.com/rvoicilas/inotify-tools/wiki) 3.14

## Usage
The default environment is 'Pendulum-v0'. To use a different environment simply pass the environment in via the `--env` argument when running the following files.
```
  $ python train.py
```
This will train the DDPG on the specified environment and periodically save checkpoints to the `/ckpts` folder.

```
  $ ./run_every_new_ckpt.sh
```
This shell script should be run alongside the training script, allowing to periodically test the latest network as it trains. This script will monitor the `/ckpts` folder and run the `test.py` script on the latest checkpoint every time a new checkpoint is saved.

```
  $ python play.py
```
Once we have a trained network, we can visualise its performance in the environment by running `play.py`. This will play the environment on screen using the trained network and save a GIF (optional).

**Note:** To reproduce the best 100-episode performance of **-123.79 +/- 6.90** that achieved the top score on the ['Pendulum-v0' OpenAI leaderboard](https://github.com/openai/gym/wiki/Leaderboard#pendulum-v0), run:
```
  $ python test.py --ckpt_file 'Pendulum-v0.ckpt-26800'
```

## Results
Result of training the DDPG on the 'Pendulum-v0' environment:

![](/video/Pendulum-v0.gif)

Result of training the DDPG on the 'BipedalWalker-v2' environment:

*To-Do*
![](/video/BipedalWalker-v2.gif)

| **Environment**      | **Best 100-episode performance** | **Ckpt file** |
|----------------------|----------------------------------|---------------|
| Pendulum-v0          |  -123.79 +- 6.90                 | ckpt-26800    |

## To-do
- Train/test on further environments, including [Mujoco](http://www.mujoco.org/)

## References
- [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book.html)

## License
MIT License
