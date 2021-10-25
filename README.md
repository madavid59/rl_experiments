Playground to get some hands-on experience with RL algorithms.
# Setup
## NVIDIA drivers for tf
https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d

## Conda
```
conda create --name rl_playground python=3.8
conda activate rl_playground
python3 -m pip install -r requirements.txt

```

# Implemented algorithms
## qlearning.py
- Monte Carlo learning
- SARSA(0)
- Q(0) learning
## dqn.py
- DQN (Deep Q-Net)

# Next steps:
- Policy gradients: Inverted pendulum
- (optional): DDQN, prioritized experience replay
- Guided policy search
