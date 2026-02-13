# ApexRL

A Reinforcement Learning library in Python featuring PPO implementation.

## Features

- PPO algorithm with GAE and clipping
- Continuous and discrete action spaces support
- Vectorized environments for parallel training
- MLP and CNN network architectures
- TensorBoard logging
- Model checkpointing

## Installation

```bash
git clone https://github.com/yourusername/apexrl.git
cd apexrl
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from apexrl.algorithms.ppo import PPO
from apexrl.algorithms.ppo.config import PPOConfig

env = gym.make("CartPole-v1")
config = PPOConfig(total_timesteps=100000, learning_rate=3e-4, num_envs=4)
model = PPO(env, config)
model.learn()
```

## Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Syntax check
python -m py_compile src/apexrl/**/*.py
```

## Code Quality

```bash
pip install pre-commit
pre-commit install
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file.
