import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt
import numpy as np
import gym
from policies.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from policies.deepq.dqn import DQN



TIME_STEPS = 250

prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
n_actions = env.n_actions
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = DQN(MlpPolicy, env, verbose=2, learning_starts=100, exploration_fraction=.75)
model.learn(total_timesteps=TIME_STEPS)
print(model.cumul_reward)



x = np.arange(TIME_STEPS+1)
plt.plot(x, model.cumul_reward)
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward")
plt.savefig("figs/{}-learning-curve.png".format(cfg.vals['prj_name']))