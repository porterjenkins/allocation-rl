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
import json
from utils import serialize_floats

TEST_T = 30

prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run



iteration_cuts = np.arange(100, 2010, 100)

results = {'mean': [],
           "std": []
           }

for ts in iteration_cuts:
    print("----------ITERATIONS = {}----------".format(ts))
    model = DQN(MlpPolicy, env, verbose=2, learning_starts=100, exploration_fraction=.75)
    model.learn(total_timesteps=ts)
    obs = env.reset()

    test_r = []
    for i in range(TEST_T):
        action, _states = model.predict(obs)
        action = AllocationEnv.check_action(obs['board_config'], action)
        obs, rewards, dones, info = env.step([action])
        test_r.append(rewards)

    test_r_mean = np.mean(test_r)
    test_r_std = np.std(test_r)
    results["mean"].append(test_r_mean)
    results["std"].append(test_r_std)


plt.plot(iteration_cuts, results["mean"])
plt.errorbar(iteration_cuts, results["mean"], yerr=results["std"], fmt='.k')
plt.xlabel("Iteration count")
plt.ylabel("Average test reward")
plt.savefig("figs/rl-learning-curve.pdf")


for k, v in results.items():
    results[k] = serialize_floats(v)


with open("output/rl-learning-curve-{}.json".format(cfg.vals['prj_name']), 'w') as f:
    json.dump(results, f)