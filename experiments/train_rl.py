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
from utils import serialize_floats
import json
import pickle

from utils import get_store_id
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from envs.state import State

TEST_T = cfg.vals["episode_len"]
TIME_STEPS = 2000
LEARNING_START = 500

store_id = get_store_id(cfg.vals["train_data"])
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
n_actions = env.n_actions
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = DQN(MlpPolicy, env, verbose=2, learning_starts=LEARNING_START, gamma=.2,
            exploration_fraction=0.35, exploration_final_eps=0.05)
model.learn(total_timesteps=TIME_STEPS, learning_curve=False, test_t=TEST_T)


obs = env.reset()
results = {'rewards': [0.0]}
buffer = ReplayBuffer(size=50000)


for j in range(100):
    for i in range(TEST_T):
        feasible_actions = AllocationEnv.get_feasible_actions(obs["board_config"])
        action_mask = AllocationEnv.get_action_mask(feasible_actions, n_actions)
        action, _states = model.predict(obs, mask=action_mask)

        action = AllocationEnv.check_action(obs['board_config'], action)
        new_obs, r, dones, info = env.step([action])

        results['rewards'].append(r[0] + results['rewards'][-1])

        # add (s, a, r, s') to buffer
        buffer.add(obs_t=State.get_vec_observation(obs),
                   action=action,
                   reward=r,
                   obs_tp1=State.get_vec_observation(new_obs),
                   done=float(dones))

        obs = new_obs



x = np.arange(TEST_T+1)
plt.plot(x, results['rewards'])
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward (test)")
plt.savefig("figs/rl-test-{}.png".format(cfg.vals['prj_name']))


for k, v in results.items():
    results[k] = serialize_floats(v)


with open("output/rl-test-{}.json".format(cfg.vals['prj_name']), 'w') as f:
    json.dump(results, f)



with open(f"../data/{store_id}-buffer-r.p", 'wb') as f:
    pickle.dump(buffer, f)
