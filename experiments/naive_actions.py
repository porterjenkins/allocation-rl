import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt
from utils import serialize_floats
import json

TIME_STEPS = cfg.vals["episode_len"]
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
results = {'rewards': [0.0]}

obs = env.reset()
for i in range(TIME_STEPS):
    action = 0
    action = AllocationEnv.check_action(obs['board_config'], action)
    obs, rew, dones, info = env.step(action)
    print("Timestep: {}".format(i))
    print("action: {} - reward: {}".format(action, rew))
    print(obs['day_vec'])
    print(obs['board_config'])

    results['rewards'].append(rew + results['rewards'][-1])

print(results['rewards'])


x = np.arange(TIME_STEPS+1)
plt.plot(x, results['rewards'])
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward")
plt.savefig("figs/naive-policy-{}.png".format(cfg.vals['prj_name']))


for k, v in results.items():
    results[k] = serialize_floats(v)


with open("output/naive-policy-{}.json".format(cfg.vals['prj_name']), 'w') as f:
    json.dump(results, f)
