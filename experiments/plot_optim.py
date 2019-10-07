import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import matplotlib.pyplot as plt
import json
import numpy as np



## Read random policy result
with open("output/random-policy-{}.json".format(cfg.vals['prj_name']), 'r') as f:
    random = json.load(f)

random['rewards'] = np.array(random['rewards'], dtype=np.float32)
n_random = len(random['rewards'])

## Read tabu policy result
with open("output/tabu-{}.json".format(cfg.vals['prj_name']), 'r') as f:
    tabu = json.load(f)
n_tabu = len(tabu['rewards'])
tabu['rewards'] = np.array(tabu['rewards'], dtype=np.float32)

## Read DQN policy result
with open("output/rl-test-{}.json".format(cfg.vals['prj_name']), 'r') as f:
    dqn = json.load(f)
n_dqn = len(dqn['rewards'])
dqn['rewards'] = np.array(dqn['rewards'], dtype=np.float32)


assert n_random == n_tabu == n_dqn

x = np.arange(n_dqn)
plt.plot(x, random['rewards'], label='random')
plt.plot(x, tabu['rewards'], label='tabu')
plt.plot(x, dqn['rewards'], label='dqn')
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward (test)")
plt.legend(loc='best')
plt.savefig("figs/optimization-all-{}.png".format(cfg.vals['prj_name']))
