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

plt.figure(figsize=(4,4))
bars = [np.max(random['rewards'])/1000, np.max(tabu['rewards'])/1000, np.max(dqn['rewards'])/1000]
barchart = plt.bar(np.arange(3),bars)
barchart[0].set_color('slategray')
barchart[1].set_color('goldenrod')
plt.ylabel("Cumulative Test Reward (thousdands of $)")
plt.xticks(np.arange(3), ['random', 'tabu', 'dqn'])
plt.savefig("figs/optimization-all-{}.png".format(cfg.vals['prj_name']))
