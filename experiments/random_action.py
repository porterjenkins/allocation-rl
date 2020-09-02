import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_store_id


from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
from utils import serialize_floats
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from envs.state import State


store_id = get_store_id(cfg.vals["train_data"])
TIME_STEPS = 5000
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
results = {'rewards': [0.0]}
buffer = ReplayBuffer(size=50000)


obs = env.reset()
for i in range(TIME_STEPS):
    action = env.action_space.sample()
    proposed_action = AllocationEnv.check_action(obs['board_config'], action)
    new_obs, rew, dones, info = env.step(proposed_action)

    if rew == -1:
        action = 0

    print("Timestep: {}".format(i))
    print("action: {} - reward: {}".format(action, rew))
    print(obs['day_vec'])
    print(obs['board_config'])

    results['rewards'].append(rew + results['rewards'][-1])

    # add (s, a, r, s') to buffer
    buffer.add(obs_t=State.get_vec_observation(obs),
               action=action,
               reward=rew,
               obs_tp1=State.get_vec_observation(new_obs),
               done=float(dones))

    obs = new_obs

print(results['rewards'])


x = np.arange(TIME_STEPS+1)
plt.plot(x, results['rewards'])
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward")
plt.savefig("figs/random-policy-{}.png".format(cfg.vals['prj_name']))


for k, v in results.items():
    results[k] = serialize_floats(v)


#with open("output/random-policy-{}.json".format(cfg.vals['prj_name']), 'w') as f:
#    json.dump(results, f)


with open(f"../data/{store_id}-buffer-r.p", 'wb') as f:
    pickle.dump(buffer, f)
