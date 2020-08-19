import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
from policies.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from policies.deepq.dqn import DQN
from utils import serialize_floats
from mopo.mopo import Mopo


TEST_T = cfg.vals["episode_len"]
TIME_STEPS = 20000
LEARNING_START = 1500
0
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True, full_posterior=True)
policy = DQN(MlpPolicy, env, batch_size=32)

mopo_dqn = Mopo(policy=policy,
                env_model=env,
                epochs=5,
                rollout=10,
                n_actions=env.n_actions,
                lmbda=1e-3,
                buffer_path="../data/random-buffer.p"

            )

mopo_dqn.learn()



obs = env.reset()
results = {'rewards': [0.0]}
for i in range(TEST_T):
    feasible_actions = AllocationEnv.get_feasible_actions(obs["board_config"])
    action_mask = AllocationEnv.get_action_mask(feasible_actions, env.n_actions)
    action, _states = mopo_dqn.policy.predict(obs, mask=action_mask)
    action = AllocationEnv.check_action(obs['board_config'], action)
    obs, r, dones, info = env.step(action)

    results['rewards'].append(r[0] + results['rewards'][-1])



x = np.arange(TEST_T+1)
plt.plot(x, results['rewards'])
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward (test)")
plt.savefig("figs/mopo-test-{}.png".format(cfg.vals['prj_name']))


for k, v in results.items():
    results[k] = serialize_floats(v)


with open("output/mopo-test.json", 'w') as f:
    json.dump(results, f)
