import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt


TIME_STEPS = 30
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
cumul_reward = [0.0]

obs = env.reset()
for i in range(TIME_STEPS):
    action = env.action_space.sample()
    action = AllocationEnv.check_action(obs['board_config'], action)
    obs, rew, dones, info = env.step(action)
    print("Timestep: {}".format(i))
    print("action: {} - reward: {}".format(action, rew))
    print(obs['day_vec'])
    print(obs['board_config'])

    cumul_reward.append(cumul_reward[-1] + rew)

print(cumul_reward)


x = np.arange(TIME_STEPS+1)
plt.plot(x, cumul_reward)
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward")
plt.savefig("figs/{}-random-policy.png".format(cfg.vals['prj_name']))