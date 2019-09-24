import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt
import numpy as np


prior = Prior(config=cfg.vals)

env = AllocationEnv(config=cfg.vals, prior=prior)
env.reset()
# a = np.zeros((4, 4))
# a[3, 3] = 1.0
# state = env.reset()
# ob, reward, epsode_over, info = env.step(a)
# print(ob)

history = []
T = 10

for i in range(T):
    print("t = {}".format(i+1))
    a = env.action_space.sample()
    ob, reward, epsode_over, info = env.step(a)
    history.append(reward)
    print(ob)
    print("r: {}".format(reward))


plt.plot(np.arange(T), history)
plt.show()