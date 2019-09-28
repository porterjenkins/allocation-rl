import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt
import numpy as np
import theano
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

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

model = DDPG(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=400000)
#model.save("tmp-agent")

ob = env.reset()
for i in range(T):
    print("t = {}".format(i+1))
    #a = env.action_space.sample()-1
    action, _states = model.predict(ob)
    print("**action**")
    # print(a)
    # TODO: check if feasible action here
    ob, reward, epsode_over, info = env.step(action)
    history.append(reward)
    print(ob)
    print("r: {}".format(reward))


plt.plot(np.arange(T), history)
plt.show()