import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import config.config as cfg

class EnsembleModel(gym.Env):

    """Environment model for training Reinforcement Learning agent"""
    metadata = {'render.modes': ['allocation'],
                'max_cnt_reward_not_reduce_round': cfg.vals['episode_len']}

    def __init__(self, config):
        pass



    def predict(self, state, mask):


        reward = -1

        return reward, None