import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import argparse

from policies.deepq.policies import MlpPolicy
from policies.deepq.dqn import DQN
from utils import serialize_floats
from experiments.exp_utils import get_simple_simulator, evaluate_policy
from experiments.logger import Logger

from utils import get_store_id, strip_reward_array

def main(args):
    store_id = get_store_id(cfg.vals["train_data"])

    policy = DQN.load(f"./models/{store_id}-off-policy-dqn.p")
    simulator = get_simple_simulator(cfg.vals)
    reward, sigma = evaluate_policy(policy, simulator, args.eval_eps)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_eps', type=int, default=10)
    args = parser.parse_args()

    main(args)