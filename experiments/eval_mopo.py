import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pickle

from utils import get_store_id
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
from policies.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from policies.deepq.dqn import DQN
from utils import serialize_floats
from mopo.mopo import Mopo

from experiments.exp_utils import get_simple_simulator, evaluate_policy
from experiments.logger import Logger

def main(args):
    store_id = get_store_id(cfg.vals["train_data"])

    policy = DQN.load(f"./models/{store_id}-mopo-policy.p")
    simulator = get_simple_simulator(cfg.vals)
    reward, sigma = evaluate_policy(policy, simulator, args.eval_eps)

    return reward, sigma


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-eps', type=int, default=10)
    args = parser.parse_args()

    main(args)

