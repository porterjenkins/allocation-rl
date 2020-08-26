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


def main(args):

    with open("../data/store-2-buffer.p", 'rb') as f:
        buffer_env = pickle.load(f)

    simulator = get_simple_simulator(cfg.vals)
    model = DQN(MlpPolicy, simulator, verbose=2)
    model.learn_off_policy(total_timesteps=args.epochs, buffer=buffer_env)



    reward, sigma = evaluate_policy(model, simulator, args.eval_eps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_eps', type=int, default=10)
    args = parser.parse_args()

    main(args)