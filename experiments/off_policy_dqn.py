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

from utils import get_store_id

def main(args):
    store_id = get_store_id(cfg.vals["train_data"])
    hyp = {
        "learning timesteps": args.epochs,
        "episode length": cfg.vals["episode_len"],
        "n simulations": args.eval_eps,
        "store": store_id
    }

    logger = Logger(hyp, "./results/", "off_policy_dqn")

    with open(f"../data/{store_id}-buffer-r.p", 'rb') as f:
        buffer_env = pickle.load(f)

    simulator = get_simple_simulator(cfg.vals)
    model = DQN(MlpPolicy, simulator, verbose=2)
    model.learn_off_policy(total_timesteps=args.epochs, buffer=buffer_env)



    reward, sigma = evaluate_policy(model, simulator, args.eval_eps)

    logger.set_result(
        {
            "reward": reward,
            "std": sigma
        }
    )
    logger.write()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_eps', type=int, default=10)
    args = parser.parse_args()

    main(args)