import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
from policies.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from policies.deepq.dqn import DQN
from utils import serialize_floats
from mopo.mopo import Mopo
from experiments.exp_utils import get_simple_simulator, evaluate_policy

def main(args):

    prior = Prior(config=cfg.vals)
    env_model = AllocationEnv(config=cfg.vals, prior=prior, load_model=True, full_posterior=True)
    policy = DQN(MlpPolicy, env_model, batch_size=args.batch_size)

    mopo_dqn = Mopo(policy=policy,
                    env_model=env_model,
                    rollout_batch_size=10,
                    epochs=args.epochs,
                    rollout=args.rollouts,
                    n_actions = env_model.n_actions,
                    lmbda=args.lmbda,
                    buffer_path="../data/store-2-buffer.p"
                    #buffer_path=None

        )


    mopo_dqn.learn()


    simulator = get_simple_simulator(cfg.vals)
    reward, sigma = evaluate_policy(mopo_dqn.policy, simulator, args.eval_eps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--rollouts', type=int, default=30)
    parser.add_argument('--lmbda', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-eps', type=int, default=10)

    args = parser.parse_args()

    main(args)