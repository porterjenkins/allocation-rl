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
    hyp = {
        "epochs": args.epochs,
        "rollout batch size": args.rollout_batch_size,
        "parameter updates": args.epochs * args.rollout_batch_size,
        "rollouts": args.rollouts,
        "lambda": args.lmbda,
        "batch size": args.batch_size,
        "posterior samples": args.posterior_samples,
        "episode length": cfg.vals["episode_len"],
        "n simulations": args.eval_eps,
        "store": store_id,
        "eps": args.eps
    }

    logger = Logger(hyp, "./results/", "pc_mopo")
    logger.write()

    prior = Prior(config=cfg.vals)
    env_model = AllocationEnv(config=cfg.vals,
                              prior=prior,
                              load_model=True,
                              full_posterior=True,
                              posterior_samples=args.posterior_samples,
                              verbose=False)

    policy = DQN(MlpPolicy, env_model, batch_size=args.batch_size)

    mopo_dqn = Mopo(policy=policy,
                    env_model=env_model,
                    rollout_batch_size=args.rollout_batch_size,
                    epochs=args.epochs,
                    rollout=args.rollouts,
                    n_actions=env_model.n_actions,
                    lmbda=args.lmbda,
                    buffer_path=f"../data/{store_id}-buffer-d-trn.p",
                    # buffer_path=None
                    eps=args.eps

                    )

    mopo_dqn.learn()

    if os.path.exists(f"./models/{store_id}-{args.file_name}"):
        os.remove(f"./models/{store_id}-{args.file_name}")
    mopo_dqn.policy.save(f"./models/{store_id}-{args.file_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--rollouts', type=int, default=30)
    parser.add_argument('--rollout-batch-size', type=int, default=10)
    parser.add_argument('--lmbda', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-eps', type=int, default=10)
    parser.add_argument('--posterior-samples', type=int, default=25)
    parser.add_argument('--file-name', type=str, default="mopo-policy.p")
    parser.add_argument('--eps', type=float, default=0.0)

    args = parser.parse_args()

    main(args)