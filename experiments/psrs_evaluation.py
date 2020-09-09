import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import argparse

from evaluators.rejection_sampler import PSRS, MLPPolicy, BCQPolicy
from evaluators.mlp_env_model import MLPClassifer
from utils import get_store_id, get_action_space

from policies.deepq.dqn import DQN
from offpolicy.bcq import BCQ


store_id = get_store_id(cfg.vals["train_data"])
buffer_path = f"../data/{store_id}-buffer-d-test.p"


def eval_dqn(policy, args):

    action_space = get_action_space(cfg.vals["n_regions"], cfg.vals["n_products"])
    model_path = f"../data/{store_id}-env_policy.pt"


    env_policy = MLPPolicy(action_space, buffer_path, model_path)

    psrs = PSRS(buffer_path, policy, env_policy, action_space, cfg.vals["n_regions"], cfg.vals["n_products"], args.epochs)
    r = psrs.evaluate()
    mean = np.mean(r)
    sigma = np.std(r)
    print(r)
    print(mean, sigma)


def eval_off_policy_dqn(args):
    policy = DQN.load(f"../experiments/models/{store_id}-off-policy-dqn.p")
    eval_dqn(policy, args)

def eval_mopo_dqn(args):

    policy = DQN.load(f"../experiments/models/{store_id}-mopo-policy.p")
    eval_dqn(policy, args)


def eval_bcq(args):
    import tensorflow as tf
    action_space = get_action_space(cfg.vals["n_regions"], cfg.vals["n_products"])

    model_path = f"../data/{store_id}-env_policy.pt"

    env_policy = MLPPolicy(action_space, buffer_path, model_path)
    state_dims = cfg.vals["n_regions"]*cfg.vals["n_products"] + 7 + 6

    with tf.Session() as sess:
        # Initialize policy
        policy = BCQ(state_dims, action_space, sess)
        policy.load(f"{store_id}-bcq.p", directory="./models")
        policy = BCQPolicy(policy)
        psrs = PSRS(buffer_path, policy, env_policy, action_space, cfg.vals["n_regions"], cfg.vals["n_products"], args.epochs)
        r = psrs.evaluate()

    mean = np.mean(r)
    sigma = np.std(r)
    print(r)
    print(mean, sigma)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--method', type=str, help="Must be {off_policy_dqn, mopo_dqn, bcq}")
    args = parser.parse_args()

    assert args.method in ["off_policy_dqn", "mopo_dqn", "bcq"]

    if args.method == "mopo_dqn":
        eval_mopo_dqn(args)
    elif args.method == "bcq":
        eval_bcq(args)
    else:
        eval_off_policy_dqn(args)
