import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import numpy as np
from policies.input import observation_input
import tensorflow as tf
import argparse
import os
import pickle

import offpolicy.utils as bcq_utils
from offpolicy.bcq import BCQ
from experiments.exp_utils import evaluate_policy, get_simple_simulator
from experiments.logger import Logger

from utils import get_store_id



# modified from BCQ in PyTorch code: https://github.com/sfujim/BCQ

def main(args):
    store_id = get_store_id(cfg.vals["train_data"])
    env = get_simple_simulator(cfg.vals)
    n_actions = env.n_actions

    tf.set_random_seed(cfg.vals["random_seed"])
    np.random.seed(cfg.vals["random_seed"])

    state_dim = observation_input(env.observation_space, batch_size=None, name='Ob', scale=False, reuse=tf.AUTO_REUSE)[0].shape[1].value
    action_dim = n_actions

    with tf.Session() as sess:
        # Initialize policy

        # Initialize policy
        policy = BCQ(state_dim, action_dim, sess)
        policy.load(f"{store_id}-{args.file_name}", directory="./models")


        evaluations = []

        episode_num = 0
        done = True
        reward, sigma = evaluate_policy(policy, env, eval_episodes=args.eval_eps)
        evaluations.append((reward, sigma))
        #np.save("./results/" + file_name, evaluations)


        # print(stats_loss)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--file-name", default="bcq.p")
    # parser.add_argument("--save_interval", default=20, type=int) # save every eval_freq intervals
    args = parser.parse_args()

    main(args)