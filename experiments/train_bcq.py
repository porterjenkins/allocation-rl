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
    hyp = {

        "episode length": cfg.vals["episode_len"],
        "n simulations": args.eval_eps,
        "store": store_id,
        "iterations:": args.iterations,
        "batch size": args.batch_size,
        "discount": args.discount,
        "tau": args.tau,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "vae_lr": args.vae_lr,
        "actor_hs": args.actor_hs,
        "critic_hs": args.critic_hs,
        "dqda_clip": args.dqda_clip,
        "clip_norm": args.clip_norm
    }

    logger = Logger(hyp, "./results/", "bcq")



    if args.actor_hs <= 0:
        actor_hs_list = [64, 64]
    else:
        actor_hs_list = [args.actor_hs] * 2
    if args.critic_hs <= 0:
        critic_hs_list = [64, 64]
    else:
        critic_hs_list = [args.critic_hs] * 2

    file_name = "BCQ_%s_%s" % (args.env_name, str(args.seed))
    buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    #prior = Prior(config=cfg.vals)
    #env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
    env = get_simple_simulator(cfg.vals)
    n_actions = env.n_actions
    #env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = observation_input(env.observation_space, batch_size=None, name='Ob', scale=False, reuse=tf.AUTO_REUSE)[0].shape[1].value
    action_dim = n_actions

    with tf.Session() as sess:
        # Initialize policy
        policy = BCQ(state_dim, action_dim, sess, args.tau, actor_hs=actor_hs_list,
                         actor_lr=args.actor_lr,
                         critic_hs=critic_hs_list, critic_lr=args.critic_lr, dqda_clipping=args.dqda_clip,
                         clip_norm=bool(args.clip_norm), vae_lr=args.vae_lr)

        # Load buffer
        with open(f"../data/{store_id}-buffer-d-trn.p", 'rb') as f:
            replay_buffer = pickle.load(f)

        #evaluations = []

        #episode_num = 0
        #done = True

        stats_loss = policy.train(replay_buffer, iterations=args.iterations, batch_size=args.batch_size,
                                  discount=args.discount)


        print("Training iterations: " + str(args.iterations))
    # print(stats_loss)

        # Save final policy
        if os.path.exists(f"./models/{store_id}-{args.file_name}"):
            os.remove(f"./models/{store_id}-{args.file_name}")
        policy.save(f"{store_id}-{args.file_name}", directory="./models")

        logger.write()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='AllocationEnv-v0')  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, TensorFlow and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")  # Prepends name to filename.
    parser.add_argument("--iterations", default=1e3, type=int)  # Max time steps to run environment for
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--actor_lr", default=0.001, type=float)
    parser.add_argument("--critic_lr", default=0.001, type=float)
    parser.add_argument("--vae_lr", default=0.001, type=float)
    parser.add_argument("--actor_hs", default=0, type=int)
    parser.add_argument("--critic_hs", default=0, type=int)
    parser.add_argument("--dqda_clip", default=None, type=float)
    parser.add_argument("--clip_norm", default=0, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument('--file-name', type=str, default="bcq.p")
    # parser.add_argument("--save_interval", default=20, type=int) # save every eval_freq intervals
    args = parser.parse_args()

    main(args)