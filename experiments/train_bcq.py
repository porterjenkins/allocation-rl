import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import matplotlib.pyplot as plt
import numpy as np
import gym
from policies.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from utils import serialize_floats
from policies.input import observation_input
import json
import tensorflow as tf
import argparse
import os
import pickle

import offpolicy.utils as bcq_utils
from offpolicy.bcq import BCQ


TEST_T = cfg.vals["episode_len"]
TIME_STEPS = 20000
LEARNING_START = 1500


# modified from BCQ in PyTorch code: https://github.com/sfujim/BCQ

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            feasible_actions = AllocationEnv.get_feasible_actions(obs["board_config"])
            action_mask = AllocationEnv.get_action_mask(feasible_actions, env.action_space.n)

            action = policy.predict(obs, mask=action_mask)
            action = AllocationEnv.check_action(obs['board_config'], action)
            obs, reward, done, _ = env.step([action])
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.1f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='AllocationEnv-v0')  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, TensorFlow and Numpy seeds
    parser.add_argument("--buffer_type", default="Robust")  # Prepends name to filename.
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
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
    # parser.add_argument("--save_interval", default=20, type=int) # save every eval_freq intervals
    args = parser.parse_args()

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

    prior = Prior(config=cfg.vals)
    env = AllocationEnv(config=cfg.vals, prior=prior, load_model=False)
    n_actions = env.n_actions
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

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
        with open(f"../data/store-2-buffer.p", 'rb') as f:
            replay_buffer  = pickle.load(f)

        evaluations = []

        episode_num = 0
        done = True

        training_iters = 0
        while training_iters < args.max_timesteps:
            stats_loss = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size,
                                      discount=args.discount)

            evaluations.append(evaluate_policy(policy))
            np.save("./results/" + file_name, evaluations)

            training_iters += args.eval_freq
            print("Training iterations: " + str(training_iters))
        # print(stats_loss)

        # Save final policy
        policy.save("%s" % (file_name), directory="./models")