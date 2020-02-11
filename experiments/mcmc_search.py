# at a state, take every possible action, find the optimal action, then  take the state at the optimal action
# repeat 30 times
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.allocation_env import AllocationEnv
from envs.prior import Prior
import config.config as cfg
import numpy as np
import copy
import theano
import matplotlib.pyplot as plt
import json
from utils import serialize_floats
from collections import deque
import random


TEST_T = cfg.vals["episode_len"]
LMBDA = 0.75
T = 5

# Initialize environment and action space
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
n_actions = env.n_actions
action_space = np.arange(n_actions)


def get_f(ae, T, penalty=None, log=True, lmbda=0.75):
    """
    compute the "energy function F".

    :param ae: Error measurement
    :param T: Temperature parameter
    :param penalty: value to penalize constrain object
    :param log: (Bool) Return f on log scale
    :param lmbda: regularization on penalty term
    :return: the 'energy' of a given state
    """
    if penalty is None:
        penalty = 0

    if log:
        return -(ae + lmbda * penalty) / T
    else:
        return np.exp(-(ae + lmbda * penalty) / T)


def get_gamma(f_current, f_proposed, symmetric=True, log=True, q_proposed_given_current=None,
              q_current_given_proposed=None):
    """
    Compute gamma to be used in acceptance probability of proposed state
    :param f_current: f ("energy function") of current state
    :param f_proposed: f ("energy function") of proposed state
    :param log: (bool) Are f_current and f_proposed on log scale?
    :return: value for gamma, or probability of accepting proposed state
    """
    if symmetric:
        if log:
            alpha = f_proposed - f_current
            gamma = np.min((0, alpha))
        else:
            alpha = f_proposed / f_current
            gamma = np.min((1, alpha))
    else:
        if log:
            alpha = f_proposed - f_current + q_current_given_proposed - q_proposed_given_current
            gamma = np.min((0, alpha))
        else:
            alpha = (f_proposed * q_current_given_proposed) / (f_current * q_proposed_given_current)
            gamma = np.min((1, alpha))

    return gamma


def map_optimal_rewards():
    state = env.reset()
    total_reward = 0
    results = {'rewards': [0.0]}
    optimal_actions = []

    curr_action = 0

    for day in range(TEST_T):

        curr_state = copy.deepcopy(env.state)
        feasible_actions = AllocationEnv.get_feasible_actions(curr_state.board_config)
        proposed_action = np.random.choice(list(feasible_actions))

        curr_state_step, curr_reward, b, i = env.step(curr_action)
        env.set_state(curr_state)

        proposed_state, proposed_reward, b, i = env.step(proposed_action)

        curr_f = get_f(ae=-curr_reward, lmbda=LMBDA, log=True, T=T)
        proposed_f = get_f(ae=-proposed_reward, lmbda=LMBDA, log=True, T=T)

        gamma = get_gamma(f_current=curr_f, f_proposed=proposed_f, log=True)
        # Generate random number on log scale
        sr = np.log(random.random())

        if sr < gamma:  # made progress
            #state, final_reward, _, _ = env.step(curr_best_action)  # update the state after each day based on the optimal action taken
            optimal_actions.append(proposed_action)
            curr_best_action = proposed_action
            final_reward = proposed_reward

        else:
            optimal_actions.append(curr_action)
            state, final_reward, _, _ = env.step(curr_action)
            curr_best_action = curr_action


        total_reward += final_reward
        results['rewards'].append(total_reward)
        print("best action: {} - reward: {}".format(curr_best_action, final_reward))
        print("total reward: {}".format(total_reward))



    return state, optimal_actions, results

state, actions, results = map_optimal_rewards()




x = np.arange(TEST_T+1)
plt.plot(x, results['rewards'])
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward (test)")
plt.savefig("figs/mcmc-{}.png".format(cfg.vals['prj_name']))


for k, v in results.items():
    results[k] = serialize_floats(v)


with open("output/mcmc-{}.json".format(cfg.vals['prj_name']), 'w') as f:
    json.dump(results, f)
