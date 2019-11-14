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

TEST_T = cfg.vals["episode_len"]


# Initialize environment and action space
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
n_actions = env.n_actions
action_space = np.arange(n_actions)

def map_optimal_rewards(tabu_len, k):
    state = env.reset()
    total_reward = 0
    results = {'rewards': [0.0]}
    optimal_actions = []




    for day in range(TEST_T):
        curr_best_val = 0.0
        curr_best_action = 0.0

        curr_state = copy.deepcopy(env.state)
        feasible_actions = AllocationEnv.get_feasible_actions(curr_state.board_config)


        for action in feasible_actions:

            print("Iteration: {}, Action: {}".format(day, action), end='\r')
            action = AllocationEnv.check_action(curr_state.board_config, action)
            proposed_state, reward, b, i = env.step(action)
            env.set_state(curr_state)

            if reward > curr_best_val:
                curr_best_action = action


        optimal_actions.append(curr_best_action)
        curr_best_action = AllocationEnv.check_action(curr_state.board_config, curr_best_action)

        state, final_reward, _ , _ = env.step(curr_best_action)  # update the state after each day based on the optimal action taken

        total_reward += final_reward
        curr_best_val = final_reward
        results['rewards'].append(total_reward)
        print("best action: {} - reward: {}".format(curr_best_action, final_reward))
        print("total reward: {}".format(total_reward))


    return state, optimal_actions, results

state, actions, results = map_optimal_rewards(tabu_len=50, k=25)




x = np.arange(TEST_T+1)
plt.plot(x, results['rewards'])
plt.xlabel("Timestep (t)")
plt.ylabel("Cumulative Reward (test)")
plt.savefig("figs/brute-force-{}.png".format(cfg.vals['prj_name']))


for k, v in results.items():
    results[k] = serialize_floats(v)


with open("output/brute-force-{}.json".format(cfg.vals['prj_name']), 'w') as f:
    json.dump(results, f)
