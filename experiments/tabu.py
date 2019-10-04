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

# I don't know why I need this locally, but it won't run without it... ¯\_(ツ)_/¯
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"



TEST_T = 30
TIME_STEPS = 30*500
LEARNING_START = int(TIME_STEPS*.3)

# Initialize environment and action space
prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
n_actions = env.n_actions
action_space = np.arange(n_actions)

def map_optimal_rewards():
    state = env.reset()
    total_reward = 0
    reward_values = []
    optimal_actions = []
    for day in range(TEST_T):
        action_to_reward = {}  # create a HashMap from action to reward

        for action in action_space:
            proposed_env = copy.deepcopy(env)
            proposed_state, reward, b, i = proposed_env.step(action)
            action_to_reward[action] = reward

        optimal_actions.insert(day, max(action_to_reward, key=action_to_reward.get)) # Save best action on ith day
        total_reward += action_to_reward.get(optimal_actions[day])
        reward_values.insert(day, total_reward)
        state = env.step(optimal_actions[day])  # update the state after each day based on the optimal action taken

    return state, optimal_actions, reward_values

print(map_optimal_rewards())


#
#
# x = np.arange(TEST_T+1)
# plt.plot(x, results['rewards'])
# plt.xlabel("Timestep (t)")
# plt.ylabel("Cumulative Reward (test)")
# plt.savefig("figs/rl-test-{}.png".format(cfg.vals['prj_name']))
#
#
# for k, v in results.items():
#     results[k] = serialize_floats(v)
#
#
# with open("output/rl-test-{}.json".format(cfg.vals['prj_name']), 'w') as f:
#     json.dump(results, f)