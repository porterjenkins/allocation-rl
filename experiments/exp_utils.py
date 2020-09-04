import os

from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import numpy as np

def get_simple_simulator(config):
    """

    :param config:
    :return:
    """
    simulator_cfg = {k: v for k, v in config.items()}
    simulator_cfg["model_type"] = "hierarchical"
    sim_nam = simulator_cfg['model_type'] + "-" + simulator_cfg['train_data'].split("/")[-1].split(".")[
        0] + "-no-precision" + ".p"
    simulator_cfg["model_path"] = os.path.join(config["prj_root"], "envs", sim_nam)

    simulator_prior = Prior(simulator_cfg)
    simulator = AllocationEnv(config=simulator_cfg,
                              prior=simulator_prior,
                              load_model=True)


    return simulator


# Runs policy for X episodes and returns average reward

def evaluate_policy(policy, env, eval_episodes=10):
    reward_arr = np.zeros(eval_episodes)

    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.
        while not done:
            feasible_actions = AllocationEnv.get_feasible_actions(obs["board_config"])
            action_mask = AllocationEnv.get_action_mask(feasible_actions, env.action_space.n)

            action, _states = policy.predict(obs, mask=action_mask)
            action = AllocationEnv.check_action(obs['board_config'], action)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        reward_arr[i] = total_reward

    avg_reward = reward_arr.mean()
    std_reward = reward_arr.std()

    print("---------------------------------------")
    print("Evaluation over {} episodes: {:.1f} ({:.2f})".format(eval_episodes, avg_reward, std_reward))
    print("---------------------------------------")
    return avg_reward, std_reward