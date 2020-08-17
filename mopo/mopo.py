from stable_baselines.deepq.replay_buffer import ReplayBuffer
import config.config as cfg
from envs.state import State
from envs.allocation_env import AllocationEnv


class Mopo(object):

    def __init__(self, policy, env_model, epochs, rollout, n_actions, buffer_size=50000):
        self.epochs = epochs
        self.env_model = env_model
        self.rollout = rollout
        self.policy = policy
        self.n_actions = n_actions
        self.buffer = ReplayBuffer(buffer_size)




    def learn(self):

        results = {'rewards': [0.0]}

        for i in range(self.epochs):
            state = State.init_state(config=cfg.vals)

            for h in range(self.rollout):
                feasible_actions = AllocationEnv.get_feasible_actions(state["board_config"])
                action_mask = AllocationEnv.get_action_mask(feasible_actions, self.n_actions)


                action, _states = self.model.predict(state, mask=action_mask)
                action = AllocationEnv.check_action(state['board_config'], action)

                #obs, r, dones, info = env.step([action])

                results['rewards'].append(r[0] + results['rewards'][-1])

