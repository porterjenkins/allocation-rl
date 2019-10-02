import os
import sys
import matplotlib.pyplot as plot

from envs.allocation_env import AllocationEnv
from envs.state import State

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
import config.config as cfg
from envs.features import Features

init_state = State.init_state(config=cfg.vals)

action = 2
action = AllocationEnv.check_action(init_state.board_config, action)

# This is the action space
# self.n_actions = 1 + self.n_regions*self.n_products*2

class SupervisedLearning(object):
    def __init__(self, config):
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.action_space = 1 + self.n_regions * self.n_products * 2

    def learn():
        train_data = pandas.read_csv(cfg.vals['train_data'])
        train_data_features = Features.feature_extraction(train_data, y_col='quantity')

        TRAIN_X = train_data_features.toarray()
        TRAIN_Y = train_data_features.y

        test_data = pandas.read_csv(cfg.vals['test_data'])
        test_data_features = Features.feature_extraction(test_data, y_col='quantity')

        TEST_X = test_data_features.toarray()
        TEST_Y = test_data_features.y

        linearRegression = LinearRegression()
        linearRegression.fit(TRAIN_X, TRAIN_Y)
        beta = linearRegression.coef_

        yPrediction = linearRegression.predict(TEST_X)

        plot.plot(TEST_Y, yPrediction, color='blue')

        plot.show()

    # init_state.board_config would be passed in as the state,
    # which is a matrix
    def predict(self, state):
        # Build an action map to be able to iterative through
        # the action space and pull the corresponding action for each key
        action_map = AllocationEnv.build_action_map(self)
        linearRegression = LinearRegression()

        V = np.array()  # Create matrix that will hold predicted data for [jth,kth] element

        for i in range(self.action_space):  # iterate for each action in action-space
            action = action_map.get(i)
            for j in range(action.length):  # iterate over each
                for k in range(state.shape(0)):
                    x = concat(state[k, :], action)
                    linearRegression.fit(x)
                    y_hat = linearRegression.predict(x)
                    V[j, k] = y_hat

        value_hat = V.sum(axis=1)
        # returns the index position of the maximum value of value_hat
        # the maximum value represents the action taken at an arbitrary row
        # in the state matrix
        return np.argmax(value_hat)
