import numpy as np
import config.config as cfg
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class State(object):
    """
    An object describing the current state of the environment. We currently implement the following state features:
        - Day of the week (int): 0 - 6
        - "Board" configuration (np.ndarray):
            - A boolean matrix (n_regions x n_products) denoting where each product has been placed
        - Sales t-1 (np.ndarray):
            - Sales at the previous time stamp:
            - (n_products x 1)
    """

    def __init__(self, day, board_config, prev_sales):
        self.day = day
        self.board_config = board_config
        self.prev_sales = prev_sales
        self._products = np.where(board_config == 1.0)[1]
        self._regions = np.where(board_config == 1.0)[0]
        self._product_mask = State.get_mask(self._products, cfg.vals['n_products'])

    def advance_state(self, a):
        self.day = (self.day + 1) % 7
        self.board_config = self.board_config + a
        # TODO: update prev_sales

        self._products = np.where(self.board_config == 1.0)[1]
        self._regions = np.where(self.board_config == 1.0)[0]

    @staticmethod
    def get_mask(items, n_items):
        mask = np.zeros(n_items, dtype=np.int32)
        for i in items:
            mask[i] = 1

        return mask


    @classmethod
    def __init_prev_sales_means(cls, df):
        prod_means = df[['product', 'sales']].groupby('product').mean()
        return prod_means.values

    @classmethod
    def init_state(cls, config):
        if "prev_sales" not in config:
            train_data = pd.read_csv(config['train_data'])
            prev_sales = State.__init_prev_sales_means(train_data)
        else:
            prev_sales = config['prev_sales']

        state = State(day=config['env_init_day'],
                      board_config=config['env_init_loc'],
                      prev_sales=prev_sales)

        return state


if __name__ == "__main__":

    init_state = State.init_state(config=cfg.vals)