import numpy as np
import config.config as cfg
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class State(object):
    """
    An object describing the current state of the environment. We currently implement the following state features:
        - Day of the week (int): 0 - 6
        - Day Vector (np.ndarray): Denotes the day of the week in a one hot vector
        - "Board" configuration (np.ndarray):
            - A boolean matrix (n_regions x n_products) denoting where each product has been placed
        - Sales t-1 (np.ndarray):
            - Sales at the previous time stamp:
            -  A matrix (n_regions x n_products)
    """

    def __init__(self, day, day_vec, board_config, prev_sales):
        self.day = day
        self.day_vec = day_vec
        self.board_config = board_config
        self.prev_sales = prev_sales
        self._products = np.where(board_config == 1.0)[1]
        self._regions = np.where(board_config == 1.0)[0]
        self._items = list(zip(self._regions, self._products))
        self._product_mask = State.get_mask(self._products, cfg.vals['n_products'])
        self.curr_sales = None

    def __str__(self):
        s = "day:\n{}\nboard:\n{}\n prev_sales:\n{}".format(self.day_vec, self.board_config, self.prev_sales)
        return s

    def update_board(self, a):
        """
        Update board configuration
        :param a: (np.ndarray) boolean matrix of action space
        :return:
        """

        # ensure that components of board configuration are in {0,1}
        self.board_config = np.minimum(self.board_config + a, 1.0)

        self._products = np.where(self.board_config == 1.0)[1]
        self._regions = np.where(self.board_config == 1.0)[0]
        self._items = list(zip(self._regions, self._products))


    def advance(self, sales_hat):
        self.increment_day()
        prev_sales_map = dict(zip(self._items, sales_hat))
        self.update_sales(prev_sales_map)


    def increment_day(self):
        self.day = (self.day + 1) % 7
        self.day_vec = State.get_day_vec(self.day)


    def update_sales(self, sales_map):
        sales_mtx = np.zeros((cfg.vals['n_regions'], cfg.vals['n_products']))
        for idx, val in sales_map.items():
            sales_mtx[idx] = val
        self.prev_sales = sales_mtx

    @staticmethod
    def get_day_vec(day):
        day_vec = np.zeros(7)
        day_vec[day] = 1.0
        return day_vec


    @staticmethod
    def get_mask(items, n_items):
        mask = np.zeros(n_items, dtype=np.int32)
        for i in items:
            mask[i] = 1

        return mask


    @classmethod
    def __init_prev_sales_means(cls, df):
        means = df[['product', 'region', 'sales']].groupby(['region', 'product']).mean()
        means_mtx = means.values.reshape(cfg.vals['n_regions'], cfg.vals['n_products'])
        return means_mtx
    @classmethod
    def init_state(cls, config):
        if "prev_sales" not in config:
            train_data = pd.read_csv(config['train_data'])
            prev_sales = State.__init_prev_sales_means(train_data)
        else:
            prev_sales = config['prev_sales']

        day_vec = State.get_day_vec(config['env_init_day'])
        state = State(day=config['env_init_day'],
                      day_vec=day_vec,
                      board_config=config['env_init_loc'],
                      prev_sales=prev_sales)


        return state


if __name__ == "__main__":

    init_state = State.init_state(config=cfg.vals)
    a = np.zeros((4,4))
    a[3,3] = 1.0
    init_state.update_board(a)
    init_state.advance([1.0, 1.0, 1.0, 1.0])

    print(init_state.board_config)
    init_state.update_board(a)
    print(init_state.board_config)