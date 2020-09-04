import numpy as np
import config.config as cfg
import pandas as pd
import ast

from envs.init_env import init_env

from utils import make_bin_mtx

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

    def __init__(self, day, day_vec, board_config, prev_sales, prices, sales_bound):
        self.day = day
        self.day_vec = day_vec
        self.board_config = board_config
        self.prev_sales = prev_sales
        self._products = np.where(board_config == 1.0)[1]
        self._regions = np.where(board_config == 1.0)[0]
        self._items = list(zip(self._regions, self._products))
        self._product_mask = State.get_mask(self._products, cfg.vals['n_products'])
        self.curr_sales = None
        self.prices = prices
        self.sales_bound = sales_bound

    def __str__(self):
        s = "day:\n{}\nboard:\n{}\n prev_sales:\n{}".format(self.day_vec, self.board_config, self.prev_sales)
        return s

    def update_board(self, a):
        """
        Update board configuration
        :param a: (np.ndarray) boolean matrix of action space
        :return:
        """

        self.board_config = self.board_config + a

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
    def clip_val(val, bound):
        return np.minimum(val, bound)


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

    @staticmethod
    def get_avg_prices(df):
        mean_prices = df[['product', 'price']].groupby('product').mean()
        return mean_prices['price'].to_dict()

    @classmethod
    def __init_prev_sales(cls, r, p):
        means_mtx = np.zeros((r, p))
        for i in range(r):
            for j in range(p):
                means_mtx[i, j] = np.random.uniform(10, 100)
        return means_mtx

    @classmethod
    def init_env_loc(cls, config):
        # setup state matrices
        if 'env_init_loc' in config:
            config['env_init_loc'] = ast.literal_eval(config['env_init_loc'])
            config['env_init_loc'] = make_bin_mtx(config['env_init_loc'], dims=(config['n_regions'], config['n_products']))
        else:
            np.random.seed(config["random_seed"])
            n_items = config['n_regions'] * config['n_products'] // 10
            init_loc = init_env(config['n_regions'], config['n_products'], n_items)
            config['env_init_loc'] = make_bin_mtx(init_loc, dims=(config['n_regions'], config['n_products']))

        return config['env_init_loc']

    @classmethod
    def init_state(cls, config):


        if "prev_sales" not in config:
            train_data = pd.read_csv(config['train_data'])
            train_data = train_data[train_data['quantity'] >= 0]
            prev_sales = State.__init_prev_sales(cfg.vals['n_regions'],
                                                       cfg.vals['n_products'])
        else:
            prev_sales = config['prev_sales']

        board_state = State.init_env_loc(config)


        mean_prices = State.get_avg_prices(train_data)
        # compute upper bound for valid, single-day product/sales value
        upper_prices = np.quantile(train_data['sales'], q=.99)
        upper_prices = .25*upper_prices + upper_prices

        prev_sales_mask = prev_sales * board_state


        day_vec = State.get_day_vec(config['env_init_day'])
        state = State(day=config['env_init_day'],
                      day_vec=day_vec,
                      board_config=board_state,
                      prev_sales=prev_sales_mask,
                      prices=mean_prices,
                      sales_bound=upper_prices)

        return state

    @staticmethod
    def get_vec_observation(obs_dict):
        assert isinstance(obs_dict, dict)
        return np.array(np.concatenate(
            ([obs_dict['day_vec'], obs_dict['board_config'], obs_dict["prev_sales"]]), axis=None))

    @staticmethod
    def get_board_config_from_vec(arr, n_regions, n_products):
        n = len(arr)
        placements = arr[7:(n-6)]
        board_cfg = np.where(placements > 0.0, 1.0, 0.0)
        board_cfg = board_cfg.reshape((n_regions, n_products))


        return board_cfg


if __name__ == "__main__":

    init_state = State.init_state(config=cfg.vals)
    a = np.zeros((4,4))
    a[3,3] = 1.0
    init_state.update_board(a)
    init_state.advance([1.0, 1.0, 1.0, 1.0])

    print(init_state.board_config)
    init_state.update_board(a)
    print(init_state.board_config)

