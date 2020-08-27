import pandas as pd
import numpy as np
import random
import pickle
from envs.state import State
from datetime import datetime
from stable_baselines.deepq.replay_buffer import ReplayBuffer

from envs.allocation_env import AllocationEnv
from preprocessing.preprocessing_utils import get_adj_mtx, init_prior, gen_weights, normalize


class ActionSpace(object):

    def __init__(self, n_regions, n_products):
        self.n_regions = n_regions
        self.n_products = n_products
        self.action_space = self.build_action_map()


    def build_action_map(self):
        m = {-1: ((), 0),
             0: ((), 0)}

        idx = 1


        for a in [-1, 1]:
            for i in range(self.n_regions):
                for j in range(self.n_products):

                    m[idx] = ((i,j), a)

                    idx += 1

        return m


    def sample(self, board_cfg, prod_next, idx_to_prod):

        keys = list(self.action_space)[1:]

        while True:
            #a_idx, action = np.random.choice(keys)
            a_idx = np.random.choice(keys)

            a_idx = AllocationEnv.check_action(board_cfg, a_idx)

            if a_idx >= 0:
                mtx_idx, action = self.action_space[a_idx]

                if a_idx == 0 or idx_to_prod[mtx_idx[1]] in prod_next:
                    break

        a_mtx = np.zeros((self.n_regions, self.n_products))
        a_mtx[mtx_idx] = action
        return a_mtx, a_idx



    """def sample(self, board_cfg, prod, prod_next, weights):

        if random.random() < .5:
            # takeaway
            rm = random.sample(prod, k=1)[0]
            prod.remove(rm)
        else:
            # add
            diff = prod.difference(prod_next)
            prod.add(random.sample(diff, k=1)[0])"""



def get_product_distributions(df):

    loc = pd.DataFrame.to_dict(df[["UPC", "SALES"]].groupby("UPC").mean())
    scale = pd.DataFrame.to_dict(df[["UPC", "SALES"]].groupby("UPC").std())

    dist = {"mean": loc, "std": scale}

    return dist




def init_state(n_regions, n_products, product_set, prod_to_idx):

    mtx = np.zeros((n_regions, n_products))
    #n_placements = n_regions // 4
    for p in product_set:
        p_idx = prod_to_idx[p]
        probs = [1 / n_regions]*n_regions

        n_placements = 0
        while n_placements < 1:
            placement = np.random.binomial(n_regions//4, probs)
            n_placements = placement.sum()

        mtx[:, p_idx] = placement

    return np.where(mtx <= 1, mtx, 1)


def get_reward(chunk, date, prod_dist, board_cfg, weights, prod_to_idx):
    chunk = chunk[chunk['DATE'] == date]


    curr_sales = np.zeros_like(board_cfg)
    reward = 0
    for p, p_idx in prod_to_idx.items():
        placement = board_cfg[:, p_idx]
        if placement.sum() == 0:
            continue

        try:
            total_sales = chunk[chunk["UPC"]==p]["SALES"].values[0]
        except:

            # TODO: sample from distribution
            mu = prod_dist["mean"]["SALES"][p]
            sig = prod_dist["std"]["SALES"][p]

            total_sales = np.random.normal(mu, sig)

        w = weights[:, p_idx]
        w = normalize(w * placement)
        est_sales = w* total_sales
        curr_sales[:, p_idx] = est_sales

        reward += est_sales.sum()

    return reward, curr_sales



def get_state(board_cfg, date, prev_sales):

    day = datetime.strptime(date, "%Y-%m-%d")
    day_vec = State.get_day_vec(day.weekday())

    if prev_sales is not None:
        prev_sales = prev_sales.sum()

    state = {
        "day_vec": day_vec,
        "board_config": board_cfg,
        "prev_sales": prev_sales,

    }

    return state

def get_state_and_reward(chunk, date, product_set, board_cfg, weights, prod_to_idx):
    # product_set to vector
    day = datetime.strptime(date, "%Y-%m-%d")
    day_vec = State.get_day_vec(day.weekday())

    chunk = chunk[chunk['DATE'] == date]
    chunk = chunk[chunk["UPC"].isin(product_set)]


    curr_sales = np.zeros_like(board_cfg)
    reward = 0
    for idx, row in chunk.iterrows():
        product = row["UPC"]
        total_sales = row["SALES"]
        p_idx = prod_to_idx[product]
        w = weights[:, p_idx]
        placement = board_cfg[:, p_idx]
        w = normalize(w * placement)
        est_sales = w* total_sales
        curr_sales[:, p_idx] = est_sales

        reward += est_sales.sum()

    state = {
            "day_vec": day_vec,
            "board_config": board_cfg,
            "curr_sales": curr_sales,

        }

    return state, reward


def get_time_stamp(dates):
    unique = dates.unique()
    n_ts = len(unique)
    date_to_idx = dict(zip(unique, range(n_ts)))
    idx_to_date = dict(zip(range(n_ts), unique))
    return date_to_idx, idx_to_date

def get_dicts(df):

    products = {}
    grouped = df.groupby("DATE")

    for date, chunk in grouped:

        product_set = set(chunk["UPC"].unique())

        products[date] = product_set

    return products



def main():
    N_PRODUCTS = 15
    # Import store, sales data
    stores = pd.read_csv("../data/store-level-data-17-19.csv")
    #stores['DATE'] = pd.to_datetime(stores['DATE'])
    # stores['day_of_week'] = stores['DATE'].dt.dayofweek
    stores = stores[stores['SALES'] > 0.0]
    # log of sales
    stores['SALES'] = stores['QUANTITY'] * stores['PRICE']
    ## Standadarze sales data ([x - mu] / sd)
    # stores['SALES'] = (stores['SALES'] - stores['SALES'].mean()) / np.std(stores['SALES'])
    # stores['SALES_2'] = np.power(stores["SALES"], 2)
    #stores['day_of_week'] = stores['DATE'].dt.dayofweek

    ## Store consideration set
    # 600055785 --> Fort Union (Midvale) - Large Store (by sales)
    # 600055679 --> Draper - Small Store (by sales)
    STORE_SET = [600055785, 600055679]

    # Get Product set
    # Top 15 products
    top_prods = stores[['UPC', 'SALES']].groupby('UPC').sum().sort_values("SALES", ascending=False).reset_index()[
                :N_PRODUCTS]

    PROD_SET = top_prods["UPC"].values
    prod_to_idx = dict(zip(PROD_SET, range(N_PRODUCTS)))
    idx_to_prod = dict(zip(range(N_PRODUCTS), PROD_SET))

    stores_keep = stores[stores["UPC"].isin(PROD_SET)]

    store1 = stores_keep[stores_keep['CUSTOMER'] == STORE_SET[0]]
    store2 = stores_keep[stores_keep['CUSTOMER'] == STORE_SET[1]]

    # Regions per store
    n_regions = dict(zip(STORE_SET, [18, 12]))
    # Adjacency Matrices

    adj = {STORE_SET[0]: get_adj_mtx("../data/store-1-adj-mtx.json"),
           STORE_SET[1]: get_adj_mtx("../data/store-2-adj-mtx.json")}

    for k, a in adj.items():
        is_symmetric = np.allclose(a, a.transpose())
        assert is_symmetric

    # Heterogeneous spatial weights
    priors = init_prior(STORE_SET, adj, n_regions)

    # (regions x products)
    weights = gen_weights(STORE_SET, priors, N_PRODUCTS)

    store_cntr = 0
    for store in [store1, store2]:
        prod_dist = get_product_distributions(store)
        buffer = ReplayBuffer(size=50000)

        store_id = STORE_SET[store_cntr]
        r = n_regions[store_id]

        actions = ActionSpace(r, N_PRODUCTS)

        date_to_idx, idx_to_date = get_time_stamp(store['DATE'])
        products_map = get_dicts(store)

        product_t = products_map[idx_to_date[0]]
        board_cfg = init_state(r, N_PRODUCTS, product_t, prod_to_idx)

        prev_sales = None



        for dt, dt_idx in date_to_idx.items():

            product_t = products_map[dt]
            t_p_1 = idx_to_date.get(dt_idx+1, None)

            if t_p_1 is None:
                break

            state = get_state(board_cfg, idx_to_date[0], prev_sales)
            _, sales_t = get_reward(store, dt, prod_dist, state["board_config"], weights[store_id], prod_to_idx)

            product_next = products_map[t_p_1]

            # select action
            a, a_idx = actions.sample(state["board_config"], product_next, idx_to_prod)
            new_board_cfg = state["board_config"] + a

            new_state = get_state(new_board_cfg, t_p_1, prev_sales=sales_t)


            reward, _ = get_reward(store, t_p_1, prod_dist, new_board_cfg, weights[store_id], prod_to_idx)

            #state = new_state
            print(state["board_config"], reward)

            buffer.add(obs_t=State.get_vec_observation(state),
                       action=a_idx,
                       reward=reward,
                       obs_tp1=State.get_vec_observation(new_state),
                       done=False)



        with open(f"../data/store-{store_cntr+1}-buffer.p", 'wb') as f:
            pickle.dump(buffer, f)
        store_cntr +=1



if __name__ == "__main__":
    main()