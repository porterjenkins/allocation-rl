
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from data.true_params import TrueParams
from experiments.plot import plot_region_product
import ast
from sklearn.preprocessing import LabelEncoder


def make_bin_mtx(arr, dims):
    mtx = np.zeros(dims)
    for idx in arr:
        mtx[idx] = 1.0
    return mtx


def get_adj_mtx(fname):
    with open(fname) as f:
        adj_store = json.load(f)
    adj_store['non_zero_entries'] = ast.literal_eval(adj_store['non_zero_entries'])
    A = make_bin_mtx(adj_store['non_zero_entries'], (adj_store['n_regions'],  adj_store['n_regions']))
    A = A + np.eye(adj_store['n_regions'])
    return A

def update_timestamps(df):
    date_map = {}
    time_stamps = np.zeros(df.shape[0])
    date_cntr = 0
    row_cntr = 0
    for idx, row in df.iterrows():
        # timestamps
        if row['date'] in date_map:
            ts = date_map[row['date']]
        else:
            ts = date_cntr
            date_map[row['date']] = date_cntr
            date_cntr += 1

        time_stamps[row_cntr] = ts
        row_cntr += 1
    df['time'] = time_stamps

    return df

# Split train/test

def split(df, train_pct=.8):
    dates = df['time'].unique()
    train_size = int(train_pct * len(dates))

    train_idx = dates[:train_size]
    test_idx = dates[train_size:]

    train = df[df["time"].isin(train_idx)]
    test = df[df["time"].isin(test_idx)]


    return train, test



class DataGenerator(object):

    def __init__(self, store, n_regions, n_products, params):
        self.store = store
        self.n_regions = n_regions
        self.n_products = n_products
        self.params = params

        self.stores = [600055785, 600055679]
        self.upc_idx_map = {49000028904: 0,
                            49000028911: 1,
                            49000028928: 2,
                            78000082166: 3,
                            49000042559: 4,
                            49000031652: 5,
                            49000050103: 6,
                            49000024692: 7,
                            78000083163: 8,
                            49000024685: 9,
                            49000000443: 10,
                            78000082401: 11,
                            70847811169: 12,
                            49000014631: 13,
                            49000005486: 14
                            }
        self.idx_upc_map = {v: k for k, v in self.upc_idx_map.items()}
        self.products = list(self.upc_idx_map.keys())


    def _get_features_with_mapping(self, idx, val, map):
        n = len(idx)
        d = len(map)
        encoding = np.zeros((n, d))

        for i, k in enumerate(idx):

            j = map[k]
            encoding[i, j] = val[i]


        return pd.DataFrame(encoding, columns = list(map.keys()))



    def get_preprocess_data(self, fname):

        stores = pd.read_csv(fname)
        stores.rename(columns={"PRICE": "price"}, inplace=True)


        stores = stores[stores["CUSTOMER"] == self.store]
        stores['DATE'] = pd.to_datetime(stores['DATE'])
        # stores['day_of_week'] = stores['DATE'].dt.dayofweek
        stores = stores[stores['SALES'] > 0.0]
        # log of sales
        stores['SALES'] = np.log(stores['QUANTITY'] * stores['price'])
        ## Standadarze sales data ([x - mu] / sd)
        # stores['SALES'] = (stores['SALES'] - stores['SALES'].mean()) / np.std(stores['SALES'])
        # stores['SALES_2'] = np.power(stores["SALES"], 2)
        stores['day_of_week'] = stores['DATE'].dt.dayofweek


        stores = stores[stores["UPC"].isin(self.products)]


        #tmp = stores[stores['CUSTOMER']==600056204].pivot(index='DATE', columns='UPC', values='QUANTITY').fillna(0)

        product_features = self._get_features_with_mapping(stores["UPC"].values.astype(int), stores["QUANTITY"].values, self.upc_idx_map)
        product_features = pd.concat([stores[['CUSTOMER', "DATE"]].reset_index(drop=True), product_features.reset_index(drop=True)], axis=1)
        self.product_features = product_features.groupby(["CUSTOMER", "DATE"]).sum()

        self.store_data = stores[['CUSTOMER', 'DATE', 'price', 'day_of_week']]



    def get_quantity_features(self, r, p, c, s):
        c_vec = np.array([c])
        s_vec = np.array([s])
        features = np.concatenate([r, p, c_vec, s_vec])
        return features

    def gen_product_weights(self):
        mu = self.params["prior_loc_w_p"]
        sig = self.params["prior_scale_w_p"]
        W = np.random.multivariate_normal(mu, sig)
        return W

    def gen_region_weights(self):
        mu = self.params["prior_loc_w_r"]
        sig = self.params["prior_scale_w_r"]
        W = np.random.multivariate_normal(mu, sig)
        return W


    def gen_customer_counts(self, beta_c, x_t):
        lambda_c = np.dot(beta_c, x_t)
        lambda_c_floor = np.maximum(0, lambda_c)
        c_j = np.random.poisson(lam=lambda_c_floor)
        return c_j

    def gen_demand_q(self, w_r_ij, x_ij):

        lambda_q = np.dot(w_r_ij, x_ij)
        q = np.random.poisson(lambda_q)

        return q


    def gen_q(self, prod, t, r):
        product_w = np.array([8.0, 4.0, 75.0, 1.5])
        time_w = 2.5

        lmbda = np.dot(product_w, prod) + t*time_w
        q = np.random.poisson(lam=lmbda, size=r)
        return q

    def get_sales(self, quantity, prices):
        sales = np.dot(np.transpose(quantity), prices)
        return sales

    def get_curr_state(self, p=.4):

        state_mask = np.zeros([self.n_regions, self.n_products])

        for i in range(self.n_regions):
            for j in range(self.n_products):

                state_mask[i, j] = np.random.binomial(1, p)

        return state_mask

    def get_prod_region_block(self):
        block = []



        for i in range(self.n_products):
            for j in range(self.n_regions):
                block.append([self.idx_upc_map[i], i, j])


        return np.array(block)




    def run(self, fname='test-data.csv'):

        n = self.product_features.shape[0]*self.n_products*self.n_regions
        block_size = self.n_regions*self.n_products
        block = self.get_prod_region_block()

        output = np.zeros((n, 6), dtype=object)
        grids = []

        w_p = self.gen_product_weights().reshape(-1,1)
        w_r = self.gen_region_weights().reshape(-1, 1)

        cntr = 0
        for idx, row in self.product_features.iterrows():

                grid = np.transpose(np.outer(row[self.products].values, (1/self.n_regions) * np.ones(self.n_regions)))
                grid = grid + w_p.transpose()
                grid = grid + w_r

                grid = np.round(np.maximum(0.0, grid))

                state_mask = self.get_curr_state(p=.4)
                grid = grid * state_mask

                output[(cntr*block_size):((cntr*block_size) + block_size), 0] = idx[0]
                output[(cntr*block_size):((cntr*block_size) + block_size), 1] = idx[1]
                output[(cntr*block_size):((cntr*block_size) + block_size), 2:5] = block
                output[(cntr*block_size):((cntr*block_size) + block_size), 5] = grid.transpose().flatten()


                cntr += 1


        df = pd.DataFrame(output, columns = ['store_id', 'date', 'UPC', 'product', 'region', 'quantity'])
        df = pd.merge(df, self.store_data, left_on=['store_id', 'date'], right_on=['CUSTOMER', 'DATE'], how='left')
        df['sales'] = df['quantity'] * df["price"]
        df.sort_values(by = ['store_id', 'date', 'product', 'region'], inplace=True)

        return df


if __name__ == "__main__":
    ## Store consideration set
    # 600055785 --> Fort Union (Midvale) - Large Store (by sales)
    # 600055679 --> Draper - Small Store (by sales)

    STORES = [600055785, 600055679]


    if "store-1" in cfg.vals["adj_mtx_fname"]:
        store = STORES[0]
        store_name = "store-1"
        A = get_adj_mtx("store-1-adj-mtx.json")
    else:
        store = STORES[1]
        store_name = "store-2"
        A = get_adj_mtx("store-2-adj-mtx.json")

    P = TrueParams()

    params = P.fixTrueParams(STORES, cfg.vals['n_products'], cfg.vals['n_regions'], A, persist=False)
    generator = DataGenerator(store,
                              cfg.vals['n_regions'],
                              cfg.vals['n_products'],
                              params=params)

    generator.get_preprocess_data("store-level-data-17-19.csv")
    store = generator.run()
    store.to_csv(store_name + "-raw.csv", index=False)

    #store = update_timestamps(store)
    """date_encoder = LabelEncoder()
    store["time"] = date_encoder.fit_transform(store["date"])
    print(store.head())


    train, test = split(store, train_pct=.8)

    #train = update_timestamps(train)
    #test = update_timestamps(test)

    train.to_csv("store-1-train.csv")
    test.to_csv("store-1-test.csv")

    #print(train.head())"""





