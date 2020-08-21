import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime, timedelta

from preprocessing.preprocessing_utils import *

N_PRODUCTS = 15
FLIP_PROB = 0.05
TRAIN_DATA_PCT = .85




def main():
    # Import store, sales data
    stores = pd.read_csv("store-level-data-17-19.csv")
    stores['DATE'] = pd.to_datetime(stores['DATE'])
    # stores['day_of_week'] = stores['DATE'].dt.dayofweek
    stores = stores[stores['SALES'] > 0.0]
    # log of sales
    stores['SALES'] = np.log(stores['QUANTITY'] * stores['PRICE'])
    ## Standadarze sales data ([x - mu] / sd)
    # stores['SALES'] = (stores['SALES'] - stores['SALES'].mean()) / np.std(stores['SALES'])
    # stores['SALES_2'] = np.power(stores["SALES"], 2)
    stores['day_of_week'] = stores['DATE'].dt.dayofweek

    stores = get_year_lookback(stores, filter_date='2018-07-31')

    mean_sales = stores[["CUSTOMER", "UPC", "SALES"]].groupby(["CUSTOMER", "UPC"]).mean()
    mean_sales = mean_sales['SALES'].to_dict()

    mean_sales_day = stores[["CUSTOMER", "UPC", "day_of_week", "SALES"]].groupby(
        ["CUSTOMER", "UPC", "day_of_week"]).mean()
    mean_sales_day = mean_sales_day["SALES"].to_dict()

    # metadata
    store_ids = pd.read_csv("stores.csv")
    products = pd.read_csv("products.csv")

    ## Store consideration set
    # 600055785 --> Fort Union (Midvale) - Large Store (by sales)
    # 600055679 --> Draper - Small Store (by sales)
    STORE_SET = [600055785, 600055679]

    # Get Product set
    # Top 15 products
    top_prods = stores[['UPC', 'SALES']].groupby('UPC').sum().sort_values("SALES", ascending=False).reset_index()[
                :N_PRODUCTS]
    PROD_SET = top_prods["UPC"].values

    global prod_to_idx
    prod_to_idx = dict(zip(PROD_SET, range(N_PRODUCTS)))

    with open("product_idx_map.json", 'w') as f:
        serialized = {}
        for k, v in prod_to_idx.items():
            serialized[int(k)] = v
        json.dump(serialized, f)

    # Regions per store
    n_regions = dict(zip(STORE_SET, [18, 12]))

    # Adjacency Matrices

    adj = {STORE_SET[0]: get_adj_mtx("store-1-adj-mtx.json"),
           STORE_SET[1]: get_adj_mtx("store-2-adj-mtx.json")}

    for k, a in adj.items():
        is_symmetric = np.allclose(a, a.transpose())
        assert is_symmetric

    # Heterogeneous spatial weights
    priors = init_prior(STORE_SET, adj, n_regions)
    weights = gen_weights(STORE_SET, priors, N_PRODUCTS)

    # Intialize Board State
    global state
    state = {STORE_SET[0]: np.zeros((n_regions[STORE_SET[0]], N_PRODUCTS)),
             STORE_SET[1]: np.zeros((n_regions[STORE_SET[1]], N_PRODUCTS))}

    for store, board in state.items():
        state[store] = init_state(board)

    # slice data
    stores_keep = stores[stores["CUSTOMER"].isin(STORE_SET)]
    stores_keep = stores[stores["UPC"].isin(PROD_SET)]
    store1 = stores_keep[stores_keep['CUSTOMER'] == STORE_SET[0]]
    store2 = stores_keep[stores_keep['CUSTOMER'] == STORE_SET[1]]

    store1 = get_prev_sales(store1, means=mean_sales, means_by_day=mean_sales_day)
    store2 = get_prev_sales(store2, means=mean_sales, means_by_day=mean_sales_day)

    # store1_clean = update_features(estimate_spatial_q(store1, STORE_SET[0]))
    # store2_clean = update_features(estimate_spatial_q(store2, STORE_SET[1]))

    store1, action1 = estimate_spatial_q(store1, STORE_SET[0], n_regions, weights)
    store2, action2 = estimate_spatial_q(store2, STORE_SET[1], n_regions, weights)

    store1_clean = update_features(store1)
    store2_clean = update_features(store2)

    store1_clean = update_timestamps(store1_clean)
    store2_clean = update_timestamps(store2_clean)

    # Split train/test


    store1_train, store1_test, a1_train, a1_test = split(store1_clean, A=action1, train_pct=TRAIN_DATA_PCT)
    store2_train, store2_test, a2_train, a2_test = split(store2_clean, A=action2, train_pct=TRAIN_DATA_PCT)

    store1_train = update_timestamps(store1_train)
    store1_test = update_timestamps(store1_test)

    store2_train = update_timestamps(store2_train)
    store2_test = update_timestamps(store2_test)

    store1_train.to_csv("store-1-train.csv", index=False)
    store2_train.to_csv("store-2-train.csv", index=False)

    store1_test.to_csv("store-1-test.csv", index=False)
    store2_test.to_csv("store-2-test.csv", index=False)

    store1_clean.to_csv("store-1-all.csv", index=False)
    store2_clean.to_csv("store-2-all.csv", index=False)

    action1.to_csv("store-1-action-all.csv", index=False)
    action2.to_csv("store-2-action-all.csv", index=False)


if __name__ == "__main__":
    main()

