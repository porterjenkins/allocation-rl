import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime, timedelta

N_PRODUCTS = 15
FLIP_PROB = 0.05
TRAIN_DATA_PCT = .85

def make_bin_mtx(arr, dims):
    mtx = np.zeros(dims)
    for idx in arr:
        mtx[idx] = 1.0
    return mtx

def get_prev_sales(df, means, means_by_day):

    prev_buff = {}
    prev_sales = np.zeros(df.shape[0])

    cntr = 0
    for idx, row in df.iterrows():

        prev_date = (row['DATE'] - timedelta(days=1)).timestamp()
        prev_day_of_week = (row['DATE'] - timedelta(days=1)).dayofweek
        curr_date = row['DATE'].timestamp()

        # find mean value by day. If doesn't exist, use customer/upc mean
        #mean_val = means_by_day.get((row['CUSTOMER'], row["UPC"], prev_day_of_week), np.nan)
        #if np.isnan(mean_val):
            #mean_val = means[(row['CUSTOMER'], row["UPC"])]
        mean_val = means[(row['CUSTOMER'], row["UPC"])]
        mean_val = means_by_day.get((row['CUSTOMER'], row["UPC"]), mean_val)

        prev_sales_i = prev_buff.get((row['CUSTOMER'], row["UPC"], prev_date), mean_val)
        prev_sales[cntr] = prev_sales_i

        prev_buff[row['CUSTOMER'], row["UPC"], curr_date] = row["SALES"]


        cntr += 1


    df['PREV_SALES'] = prev_sales
    return df



def update_product_state(arr):

    scan = True
    while scan:

        for i in range(len(arr)):
            flip = np.random.random()

            if flip < FLIP_PROB:
                arr[i] = 1 - arr[i]

        if arr.sum() > 0:
            scan = False

    return arr


def init_state(mtx):
    dims = mtx.shape

    for j in range(dims[1]):
        for i in range(dims[0]):

            r = mtx[:, j]
            r_update = update_product_state(r)
            mtx[:, j] = r_update

    return mtx

def get_adj_mtx(fname):
    with open(fname) as f:
        adj_store = json.load(f)
    adj_store['non_zero_entries'] = ast.literal_eval(adj_store['non_zero_entries'])
    A = make_bin_mtx(adj_store['non_zero_entries'], (adj_store['n_regions'],  adj_store['n_regions']))
    A = A + np.eye(adj_store['n_regions'])
    return A

# Import store, sales data
stores = pd.read_csv("store-level-data.csv")
stores['DATE'] = pd.to_datetime(stores['DATE'])
#stores['day_of_week'] = stores['DATE'].dt.dayofweek
stores = stores[stores['SALES'] > 0.0]
# log of sales
stores['SALES'] = np.log(stores['QUANTITY']*stores['PRICE'])
## Standadarze sales data ([x - mu] / sd)
#stores['SALES'] = (stores['SALES'] - stores['SALES'].mean()) / np.std(stores['SALES'])
#stores['SALES_2'] = np.power(stores["SALES"], 2)
stores['day_of_week'] = stores['DATE'].dt.dayofweek

mean_sales = stores[["CUSTOMER", "UPC", "SALES"]].groupby(["CUSTOMER", "UPC"]).mean()
mean_sales = mean_sales['SALES'].to_dict()

mean_sales_day = stores[["CUSTOMER", "UPC", "day_of_week", "SALES"]].groupby(["CUSTOMER", "UPC", "day_of_week"]).mean()
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
top_prods = stores[['UPC', 'SALES']].groupby('UPC').sum().sort_values("SALES", ascending=False)[:N_PRODUCTS].reset_index()
PROD_SET = top_prods["UPC"].values

global prod_to_idx
prod_to_idx = dict(zip(PROD_SET, range(N_PRODUCTS)))

# Regions per store
n_regions = dict(zip(STORE_SET, [18, 5]))

# Adjacency Matrices

adj = {STORE_SET[0]: get_adj_mtx("store-1-adj-mtx.json"),
       STORE_SET[1]: get_adj_mtx("store-2-adj-mtx.json")}

for k, a in adj.items():
    is_symmetric = np.allclose(a, a.transpose())
    assert is_symmetric

# Heterogeneous spatial weights
gamma = 50
priors = {STORE_SET[0]: {'loc': np.ones(n_regions[STORE_SET[0]])*25,
                         'scale': adj[STORE_SET[0]]*gamma},
          STORE_SET[1]: {'loc': np.ones(n_regions[STORE_SET[1]])*25,
                         'scale':  adj[STORE_SET[1]]*gamma}}


weights = {STORE_SET[0]: np.random.multivariate_normal(mean=priors[STORE_SET[0]]['loc'],
                                                       cov=priors[STORE_SET[0]]['scale'],
                                                       size=N_PRODUCTS).transpose(),
           STORE_SET[1]: np.random.multivariate_normal(mean=priors[STORE_SET[1]]['loc'],
                                                       cov=priors[STORE_SET[1]]['scale'], size=N_PRODUCTS).transpose()}


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

def normalize(x):
    x_norm = x / x.sum()
    return x_norm


def estimate_spatial_q(df, cust):
    new_cols = ['Q_R', 'REGION'] + list(df.columns)
    n_rows = df.shape[0]*n_regions[cust]
    n_cols = len(new_cols)
    mtx = np.zeros((n_rows, n_cols), dtype=object)


    cntr = 0
    start_idx = 0
    end_idx = start_idx + n_regions[cust]
    for idx, row in df.iterrows():
        prod_idx = prod_to_idx[row["UPC"]]
        try:
            w = weights[cust][:, prod_idx]
        except:
            print(prod_idx)
            print(row["UPC"])
            print(prod_to_idx)
        state_vec = state[cust][:, prod_idx]
        w = normalize(w * state_vec)
        updated_state = update_product_state(state_vec)
        state[cust][:, prod_idx] = updated_state

        q = w*row['QUANTITY']

        mtx[start_idx:end_idx, 0] = q.round()
        mtx[start_idx:end_idx, 1] = range(n_regions[cust])
        mtx[start_idx:end_idx, 2:] = row.values

        cntr += 1
        start_idx += n_regions[cust]
        end_idx += n_regions[cust]

    return pd.DataFrame(mtx, columns=new_cols)

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


def update_features(df):

    # compute correct sales
    df['SALES'] = df['Q_R']*df['PRICE']
    # drop na - prev_sales
    df.dropna(inplace=True)


    # get product codes
    prod_map = {}
    prods = np.zeros(df.shape[0])
    prod_cntr = 0
    row_cntr = 0
    for idx, row in df.iterrows():
        # products
        if row['UPC'] in prod_map:
            prod_idx = prod_map[row['UPC']]
        else:
            prod_idx = prod_cntr
            prod_map[row['UPC']] = prod_cntr
            prod_cntr += 1

        prods[row_cntr] = prod_idx
        row_cntr += 1
    df['product'] = prods
    # Rename features
    df.drop(labels=['PROMO', 'QUANTITY'], axis=1, inplace=True)
    df.rename(columns={'Q_R': 'quantity',
               'REGION': 'region',
               'CUSTOMER': 'store_id',
               'DATE': 'date',
               'PRICE': 'price',
               'SALES': 'sales',
               'PREV_SALES': 'prev_sales', 'PREV_SALES_2': 'prev_sales_'}, inplace=True)


    return df

store1 = get_prev_sales(store1, means=mean_sales, means_by_day=mean_sales_day)
store2 = get_prev_sales(store2, means=mean_sales, means_by_day=mean_sales_day)

store1_clean = update_features(estimate_spatial_q(store1, STORE_SET[0]))
store2_clean = update_features(estimate_spatial_q(store2, STORE_SET[1]))

store1_clean = update_timestamps(store1_clean)
store2_clean = update_timestamps(store2_clean)

# Split train/test

def split(df, train_pct=.8):
    dates = df['time'].unique()
    train_size = int(train_pct * len(dates))
    train_idx = dates[:train_size]
    test_idx = dates[train_size:]

    train = df[df["time"].isin(train_idx)]
    test = df[df["time"].isin(test_idx)]

    return train, test

store1_train, store1_test = split(store1_clean, train_pct=TRAIN_DATA_PCT)
store2_train, store2_test = split(store2_clean, train_pct=TRAIN_DATA_PCT)

store1_train = update_timestamps(store1_train)
store1_test = update_timestamps(store1_test)

store2_train = update_timestamps(store2_train)
store2_test = update_timestamps(store2_test)



store1_train.to_csv("store-1-train.csv")
store2_train.to_csv("store-2-train.csv")

store1_test.to_csv("store-1-test.csv")
store2_test.to_csv("store-2-test.csv")

store1_clean.to_csv("store-1-all.csv")
store2_clean.to_csv("store-2-all.csv")