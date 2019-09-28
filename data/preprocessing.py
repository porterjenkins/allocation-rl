import pandas as pd
import numpy as np

N_PRODUCTS = 15
FLIP_PROB = 0.05

def get_prev_sales(df):

    prev_buff = {}
    prev_sales = np.zeros(df.shape[0])

    cntr = 0
    for idx, row in df.iterrows():

        prev_sales_i = prev_buff.get(row["UPC"], np.nan)
        prev_sales[cntr] = prev_sales_i

        prev_buff[row["UPC"]] = row["SALES"]

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


# Import store, sales data
stores = pd.read_csv("store-level-data.csv")
stores['DATE'] = pd.to_datetime(stores['DATE'])
#stores['day_of_week'] = stores['DATE'].dt.dayofweek
stores = stores[stores['SALES'] > 0.0]
stores['SALES'] = stores['QUANTITY']*stores['PRICE']


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

adj = {STORE_SET[0]: np.eye(n_regions[STORE_SET[0]]),
       STORE_SET[1]: np.eye(n_regions[STORE_SET[1]])}

# Heterogeneous spatial weights
gamma = 25
priors = {STORE_SET[0]: {'loc': np.ones(n_regions[STORE_SET[0]])*25,
                         'scale': adj[STORE_SET[0]]*gamma},
          STORE_SET[1]: {'loc': np.ones(n_regions[STORE_SET[1]])*10,
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


def update_sales(df):
    df['SALES'] = df['Q_R']*df['PRICE']
    return df

def rename_features(df):
    df.drop(labels=['PROMO', 'QUANTITY'], axis=1, inplace=True)
    df.rename(columns={'Q_R': 'quantity',
               'REGION': 'region',
               'CUSTOMER': 'store_id',
               'DATE': 'time',
               'UPC': 'product',
               'PRICE': 'price',
               'SALES': 'sales',
               'PREV_SALES': 'prev_sales'}, inplace=True)

    return df

store1 = get_prev_sales(store1)
store2 = get_prev_sales(store2)

store1_clean = update_sales(estimate_spatial_q(store1, STORE_SET[0]))
store2_clean = update_sales(estimate_spatial_q(store2, STORE_SET[1]))

store1_clean = rename_features(store1_clean)
store2_clean = rename_features(store2_clean)
# Split train/test

def split(df, train_pct=.8):
    dates = df['time'].unique()
    train_size = int(train_pct * len(dates))
    train_idx = dates[:train_size]
    test_idx = dates[train_size:]

    train = df[df["time"].isin(train_idx)]
    test = df[df["time"].isin(test_idx)]

    return train, test

store1_train, store1_test = split(store1_clean)
store2_train, store2_test = split(store2_clean)

store1_train.to_csv("store-1-train.csv")
store2_train.to_csv("store-2-train.csv")

store1_test.to_csv("store-1-test.csv")
store2_test.to_csv("store-2-test.csv")