import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime, timedelta



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
    arr_update = arr.copy()
    scan = True
    while scan:

        for i in range(len(arr)):
            flip = np.random.random()

            if flip < FLIP_PROB:
                arr_update[i] = 1 - arr[i]


        if arr_update.sum() > 0:
            scan = False

    actions = arr - arr_update
    return arr_update, actions


def init_state(mtx):
    dims = mtx.shape

    for j in range(dims[1]):
        for i in range(dims[0]):

            r = mtx[:, j]
            r_update, _ = update_product_state(r)
            mtx[:, j] = r_update

    return mtx

def get_adj_mtx(fname):
    with open(fname) as f:
        adj_store = json.load(f)
    adj_store['non_zero_entries'] = ast.literal_eval(adj_store['non_zero_entries'])
    A = make_bin_mtx(adj_store['non_zero_entries'], (adj_store['n_regions'],  adj_store['n_regions']))
    A = A + np.eye(adj_store['n_regions'])
    return A


def get_year_lookback(df, filter_date):
    ts = df[['DATE','CUSTOMER','UPC','SALES']]
    ts['year'] = df['DATE'].dt.year
    ts['week'] = df['DATE'].dt.week

    means = ts.groupby(['CUSTOMER', 'UPC', 'year', 'week']).mean().reset_index()
    means.rename(columns={'SALES': "SALES_PREV_YR", 'year': 'year_prev', 'week': 'week_prev'}, inplace=True)
    df['DATE_PREV'] = df['DATE'] - timedelta(days=365)
    df['year_prev'] = df['DATE_PREV'].dt.year
    df['week_prev'] = df['DATE_PREV'].dt.week

    out = pd.merge(df, means, left_on=['CUSTOMER', 'UPC','year_prev', 'week_prev'], right_on=['CUSTOMER', 'UPC', 'year_prev', 'week_prev'], how='left')
    filter_date = pd.to_datetime(filter_date)
    out = out[out['DATE'] >= filter_date]

    prod_means = ts[['CUSTOMER', 'UPC','SALES']].groupby(['CUSTOMER', 'UPC']).mean()
    prod_means.rename(columns={'SALES': "SALES_MEAN"}, inplace=True)

    out = pd.merge(out, prod_means, on = ['CUSTOMER', 'UPC'])
    out['SALES_PREV_YR'] = np.where(np.isnan(out['SALES_PREV_YR']), out['SALES_MEAN'], out['SALES_PREV_YR'])

    return out.sort_values(by='DATE')





def normalize(x):
    x_norm = x / x.sum()
    return x_norm


def estimate_spatial_q(df, cust, n_regions, weights):
    new_cols = ['Q_R', 'REGION'] + list(df.columns)
    n_rows = df.shape[0]*n_regions[cust]
    n_cols = len(new_cols)
    mtx = np.zeros((n_rows, n_cols), dtype=object)
    action_mtx = np.zeros((n_rows, n_regions[cust]))

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
        updated_state, action = update_product_state(state_vec)
        state[cust][:, prod_idx] = updated_state

        q = w*row['QUANTITY']

        mtx[start_idx:end_idx, 0] = q.round()
        mtx[start_idx:end_idx, 1] = range(n_regions[cust])
        mtx[start_idx:end_idx, 2:] = row.values
        action_mtx[start_idx:end_idx, :] = action

        cntr += 1
        start_idx += n_regions[cust]
        end_idx += n_regions[cust]

    df = pd.DataFrame(mtx, columns=new_cols)
    action_df = pd.DataFrame(action_mtx, columns=["action_region_{}".format(x) for x in range(n_regions[cust])])

    return df, action_df

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
                'SALES_PREV_YR': 'sales_prev_yr',
               'SALES': 'sales',
               'PREV_SALES': 'prev_sales', 'PREV_SALES_2': 'prev_sales_'}, inplace=True)


    return df

def split(df, A, train_pct=.8):
    dates = df['time'].unique()
    train_size = int(train_pct * len(dates))

    train_idx = dates[:train_size]
    test_idx = dates[train_size:]

    train = df[df["time"].isin(train_idx)]
    test = df[df["time"].isin(test_idx)]

    A_train = A.loc[train.index]
    A_test = A.loc[test.index]

    return train, test, A_train, A_test


def gen_weights(store_set, priors, n_products):
    weights = {store_set[0]: np.random.multivariate_normal(mean=priors[store_set[0]]['loc'],
                                                           cov=priors[store_set[0]]['scale'],
                                                           size=n_products).transpose(),
               store_set[1]: np.random.multivariate_normal(mean=priors[store_set[1]]['loc'],
                                                           cov=priors[store_set[1]]['scale'],
                                                           size=n_products).transpose()}

    return weights


def init_prior(store_set, adj, n_regions, gamma=50):

    priors = {store_set[0]: {'loc': np.ones(n_regions[store_set[0]]) * 25,
                             'scale': adj[store_set[0]] * gamma},
              store_set[1]: {'loc': np.ones(n_regions[store_set[1]]) * 25,
                             'scale': adj[store_set[1]] * gamma}}

    return priors