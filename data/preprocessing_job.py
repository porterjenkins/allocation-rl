import pandas as pd
import numpy as np
import json
import ast
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder


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

def get_prev_sales(df, means, means_by_day):

    prev_buff = {}
    prev_sales = np.zeros(df.shape[0])

    cntr = 0
    for idx, row in df.iterrows():

        prev_date = (row['date'] - timedelta(days=1)).timestamp()
        prev_day_of_week = (row['date'] - timedelta(days=1)).dayofweek
        curr_date = row['date'].timestamp()

        # find mean value by day. If doesn't exist, use customer/upc mean
        #mean_val = means_by_day.get((row['CUSTOMER'], row["UPC"], prev_day_of_week), np.nan)
        #if np.isnan(mean_val):
            #mean_val = means[(row['CUSTOMER'], row["UPC"])]
        mean_val = means[(row['store_id'], row["UPC"])]
        mean_val = means_by_day.get((row['store_id'], row["UPC"]), mean_val)

        prev_sales_i = prev_buff.get((row['store_id'], row["UPC"], prev_date), mean_val)
        prev_sales[cntr] = prev_sales_i

        prev_buff[row['store_id'], row["UPC"], curr_date] = row["sales"]


        cntr += 1


    df['prev_sales'] = np.log(prev_sales)
    df['prev_sales'] = np.where(np.isinf(df['prev_sales']), 0, df["prev_sales"])
    return df

def split(df, train_pct=.8):
    dates = df['time'].unique()
    train_size = int(train_pct * len(dates))

    train_idx = dates[:train_size]
    test_idx = dates[train_size:]

    train = df[df["time"].isin(train_idx)]
    test = df[df["time"].isin(test_idx)]

    return train, test


store1 = pd.read_csv("store-1-raw.csv")
store2 = pd.read_csv("store-2-raw.csv")

store1['date'] = pd.to_datetime(store1['date'])

print(store1.head())


mean_sales_1 = store1[["store_id", "UPC", "sales"]].groupby(["store_id", "UPC"]).mean()
mean_sales_1 = mean_sales_1['sales'].to_dict()

mean_sales_day_1 = store1[["store_id", "UPC", "day_of_week", "sales"]].groupby(["store_id", "UPC", "day_of_week"]).mean()
mean_sales_day_1 = mean_sales_day_1["sales"].to_dict()


store1 = get_prev_sales(store1, mean_sales_1, mean_sales_day_1)

date_encoder = LabelEncoder()
store1["time"] = date_encoder.fit_transform(store1["date"])


store_1_train, store_1_test = split(store1)

store_1_train = update_timestamps(store_1_train)
store_1_test = update_timestamps(store_1_test)


store_1_train.to_csv("store-1-train.csv", index=False)
store_1_test.to_csv("store-1-test.csv", index=False)