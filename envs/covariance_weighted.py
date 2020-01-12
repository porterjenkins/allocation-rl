import pandas as pd
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
from sklearn.preprocessing import MinMaxScaler


ASIN_FNAME = "../data/top_k_upc_asin.json"
AMAZON_COLS = ['asin', 'user_id', 'rating']

def z_score(arr):
    mu = np.mean(arr)
    sigma = np.std(arr)

    return (arr - mu) / sigma

with open(ASIN_FNAME, 'r') as f:

    asin_upc_map = json.load(f)

n_products = len(asin_upc_map)



pantry = pd.read_csv(cfg.vals['amazon_dir'] + "Prime_Pantry.csv", header=None)
pantry.columns = AMAZON_COLS



grocery = pd.read_csv(cfg.vals['amazon_dir'] + "Grocery_and_Gourmet_Food.csv", header=None)
grocery.columns = AMAZON_COLS


df = pd.concat([pantry, grocery], axis=0)

min_max = MinMaxScaler(feature_range=(0,1))
df['rating'] = min_max.fit_transform(df['rating'].values.reshape(-1,1))


df = df[df['asin'].isin(list(asin_upc_map.values()))]


c_matrix = np.zeros((n_products, n_products))
asin_to_idx = dict(zip(list(asin_upc_map.values()), np.arange(n_products)))

for user_id, user_data in df.groupby(['user_id']):
    for idx1, row1 in user_data.iterrows():
        for idx2, row2 in user_data.iterrows():
            if row1['asin'] != row2['asin']:
                weighted_rating = row1.rating*row2.rating
                c_matrix[asin_to_idx[row1.asin], asin_to_idx[row2.asin]] += weighted_rating

c_matrix = z_score(c_matrix)
print(c_matrix)