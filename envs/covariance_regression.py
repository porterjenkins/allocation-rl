import pandas as pd
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



ASIN_FNAME = "../data/top_k_upc_asin.json"
AMAZON_COLS = ['asin', 'user_id', 'rating']


def compute_pariwise_mrs(grad):

    n = len(grad)
    mrs_mat = np.zeros((n, n))

    for i, g_i in enumerate(grad):
        for j, g_j in enumerate(grad):

            mrs_mat[i, j] = -(g_i / g_j)

    return mrs_mat

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
df = df[df['asin'].isin(list(asin_upc_map.values()))]


X = pd.get_dummies(df['asin'], prefix='product')
y = df['rating']


ols = LinearRegression(fit_intercept=False)
ols.fit(X, y)
y_hat = ols.predict(X)
rmse = np.sqrt(mean_squared_error(y_true=y, y_pred=y_hat))
print(rmse)
print(ols.coef_)


scaler = StandardScaler()
mrs = compute_pariwise_mrs(ols.coef_)
#mrs = scaler.fit_transform(mrs)
mrs = z_score(mrs)
print(mrs)
