import pandas as pd
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json


ASIN_FNAME = "../data/top_k_upc_asin.json"
AMAZON_COLS = ['asin', 'user_id', 'rating']
UPC_IDX_MAP = "../data/product_idx_map.json"

def z_score(arr):
    mu = np.mean(arr)
    sigma = np.std(arr)

    return (arr - mu) / sigma

def is_pos_def(x):
    eigen_vals = np.linalg.eigvals(x)
    return np.all(eigen_vals > 0)

def get_distance(X, theta=1):
    return np.exp(-X/theta)

def ones_diag(X):
    for i in range(X.shape[0]):
        X[i, i] = 1.0
    return X


def cov_weighted_sum(dist_mat, alpha, theta1, theta2):

    Sigma = alpha*get_distance(dist_mat,theta1) + (1-alpha)*get_distance(dist_mat,theta2)
    return Sigma


with open(ASIN_FNAME, 'r') as f:

    asin_upc_map = json.load(f)

with open(UPC_IDX_MAP, 'r') as f:
    upc_idx_map = json.load(f)




n_products = len(upc_idx_map)



pantry = pd.read_csv(cfg.vals['amazon_dir'] + "Prime_Pantry.csv", header=None)
pantry.columns = AMAZON_COLS



grocery = pd.read_csv(cfg.vals['amazon_dir'] + "Grocery_and_Gourmet_Food.csv", header=None)
grocery.columns = AMAZON_COLS


df = pd.concat([pantry, grocery], axis=0)

min_max = MinMaxScaler(feature_range=(0,1))
#df['rating'] = min_max.fit_transform(df['rating'].values.reshape(-1,1))


df = df[df['asin'].isin(list(asin_upc_map.values()))]


c_matrix_init = np.random.uniform(0, .05, size=n_products**2)
c_matrix_init = c_matrix_init.reshape((n_products, n_products))




count_matrix = np.zeros((n_products, n_products))
max_score = 5.0


asin_to_idx = {}
idx_to_asin = {}

for upc, upc_id in upc_idx_map.items():

    asin = asin_upc_map[upc]

    asin_to_idx[asin] = upc_id
    idx_to_asin[upc_id] = asin


basket_counts = df.groupby('asin').count()

D = np.zeros((n_products, n_products)) + np.random.uniform(0, .05)
# seed matrix with counts

for idx, row in basket_counts.iterrows():
    for idx2, row2 in basket_counts.iterrows():

        if idx == idx2:
            continue
        else:
            i = asin_to_idx[idx]
            j = asin_to_idx[idx2]

            #c_matrix_init[i,j] = np.log(row['rating'] + row2['rating'])
            D[i, j] = np.sqrt(row['rating'] * row2['rating'])


c_matrix = np.zeros((n_products, n_products)) + c_matrix_init
Z = np.zeros((n_products, n_products)) + c_matrix_init



for user_id, user_data in df.groupby(['user_id']):
    for idx1, row1 in user_data.iterrows():
        for idx2, row2 in user_data.iterrows():
            if row1['asin'] != row2['asin']:

                weighted_rating = row1.rating*row2.rating
                c_matrix[asin_to_idx[row1.asin], asin_to_idx[row2.asin]] += weighted_rating

                count_matrix[asin_to_idx[row1.asin], asin_to_idx[row2.asin]] += 1.0

            Z[asin_to_idx[row1.asin], asin_to_idx[row2.asin]] += c_matrix_init[asin_to_idx[row1.asin], asin_to_idx[row2.asin]] + max_score ** 2

n_user = len(df.user_id.unique())

c_is_pos_def = is_pos_def(c_matrix)

"""c_matrix_plus_eps = c_matrix + np.random.uniform(0, 10)
c_is_pos_def = is_pos_def(c_matrix_plus_eps)

Z = n_user*max_score*max_score
#c_matrix = z_score(c_matrix)
c_is_pos_def = is_pos_def(c_matrix)
# set diag to 1
c_matrix = c_matrix - np.diag(np.diag(c_matrix)) + np.eye(n_products)
c_is_pos_def = is_pos_def(c_matrix)"""

# Experiment with normalized score as distance metric
c_mat_norm = c_matrix / D
c_mat_norm = ones_diag(c_mat_norm)
print(c_mat_norm)
print("is pos. def: {}".format(is_pos_def(c_mat_norm)))


np.savetxt('../data/item-covariance.txt', c_mat_norm)

with open("../data/item-covariance-idx-map.json", 'w') as f:
    ids_as_str = {}
    for k, v in asin_to_idx.items():
        ids_as_str[k] = str(v)
    json.dump(ids_as_str, f)



dist_mat = 1 - c_mat_norm



#c_mat_norm = get_distance(c_mat_norm, theta=1)

#for i in range(n_products):
#    c_mat_norm[i,i]=1.0




c_is_pos_def = is_pos_def(c_mat_norm)

x_i = np.random.multivariate_normal(np.zeros(n_products), cov=c_mat_norm)
print(x_i)





"""
theta1 = 1
theta2 = 1
M = 100
m = np.sqrt(theta2/theta1)

lower_bound = 1 - max(1, M)
upper_bound = 1 - np.max([1, m])

#alpha = np.random.uniform(low=lower_bound, high=upper_bound)



Sigma_with_alpha = cov_weighted_sum(dist_mat, alpha=alpha, theta1=2, theta2=1)
Sigma_with_alpha = ones_diag(Sigma_with_alpha)

print(Sigma_with_alpha)
print(is_pos_def(Sigma_with_alpha))
"""