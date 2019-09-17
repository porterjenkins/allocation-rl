import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import pandas as pd
import json
from plot import plot_posterior_predictive_check


with open('config.json') as f:
    config = json.load(f)

config['adj_mtx'] = np.eye(config['n_regions'])

data = pd.read_csv('test-data-simple.csv', index_col=0)
data['day_of_week'] = data['time'] % 7

day_features_grouped = data[['time', 'day_of_week']].groupby('time').max()

region_features = pd.get_dummies(data.region, prefix='region')
product_features = pd.get_dummies(data['product'], prefix='product')
day_features = pd.get_dummies(day_features_grouped['day_of_week'], prefix='day')


X_train = pd.concat([region_features, product_features], axis=1)

X = X_train.values.astype(theano.config.floatX)
y = data['quantity'].values.astype(theano.config.floatX)

X_region = region_features.values.astype(theano.config.floatX)
X_product = product_features.values.astype(theano.config.floatX)
X_temporal = day_features.values.astype(theano.config.floatX)
X_lagged = data['prev_sales'].values.astype(theano.config.floatX)
prices = np.dot(product_features.values, config['prices']).reshape(1,-1)



N_PRODUCTS = config['n_products']
N_REGIONS = config['n_regions']
GAMMA = 100
N_QUANTITY_FEATURES = N_REGIONS + N_PRODUCTS
N_TEMPORAL_FEATURES = 7
N = X_train.shape[0]
T = len(data['time'].unique())
TIME_STAMPS = data['time'].values.astype(int)


with pm.Model() as env_model:


    # Prior for region weights
    prior_loc_w_r = np.ones(N_REGIONS)*50
    prior_scale_w_r = config['adj_mtx']*GAMMA
    # Generate region weights
    w_r = pm.MvNormal('w_r', mu=prior_loc_w_r, cov=prior_scale_w_r, shape=N_REGIONS)

    # Prior for product weights
    prior_loc_w_p = np.ones(N_PRODUCTS)*10
    prior_scale_w_r = np.eye(N_PRODUCTS)*25
    # Generate Product weights
    w_p = pm.MvNormal('w_p', mu=prior_loc_w_p, cov=prior_scale_w_r, shape=N_PRODUCTS)

    # Prior for customer weight
    prior_loc_w_c = 100
    prior_scale_w_c = 20
    # Generate customer weight
    w_c = pm.Normal('w_c', mu=prior_loc_w_c, sigma=prior_scale_w_c)

    # prior for previous sales (s_t-1)
    prior_loc_w_s = .1
    prior_scale_w_s = .25
    # Generate customer weight
    w_s = pm.Gamma('w_s', mu=prior_loc_w_s, sigma=prior_scale_w_s)


    # Prior for temporal weights
    prior_loc_w_t = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 5.0, 7.0])
    prior_scale_w_t = np.eye(N_TEMPORAL_FEATURES) * 10
    # Generate temporal weights
    w_t = pm.MvNormal('w_t', mu=prior_loc_w_t, cov=prior_scale_w_t, shape=N_TEMPORAL_FEATURES)
    #print("w_t", w_t.tag.test_value)
    lambda_c_t = pm.math.dot(X_temporal, w_t.T)
    #print("lambda_c_t", lambda_c_t.tag.test_value)
    #lambda_c_t = 5.0
    c_t = pm.Poisson("customer_t", mu=lambda_c_t, shape=T)
    #print("c_t", c_t.tag.test_value)


    c_all = c_t[TIME_STAMPS] * w_c
    #print("c_all: ", c_all.tag.test_value.shape)

    lambda_q = pm.math.dot(X_region, w_r.T) + pm.math.dot(X_product, w_p.T) + c_all + w_s*X_lagged
    #print("lambda_q", lambda_q.tag.test_value.shape)

    q_ij = pm.Poisson('quantity_ij', mu=lambda_q, observed=y)


"""for RV in env_model.basic_RVs:
    try:
        print(RV.name, RV.logp(env_model.test_point), RV.dshape)
    except AttributeError:
        print(RV.name, RV.logp(env_model.test_point))
"""

with env_model:
    trace = pm.sample(1000, tune=1000, init='advi+adapt_diag')
    posterior_pred = pm.sample_posterior_predictive(trace)
    #mean_field = pm.fit(method='advi')
    #posterior_pred = pm.sample_posterior_predictive(mean_field)


y_hat = posterior_pred['quantity_ij'].mean(axis=0)
sales = posterior_pred['quantity_ij'] * prices
y_hat_sales = sales.mean(axis=0)
data['sales_pred_upper'] = np.percentile(sales, q=95.0, axis=0)
data['sales_pred_lower'] = np.percentile(sales, q=5.0, axis=0)

data['sales_pred'] = y_hat_sales.flatten()

print(data.head())


err = y-y_hat
mse = np.mean(np.abs((y - y_hat)))

print("mse: {}".format(mse))


mse_sales = np.mean(np.abs((data.sales.values - y_hat_sales)))
print("sales mse: {}".format(mse_sales))


data.to_csv("model-output.csv")

plt.figure(figsize=(7, 7))
pm.traceplot(trace[::10], var_names=['w_s', 'w_p','w_c','w_r'])
plt.savefig("trace-plot.pdf")
plt.clf()
plt.close()

plt.figure(figsize=(7, 7))
pm.traceplot(trace[::10], var_names=['w_t'])
plt.savefig("trace-plot-temporal.pdf")
