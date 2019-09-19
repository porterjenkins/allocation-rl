import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import pandas as pd
import json
from plot import plot_posterior_predictive_check

train_data = pd.read_csv('train-data-simple.csv', index_col=0)
test_data = pd.read_csv('test-data-simple.csv', index_col=0,nrows=16)

def feature_extraction(df):
    df['day_of_week'] = df['time'] % 7

    day_features_grouped = df[['time', 'day_of_week']].groupby('time').max()

    region_features = pd.get_dummies(df.region, prefix='region')
    product_features = pd.get_dummies(df['product'], prefix='product')
    day_features = pd.get_dummies(day_features_grouped['day_of_week'], prefix='day')

    features = {}

    features['region'] = region_features.values.astype(theano.config.floatX)
    features['product'] = product_features.values.astype(theano.config.floatX)
    features['temporal'] = day_features.values.astype(theano.config.floatX)
    features['lagged'] = df['prev_sales'].values.astype(theano.config.floatX)
    features['prices'] = np.dot(product_features.values, config['prices']).reshape(1, -1)

    return features

def get_target(df):
    y = df['quantity'].values.astype(theano.config.floatX)
    return y

with open('config/config.json') as f:
    config = json.load(f)
config['adj_mtx'] = np.eye(config['n_regions'])


N_PRODUCTS = config['n_products']
N_REGIONS = config['n_regions']
GAMMA = 100
N_QUANTITY_FEATURES = N_REGIONS + N_PRODUCTS
N_TEMPORAL_FEATURES = 7
N = train_data.shape[0]
T = len(train_data['time'].unique())
TIME_STAMPS = train_data['time'].values.astype(int)

train_features = feature_extraction(train_data)
test_features = feature_extraction(test_data)

y_train = get_target(train_data)
y_test = get_target(test_data)

def build_env_model(X_region, X_product, X_temporal, X_lagged, y=None):

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

    return env_model


X_region = theano.shared(train_features['region'])
X_product = theano.shared(train_features['product'])
X_temporal = theano.shared(train_features['temporal'])
X_lagged = theano.shared(train_features['lagged'])
y = theano.shared(y_train)

env_model = build_env_model(X_region, X_product, X_temporal, X_lagged, y)


with env_model:
    trace = pm.sample(100, tune=100, init='advi+adapt_diag')
    posterior_pred_train = pm.sample_posterior_predictive(trace)
    #mean_field = pm.fit(method='advi')
    #posterior_pred = pm.sample_posterior_predictive(mean_field)


y_hat = posterior_pred_train['quantity_ij'].mean(axis=0)
sales = posterior_pred_train['quantity_ij'] * train_features['prices']
y_hat_sales = sales.mean(axis=0)
train_data['sales_pred_upper'] = np.percentile(sales, q=95.0, axis=0)
train_data['sales_pred_lower'] = np.percentile(sales, q=5.0, axis=0)

train_data['sales_pred'] = y_hat_sales.flatten()

print(train_data.head())


err = y-y_hat
mse = np.mean(np.abs((y_train - y_hat)))

print("mse: {}".format(mse))


mse_sales = np.mean(np.abs((train_data.sales.values - y_hat_sales)))
print("sales mse: {}".format(mse_sales))


"""plt.figure(figsize=(7, 7))
pm.traceplot(trace[::10], var_names=['w_s', 'w_p','w_c','w_r'])
plt.savefig("trace-plot.png")
plt.clf()
plt.close()

plt.figure(figsize=(7, 7))
pm.traceplot(trace[::10], var_names=['w_t'])
plt.savefig("trace-plot-temporal.png")"""

## Test data ##

X_region.set_value(test_features['region'])
X_product.set_value(test_features['product'])
X_temporal.set_value(test_features['temporal'])
X_lagged.set_value(test_features['lagged'])
#y.set_value(y_test)

with env_model:
    posterior_pred_test = pm.sample_posterior_predictive(trace, samples=50)

y_hat = posterior_pred_test['quantity_ij'].mean(axis=0)
sales = posterior_pred_test['quantity_ij'] * train_features['prices']
y_hat_sales = sales.mean(axis=0)
test_data['sales_pred_upper'] = np.percentile(sales, q=95.0, axis=0)
test_data['sales_pred_lower'] = np.percentile(sales, q=5.0, axis=0)

test_data['sales_pred'] = y_hat_sales.flatten()

print(train_data.head())
err = y_test-y_hat
mse = np.mean(np.abs((y_test - y_hat)))

print("test mae: {}".format(mse))


mse_sales = np.mean(np.abs((test_data.sales.values - y_hat_sales)))
print("test sales mae: {}".format(mse_sales))


np.savetxt('sales-draws.csv', sales.transpose(), delimiter=',')


test_data.to_csv("model-output.csv")