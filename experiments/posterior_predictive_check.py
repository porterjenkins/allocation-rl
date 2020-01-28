import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.features import Features
from envs.prior import Prior
import config.config as cfg
from envs.allocation_env import AllocationEnv
import numpy as np
import matplotlib.pyplot
import pandas as pd
from utils import mae, rmse, check_draws_inf, mape
from experiments.plot import plot_total_ppc


def trim_draws(X):
    n = X.shape[0]
    for i in range(n):
        x_i = X[i, :]
        lower, upper = np.quantile(x_i, q=[0.05, 0.95])
        X[i, :] = np.clip(x_i, a_min=lower, a_max=upper)

    return X

train_data = pd.read_csv(cfg.vals['train_data'])
train_data = train_data[train_data.quantity > 0.0]
train_data['date'] = pd.to_datetime(train_data['date'])
train_data_features = Features.feature_extraction(train_data, y_col='quantity')

y_train = train_data['sales'].values

test_data = pd.read_csv(cfg.vals['test_data'])
test_data = test_data[test_data.quantity > 0.0]
test_data['date'] = pd.to_datetime(test_data['date'])
test_data_features = Features.feature_extraction(test_data, y_col='quantity')

y_test = test_data['sales'].values


prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
env.reset()

# Training Data
y_hat_draws_train = env._predict(features=train_data_features, n_samples=100)
y_hat_draws_train = check_draws_inf(y_hat_draws_train)
y_hat = y_hat_draws_train.mean(axis=0)
y_hat_draws_train = y_hat_draws_train.transpose()

train_data['y_hat'] = y_hat_draws_train.mean(axis=1)
#test_data['y_hat'] = np.median(y_hat_draws, axis=1)
train_data['y_hat_upper'] = np.percentile(y_hat_draws_train, q=95.0, axis=1)
train_data['y_hat_lower'] = np.percentile(y_hat_draws_train, q=5.0, axis=1)



train_mae = mae(y_hat, y_train)
train_rmse = rmse(y_hat, y_train)
train_mape = mape(y_hat, y_train)


# Test Data
y_hat_draws_test = env._predict(features=test_data_features, n_samples=100)
y_hat_draws_test = check_draws_inf(y_hat_draws_test)
y_hat_draws_test = y_hat_draws_test.transpose()

test_data['y_hat'] = y_hat_draws_test.mean(axis=1)
#test_data['y_hat'] = np.median(y_hat_draws, axis=1)
test_data['y_hat_upper'] = np.percentile(y_hat_draws_test, q=95.0, axis=1)
test_data['y_hat_lower'] = np.percentile(y_hat_draws_test, q=5.0, axis=1)

test_mae = mae(test_data.y_hat, y_test)
test_rmse = rmse(test_data.y_hat, y_test)
test_mape = mape(test_data.y_hat, y_test)

print("MAE - (test):  {:.2f}".format(test_mae))
print("RMSE - (test):  {:.2f}".format(test_rmse))
print("MAPE: - (test):  {:.4f}".format(test_mape))



# product-level errors:
test_data["err"] = test_data.sales - test_data.y_hat
prod_errors = test_data[["product", "err"]].groupby("product").agg(lambda x: np.mean(np.abs(x)))
print(prod_errors)


# product-level errors:
"""prod_errors = test_data[["product", "sales", "y_hat"]].groupby("product").sum()
prod_errors["err"] = prod_errors.sales - prod_errors.y_hat

test_mae = mae(prod_errors.sales, prod_errors.y_hat)
test_rmse = rmse(prod_errors.sales, prod_errors.y_hat)
test_mape = mape(prod_errors.sales, prod_errors.y_hat)

print("product-level errors:")
print("MAE - (test):  {:.2f}".format(test_mae))
print("RMSE - (test):  {:.2f}".format(test_rmse))
print("MAPE: - (test):  {:.2f}".format(test_mape))"""

prod_errors = test_data[['region', 'time', 'sales', 'y_hat']].groupby(['time', "region"]).sum()
prod_mae = mae(prod_errors.y_hat, prod_errors.sales)
prod_rmse = rmse(prod_errors.y_hat, prod_errors.sales)
prod_mape = mape(prod_errors.y_hat, prod_errors.sales)
print("Region MAE - (test):  {:.2f}".format(prod_mae))
print("Region RMSE - (test):  {:.2f}".format(prod_rmse))
print("Region MAPE - (test):  {:.4f}".format(prod_mape))

all_data = pd.concat([train_data, test_data], axis=0)
all_draws = np.concatenate([y_hat_draws_train, y_hat_draws_test],axis=0)

test_date = test_data.iloc[0]['date']

plot_total_ppc(test_data, pd.DataFrame(y_hat_draws_test), fname="figs/total-ppc-{}".format(cfg.vals['prj_name']))
