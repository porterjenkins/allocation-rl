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
from utils import mae, rmse, check_draws_inf
from experiments.plot import plot_total_ppc

train_data = pd.read_csv(cfg.vals['train_data'])
train_data = train_data[train_data.quantity > 0.0]
train_data_features = Features.feature_extraction(train_data, y_col='quantity')

y_train = train_data['sales'].values

test_data = pd.read_csv(cfg.vals['test_data'])
test_data = test_data[test_data.quantity > 0.0]
#test_data = test_data[test_data.time > 3.0]
test_data_features = Features.feature_extraction(test_data, y_col='quantity')

y_test = test_data['sales'].values


prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)

# Training Data
y_hat_draws = env._predict(features=train_data_features, n_samples=100)
y_hat_draws = check_draws_inf(y_hat_draws)
y_hat = y_hat_draws.mean(axis=0)
train_mae = mae(y_hat, y_train)
train_rmse = rmse(y_hat, y_train)

print("MAE - (train): {}".format(train_mae))
print("RMSE - (train): {}".format(train_rmse))

# Test Data
y_hat_draws = env._predict(features=test_data_features, n_samples=100)
y_hat_draws = check_draws_inf(y_hat_draws)
y_hat_draws = y_hat_draws.transpose()
test_data['y_hat'] = y_hat_draws.mean(axis=1)
test_data['y_hat_upper'] = np.percentile(y_hat_draws, q=95.0, axis=1)
test_data['y_hat_lower'] = np.percentile(y_hat_draws, q=5.0, axis=1)

test_mae = mae(test_data.y_hat, y_test)
test_rmse = rmse(test_data.y_hat, y_test)

print("MAE - (test): {}".format(test_mae))
print("RMSE - (test): {}".format(test_rmse))

plot_total_ppc(test_data, pd.DataFrame(y_hat_draws), fname="figs/total-ppc-{}".format(cfg.vals['prj_name']))
