import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
from envs.features import Features
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from utils import mae, rmse, mape
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np



train_data = pd.read_csv(cfg.vals['train_data'])
train_data = train_data[train_data.quantity > 0.0]
train_data_features = Features.feature_extraction(train_data, y_col='quantity')
X_train = train_data_features.toarray()
y_train = train_data['sales'].values


test_data = pd.read_csv(cfg.vals['test_data'])
test_data = test_data[test_data.quantity > 0.0]
test_data_features = Features.feature_extraction(test_data, y_col='quantity')
X_test = test_data_features.toarray()
y_test = test_data['sales'].values


print("test data shape: {}".format(test_data.shape))

## Linear Regression

ols = LinearRegression(fit_intercept=True)
ols.fit(X_train, y_train)
y_hat = ols.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--OLS--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.4f}".format(test_mape))

prod_errors = test_data[['region', 'time', 'sales', 'y_hat']].groupby(['time', "region"]).sum()
prod_mae = mae(prod_errors.y_hat, prod_errors.sales)
prod_rmse = rmse(prod_errors.y_hat, prod_errors.sales)
prod_mape = mape(prod_errors.y_hat, prod_errors.sales)
print("Region MAE - (test):  {:.2f}".format(prod_mae))
print("Region RMSE - (test):  {:.2f}".format(prod_rmse))
print("Region MAPE - (test):  {:.4f}".format(prod_mape))




## Random Forest
rf = RandomForestRegressor(n_estimators=10)
rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--RF--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.4f}".format(test_mape))

prod_errors = test_data[['region', 'time', 'sales', 'y_hat']].groupby(['time', "region"]).sum()
prod_mae = mae(prod_errors.y_hat, prod_errors.sales)
prod_rmse = rmse(prod_errors.y_hat, prod_errors.sales)
prod_mape = mape(prod_errors.y_hat, prod_errors.sales)
print("Region MAE - (test):  {:.2f}".format(prod_mae))
print("Region RMSE - (test):  {:.2f}".format(prod_rmse))
print("Region MAPE - (test):  {:.4f}".format(prod_mape))


mlp = MLPRegressor(hidden_layer_sizes=(128, 64), solver='adam')
mlp.fit(X_train, y_train)
y_hat = mlp.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--MLP--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.4f}".format(test_mape))
#
#
#
prod_errors = test_data[['region', 'time', 'sales', 'y_hat']].groupby(['time', "region"]).sum()
prod_mae = mae(prod_errors.y_hat, prod_errors.sales)
prod_rmse = rmse(prod_errors.y_hat, prod_errors.sales)
prod_mape = mape(prod_errors.y_hat, prod_errors.sales)
print("Region MAE - (test):  {:.2f}".format(prod_mae))
print("Region RMSE - (test):  {:.2f}".format(prod_rmse))
print("Region MAPE - (test):  {:.4f}".format(prod_mape))


## GBRT
GBR = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=0, loss='ls')
GBR.fit(X_train, y_train)
y_hat = GBR.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--GBRT--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.4f}".format(test_mape))



prod_errors = test_data[['region', 'time', 'sales', 'y_hat']].groupby(['time', "region"]).sum()
prod_mae = mae(prod_errors.y_hat, prod_errors.sales)
prod_rmse = rmse(prod_errors.y_hat, prod_errors.sales)
prod_mape = mape(prod_errors.y_hat, prod_errors.sales)
print("Region MAE - (test):  {:.2f}".format(prod_mae))
print("Region RMSE - (test):  {:.2f}".format(prod_rmse))
print("Region MAPE - (test):  {:.4f}".format(prod_mape))


## KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_hat = knn.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--KNN--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.4f}".format(test_mape))


## SVR
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_rbf.fit(X_train, y_train)
y_hat = svr_rbf.predict(X_test)
test_data["y_hat"] = y_hat
test_mae = mae(y_hat, y_test)
test_rmse = rmse(y_hat, y_test)
test_mape = mape(y_hat, y_test)

print("--SVR--")
print("MAE - (test): {:.2f}".format(test_mae))
print("RMSE - (test): {:.2f}".format(test_rmse))
print("MAPE: - (test): {:.4f}".format(test_mape))
