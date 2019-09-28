import os
import sys
import numpy
import matplotlib.pyplot as plot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import config.config as cfg
from envs.features import Features

train_data = pandas.read_csv('../data/train-data-simple.csv')

train_data_features = Features.feature_extraction(train_data, prices=cfg.vals['prices'], y_col='quantity')

train_data_features.toarray()
