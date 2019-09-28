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

train_data = pandas.read_csv(cfg.vals['train_data'])
train_data_features = Features.feature_extraction(train_data, y_col='quantity')

TRAIN_X = train_data_features.toarray()
TRAIN_Y = train_data_features.y

test_data = pandas.read_csv(cfg.vals['test_data'])
test_data_features = Features.feature_extraction(test_data, y_col='quantity')

TEST_X = test_data_features.toarray()
TEST_Y = test_data_features.y

print(TRAIN_Y)
print(TEST_Y)