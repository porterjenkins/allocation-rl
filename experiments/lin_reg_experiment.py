import os
import sys
import matplotlib.pyplot as plot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas
from sklearn.linear_model import LinearRegression
import config.config as cfg
from envs.features import Features


def learn():
    train_data = pandas.read_csv(cfg.vals['train_data'])
    train_data_features = Features.feature_extraction(train_data, y_col='quantity')

    TRAIN_X = train_data_features.toarray()
    TRAIN_Y = train_data_features.y

    test_data = pandas.read_csv(cfg.vals['test_data'])
    test_data_features = Features.feature_extraction(test_data, y_col='quantity')

    TEST_X = test_data_features.toarray()
    TEST_Y = test_data_features.y

    linearRegression = LinearRegression()
    linearRegression.fit(TRAIN_X, TRAIN_Y)
    beta = linearRegression.coef_

    yPrediction = linearRegression.predict(TEST_X)

    plot.plot(TEST_Y, yPrediction, color='blue')

    plot.show()


class PredictOptimalAction:
    def __init__(self, ):

    def predict(self, state):

        print()

    predict()