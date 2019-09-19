import numpy as np
import pandas as pd
import theano



class Features(object):

    def __init__(self, region, product, temporal, lagged, prices):
        self.region = region
        self.product = product
        self.temporal = temporal
        self.lagged = lagged
        self.prices = prices




def feature_extraction(df, prices):
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
    features['prices'] = np.dot(product_features.values, prices).reshape(1, -1)

    return features


def get_target(df):
    y = df['quantity'].values.astype(theano.config.floatX)
    return y