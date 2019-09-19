import numpy as np
import pandas as pd
import theano



class Features(object):

    def __init__(self, region, product, temporal, lagged, prices, time_stamps, y=None):
        self.region = region
        self.product = product
        self.temporal = temporal
        self.lagged = lagged
        self.prices = prices
        self.time_stamps = time_stamps
        self.y = y
        


    @classmethod
    def feature_extraction(cls, df, prices, y_col=None):
        df['day_of_week'] = df['time'] % 7

        day_features_grouped = df[['time', 'day_of_week']].groupby('time').max()

        region_features = pd.get_dummies(df.region, prefix='region')
        product_features = pd.get_dummies(df['product'], prefix='product')
        day_features = pd.get_dummies(day_features_grouped['day_of_week'], prefix='day')

        if y_col is not None:
            y = df[y_col].values.astype(theano.config.floatX)
        else:
            y = np.ones(df.shape[0]).astype(theano.config.floatX)

        features = Features(region=region_features.values.astype(theano.config.floatX),
                            product=product_features.values.astype(theano.config.floatX),
                            temporal=day_features.values.astype(theano.config.floatX),
                            lagged=df['prev_sales'].values.astype(theano.config.floatX),
                            prices=np.dot(product_features.values, prices).reshape(1, -1),
                            time_stamps=df['time'].values.astype(int),
                            y=y)

        return features
