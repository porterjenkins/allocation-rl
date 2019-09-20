import numpy as np
import pandas as pd
import theano
import config.config as cfg
from envs.state import State
from sklearn.preprocessing import OneHotEncoder


class Features(object):

    def __init__(self, region, product, temporal, lagged, prices, time_stamps, y=None):
        self.region = region
        self.product = product
        self.temporal = temporal
        self.lagged = lagged
        self.prices = prices
        self.time_stamps = time_stamps
        self.y = y


    def toarray(self):
        features = [self.product, self.region, self.temporal, self.lagged.reshape(-1,1)]

        feature_mtx = np.concatenate(features, axis=1)
        return feature_mtx


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

    @classmethod
    def _get_day_features(cls, n_rows, day_idx):
        day_features = np.zeros((n_rows, cfg.vals['n_temporal_features']))
        day_features[:, day_idx] = 1.0
        return day_features

    @classmethod
    def _get_lagged_features(cls, prev_sales, items):
        prev_sales = np.array([prev_sales[i] for i in items])
        return prev_sales

    @classmethod
    def _get_one_hot_features_multiple_ts(cls, n_rows, items, axis_1_size, strat_dim_size):
        one_hot_mtx = np.zeros((n_rows, axis_1_size))

        row_cntr = 0
        item_idx = 0

        while row_cntr < n_rows:
            col_idx = items[item_idx]
            one_hot_mtx[row_cntr, col_idx] = 1.0
            item_idx += 1

            if item_idx == strat_dim_size:
                item_idx = 0

            row_cntr += 1

        return one_hot_mtx

    @classmethod
    def _get_one_hot_features_single_ts(cls, items, n_items):
        encoder = OneHotEncoder(categories=[range(n_items)])
        one_hot = encoder.fit_transform(items.reshape(-1,1)).toarray()
        return one_hot


    @classmethod
    def featurize_state(cls, state):
        # number of unique item/product combinations in board config. Can use length of products from state
        n_rows = len(state._products)

        time_stamps = np.array([state.day]*n_rows)
        day_features = Features._get_one_hot_features_single_ts(time_stamps, cfg.vals['n_temporal_features'])
        product_features = Features._get_one_hot_features_single_ts(state._products, cfg.vals['n_products'])
        region_features = Features._get_one_hot_features_single_ts(state._regions, cfg.vals['n_regions'])
        prices = np.array(cfg.vals['prices'])[state._products]
        prev_sales = Features._get_lagged_features(state.prev_sales, state._items)
        y = np.ones(n_rows).astype(theano.config.floatX)


        features = Features(temporal=day_features,
                            product=product_features,
                            region=region_features,
                            time_stamps=time_stamps,
                            lagged=prev_sales,
                            prices=prices,
                            y=y)

        return features


if __name__ == "__main__":

    init_state = State.init_state(config=cfg.vals)
    state_features = Features.featurize_state(init_state)

    stop = 0