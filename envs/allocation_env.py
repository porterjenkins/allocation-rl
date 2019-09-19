import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import pandas as pd
import json
from plot import plot_posterior_predictive_check
from envs.prior import Prior
import config.config as cfg
from envs.features import feature_extraction, get_target


class AllocationEnv(object):
    """Environment model for training Reinforcement Learning agent"""

    def __init__(self, config, prior):
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.n_temporal_features = config['n_temporal_features']
        self.adj_mtx = config['adj_mtx']
        self.prior = prior
        self.model = None
        self.trace = None

    def build_env_model(self, X_region, X_product, X_temporal, X_lagged, time_stamps, y=None):

        num_time_stamps = len(np.unique(time_stamps))

        with pm.Model() as env_model:

            # Generate region weights
            w_r = pm.MvNormal('w_r', mu=self.prior.loc_w_r, cov=self.prior.scale_w_r,
                              shape=self.n_regions)

            # Generate Product weights
            w_p = pm.MvNormal('w_p', mu=self.prior.loc_w_p, cov=self.prior.scale_w_p,
                              shape=self.n_products)

            # Prior for customer weight
            w_c = pm.Normal('w_c', mu=self.prior.loc_w_c, sigma=self.prior.scale_w_c)

            # Generate customer weight
            w_s = pm.Gamma('w_s', mu=self.prior.loc_w_s, sigma=self.prior.scale_w_s)

            # Generate temporal weights
            w_t = pm.MvNormal('w_t', mu=self.prior.loc_w_t, cov=self.prior.scale_w_t,
                              shape=self.n_temporal_features)
            lambda_c_t = pm.math.dot(X_temporal, w_t.T)
            c_t = pm.Poisson("customer_t", mu=lambda_c_t, shape=num_time_stamps)

            c_all = c_t[time_stamps] * w_c

            lambda_q = pm.math.dot(X_region, w_r.T) + pm.math.dot(X_product, w_p.T) + c_all + w_s * X_lagged

            q_ij = pm.Poisson('quantity_ij', mu=lambda_q, observed=y)

        self.model = env_model

    def __check_model(self):
        if self.model is not None:
            return None
        else:
            raise ValueError("environment model has not been built. run build_env_mode()")

    def train(self, samples, tune):
        self.__check_model()

        with self.model:
            self.trace = pm.sample(samples, tune=tune, init='advi+adapt_diag')

    def predict(self):
        self.__check_model()
        with self.model:
            posterior_pred = pm.sample_posterior_predictive(self.trace)
        return posterior_pred['quantity_ij']




if __name__ == "__main__":
    prior = Prior(config=cfg.vals,
                  fname='prior.json')
    train_data = pd.read_csv('../train-data-simple.csv', index_col=0)

    y_train = get_target(train_data)
    train_features = feature_extraction(train_data, prices=cfg.vals['prices'])

    X_region = theano.shared(train_features['region'])
    X_product = theano.shared(train_features['product'])
    X_temporal = theano.shared(train_features['temporal'])
    X_lagged = theano.shared(train_features['lagged'])
    y = theano.shared(y_train)

    env = AllocationEnv(config=cfg.vals, prior=prior)
    env.build_env_model(X_region, X_product, X_temporal, X_lagged, y=y_train, time_stamps=train_data['time'].astype(int))
    env.train(samples=100, tune=100)
    q_ij = env.predict()
    print(q_ij)