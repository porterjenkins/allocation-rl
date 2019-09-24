import pymc3 as pm


class Model(object):

    def __init__(self, prior, n_regions, n_products, n_temporal_features, X_region, X_product, X_temporal, X_lagged, y,
                 time_stamps):
        self.prior = prior
        self.n_regions = n_regions
        self.n_products = n_products
        self.n_temporal_features = n_temporal_features
        self.X_region = X_region
        self.X_product = X_product
        self.X_temporal = X_temporal
        self.X_lagged = X_lagged
        self.y = y
        self.time_stamps = time_stamps

    def build(self):
        pass


class LinearModel(Model):
    def __init__(self, prior, n_regions, n_products, n_temporal_features, X_region, X_product, X_temporal, X_lagged, y,
                 time_stamps):
        super().__init__(prior, n_regions, n_products, n_temporal_features, X_region, X_product, X_temporal, X_lagged, y,
                 time_stamps)

    def build(self):
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
            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)
            c_t = pm.Normal("customer_t", mu=lambda_c_t, sigma=25.0, shape=self.X_temporal.shape.eval()[0])

            c_all = c_t[self.time_stamps] * w_c

            lambda_q = pm.math.dot(self.X_region, w_r.T) + pm.math.dot(self.X_product, w_p.T) + c_all + w_s * self.X_lagged

            q_ij = pm.Normal('quantity_ij', mu=lambda_q, sigma=25.0, observed=self.y)

        return env_model


class HierarchicalModel(Model):

    def __init__(self):
        super(Model, self).__init__()

    def build(self):
        with pm.Model() as env_model:

            # Generate upper-level region weights
            w_r = pm.MvNormal('w_r', mu=self.prior.loc_w_r, cov=self.prior.scale_w_r,
                              shape=self.n_regions)

            # Generate lower-level region weights
            w_r_ij = pm.MvNormal('w_r_ij', mu=w_r, cov=self.prior.scale_w_r, shape=(self.n_products, self.n_regions))
            #w_r_ij = w_r_ij_gen[self.product_idx, :]

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
            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)

            c_t = pm.Normal("customer_t", mu=lambda_c_t, sigma=25.0,shape=self.X_temporal.shape.eval()[0])
            c_all = c_t[self.time_stamps] * w_c

            lambda_q = pm.math.sum(self.X_region * w_r_ij[self.product_idx], axis=1) + pm.math.dot(self.X_product, w_p.T) + \
                       c_all + w_s * self.X_lagged

            q_ij = pm.Normal('quantity_ij', mu=lambda_q, sigma=25.0, observed=self.y)

        return env_model