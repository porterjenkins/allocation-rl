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
            w_r = pm.MvNormal('w_r', mu=self.prior.loc_w_r, tau=self.prior.scale_w_r,
                              shape=self.n_regions)

            # Generate Product weights
            w_p = pm.MvNormal('w_p', mu=self.prior.loc_w_p, cov=self.prior.scale_w_p,
                              shape=self.n_products)
            # Prior for customer weight
            #w_c = pm.Normal('w_c', mu=self.prior.loc_w_c, sigma=self.prior.scale_w_c)
            # Generate sales weight
            w_s = pm.TruncatedNormal('w_s', mu=self.prior.loc_w_s, sigma=self.prior.scale_w_s,lower=0.0)
            # Generate temporal weights
            w_t = pm.MvNormal('w_t', mu=self.prior.loc_w_t, cov=self.prior.scale_w_t,
                              shape=self.n_temporal_features)

            #bias_c_t = pm.Normal("bias_c_t", mu=0.0, sigma=25.0)
            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)
            print("lambda_c_t", lambda_c_t.tag.test_value.shape)

            #sigma_c_t = pm.InverseGamma("sigma_c_t",alpha=self.prior.loc_sigma_c_t, beta=self.prior.scale_sigma_c_t)
            #c_t = pm.Normal("customer_t", mu=lambda_c_t, sigma=sigma_c_t, shape=self.X_temporal.shape.eval()[0])
            #c_all = c_t[self.time_stamps] * w_c

            bias_q = pm.Normal("bias_q", mu=0.0, sigma=25.0)
            # TODO: should force mean to be positive ? exp(mu)
            lambda_q = pm.math.exp(bias_q + lambda_c_t[self.time_stamps] + pm.math.dot(self.X_region, w_r.T) + pm.math.dot(self.X_product, w_p.T)  + w_s * self.X_lagged)


            sigma_q_ij = pm.InverseGamma("sigma_q_ij",alpha=self.prior.loc_sigma_q_ij, beta=self.prior.scale_sigma_q_ij)
            q_ij = pm.TruncatedNormal('quantity_ij', mu=lambda_q, sigma=sigma_q_ij, lower=0.0, observed=self.y)

        return env_model


class HierarchicalModel(Model):

    def __init__(self, prior, n_regions, n_products, n_temporal_features, X_region, X_product, X_temporal, X_lagged, y,
                 time_stamps, product_idx):
        self.product_idx = product_idx
        super().__init__(prior, n_regions, n_products, n_temporal_features, X_region, X_product, X_temporal, X_lagged, y,
                 time_stamps)

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
            #w_c = pm.Normal('w_c', mu=self.prior.loc_w_c, sigma=self.prior.scale_w_c)

            # Generate customer weight
            w_s = pm.TruncatedNormal('w_s', mu=self.prior.loc_w_s, sigma=self.prior.scale_w_s, lower=0.0)

            # Generate temporal weights
            w_t = pm.MvNormal('w_t', mu=self.prior.loc_w_t, cov=self.prior.scale_w_t,
                              shape=self.n_temporal_features)
            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)

            #c_t = pm.Normal("customer_t", mu=lambda_c_t, sigma=25.0,shape=self.X_temporal.shape.eval()[0])
            #c_all = c_t[self.time_stamps] * w_c

            bias_q = pm.Normal("bias_q", mu=0.0, sigma=25.0)
            lambda_q = bias_q + pm.math.sum(self.X_region * w_r_ij[self.product_idx], axis=1) + pm.math.dot(self.X_product, w_p.T) + \
                       lambda_c_t[self.time_stamps] + w_s * self.X_lagged

            sigma_q_ij = pm.InverseGamma("sigma_q_ij", alpha=self.prior.loc_sigma_q_ij,
                                         beta=self.prior.scale_sigma_q_ij)
            q_ij = pm.TruncatedNormal('quantity_ij', mu=lambda_q, sigma=sigma_q_ij, lower=0.0, observed=self.y)

        return env_model