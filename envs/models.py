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
            packed_L_p = pm.LKJCholeskyCov('packed_L_p', n=self.n_products,
                                           eta=2., sd_dist=pm.HalfCauchy.dist(2.5))

            L_p = pm.expand_packed_triangular(self.n_products, packed_L_p)

            mu_p = pm.MvNormal("mu_p", mu=self.prior.loc_w_p, cov=self.prior.scale_w_p,
                               shape=self.n_products)

            w_p = pm.MvNormal('w_p', mu=mu_p, chol=L_p,
                              shape=self.n_products)


            # Generate previous sales weight
            loc_w_s = pm.HalfCauchy('loc_w_s',1.0)
            scale_w_s = pm.HalfCauchy('scale_w_s', 2.5)

            w_s = pm.TruncatedNormal('w_s', mu=loc_w_s, sigma=scale_w_s,lower=0.0)

            # Generate temporal weights
            packed_L_t = pm.LKJCholeskyCov('packed_L_t', n=self.n_temporal_features,
                                         eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
            L_t = pm.expand_packed_triangular(self.n_temporal_features, packed_L_t)
            mu_t = pm.MvNormal("mu_t", mu=self.prior.loc_w_t, cov=self.prior.scale_w_t,
                               shape = self.n_temporal_features)

            w_t = pm.MvNormal('w_t', mu=mu_t, chol=L_t,
                              shape=self.n_temporal_features)

            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)


            bias_q_loc = pm.Normal('bias_q_loc', mu=0.0, sigma=1.0)
            bias_q_scale = pm.HalfCauchy('bias_q_scale', 5.0)

            bias_q = pm.Normal("bias_q", mu=bias_q_loc, sigma=bias_q_scale)
            # TODO: should force mean to be positive ? exp(mu)
            lambda_q = bias_q + lambda_c_t[self.time_stamps] + pm.math.dot(self.X_region, w_r.T) + pm.math.dot(self.X_product, w_p.T)  + w_s * self.X_lagged


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
            w_r = pm.MvNormal('w_r', mu=self.prior.loc_w_r, tau=self.prior.scale_w_r,
                              shape=self.n_regions)

            # Generate lower-level region weights
            w_r_ij = pm.MvNormal('w_r_ij', mu=w_r, tau=self.prior.scale_w_r, shape=(self.n_products, self.n_regions))
            #w_r_ij = w_r_ij_gen[self.product_idx, :]

            # Generate Product weights
            packed_L_p = pm.LKJCholeskyCov('packed_L_p', n=self.n_products,
                                           eta=2., sd_dist=pm.HalfCauchy.dist(2.5))

            L_p = pm.expand_packed_triangular(self.n_products, packed_L_p)

            mu_p = pm.MvNormal("mu_p", mu=self.prior.loc_w_p, cov=self.prior.scale_w_p,
                               shape=self.n_products)

            w_p = pm.MvNormal('w_p', mu=mu_p, chol=L_p,
                              shape=self.n_products)

            # Generate previous sales weight
            loc_w_s = pm.HalfCauchy('loc_w_s', 1.0)
            scale_w_s = pm.HalfCauchy('scale_w_s', 2.5)

            w_s = pm.TruncatedNormal('w_s', mu=loc_w_s, sigma=scale_w_s, lower=0.0)

            # Generate temporal weights
            packed_L_t = pm.LKJCholeskyCov('packed_L_t', n=self.n_temporal_features,
                                           eta=2., sd_dist=pm.HalfCauchy.dist(2.5))
            L_t = pm.expand_packed_triangular(self.n_temporal_features, packed_L_t)
            mu_t = pm.MvNormal("mu_t", mu=self.prior.loc_w_t, cov=self.prior.scale_w_t,
                               shape=self.n_temporal_features)

            w_t = pm.MvNormal('w_t', mu=mu_t, chol=L_t,
                              shape=self.n_temporal_features)

            lambda_c_t = pm.math.dot(self.X_temporal, w_t.T)

            bias_q_loc = pm.Normal('bias_q_loc', mu=0.0, sigma=1.0)
            bias_q_scale = pm.HalfCauchy('bias_q_scale', 5.0)

            bias_q = pm.Normal("bias_q", mu=bias_q_loc, sigma=bias_q_scale)
            lambda_q = bias_q + pm.math.sum(self.X_region * w_r_ij[self.product_idx], axis=1) + pm.math.dot(self.X_product, w_p.T) + \
                       lambda_c_t[self.time_stamps] + w_s * self.X_lagged

            sigma_q_ij = pm.InverseGamma("sigma_q_ij", alpha=self.prior.loc_sigma_q_ij,
                                         beta=self.prior.scale_sigma_q_ij)
            q_ij = pm.TruncatedNormal('quantity_ij', mu=lambda_q, sigma=sigma_q_ij, lower=0.0, observed=self.y)

        return env_model