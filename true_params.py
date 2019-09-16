import numpy as np
import json


def getStructuredCovariance(A, n_regions, n_quantity_features, jitter=True):
    # Covariance for region features
    # assume region features are the first k of weight vector
    if jitter:
        sigma_i_regions = np.random.uniform(0, 5, size=n_regions)
    else:
        sigma_i_regions = np.ones(n_regions)
    sigma_regions = np.multiply(sigma_i_regions, A)

    # covariance for other features (products, n_cust, etc...)
    if jitter:
        sigma_i_other = np.random.uniform(0, 1, size= n_quantity_features-n_regions)
    else:
        sigma_i_other = np.ones(n_quantity_features - n_regions)

    Sigma = np.eye(n_quantity_features)
    Sigma[:n_regions, :n_regions] = sigma_regions
    sigma_other = np.multiply(Sigma[n_regions:n_quantity_features, n_regions:n_quantity_features], sigma_i_other)
    Sigma[n_regions:n_quantity_features, n_regions:n_quantity_features] = sigma_other


    return Sigma


class TrueParams(object):

    #def __init__(self):

    def fixTrueParams(self, n_products, n_regions, A, random_seed=1990, gamma=10, persist=True):
        np.random.seed(random_seed)
        params = {}

        # generate day of week weights (one for each day of the week. weekends recieve heavier weight)
        params['prior_loc_w_t'] = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 5.0, 7.0, ])
        params['prior_scale_w_t'] = np.eye(len(params['prior_loc_w_t']))*5
        params['w_t'] = np.random.multivariate_normal(mean=params['prior_loc_w_t'],
                                                      cov=params['prior_scale_w_t'])

        # generate product weights
        params['prior_loc_w_p'] = np.random.uniform(5, 25, n_products)
        params['prior_scale_w_p'] = np.eye(n_products )*5
        params['w_p'] = np.random.multivariate_normal(mean=params['prior_loc_w_p'],
                                                      cov=params['prior_scale_w_p'])

        # generate customer weight
        params['prior_loc_w_p'] = 5.0
        params['prior_scale_w_p'] = 1.0
        params['w_c'] = np.array([np.random.normal(loc=params['prior_loc_w_p'],
                                                   scale=params['prior_scale_w_p'])])

        # generate region weights
        params['prior_loc_w_r'] = np.ones(n_regions)*10
        params['prior_scale_w_r'] = A*gamma
        params['w_r'] = np.zeros([n_products, n_regions])
        w_r = np.random.multivariate_normal(mean=params['prior_loc_w_r'],
                                            cov=params['prior_scale_w_r'])
        for i in range(n_products):
            params['w_r'][i, :] = w_r

        if persist:
            paramsSerializable = {}
            for key, item in params.items():
                if isinstance(item, np.ndarray):
                    paramsSerializable[key] = params[key].tolist()
                else:
                    paramsSerializable[key] = params[key]
            with open("params.json", 'w') as f:
                json.dump(paramsSerializable, f)

        return params


def fixTrueParamsHierachical(n_products, n_regions, A, random_seed=1990, persist=True):
    np.random.seed(random_seed)

    params = {}

    ## Customer generation

    # day of week features (one for each day of the week. weekends recieve heavier weight)
    params['prior_loc_w_t'] = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 5.0, 7.0, ])
    params['prio_scale_w_t'] = np.eye(len(params['mu_c']))
    params['w_t'] = np.random.multivariate_normal(mean=params['mu_c'], cov=params['sigma_c'])

    ## quantity demanded
    # number of regions + number of products + a parameter for the number of customers
    n_quantity_features = n_regions + n_products + 1

    params['phi'] = np.ones(n_quantity_features) * 50
    params['psi'] = np.eye(n_quantity_features) * 25

    params['mu_i'] = np.random.multivariate_normal(mean=params['phi'], cov=params['psi'], size=n_products)
    params['sigma_i'] = getStructuredCovariance(A, n_regions, n_quantity_features)
    params['beta_ij'] = np.zeros(shape=(n_products, n_regions, n_quantity_features))

    for i in range(n_products):
        params['beta_ij'][i, :, :] = np.random.multivariate_normal(mean=params['mu_i'][i, :],
                                                                   cov=params['sigma_i'],
                                                                   size=n_regions)

    if persist:
        paramsSerializable = {}
        for key, item in params.items():
            if isinstance(item, np.ndarray):
                paramsSerializable[key] = params[key].tolist()
            else:
                paramsSerializable[key] = params[key]
        with open("params.json", 'w') as f:
            json.dump(paramsSerializable, f)

    return params