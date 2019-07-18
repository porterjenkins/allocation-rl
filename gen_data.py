import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def getStructuredCovariance(A, n_regions, n_quantity_features):
    # Covariance for region features
    sigma_i_regions = np.random.uniform(0, 5, size=n_regions)
    sigma_regions = np.multiply(sigma_i_regions, A)

    # covariance for other features (products, n_cust, etc...)
    sigma_i_other = np.random.uniform(0, 1, size=n_quantity_features-n_regions)

    Sigma = np.eye(n_quantity_features)
    Sigma[:n_regions, :n_regions] = sigma_regions
    sigma_other = np.multiply(Sigma[n_regions:n_quantity_features, n_regions:n_quantity_features], sigma_i_other)
    Sigma[n_regions:n_quantity_features, n_regions:n_quantity_features] = sigma_other


    return Sigma

def fixTrueParams(n_products, n_regions, A, random_seed=1990, persist=True):
    np.random.seed(random_seed)

    params = {}

    ## Customer generation

    # day of week features (one for each day of the week. weekends recieve heavier weight)
    params['mu_c'] = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 5.0, 7.0,])
    params['sigma_c'] = np.eye(len(params['mu_c']))
    params['beta_c'] = np.random.multivariate_normal(mean=params['mu_c'], cov=params['sigma_c'])


    ## quantity demanded
    # number of products + a parameter for the number of customers
    n_quantity_features = n_regions + n_products + 1

    params['phi'] = np.random.uniform(10, 100, size=n_quantity_features)
    params['psi'] = np.eye(n_quantity_features)


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


class DataGenerator(object):

    def __init__(self,  n_regions, n_products, max_t, prices, params, plot=True):
        self.n_regions = n_regions
        self.n_products = n_products
        self.max_t = max_t
        self.prices = prices
        self.params = params
        self.plot = plot

    def get_quantity_features(self, r, p, c):
        c_vec = np.array([c])
        features = np.concatenate([r, p, c_vec])
        return features


    def gen_customer_counts(self, beta_c, x_t):
        lambda_c = np.dot(beta_c, x_t)
        lambda_c_floor = np.maximum(0, lambda_c)
        c_j = np.random.poisson(lam=lambda_c_floor)
        return c_j

    def gen_demand_q(self, beta_ij, x_ij):

        lambda_q = np.dot(beta_ij, x_ij)
        q = np.random.poisson(lambda_q)

        return q


    def gen_q(self, prod, t, r):
        product_w = np.array([8.0, 4.0, 75.0, 1.5])
        time_w = 2.5

        lmbda = np.dot(product_w, prod) + t*time_w
        q = np.random.poisson(lam=lmbda, size=r)
        return q

    def get_sales(self, quantity, prices):
        sales = np.dot(np.transpose(quantity), prices)
        return sales

    def run(self):
        quantity_data = np.zeros(shape=(self.max_t*self.n_products*self.n_regions, 4))
        sales_data = np.zeros(shape=(self.max_t*self.n_products*self.n_regions))

        cntr = 0
        for t in range(self.max_t):
            day_of_week_vec = np.zeros(7)
            day_of_week = t % 7
            day_of_week_vec[day_of_week] = 1.0
            c_j = self.gen_customer_counts(params['beta_c'], day_of_week_vec)

            p_vec = np.eye(self.n_products)
            for p in range(self.n_products):
                r_vec = np.eye(self.n_regions)
                for r in range(self.n_regions):


                    x_pr = self.get_quantity_features(r_vec[r], p_vec[p], c_j)
                    beta_ij = params['beta_ij'][p, r]
                    q_tpr = self.gen_demand_q(beta_ij, x_pr)
                    s_tpr = self.prices[p] * q_tpr

                    quantity_data[cntr, 0] = t
                    quantity_data[cntr, 1] = p
                    quantity_data[cntr, 2] = r
                    quantity_data[cntr, 3] = q_tpr


                    sales_data[cntr] = s_tpr

                    cntr+=1


        cols = ['time', 'product', 'region', 'quantity']
        data = pd.DataFrame(quantity_data, columns=cols)
        data['sales'] = sales_data
        print(data.head())

        data.to_csv("test-data.csv")

        if self.plot:

            sns.set(style="darkgrid")
            sns.relplot(x="time", y="quantity",
                         style="region",  kind="line",
                         data=data)

            plt.show()



if __name__ == "__main__":

    with open('config.json') as f:
        config = json.load(f)

    config['adj_mtx'] = np.eye(config['n_products'])

    params = fixTrueParams(config['n_products'], config['n_regions'], config['adj_mtx'])



    generator = DataGenerator(config['n_regions'],
                              config['n_products'],
                              config['max_t'],
                              config['prices'],
                              params)

    generator.run()