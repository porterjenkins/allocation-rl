import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def fixTrueParams(n_products, n_regions, A, random_seed=1990, persist=True):
    np.random.seed(random_seed)

    params = {}

    ## Customer generation

    # day of week features (one for each day of the week. weekends recieve heavier weight)
    params['mu_c'] = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 5.0, 7.0,])
    params['sigma_c'] = np.eye(len(params['mu_c']))
    params['beta_c'] = np.random.multivariate_normal(mean=params['mu_c'], cov=params['sigma_c'])


    ## quantity demanded

    params['phi'] = np.random.uniform(-1, 1, size=n_products)
    params['psi'] = np.eye(n_products)
    #for i in range(n_products):
    #    params['psi'] += np.random.randint(-5, 5)

    params['mu_i'] = np.random.multivariate_normal(mean=params['phi'], cov=params['psi'])
    sigma_i = np.random.multivariate_normal(np.zeros(n_regions), cov=np.eye(n_regions))
    params['sigma_i'] = np.multiply(sigma_i, A)

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

    def __init__(self,  n_regions, n_products, max_t, prices, plot=True):
        self.n_regions = n_regions
        self.n_products = n_products
        self.max_t = max_t
        self.prices = prices
        self.plot = plot



    def gen_q(self, prod, t, r):
        product_w = np.array([8.0, 4.0, 75.0, 1.5])
        time_w = 2.5

        lmbda = np.dot(product_w, prod) + t*time_w
        q = np.random.poisson(lam=lmbda, size=r)
        return q

    def run(self):
        quantity_data = np.zeros(shape=(self.max_t*self.n_products*self.n_regions, 4))
        sales_data = np.zeros(shape=(self.max_t*self.n_products*self.n_regions))

        cntr = 0
        for t in range(self.max_t):
            for p in range(self.n_products):
                for r in range(self.n_regions):
                    p_vec = np.zeros(self.n_products)
                    p_vec[p] = 1.0
                    q_tpr = self.gen_q(p_vec, t, 1)
                    s_tpr = q_tpr*self.prices[int(p)]

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
                              config['prices'])