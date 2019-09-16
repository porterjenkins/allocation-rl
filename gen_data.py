import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from true_params import TrueParams

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

    def get_weights(self, w_r, w_p, w_c):
        W = np.concatenate((w_r, w_p, w_c))
        return W


    def gen_customer_counts(self, beta_c, x_t):
        lambda_c = np.dot(beta_c, x_t)
        lambda_c_floor = np.maximum(0, lambda_c)
        c_j = np.random.poisson(lam=lambda_c_floor)
        return c_j

    def gen_demand_q(self, w_r_ij, x_ij):

        lambda_q = np.dot(w_r_ij, x_ij)
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


    def run(self, fname='test-data.csv'):
        quantity_data = np.zeros(shape=(self.max_t*self.n_products*self.n_regions, 4))
        sales_data = np.zeros(shape=(self.max_t*self.n_products*self.n_regions))

        cntr = 0
        for t in range(self.max_t):
            print("Time stamp: t={}".format(t))
            day_of_week_vec = np.zeros(7)
            day_of_week = t % 7
            day_of_week_vec[day_of_week] = 1.0
            c_t = self.gen_customer_counts(params['w_t'], day_of_week_vec)

            p_vec = np.eye(self.n_products)
            for p in range(self.n_products):
                r_vec = np.eye(self.n_regions)
                for r in range(self.n_regions):


                    x_pr = self.get_quantity_features(r_vec[r], p_vec[p], c_t)

                    w_pr = self.get_weights(w_r=params['w_r'][p, :],
                                            w_p=params['w_p'],
                                            w_c=params['w_c'])

                    q_tpr = self.gen_demand_q(w_pr, x_pr)
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
        print(data.head(25))

        data.to_csv(fname)

        if self.plot:

            sns.set(style="darkgrid")
            sns.relplot(x="time", y="quantity",
                         style="product",  kind="line",
                         data=data)

            plt.show()



if __name__ == "__main__":

    with open('config.json') as f:
        config = json.load(f)

    config['adj_mtx'] = np.eye(config['n_products'])
    P = TrueParams()

    params = P.fixTrueParams(config['n_products'], config['n_regions'], config['adj_mtx'])



    generator = DataGenerator(config['n_regions'],
                              config['n_products'],
                              config['max_t'],
                              config['prices'],
                              params)

    generator.run("test-data-simple.csv")