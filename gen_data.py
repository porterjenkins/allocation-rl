import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class DataGenerator(object):

    def __init__(self,  n_regions, n_products, max_t, prices, plot):
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
    n_regions = 4
    n_products = 4
    T = 20
    prices = [2.50, 4.99, 3.00, 1.20]
    plot = True

    generator = DataGenerator(n_regions, n_products, T, prices, plot)