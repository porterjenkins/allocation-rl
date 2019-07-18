import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

N_REGIONS = 4
N_PRODUCTS = 4
T = 20
PRICES = [2.50, 4.99, 3.00, 1.20]
PLOT = True

def gen_q(prod, t, r):
    product_w = np.array([8.0, 4.0, 75.0, 1.5])
    time_w = 2.5

    lmbda = np.dot(product_w, prod) + t*time_w
    q = np.random.poisson(lam=lmbda, size=r)
    return q

quantity_data = np.zeros(shape=(T*N_PRODUCTS*N_REGIONS, 4))
sales_data = np.zeros(shape=(T*N_PRODUCTS*N_REGIONS))

cntr = 0
for t in range(T):
    for p in range(N_PRODUCTS):
        for r in range(N_REGIONS):
            p_vec = np.zeros(N_PRODUCTS)
            p_vec[p] = 1.0
            q_tpr = gen_q(p_vec, t, 1)
            s_tpr = q_tpr*PRICES[int(p)]

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

if PLOT:

    sns.set(style="darkgrid")
    sns.relplot(x="time", y="quantity",
                 style="region",  kind="line",
                 data=data)

    plt.show()

