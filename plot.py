import matplotlib.pyplot as plt
import pandas as pd
import json


def plot_region_product(df, n_product, n_region, fname, y_col='sales'):
    x_col = 'time'
    fig, ax = plt.subplots(nrows=n_product, ncols=n_region, sharex=True, sharey=True, figsize=(12,12))

    for i in range(n_product):
        prod_data = df[df['product'] == i]
        for j in range(n_region):
            prod_reg_data = prod_data[df['region'] == j].reset_index()
            x = prod_reg_data[x_col]
            y = prod_reg_data[y_col]
            ax[i,j].plot(x, y, label=y_col)
            #ax[i, j].fill_between(x, y + y*.5, y-y*.5, color='gray', alpha=0.2)
            ax[i,j].legend(loc='best')

    fig.text(0.5, 0.04, 'Regions', ha='center')
    fig.text(0.04, 0.5, 'Products', va='center', rotation='vertical')
    plt.savefig(fname)



def plot_posterior_predictive_check(df, n_product, n_region, fname, y_col, y_hat_col):
    x_col = 'time'

    #fig, ax = plt.subplots(nrows=n_product, ncols=n_region, sharex=True, sharey=True, figsize=(12,12))

    for i in range(n_product):
        prod_data = df[df['product'] == i]
        for j in range(n_region):
            prod_reg_data = prod_data[df['region'] == j].reset_index()

            x = prod_reg_data[x_col]
            y_true = prod_reg_data[y_col]
            y_hat = prod_reg_data[y_hat_col]
            y_hat_col_upper = prod_reg_data[y_hat_col + "_upper"]
            y_hat_col_lower = prod_reg_data[y_hat_col + "_lower"]

            plt.plot(x, y_true, label=y_col)
            plt.plot(x, y_hat, label=y_hat_col)
            plt.fill_between(x, y_hat_col_upper, y_hat_col_lower, color='gray', alpha=0.2)
            plt.legend(loc='best')


            plt.savefig(fname + "-{}-{}.pdf".format(i,j))
            plt.clf()
            plt.close()



if __name__ == "__main__":
    import numpy as np

    with open('config.json') as f:
        config = json.load(f)

    config['adj_mtx'] = np.eye(config['n_regions'])
    data = pd.read_csv("model-output.csv")
    plot_posterior_predictive_check(data, n_product=config['n_products'], n_region=config['n_regions'],
                                    fname='figs/posterior_predictive_check.pdf', y_col='sales',
                                    y_hat_col='sales_pred')