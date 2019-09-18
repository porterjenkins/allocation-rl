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
            ax[i,j].legend(loc='best')

    fig.text(0.5, 0.04, 'Regions', ha='center')
    fig.text(0.04, 0.5, 'Products', va='center', rotation='vertical')
    plt.savefig(fname)



def plot_posterior_predictive_check(df, n_product, n_region, fname, y_col, y_hat_col):
    x_col = 'time'

    #fig, ax = plt.subplots(nrows=n_product, ncols=n_region, sharex=True, sharey=True, figsize=(12,12))


    for j in range(n_region):
        reg_data = df[df['region'] == j]
        #plt.figure(figsize=(10, 10))
        for i in range(n_product):
            prod_reg_data = reg_data[reg_data['product'] == i].reset_index()

            x = prod_reg_data[x_col]
            y_true = prod_reg_data[y_col]
            y_hat = prod_reg_data[y_hat_col]
            y_hat_col_upper = prod_reg_data[y_hat_col + "_upper"]
            y_hat_col_lower = prod_reg_data[y_hat_col + "_lower"]
            #err = np.concatenate((y_hat_col_lower, y_hat_col_upper),axis=0)

            plt.plot(x, y_true, label='product: {} - true'.format(i))
            plt.plot(x, y_hat, label='product: {} - pred'.format(i), linestyle='--')
            plt.fill_between(x, y_hat_col_upper, y_hat_col_lower, color='gray', alpha=0.2)

        #plt.legend(loc='best')
        plt.xlabel("Time")
        plt.ylabel("Revenue")
        plt.savefig(fname + "-{}.pdf".format(j))
        plt.clf()
        plt.close()

def plot_total_ppc(df, draws, y_col='sales', fname='total-ppc.png'):
    totals = df[['time', y_col]].groupby('time').sum()
    draws_all = pd.concat((df, draws), axis=1)
    draw_cols = list(range(draws.shape[1]))
    draw_sums = draws_all[['time'] + draw_cols].groupby('time').sum().values
    y_hat = draw_sums.mean(axis=1)
    y_hat_lower = np.percentile(draw_sums, q=2.5, axis=1)
    y_hat_upper = np.percentile(draw_sums, q=97.5, axis=1)

    x = totals.index
    y = totals[y_col]
    plt.plot(x, y, label='observed')
    plt.plot(x, y_hat, label= 'predicted')
    plt.fill_between(x, y_hat_lower, y_hat_upper, color='gray', alpha=0.2)
    plt.legend(loc='best')
    plt.savefig(fname)


if __name__ == "__main__":
    import numpy as np

    with open('config.json') as f:
        config = json.load(f)

    config['adj_mtx'] = np.eye(config['n_regions'])
    data = pd.read_csv("model-output.csv")
    draws = pd.read_csv('sales-draws.csv',header=None)
    #plot_total_ppc(data, draws=draws ,y_col='sales')
    plot_posterior_predictive_check(data, n_product=config['n_products'], n_region=config['n_regions'],
                                    fname='figs/posterior_predictive_check.pdf', y_col='sales',
                                    y_hat_col='sales_pred')