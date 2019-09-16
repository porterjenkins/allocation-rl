import matplotlib.pyplot as plt
import pandas as pd


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
