import pandas as pd
import numpy as np


from utils import get_store_id

def main():
    dir = "../data/"
    files = ["store-1-train.csv", "store-2-train.csv"]

    for f in files:
        store_id = get_store_id(f)

        data = pd.read_csv(dir + f)

        sales = data[["date", "sales"]].groupby("date").sum()

        quantiles = np.quantile(sales["sales"].values, [.2, .4, .6, .8, 1.0])

        with open(dir + f"{store_id}-quantiles.txt", "w") as fp:

            for q in quantiles:
                fp.write("{:.4f}\n".format(q))


if __name__ == "__main__":
    main()