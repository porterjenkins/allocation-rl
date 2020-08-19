import pandas as pd
import datetime
import numpy as np

from stable_baselines.deepq.replay_buffer import ReplayBuffer

import config.config as cfg
from envs.state import State


def main(fpath):
    train_data = pd.read_csv(fpath)
    n_products = train_data['product'].max() + 1
    n_regions = train_data['region'].max() + 1

    buffer = ReplayBuffer(size=100000)
    grouped = train_data.groupby(by='date')

    #prev


    for date, chunk in grouped:
        board_config = np.zeros([n_regions, n_products])
        prev_sales = np.zeros([n_regions, n_products])

        day = chunk.iloc[0, 8]

        prev_sales_product = {}
        prev_placement_cnts = {}
        for idx, row in chunk.iterrows():

            region = row['region']
            product = row['product']

            prev_sales_product[product] = row['prev_sales']


            if row['quantity'] > 0:
                board_config[region, product] = 1.0

                if product not in prev_placement_cnts:
                    prev_placement_cnts[product] = 0

                prev_placement_cnts[product] += 1


        for p in range(n_products):

            if p not in prev_placement_cnts:
                continue

            sales = prev_sales_product[p]
            cnt = prev_placement_cnts[p]
            avg_spatial_sales = sales / cnt
            regions = board_config[:, p]

            prev_sales[:, p] = regions * avg_spatial_sales




        day_vec = State.get_day_vec(day)

        state = {"day_vec": day_vec, "prev_sales": prev_sales}
        print(prev_sales)




if __name__ == "__main__":
    main(cfg.vals['train_data'])
