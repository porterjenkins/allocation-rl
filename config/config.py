import json
import numpy as np
import os
import ast

EVAL_KEYS = ['adj_mtx', 'env_init_loc']

def make_bin_mtx(arr, dims):
    mtx = np.zeros(dims)
    for idx in arr:
        mtx[idx] = 1.0
    return mtx

with open('../config/config.json') as f:
    vals = json.load(f)

for k in EVAL_KEYS:
    vals[k] = ast.literal_eval(vals[k])

vals['env_init_loc'] = make_bin_mtx(vals['env_init_loc'], dims=(vals['n_regions'], vals['n_products']))
vals['adj_mtx'] = make_bin_mtx(vals['adj_mtx'], dims=(vals['n_regions'], vals['n_regions']))
vals['model_path'] = "{}.p".format(vals['model_type'])


assert vals['model_type'] in ['hierarchical', 'linear']