import json
import numpy as np
import os
import ast
import sys

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

# setup path variables
vals['prj_name'] =  vals['model_type'] + "-" + vals['train_data'].split("/")[1].split(".")[0]
vals['prj_root'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
vals['model_path'] = "{}/envs/{}.p".format(vals['prj_root'],vals['prj_name'])
vals['train_data'] = vals['prj_root'] + "/" + vals['train_data']
vals['prior_fname'] = vals['prj_root'] + "/" + vals['prior_fname']


# setup state matrices
vals['env_init_loc'] = make_bin_mtx(vals['env_init_loc'], dims=(vals['n_regions'], vals['n_products']))
vals['adj_mtx'] = make_bin_mtx(vals['adj_mtx'], dims=(vals['n_regions'], vals['n_regions']))



assert vals['model_type'] in ['hierarchical', 'linear']