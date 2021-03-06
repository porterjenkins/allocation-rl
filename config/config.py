import json
import numpy as np
import os
import ast
import sys
from envs.init_env import init_env


def make_bin_mtx(arr, dims):
    mtx = np.zeros(dims)
    for idx in arr:
        mtx[idx] = 1.0
    return mtx

with open('../config/config.json') as f:
    vals = json.load(f)




# flag to use precision matrix from graph laplacian
if 'precision_mtx' not in vals:
    vals['precision_mtx'] = True

# setup path variables
if vals['precision_mtx']:
    vals['prj_name'] = vals['model_type'] + "-" + vals['train_data'].split("/")[1].split(".")[0]
else:
    vals['prj_name'] =  vals['model_type'] + "-" + vals['train_data'].split("/")[1].split(".")[0] + "-no-precision"

if vals["log_linear"]:
    vals["prj_name"] += "-loglinear"

vals['prj_root'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
vals['model_path'] = "{}/envs/{}.p".format(vals['prj_root'],vals['prj_name'])
vals['train_data'] = vals['prj_root'] + "/" + vals['train_data']
vals['test_data'] = vals['prj_root'] + "/" + vals['test_data']
vals['prior_fname'] = vals['prj_root'] + "/" + vals['prior_fname']
vals['adj_mtx_fname'] = vals['prj_root'] + "/" + vals['adj_mtx_fname']

# project random seed
vals["random_seed"] = int(vals["random_seed"])
#vals["random_seed"] = None


with open(vals['adj_mtx_fname']) as f:
    adj_mtx_vals = json.load(f)
if 'n_regions' not in vals:
    vals['n_regions'] = adj_mtx_vals['n_regions']
adj_mtx_vals['non_zero_entries'] = ast.literal_eval(adj_mtx_vals['non_zero_entries'])

vals['adj_mtx'] = make_bin_mtx(adj_mtx_vals['non_zero_entries'], dims=(vals['n_regions'], vals['n_regions']))
# A + I - multiply by identity matrix
#vals['adj_mtx'] = vals['adj_mtx'] + np.eye(vals['n_regions'])



assert vals['model_type'] in ['hierarchical', 'linear']