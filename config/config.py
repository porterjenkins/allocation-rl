import json
import numpy as np
import os


with open('../config/config.json') as f:
    vals = json.load(f)

vals['adj_mtx'] = np.eye(vals['n_regions'])