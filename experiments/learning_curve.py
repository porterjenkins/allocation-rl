import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import normalize, MinMaxScaler

scaler = MinMaxScaler()


eps_30_fname = "output/eps_30/rl-learning-curve-{}.json".format(cfg.vals['prj_name'])

with open(eps_30_fname, 'r') as f:
    eps_30 = json.load(f)
    vals = []
    #for i, x in enumerate(eps_30["sum"]):
    #    if i % 3.0 == 0.0:
    #        vals.append(float(x))

    #eps_30["sum"] = np.array(vals)

    eps_30["sum"] = scaler.fit_transform(np.array([float(x) for x in eps_30["sum"]]  ).reshape(-1,1))


eps_60_fname = "output/eps_60/rl-learning-curve-{}.json".format(cfg.vals['prj_name'])

with open(eps_60_fname, 'r') as f:
    eps_60 = json.load(f)
    eps_60["sum"] = scaler.fit_transform(np.array([float(x) for x in eps_60["sum"]]).reshape(-1,1))



eps_90_fname = "output/eps_90/rl-learning-curve-{}.json".format(cfg.vals['prj_name'])

with open(eps_90_fname, 'r') as f:
    eps_90 = json.load(f)
    eps_90["sum"] = scaler.fit_transform(np.array([float(x) for x in eps_90["sum"]]).reshape(-1,1))




n = len(eps_90["sum"])
x = range(0, n*3, 3)

plt.plot(x,  eps_30["sum"].flatten()[:n], c='b', label='30 days',alpha=.6)
plt.plot(x,  eps_60["sum"].flatten()[:n], c='r', label='60 days',alpha=.6)
plt.plot(x,  eps_90["sum"].flatten(), c='g', label='90 days',alpha=.6)

plt.xlabel("Episodes")
plt.ylabel("Normalized Test Reward")
plt.legend(loc='best')
plt.savefig("figs/learning-curves-all-{}".format(cfg.vals['prj_name']))
