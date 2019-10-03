import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import pymc3 as pm
import matplotlib.pyplot as plt


prior = Prior(config=cfg.vals)

env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)

pm.traceplot(env.trace, var_names=['w_t','w_r'])
plt.savefig("figs/trace-plots-t-r.pdf")
plt.clf()
plt.close()

pm.traceplot(env.trace, var_names=['w_p','w_s'])
plt.savefig("figs/trace-plots-p-s.pdf")