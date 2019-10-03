import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
import config.config as cfg
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns

PLOT = True
n_p = cfg.vals["n_products"]
n_r = cfg.vals["n_regions"]


prior = Prior(config=cfg.vals)

env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)

mod_summary = pm.summary(env.trace)

mod_summary.to_csv("output/model-summary-{}.csv".format(cfg.vals['prj_name']))
w_r_names = []

for i in range(n_p):
    for j in range(n_r):
        name = "w_r_ij__{}_{}".format(i, j)
        w_r_names.append(name)

if cfg.vals['model_type'] == 'hierarchical':

    w_r_means = mod_summary.loc[w_r_names]['mean'].values.reshape((n_p, n_r))


    sns.heatmap(w_r_means.transpose(), linewidth=0.5, cmap="YlGnBu")
    plt.xlabel('Products')
    plt.ylabel("Regions")
    plt.savefig("figs/weight-heatmap-{}.pdf".format(cfg.vals['prj_name']))
    plt.clf()
    plt.close()


if PLOT:

    pm.traceplot(env.trace, var_names=['w_r'])
    plt.savefig("figs/trace-plots-w-r-{}.pdf".format(cfg.vals['prj_name']))
    plt.clf()
    plt.close()

    pm.traceplot(env.trace, var_names=['w_t'])
    plt.savefig("figs/trace-plots-w-t-{}.pdf".format(cfg.vals['prj_name']))
    plt.clf()
    plt.close()



    pm.traceplot(env.trace, var_names=['w_p','w_s'])
    plt.savefig("figs/trace-plots-p-s-{}.pdf".format(cfg.vals['prj_name']))



