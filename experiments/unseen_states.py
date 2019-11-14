import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
from envs.allocation_env import AllocationEnv
from envs.state import State
from envs.features import Features
import config.config as cfg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl


prior = Prior(config=cfg.vals)
env = AllocationEnv(config=cfg.vals, prior=prior, load_model=True)
env.reset()

print("n comp: {}".format(env.state.board_config.sum()))

s = State.init_state(config=cfg.vals)
state_features = Features.featurize_state(s)


y_hat = env._predict(state_features, n_samples=500)

print(y_hat)
print(y_hat.shape)

agg_sales = y_hat.sum(axis=1)
agg_mean = np.mean(agg_sales)
agg_lower, agg_upper = np.quantile(agg_sales, q=[.05, .95])

x_lim = [np.min(agg_sales), np.max(np.max(agg_sales))]

#plt.hist(agg_sales)
plt.axvline(agg_mean,linestyle='--',c='blue')
plt.axvline(agg_lower,linestyle='dotted',c='red')
plt.axvline(agg_upper,linestyle='dotted',c='red')

sns.distplot(agg_sales, hist=True, kde=True, color = 'blue',
             hist_kws={'edgecolor':'black'}, norm_hist=True)

plt.xlabel("Revenue ($)")
plt.savefig("figs/unseen_state-dist.pdf")
plt.clf()
plt.close()

board_config = s.board_config
fig, ax = plt.subplots()
# define the colors
cmap = mpl.colors.ListedColormap(['lightgray', 'blue'])
# create a normalize object the describes the limits of
# each color
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#plt.imshow(board_config.astype(int), interpolation='none', cmap=cmap, norm=norm)
sns.heatmap(board_config.astype(int), cmap=cmap,cbar=False)

plt.xlabel('Products')
plt.ylabel("Regions")
plt.savefig("figs/unseen_states-heatmap.pdf")
plt.clf()
plt.close()
