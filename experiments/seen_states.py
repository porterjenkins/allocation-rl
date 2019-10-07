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

# get seen state

ts_train = env.time_stamps.container.data
ts_unique = np.unique(ts_train)
ts = np.random.choice(ts_unique, )

idx = np.where(ts_train == ts)[0]

p_train = env.X_product.container.data
r_train = env.X_region.container.data

p_state = p_train[idx, :]
r_state = r_train[idx, :]

#products = np.unique(np.where(p_state ==1.0)[1])
#regions = np.unique(np.where(r_state ==1.0)[1])

observed_board = np.zeros([cfg.vals["n_regions"], cfg.vals["n_products"]])

for row in range(p_state.shape[0]):
    i = np.where(r_state[row, :] ==1.0)[0][0]
    j = np.where(p_state[row, :] ==1.0)[0][0]
    observed_board[i, j] = 1.0


#for i in regions:
#    for j in products:
#        observed_board[i,j] = 1.0

a = observed_board - env.state.board_config
env._take_action(a)

# heat map for seen state

fig, ax = plt.subplots()
# define the colors
cmap = mpl.colors.ListedColormap(['lightgray', 'blue'])
# create a normalize object the describes the limits of
# each color
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#plt.imshow(board_config.astype(int), interpolation='none', cmap=cmap, norm=norm)
sns.heatmap(observed_board.astype(int), cmap=cmap,cbar=False)

plt.xlabel('Products')
plt.ylabel("Regions")
plt.savefig("figs/seen_state-heatmap.pdf")
plt.clf()
plt.close()

# Plot posterior for seen state
state_features = Features.featurize_state(env.state)
y_hat = env._predict(state_features, n_samples=500)

print(y_hat)
print(y_hat.shape)

agg_sales = y_hat.sum(axis=1)
agg_mean = np.mean(agg_sales)
agg_lower, agg_upper = np.quantile(agg_sales, q=[.05, .95])

#plt.hist(agg_sales)
plt.axvline(agg_mean,linestyle='--',c='blue')
plt.axvline(agg_lower,linestyle='dotted',c='red')
plt.axvline(agg_upper,linestyle='dotted',c='red')

sns.distplot(agg_sales, hist=True, kde=True, color = 'blue',
             hist_kws={'edgecolor':'black'}, norm_hist=True)

plt.xlabel("Revenue ($)")
plt.savefig("figs/seen_state-dist.pdf")
plt.clf()
plt.close()