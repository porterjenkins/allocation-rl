import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



board_config = np.zeros((4, 5))
board_config[(0,3)] = 1.0
board_config[(3,1)] = 1.0
board_config[(1,2)] = 1.0
board_config[(2,0)] = 1.0
board_config[(3,4)] = 1.0


fig, ax = plt.subplots()
# define the colors
cmap = mpl.colors.ListedColormap(['lightgray', 'blue'])
# create a normalize object the describes the limits of
# each color
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#plt.imshow(board_config.astype(int), interpolation='none', cmap=cmap, norm=norm)
sns.heatmap(board_config.astype(int), cmap=cmap,cbar=False, linewidths=.5)

plt.xlabel('Products', fontsize=16)
plt.ylabel("Regions", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figs/example-fig-heatmap.pdf")


A = np.zeros((4, 4))
A[0, 1] = 1.0
A[0, 3] = 1.0
A[1, 0] = 1.0
A[1, 3] = 1.0
A[1, 2] = 1.0
A[2, 1] = 1.0
A[2, 3] = 1.0
A[3, 0] = 1.0
A[3, 1] = 1.0
A[3, 2] = 1.0


# define the colors
cmap = mpl.colors.ListedColormap(['lightgray', 'mediumseagreen'])
# create a normalize object the describes the limits of
# each color
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#plt.imshow(board_config.astype(int), interpolation='none', cmap=cmap, norm=norm)
sns.heatmap(A.astype(int), cmap=cmap,cbar=False, linewidths=.5)

plt.xlabel('Regions', fontsize=16)
plt.ylabel("Regions", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figs/example-adj-mtx.pdf")


np.random.seed(12345)
board_config = np.random.uniform(size=4*5).reshape(4,5)
board_config = board_config / board_config.sum(axis=0)
#for i in range(4):
#    board_config[i, :] = np.random.uniform(5)

sns.heatmap(board_config, cmap='YlOrRd', linewidths=.5)


plt.xlabel('Products', fontsize=16)
plt.ylabel("Regions", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figs/example-fig-heatmap-sales.pdf")

