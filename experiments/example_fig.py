import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


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
sns.heatmap(board_config.astype(int), cmap=cmap,cbar=False)

plt.xlabel('Products', fontsize=16)
plt.ylabel("Regions", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figs/example-fig-heatmap.pdf")
