import numpy as np
import json
import sys
import random

random.seed(1990)


store_id = sys.argv[1]
R = int(sys.argv[2])


#R = 18
A = np.zeros((R, R))
edge_prob = .4

for i in range(R):
    for j in range(i+1, R):
        if np.random.random() < edge_prob:
            A[i, j]=1.0


A = A + A.transpose()
print(A)




rows, cols = np.where(A > 0.0)
non_zeros = list(zip(rows, cols))
non_zeros_str = [str(x) for x in non_zeros]
non_zeros_str = ",".join(non_zeros_str)
print(non_zeros_str)

output = {'n_regions': R,
          'non_zero_entries': non_zeros_str}

with open("store-{}-adj-mtx.json".format(store_id), 'w') as f:
    json.dump(output, f)