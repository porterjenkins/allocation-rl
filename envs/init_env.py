import numpy as np


def init_env(n_regions, n_products, n_components=None):
    if n_components is None:
        n_components = np.random.randint(0, n_regions*n_products)

    cntr = 0
    seen = set()

    while cntr < n_components:
        i = np.random.randint(0, n_regions)
        j = np.random.randint(0, n_products)

        if (i, j) in seen:
            pass
        else:
            seen.add((i, j))
            cntr += 1


    return seen
