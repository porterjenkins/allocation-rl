import numpy as np
import datetime


def check_pos_def(x):
    is_pos_def = np.all(np.linalg.eigvals(x) >= 0)
    if not is_pos_def:
        raise Exception("Prior matrix is not positive semidefinite")


def get_norm_laplacian(A, n_regions):
    """
    Normalized Graph Laplacian - used as precision matrix
    :param A:
    :return:
    """
    I = np.eye(n_regions)
    #A = self.adj_mtx - I
    D = np.eye(n_regions) * A.sum(axis=1)
    D_inv = np.power(np.linalg.inv(D), 0.5)
    L_norm = I - np.dot(np.dot(D_inv, A), D_inv)
    check_pos_def(L_norm)

    return L_norm


def mae(y_hat, y_true):
    err = y_true - y_hat
    return np.mean(np.abs(err))

def rmse(y_hat, y_true):
    err = y_true - y_hat
    return np.mean(np.power(err, 2))

def mape(y_hat, y_true):
    err = np.abs((y_true - y_hat) / y_true)
    return np.mean(err)



def check_draws_inf(draws):
    is_inf = np.isinf(draws)
    is_inf_sums = is_inf.sum(axis=0)
    draws[is_inf] = np.nan
    means = np.nanmean(draws, axis=0)
    for col in range(draws.shape[1]):
        draws[is_inf[:, col], col] = means[col]
        if is_inf_sums[col] == draws.shape[0]:
            raise Exception("All draws are inf for sample: {}".format(col))
    return draws

def serialize_floats(arr):
    return ["{}".format(x) for x in arr]


def get_store_id(trn_data_path):
    store = trn_data_path.split("/")[-1].split(".")[0]
    store_id = "-".join(store.split("-")[:2])
    return store_id