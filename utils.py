import numpy as np
import datetime

from stable_baselines.deepq.replay_buffer import ReplayBuffer

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


def get_reward(r):

    if isinstance(r, np.ndarray):
        return r[0]
    else:
        return r

def strip_reward_array(buffer):

    fresh_buffer = ReplayBuffer(len(buffer))


    print("Copying environment buffer: ")
    for i in range(len(buffer)):
        obs_t, action, reward, obs_tp1, done = buffer._storage[i]
        fresh_buffer.add(obs_t, action, reward[0], obs_tp1, done)

    return fresh_buffer


def get_action_space(n_regions, n_products):
    return (n_regions * n_products)*2 + 1


def make_bin_mtx(arr, dims):
    mtx = np.zeros(dims)
    for idx in arr:
        mtx[idx] = 1.0
    return mtx