import numpy as np


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