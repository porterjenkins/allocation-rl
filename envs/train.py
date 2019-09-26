import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.prior import Prior
import config.config as cfg
from envs.allocation_env import AllocationEnv


N_ITER = int(sys.argv[1])
N_SAMPLES = int(sys.argv[2])
LOAD_MODEL = False

prior = Prior(config=cfg.vals)

env = AllocationEnv(config=cfg.vals, prior=prior, load_model=LOAD_MODEL)
y_hat = env.train(n_iter=N_ITER, n_samples=N_SAMPLES, fname=cfg.vals['model_path'])