from envs.prior import Prior
import config.config as cfg
from envs.allocation_env import AllocationEnv
import sys

BURN_IN = sys.argv[1]
N_SAMPLES = sys.argv[2]
LOAD_MODEL = False

prior = Prior(config=cfg.vals,
              fname='prior.json')

env = AllocationEnv(config=cfg.vals, prior=prior, load_model=LOAD_MODEL)
y_hat = env.train(n_samples=N_SAMPLES, tune=BURN_IN, fname=cfg.vals['model_path'])