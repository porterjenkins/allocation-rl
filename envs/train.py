from envs.prior import Prior
import config.config as cfg
from envs.allocation_env import AllocationEnv

TRAIN_FLAG = True

prior = Prior(config=cfg.vals,
              fname='prior.json')

env = AllocationEnv(config=cfg.vals, prior=prior, data_model_path='../train-data-simple.csv', train=TRAIN_FLAG)