import os

from envs.prior import Prior
from envs.allocation_env import AllocationEnv


def get_simple_simulator(config):
    """

    :param config:
    :return:
    """
    simulator_cfg = {k: v for k, v in config.items()}
    simulator_cfg["model_type"] = "linear"
    sim_nam = simulator_cfg['model_type'] + "-" + simulator_cfg['train_data'].split("/")[-1].split(".")[
        0] + "-no-precision" + ".p"
    simulator_cfg["model_path"] = os.path.join(config["prj_root"], "envs", sim_nam)

    simulator_prior = Prior(simulator_cfg)
    simulator = AllocationEnv(config=simulator_cfg,
                              prior=simulator_prior,
                              load_model=True)


    return simulator