import numpy as np
import json
import config.config as cfg


class Prior(object):
    """Prior specification for world model"""

    def __init__(self, fname, config):
        self.fname = fname
        self.n_regions = config['n_regions']
        self.n_products = config['n_products']
        self.n_temporal_features = config['n_temporal_features']
        self.adj_mtx = config['adj_mtx']
        self.__get_priors(self.fname)


    def __load_file(self, fname):
        with open(fname) as f:
            prior = json.load(f)
        return prior

    def __get_loc_val(self, loc_val, size):
        if isinstance(loc_val, list):
            return np.array(loc_val)
        else:
            return np.ones(size)*loc_val

    def __get_scale_val(self, scale_val, size):
        if isinstance(scale_val, np.ndarray):
            return scale_val
        else:
            return np.eye(size)*scale_val


    def __get_priors(self, fname):
        prior = self.__load_file(fname)

        # prior for region weights
        self.loc_w_r = self.__get_loc_val(prior['loc_w_r'], self.n_regions)
        self.scale_w_r = self.adj_mtx*prior['scale_w_r']
        # prior for product wieghts
        self.loc_w_p = self.__get_loc_val(prior['loc_w_p'], self.n_products)
        self.scale_w_p = self.__get_scale_val(prior['loc_w_p'], self.n_products)
        #prior for customer weight
        self.loc_w_c = prior['loc_w_c']
        self.scale_w_c = prior['scale_w_c']
        # prior for prev sales weight
        self.loc_w_s = prior['loc_w_s']
        self.scale_w_s = prior['scale_w_s']
        # prior for day
        self.loc_w_t = self.__get_loc_val(prior['loc_w_t'], self.n_products)
        self.scale_w_t = self.__get_scale_val(prior['loc_w_p'], self.n_products)


if __name__ == "__main__":
    prior = Prior(config=cfg.vals,
                  fname='prior.json')

    stop = 0