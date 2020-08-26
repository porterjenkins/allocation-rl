import os
import datetime

class Logger(object):

    def __init__(self, hyp, dir, prj_name, result=None):

        self.hyp = hyp
        self.result = result
        self.dir = dir
        self.prj_name = prj_name

    def set_result(self, result):
        self.result = result


    def write(self):

        now = str(datetime.datetime.now())
        fpath = self.dir + f"{self.prj_name}-{now}.txt"

        with open(fpath, "w") as f:

            f.write("------------------\n")
            f.write("Hyperparameters\n")
            f.write("-------------------\n\n")

            for k, v in self.hyp.items():
                f.write(f"{k}: {v}\n")

            f.write("\n")

            f.write("------------------\n")
            f.write("Results\n")
            f.write("-------------------\n\n")

            for k, v in self.result.items():
                f.write(f"{k}: {v}\n")

