import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd



s1 = pd.read_csv("output/store-1-optim-workshop.csv",index_col=0)
s1.columns = ["30 days", '60 days', '90 days']

s1 = s1.transpose()

s1 = s1 /1000.0


fig = plt.figure(figsize=(4,6))


s1.plot(kind='bar', rot=0, color=['slategray', 'goldenrod', 'cornflowerblue', 'seagreen'])
plt.tick_params(axis='x',labelsize=16)
plt.legend(fontsize=16)
plt.xlabel("")
plt.ylabel('Cumulative reward (in thousands of $)', fontsize=14)
plt.savefig("figs/optimization-store-1.pdf")
plt.clf()
plt.close()


s2 = pd.read_csv("output/store-2-optim-workshop.csv",index_col=0)
s2.columns = ["30 days", '60 days', '90 days']

s2 = s2.transpose()

s2 = s2 /1000.0


fig = plt.figure(figsize=(4,6))


s2.plot(kind='bar', rot=0, color=['slategray', 'goldenrod', 'cornflowerblue', 'seagreen'])
plt.tick_params(axis='x',labelsize=16)
plt.legend(fontsize=16)
plt.xlabel("")
plt.ylabel('Cumulative reward (in thousands of $)', fontsize=14)
plt.savefig("figs/optimization-store-2.pdf")
plt.clf()
plt.close()