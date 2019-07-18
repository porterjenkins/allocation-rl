import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import pandas as pd

data = pd.read_csv('test-data.csv')

train = data[['time', 'product', 'quantity', 'sales']]

X = data[['time', 'product']].values.astype(theano.config.floatX)
y = data['quantity'].values.astype(theano.config.floatX)


with pm.Model() as env_model:

    #weights = pm.Normal('weights', mu=0, sigma=1, shape=(2,1))
    weights = pm.MvNormal('weights', mu=np.zeros(2), cov=np.eye(2), shape=(1,2))
    mean = pm.math.dot(X, weights.T)

    variance = pm.Gamma('noise', alpha=2, beta=1)

    y_observed = pm.Normal('y_observed', mu=mean, sigma=variance, observed=y)



# Inference button (TM)!
with env_model:
    trace = pm.sample(500, tune=500, target_accept=.9)
    posterior_pred = pm.sample_posterior_predictive(trace)

y_hat = posterior_pred['y_observed'].mean(axis=0)

print(y_hat)
print(y)
print(y_hat.shape)
print(y.shape)

mse = np.mean(np.power((y - y_hat),2))
print("mse: {}".format(mse))

plt.figure(figsize=(7, 7))
pm.traceplot(trace)
plt.tight_layout()
plt.show()