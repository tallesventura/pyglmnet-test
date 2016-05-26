"""
==============================
Communities and Crime - scikit and pyglmnet
==============================

This is a real life example demonstrating how pyglmnet with
poisson exponential distribution works and comparing it against scikit.

"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

# Split data in train set and test set
from sklearn.cross_validation import train_test_split

ds = pd.read_csv('community_crime.csv',header=0)
X = ds.values # it returns a numpy array
n_samples, n_features = X.shape

X, y = np.array(ds.drop(['att128'],axis=1)), np.array(ds['att128'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

########################################################
# Work with pyglmnet

########################################################

from pyglmnet import GLM

# Think about this variable
# reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
glm = GLM(distr='poissonexp', alpha=0.2, learning_rate=1e-3, verbose=False)

glm.fit(X_train, y_train)

y_pred_glm = glm[-1].predict(X_test)

r2_score_glm = glm[-1].score(y_test, y_pred_glm, np.mean(y_train), method='pseudo_R2')
# r2_score_glm = r2_score(y_test, y_pred_glm)
print("r^2 on test data using pyglmnet : %f" % r2_score_glm)

########################################################
# Work with scikit

########################################################
from sklearn.linear_model import ElasticNet

lambda_ = glm[-1].reg_lambda

enet = ElasticNet(alpha=0.2, l1_ratio=lambda_)
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
# print(enet)
print("r^2 on test data using sklearn : %f" % r2_score_enet)

########################################################
# Plot ...

########################################################

#plotting the predictions
plt.plot(y_test, label='tr')
plt.plot(y_pred_enet, 'r', label='sklearn-pr')
# Add the result for GLM
plt.plot(y_pred_glm, 'g', label='glm-pr')
plt.xlabel('')
plt.ylabel('')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()

