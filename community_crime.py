import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

alpha = 0.1
l_ratio = 1e-4

# Split data in train set and test set
from sklearn.cross_validation import train_test_split

ds = pd.read_csv('community_crime.csv',header=0)
X = ds.values # it returns a numpy array
n_samples, n_features = X.shape

X, y = np.array(ds.drop(['att128'],axis=1)), np.array(ds['att128'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

########################################################
# Work with scikit

########################################################
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=l_ratio)
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
# print(enet)
print("r^2 on test data using sklearn : %f" % r2_score_enet)

########################################################
# Work with pyglmnet

########################################################

from pyglmnet import GLM

reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
glm = GLM(distr='poisson', alpha=alpha, reg_lambda=reg_lambda, learning_rate=l_ratio ,verbose=False)

########################################################
# Plot ...

########################################################

#plotting the predictions
plt.plot(y_test, label='tr')
plt.plot(y_pred_enet, 'r', label='sklearn-pr')
# Add the result for GLM
plt.xlabel('')
plt.ylabel('')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)

# plt.plot(enet.coef_, label='Elastic net coefficients')
# plt.plot(, '--', label='original coefficients')
# plt.legend(loc='best')
# plt.title("Elastic Net R^2: %f"
#           % ( r2_score_enet))
plt.show()

