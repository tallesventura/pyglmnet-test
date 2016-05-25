import csv
import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pyglmnet import GLM
from sklearn.datasets import make_classification


# reading the dataset
df = pd.read_csv('winequalityred.csv',header=0)


y = np.array(df['quality'].astype(int))
X = np.array(df.drop(['quality'],axis=1))


#Splitting the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

# Defining the model
model = GLM(distr='multinomial', alpha=0.01,
               reg_lambda=np.array([0.02, 0.01]), learning_rate=3e-3 ,verbose=False,)


#initial values for the coefficients
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())


model.threshold = 1e-5

#scaler = StandardScaler().fit(X_train)
#model.fit(scaler.transform(X_train),y_train)

# Fitting the model
model.fit(X_train,y_train)


#ploting the fit coefficients
# TODO: fix this graph
fit_param = model[0].fit_
plt.plot(beta[:], 'bo', label ='bo')
plt.plot(fit_param['beta'][:], 'ro', label='ro')
plt.xlabel('samples')
plt.ylabel('outputs')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()

# Makin the predictions base on fit model
yt_predicted = model[-1].predict(X_test)
yr_predicted = model[-1].predict(X_train)
y_test_predicted = yt_predicted.argmax(axis=1)
y_train_predicted = yr_predicted.argmax(axis=1)

# Can we use this??
print('Output performance = %f percent.' % (y_test_predicted == y_test).mean())


#plotting the predictions
plt.plot(y_test[:500], label='tr')
plt.plot(y_test_predicted[:500], 'r', label='pr')
plt.xlabel('samples')
plt.ylabel('true and predicted outputs')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()


"""
# Compute model deviance
Dr = model[0].score(y_train, yr_predicted)
Dt = model[0].score(y_test, yt_predicted)
print('Dr = %f' % Dr, 'Dt = %f' % Dt)


# Compute pseudo-R2s
R2r = model[0].score(y_train, yr_predicted, np.mean(y_train), method='pseudo_R2')
R2t = model[0].score(y_test, yt_predicted, np.mean(y_train), method='pseudo_R2')
print('  R2r =  %f' % R2r, ' R2r = %f' % R2t)
"""



