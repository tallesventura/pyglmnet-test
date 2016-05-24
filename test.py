import csv
import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pyglmnet import GLM
from sklearn.datasets import make_classification


df = pd.read_csv('winequalityred.csv',header=0)
ndarray_data = np.array(df)


X = ndarray_data[1:,:ndarray_data.shape[1]-1]
y = ndarray_data[1:,ndarray_data.shape[1]-1]

n_features = X.shape[1]
n_samples = X.shape[0]
print 'n_features: ',n_features
print 'n_samples: ',n_samples

X, y = make_classification(n_samples=X.shape[0], n_classes=10,
                           n_informative=n_features, n_features=n_features, n_redundant=0)


#Splitting the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
#print 'true X ',X_test.shape

"""
#creating the model
reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
model = GLM(distr='multinomial', verbose=False, alpha=0.05,
            max_iter=1000, learning_rate=1e-4,
            reg_lambda=reg_lambda)
"""
model = GLM(distr='multinomial', alpha=0.01,
               reg_lambda=np.array([0.02, 0.01]), verbose=False,)


#initial values for the coefficients
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())


model.threshold = 1e-5
#scaler = StandardScaler().fit(X_train)
#model.fit(scaler.transform(X_train),y_train)


# Fitting the model
model.fit(X_train,y_train)
#print 'reg_lambda: ',model.reg_lambda


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


yt_predicted = model[0].predict(X_test)
yr_predicted = model[0].predict(X_train)
y_test_predicted = yt_predicted.argmax(axis=1)
y_train_predicted = yr_predicted.argmax(axis=1)

##y_test_predicted = model[0].predict(X_test).argmax(axis=1)
##y_train_predicted = model[0].predict(X_train).argmax(axis=1)
#print 'true y ',y_test.shape
#print 'fit_: ',model[0].fit_['beta'].shape
#print 'predicted y ',y_test_predicted.shape


#predicting values
#y_test_predicted = model[0].predict(scaler.transform(X_test))
#y_train_predicted = model[0].predict(scaler.transform(X_train))


#plotting the predictions
plt.plot(y_test[:100], label='tr')
plt.plot(y_test_predicted[:100], 'r', label='pr')
plt.xlabel('samples')
plt.ylabel('true and predicted outputs')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()


# Compute model deviance
Dr = model[0].score(y_train, yr_predicted)
Dt = model[0].score(y_test, yt_predicted)
print('Dr = %f' % Dr, 'Dt = %f' % Dt)


# Compute pseudo-R2s
R2r = model[0].score(y_train, yr_predicted, np.mean(y_train), method='pseudo_R2')
R2t = model[0].score(y_test, yt_predicted, np.mean(y_train), method='pseudo_R2')
print('  R2r =  %f' % R2r, ' R2r = %f' % R2t)




