import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from pyglmnet import GLM


#file: 			the name of the file
#header_index: 	the index of the row which contains the names of the features
def read_dataset(file,header_index):
	return pd.read_csv(file,header=header_index)


#dataframe:				the variable that contains the dataset
#label_column_index:	the index of the column that contain the labels (values of y)
def build_Xy(dataframe,label_column_index):
	return np.array(df.drop([label_column_index],axis=1)), np.array(df[label_column_index])


# reading the dataset
df = read_dataset('community_crime.csv',0)

# separating the dependent variables from the independent variables
X, y = build_Xy(df,'att128')

n_features = X.shape[1]
n_samples = X.shape[0]
#print n_samples
#print n_features


#Splitting the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5,random_state=0)

# Defining the model
reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
model = GLM(distr='poisson', verbose=False, alpha=0.01, learning_rate=1e-4, reg_lambda = reg_lambda)

print 'alpha: ', model.alpha
print 'learning rate: ', model.learning_rate


#model.threshold = 1e-5

scaler = StandardScaler().fit(X_train)
model.fit(scaler.transform(X_train),y_train)

# Fitting the model
#model.fit(X_train,y_train)


# Making the predictions base on fit model
if(model.distr == 'multinomial'):
	yt_predicted = model[-1].predict(X_test).argmax(axis=1)
	yr_predicted = model[-1].predict(X_train).argmax(axis=1)
else:
	yt_predicted = model[-1].predict(scaler.transform(X_test))
	yr_predicted = model[-1].predict(scaler.transform(X_train))


# Can we use this??
#print('Output performance = %f percent.' % (yt_predicted == y_test).mean())


#plotting the predictions
plt.plot(y_test[:500], label='tr')
plt.plot(yt_predicted[:500], 'r', label='pr')
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




