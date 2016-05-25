import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

alpha = 0.5
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


