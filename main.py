import numpy as np
import pandas as pd
import pdb, sys, os, itertools
np.random.seed(1006)
np.set_printoptions(linewidth=200,threshold=50)

from sklearn.model_selection import train_test_split, LeavePGroupsOut, KFold
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_squared_error as mse 

from scipy import stats

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15,10
rcParams['font.size'] = 10
rcParams['xtick.labelsize'] = 10 
rcParams['ytick.labelsize'] = 10 
plt.style.use('ggplot')


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_id = test.Id
y_name = 'SalePrice'

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train = train.select_dtypes(numerics)
train = train.dropna(axis=1)
train.drop('Id',axis=1,inplace=True)

X = train.drop([y_name],axis=1).values
y = train[y_name].values
y = np.log(y)

train_cols = train.drop([y_name],axis=1).columns
test = test[train_cols]
test = test.fillna(test.mean())


fold = 0

rmse = lambda x,y: np.sqrt(mse(x,y))

n_folds =  5
kf = KFold(n_splits=n_folds,random_state = 1006, shuffle = True)
for tr_idx, val_idx in kf.split(X,y):

    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    rf = RandomForestRegressor(n_estimators=100,
            max_depth=10,
            criterion = 'mse')
    rf.fit(X_tr,y_tr)
    pred_tr = rf.predict(X_tr)
    pred_val = rf.predict(X_val)
    train_mse = rmse(y_tr,pred_tr)
    test_mse = rmse(y_val,pred_val)
    print("FOLD = {0}, train, val error = {1:.3f}, {2:.3f}.".format(fold,train_mse,test_mse))


X_te = test.values 
rf.fit(X,y) # fit all
predictions = np.exp(rf.predict(X_te))
predictions = np.vstack((test_id,predictions)).transpose()
predictions = pd.DataFrame(predictions,columns=['Id','SalePrice'])
predictions.Id = predictions.Id.astype(int)
predictions.to_csv("preds.csv",index=0)
pdb.set_trace()





