import numpy as np
import pandas as pd
import pdb, sys, os, itertools
np.random.seed(1006)
np.set_printoptions(linewidth=200,threshold=50)

from sklearn.model_selection import train_test_split, LeavePGroupsOut, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
y = train[y_name].values
y = np.log(y)


n_train = train.shape[0]
n_test = test.shape[0]

# Join train and test to perform same preprocessing on both
train_test = pd.concat((train,test),axis=0).reset_index(drop=True)
train_test.drop('Id',axis=1,inplace=True)

assert np.isnan(train_test.iloc[:n_train].SalePrice).sum()==0, "Make sure n_train signifys end of train set"
assert np.isnan(train_test.iloc[n_train:].SalePrice).sum()==n_test, "Make sure n_train signifys end of train set"

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
non_numeric = ['object']

train_test_cat = train_test.select_dtypes(non_numeric)
train_test = train_test.select_dtypes(numerics)
train_test = train_test.dropna(axis=1)

def inspect():
    train = pd.read_csv("train.csv")
    tr = train.select_dtypes(non_numeric)
    tr[y_name] = train[y_name]

    for col in tr.columns:
        print("*"*100)
        print(col)
        subset = tr[[col,y_name]].groupby(col)
        print(subset.mean())
        print(subset.count())
        print("*"*100)


#inspect()
    
cols_to_add = ['MSZoning','LotShape','SaleCondition','SaleType']
to_dummy = pd.get_dummies(train_test_cat[cols_to_add])
train_test = pd.concat((train_test,to_dummy),axis=1)


# Now split them up again
train, test = train_test[:n_train], train_test[n_train:]
X = train.values



# KFOLD validation
fold = 0
n_folds =  5
kf = KFold(n_splits=n_folds,random_state = 1006, shuffle = True)
train_mses = []
val_mses = []
rmse = lambda x,y: np.sqrt(mse(x,y))

def ensemble_pred(ensemble,X):
    predictions= []
    for algorithm in ensemble:
        predictions.append(algorithm.predict(X))
    predictions = np.array(predictions).mean(0)
    return predictions


for tr_idx, val_idx in kf.split(X,y):

    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model1 = RandomForestRegressor(n_estimators=150, max_depth=None, criterion = 'mse')
    model2 = GradientBoostingRegressor(n_estimators=150,learning_rate=0.1,loss='ls')
    model1.fit(X_tr,y_tr)
    model2.fit(X_tr,y_tr)
    ensemble = [model1,model2]
    pred_tr = ensemble_pred(ensemble,X_tr)
    pred_val = ensemble_pred(ensemble,X_val)
    train_mse = rmse(y_tr,pred_tr)
    val_mse = rmse(y_val,pred_val)
    train_mses.append(train_mse)
    val_mses.append(val_mse)
    print("FOLD = {0}, train, val error = {1:.4f}, {2:.4f}.".format(fold,train_mse,val_mse))

train_mses, val_mses = [np.array(x).mean() for x in [train_mses,val_mses]]
print("Average train, val error = {0:.3f}, {1:.3f}.".format(train_mses,val_mses))

X_te = test.values 
model1.fit(X,y)
model2.fit(X,y)
ensemble = [model1,model2]
predictions = np.exp(ensemble_pred(ensemble,X_te))
predictions = np.vstack((test_id,predictions)).transpose()
predictions = pd.DataFrame(predictions,columns=['Id','SalePrice'])
predictions.Id = predictions.Id.astype(int)
predictions.to_csv("preds.csv",index=0)
pdb.set_trace()





