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

#train_test_cat = train_test.select_dtypes(non_numeric)

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
    

#cols_to_add = ['MSZoning','LotShape','SaleCondition','SaleType','PavedDrive','GarageFinish','KitchenQual','BsmtFinType1', 'BsmtQual','Foundation','MasVnrType','Neighborhood']
cols_to_add = train_test.select_dtypes(non_numeric)
train_test_numeric = train_test.select_dtypes(numerics)
to_dummy = pd.get_dummies(train_test[cols_to_add.columns])
train_test = pd.concat((train_test_numeric,to_dummy),axis=1)

# KFOLD validation
fold = 0
n_folds =  10
kf = KFold(n_splits=n_folds,random_state = 1006, shuffle = True)
rmse = lambda x,y: np.sqrt(mse(x,y))

def ensemble_pred(ensemble,X):
    predictions= []
    for algorithm in ensemble:
        predictions.append(algorithm.predict(X))
    predictions = np.array(predictions).mean(0)
    return predictions

feat_importance = lambda model: model.feature_importances_.argsort()[::-1]

# Now split up train and test them up again
n_important = 5 

tr_loss =  []
val_loss = []
n_feats = []
feat_val = []

while True:
    print("*"*100)
    feat_importance_names = []
    worst_features = []
    worst_feature_values = []

    features_to_remove = []
    train, test = train_test[:n_train].copy(), train_test[n_train:].copy()
    X = train.values

    fold = 0
    train_mses = []
    val_mses = []

    for tr_idx, val_idx in kf.split(X,y):

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        #model1 = RandomForestRegressor(n_estimators=150, max_depth=None, criterion = 'mse')
        model2 = GradientBoostingRegressor(n_estimators=150,learning_rate=0.1,loss='ls')

        #model1.fit(X_tr,y_tr)
        model2.fit(X_tr,y_tr)
        ensemble = [model2]
        pred_tr = ensemble_pred(ensemble,X_tr)
        pred_val = ensemble_pred(ensemble,X_val)
        train_mse = rmse(y_tr,pred_tr)
        val_mse = rmse(y_val,pred_val)
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        print("FOLD = {0}, train, val error = {1:.4f}, {2:.4f}.".format(fold,train_mse,val_mse))

        feature_importance =  model2.feature_importances_.argsort()[::-1] # in descending order of importance


        feat_importance_names.append(train_test.columns[feature_importance][:n_important].values)

        worst_features_ = train_test.columns[feature_importance][-n_important:].values
        worst_features.append(worst_features_)
        worst_feature_values_ = model2.feature_importances_[feature_importance[-n_important:]]
        worst_feature_values.append(worst_feature_values_)
        fold += 1


    train_mses, val_mses = [np.array(x).mean() for x in [train_mses,val_mses]]
    worst_feature_values = np.array(worst_feature_values).mean()

    print("Average train, val error = {0:.4f}, {1:.4f}. Worst feature value mean = {2:.5f}".format(train_mses,val_mses,worst_feature_values))

    worst_features = pd.Series(pd.DataFrame(worst_features).values.flatten()).value_counts()
    worst_features = worst_features[worst_features>1].index

    if worst_feature_values > 0.0:
        break

    tr_loss.append(train_mses) 
    val_loss.append(val_mses)
    n_feats.append(train_test.shape[1])
    feat_val.append(worst_feature_values)

    feat_importance_names = pd.DataFrame(feat_importance_names)
    train_test.drop(worst_features,axis=1,inplace=True)
    shape = train_test.shape
    print("Shape is now {0}".format(shape))
    print("*"*100)


x = np.arange(len(val_loss))
plt.subplot(221); plt.plot(x,tr_loss); plt.plot(x,val_loss); plt.title("Train/Validation loss");
plt.subplot(223); plt.plot(x,n_feats); plt.title("n feats");
plt.subplot(224); plt.plot(x,feat_val); plt.title("Feature importance of ones removed");
plt.savefig("perf.jpg")

def fit_test():
    # now final fit and test
    train, test = train_test[:n_train].copy(), train_test[n_train:].copy()
    X = train.values
    X_te = test.values 
    #model1.fit(X,y)
    model2.fit(X,y)
    ensemble = [model2]
    predictions = np.exp(ensemble_pred(ensemble,X_te))
    predictions = np.vstack((test_id,predictions)).transpose()
    predictions = pd.DataFrame(predictions,columns=['Id','SalePrice'])
    predictions.Id = predictions.Id.astype(int)
    predictions.to_csv("preds.csv",index=0)

fit_test()




