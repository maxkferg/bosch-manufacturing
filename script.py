# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from mcc import mcc_xgboost


DATA_DIR = "~/data/bosch"
TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)
TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)






import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


# I'm limited by RAM here and taking the first N rows is likely to be
# a bad idea for the date data since it is ordered.
# Sample the data in a roundabout way:
print("Reading data chunks")
date_chunks = pd.read_csv(TRAIN_DATE, index_col=0, chunksize=100000, dtype=np.float32)
num_chunks = pd.read_csv(TRAIN_NUMERIC, index_col=0,
                         usecols=list(range(969)), chunksize=100000, dtype=np.float32)

X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)
               for dchunk, nchunk in zip(date_chunks, num_chunks)])

print("Reading data chunks [y]")
y = pd.read_csv(TRAIN_NUMERIC, index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index].values.ravel()
X = X.values


# load entire dataset for these features.
# note where the feature indices are split so we can load the correct ones straight from read_csv
print("Fitting XGBClassifier")
clf = XGBClassifier(base_score=0.005, feval=mcc_xgboost)
clf.fit(X, y)


plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices = np.where(clf.feature_importances_>0.005)[0]
print(important_indices)


print("Reading Date Features")
n_date_features = 1156
X = np.concatenate([
    pd.read_csv(TRAIN_DATE, index_col=0, dtype=np.float32,
                usecols=np.concatenate([[0], important_indices[important_indices < n_date_features] + 1])).values,
    pd.read_csv(TRAIN_NUMERIC, index_col=0, dtype=np.float32,
                usecols=np.concatenate([[0], important_indices[important_indices >= n_date_features] + 1 - 1156])).values
], axis=1)
y = pd.read_csv(TRAIN_NUMERIC, index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()



print("Reading Date Features")
clf = XGBClassifier(max_depth=5, base_score=0.005, feval=mcc_xgboost,)
cv = StratifiedKFold(y, n_folds=3)
preds = np.ones(y.shape[0])
#for i, (train, test) in enumerate(cv):
#    preds[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
#    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], preds[test])))
#print(roc_auc_score(y, preds))


# pick the best threshold out-of-fold
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(y, preds>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())

# load test data
X = np.concatenate([
    pd.read_csv(TEST_DATE, index_col=0, dtype=np.float32,
                usecols=np.concatenate([[0], important_indices[important_indices<1156]+1])).values,
    pd.read_csv(TEST_NUMERIC, index_col=0, dtype=np.float32,
                usecols=np.concatenate([[0], important_indices[important_indices>=1156] +1 - 1156])).values
], axis=1)


# generate predictions at the chosen threshold
preds = (clf.predict_proba(X)[:,1] > best_threshold).astype(np.int8)










###########################################################################################









ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'

SEED = 0
CHUNKSIZE = 50000
NROWS = 250000


FILENAME = "etimelhoods"

print("Reading Numeric Data")
train = pd.read_csv(TRAIN_NUMERIC, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)
test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)

train["StartTime"] = -1
test["StartTime"] = -1


nrows = 0
print("Reading Date Data")
for tr, te in zip(pd.read_csv(TRAIN_DATE, chunksize=CHUNKSIZE), pd.read_csv(TEST_DATE, chunksize=CHUNKSIZE)):
    feats = np.setdiff1d(tr.columns, [ID_COLUMN])

    stime_tr = tr[feats].min(axis=1).values
    stime_te = te[feats].min(axis=1).values

    train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
    test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te

    nrows += CHUNKSIZE
    if nrows >= NROWS:
        break


ntrain = train.shape[0]
print('-->',train.shape)
print('-->',test.shape)
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)

train_test['0_¯\_(ツ)_/¯_1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['0_¯\_(ツ)_/¯_2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)

train_test['0_¯\_(ツ)_/¯_3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['0_¯\_(ツ)_/¯_4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)
train = train_test.iloc[:ntrain, :]

features = np.setdiff1d(list(train.columns), [TARGET_COLUMN, ID_COLUMN])

y = train.Response.ravel()
train = np.array(train[features])

print('train: {0}'.format(train.shape))
print('trainX: {0}'.format(X.shape))
prior = np.sum(y) / (1.*len(y))

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 2,
    'eval_metric': 'auc',
    'base_score': prior
}






# Concatenate all numeric features
train = pd.concat([X, train], axis=1)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 2,
    'eval_metric': 'auc',
    'base_score': prior
}




print('Training network')
dtrain = xgb.DMatrix(train, label=y)
res = xgb.cv(xgb_params, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True, early_stopping_rounds=1, verbose_eval=1, feval=mcc_xgboost, show_stdv=True)

cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))


