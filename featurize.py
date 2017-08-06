# -*- coding: utf-8 -*-
"""
@author: Faron
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from mcc import mcc_xgboost, matthews_correlation
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "~/data/bosch"
TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)
TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'

SEED = 0
CHUNKSIZE = 300
#NROWS = 500000
NROWS = 600000



def get_important_data(n):
	"""Fit a classifier and select the important features"""
	print("Sampling data chunks")
	n_date_features = 1156
	date_chunks = pd.read_csv(TRAIN_DATE, index_col=0, nrows=n, chunksize=100000, dtype=np.float32)
	num_chunks = pd.read_csv(TRAIN_NUMERIC, index_col=0, nrows=n, usecols=list(range(969)), chunksize=100000, dtype=np.float32)

	X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05) for dchunk, nchunk in zip(date_chunks, num_chunks)])

	print("Sampling data chunks [y]")
	y = pd.read_csv(TRAIN_NUMERIC, index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index].values.ravel()

	print("Fitting XGBClassifier to select features")
	clf = XGBClassifier(base_score=0.005, feval=mcc_xgboost)
	clf.fit(X.values, y)

	plt.hist(clf.feature_importances_[clf.feature_importances_>0])
	important_indices = np.where(clf.feature_importances_>0.005)[0]

	#important_num_indices = np.concatenate([[0], important_indices[important_indices >= n_date_features] + 1 - n_date_features])
	#important_date_indices = np.concatenate([[0], important_indices[important_indices < n_date_features] + 1])
	#important_num_names = list(X.columns[important_num_indices])
	#important_date_names = list(X.columns[important_date_indices])
	print("Reading Numeric Data")

	important_date = np.concatenate([[0], important_indices[important_indices < n_date_features] + 1])
	important_num = np.concatenate([[0], important_indices[important_indices >= n_date_features] + 1 - n_date_features])

	train = pd.concat([
	    pd.read_csv(TRAIN_DATE, nrows=n, dtype=np.float32, usecols=important_date),
	    pd.read_csv(TRAIN_NUMERIC, nrows=n, dtype=np.float32, usecols=important_num),
	], axis=1)

	test = pd.concat([
	    pd.read_csv(TEST_DATE, nrows=n, dtype=np.float32, usecols=important_date),
	    pd.read_csv(TEST_NUMERIC, nrows=n, dtype=np.float32, usecols=important_num),
	], axis=1)

	return train, test



# Find all the names that we want to load
print("Reading Column Indexes")
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

# Extend Farons magic features with good features
ntrain, ntest = get_important_data(NROWS)

print('ntrain test', ntrain.shape)
print('ntrain test', ntest.shape)

train = pd.concat((train,ntrain),axis=1)
test = pd.concat((test,ntest),axis=1)

print('train test', train.shape)
print('train test', test.shape)

# Remove duplicates
train = train.loc[:, ~train.columns.duplicated()]
test = test.loc[:, ~test.columns.duplicated()]

# Process test and train features together
ntrain = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)

train_test['f1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['f2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)

train_test['f3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
train_test['f4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)

train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)

# Split back into train and test
train = train_test.iloc[:ntrain, :]
test  = train_test.iloc[ntrain:, :]

print('train test', train.shape)
print('train test', test.shape)

# Extract y from the dataframes
y = train.Response.ravel()

features = np.setdiff1d(list(train.columns), [TARGET_COLUMN, ID_COLUMN])

# Convert pandas to Numpy
trainX = np.array(train[features])
testX = np.array(test[features])





"""


# Test it out
print('Training network')

prior = np.sum(y) / (1.*len(y))

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 2,
    'eval_metric': 'auc',
    'base_score': prior
}

dtrain = xgb.DMatrix(trainX, label=y)
res = xgb.cv(xgb_params, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True, early_stopping_rounds=1, verbose_eval=1, feval=mcc_xgboost, show_stdv=True)

cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]

print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

"""
positive = np.sum(y==1)
negative = np.sum(y==0)
print('Total {0} positive and {1} negative samples'.format(positive,negative))


def weighted_loss(y_true, y_pred):
	num_classes = 2
	epsilon = tf.constant(value=1e-3)

	with tf.name_scope('loss_1'):
		# Make y_pred into a tensor with one column for each class
		y_pred = tf.reshape(y_pred, (-1, 1))
		logits = tf.concat((1-y_pred, y_pred), axis=1)
		logits = logits + epsilon

		# Make y_true into a tensor with one column for each class
		labels = tf.cast(y_true, tf.uint8)
		labels = tf.reshape(labels, (-1, 1))
		labels = tf.reshape(tf.one_hot(labels, depth=num_classes), (-1, num_classes))

		#counts = tf.reduce_sum(labels, axis=0) + epsilon
		counts = tf.constant([negative, positive], dtype=tf.float32)
		losses = tf.square(labels-logits)
		losses = tf.reduce_sum(losses, axis=0)
		losses = tf.square(losses/counts) # Square again as in Wang 2016
		loss = tf.reduce_mean(losses)

		# Return mean cross entropy loss
		#cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(labels + epsilon), coefficients), reduction_indices=[1])
		#cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		return loss




# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense

# Some preprocessing
trainX = np.nan_to_num(trainX)
trainX = normalize(trainX, axis=0, norm='max')

# Extract the number of features
nfeatures = trainX.shape[1]

# create model
model = Sequential()
model.add(Dense(12, input_dim=nfeatures, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss=weighted_loss, optimizer='adam', metrics=['accuracy',matthews_correlation])

# Fit the model
model.fit(trainX, y, epochs=200, validation_split=0.1, batch_size=256)

# evaluate the model
scores = model.evaluate(trainX, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))







