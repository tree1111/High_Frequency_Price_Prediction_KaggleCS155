import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing

train_data = pd.read_csv("data/train.csv", index_col=0)
train_data = train_data.fillna(0)
features = train_data.iloc[:, :-1]
labels = train_data.iloc[:, -1:]
features = features[["last_price", "d_open_interest", "bid1", "ask1", "bid1vol", "ask1vol"]]
features["bid"] = features["bid1"] <= features["last_price"]
features["ask"] = features["ask1"] >= features["last_price"]
features = features[["d_open_interest", "bid1vol", "ask1vol", "bid", "ask"]]

X = preprocessing.scale(features.values)
X = features.values
Y = labels.values
Y = Y.reshape(len(Y))

import xgboost as xgb

from sklearn.metrics import roc_auc_score

score = []
for i in range(10):
    index = np.random.permutation(len(X))
    cur_x = X[index]
    cur_y = Y[index]
    x_train = cur_x[:530000]
    x_val = cur_x[530000:]
    y_train = cur_y[:530000]
    y_val = cur_y[530000:]
    model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    model.fit(x_train, y_train)
    score.append(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))

'''
model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
model.fit(X, Y)

test_data = pd.read_csv("data/test.csv", index_col=0)
test_data = test_data.fillna(0)
features = test_data
features = features[["last_price", "d_open_interest", "bid1", "ask1", "bid1vol", "ask1vol"]]
features["bid"] = features["bid1"] <= features["last_price"]
features["ask"] = features["ask1"] >= features["last_price"]
features = features[["d_open_interest", "bid1vol", "ask1vol", "bid", "ask"]]

X = preprocessing.scale(features.values)
X = features.values
df_test = pd.read_csv('data/test.csv', index_col=0)
df_test['Predicted'] = model.predict_proba(X)[:, 1]
df_test[['Predicted']].to_csv('submission.csv')
'''