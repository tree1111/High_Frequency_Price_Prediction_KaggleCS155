import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing

train_data = pd.read_csv("data/train.csv", index_col=0)
# train_data = train_data.fillna(0)
features = train_data.iloc[:, :-1]
labels = train_data.iloc[:, -1:]

features["amountbid1"] = features["bid1"] * features["bid1vol"]
features["amountask1"] = features["bid2"] * features["bid2vol"]
# features["bid1", "ask1", "bid2", "ask2",  "bid3", "ask3",  "bid4", "ask4",  "bid5", "ask5"] \
#    = features["bid1", "ask1", "bid2", "ask2",  "bid3", "ask3",  "bid4", "ask4",  "bid5", "ask5"] - features["last_price"]

features["ask1"] = features["ask1"] / features["last_price"]
features["bid2"] = features["bid1"] / features["last_price"]
features["ask2"] = features["ask1"] / features["last_price"]
features["bid3"] = features["bid1"] / features["last_price"]
features["ask3"] = features["ask1"] / features["last_price"]
features["bid4"] = features["bid1"] / features["last_price"]
features["ask4"] = features["ask1"] / features["last_price"]
features["bid5"] = features["bid1"] / features["last_price"]
features["ask5"] = features["ask1"] / features["last_price"]
features["bid345"] = (features["bid3"] + features["bid4"] + features["bid5"]) / 3
features["ask345"] = (features["ask3"] + features["ask4"] + features["ask5"]) / 3
features["bid345vol"] = (features["bid3vol"] + features["bid4vol"] + features["bid5vol"]) / 3
features["ask345vol"] = (features["ask3vol"] + features["ask4vol"] + features["ask5vol"]) / 3
features = features[["transacted_qty", "d_open_interest", "bid1", "bid2", "bid345", "ask1", "ask2",
                     "ask345", "bid1vol", "bid2vol", "bid345vol", "ask1vol", "ask2vol",
                     "ask345vol", "amountbid1", "amountask1"]]

# features = features.fillna(0)
for column in list(features.columns[features.isnull().sum() > 0]):
    mean_val = features[column].mean()
    features[column].fillna(mean_val, inplace=True)

X = preprocessing.scale(features.values)
Y = labels.values
Y = Y.reshape(len(Y))

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

score = []
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=1)
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
    model = lgb.LGBMClassifier(random_state=1, num_leaves=511, n_estimators=500)
    model.fit(X_train, y_train)
    score.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
print(sum(score) / 10)

print(score)
