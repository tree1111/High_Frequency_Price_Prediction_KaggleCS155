import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import KFold
import math
from lightgbm import LGBMClassifier

train_data = pd.read_csv("data/train.csv", index_col=0)
# train_data = train_data.fillna(0)
features = train_data.iloc[:, :-1]
labels = train_data.iloc[:, -1:]

features["amountbid1"] = features["bid1"] * features["bid1vol"] / features["last_price"]
features["amountask1"] = features["bid2"] * features["bid2vol"] / features["last_price"]

# features["bid1", "ask1", "bid2", "ask2",  "bid3", "ask3",  "bid4", "ask4",  "bid5", "ask5"] \
features["bid1"] = features["bid1"] / features["last_price"]
features["ask1"] = features["ask1"] / features["last_price"]
features["bid2"] = features["bid2"] / features["last_price"]
features["ask2"] = features["ask2"] / features["last_price"]
features["bid3"] = features["bid3"] / features["last_price"]
features["ask3"] = features["ask3"] / features["last_price"]
features["bid4"] = features["bid4"] / features["last_price"]
features["ask4"] = features["ask4"] / features["last_price"]
features["bid5"] = features["bid5"] / features["last_price"]
features["ask5"] = features["ask5"] / features["last_price"]

features["maxbid"] = features[["bid1vol", "bid2vol", "bid3vol", "bid4vol", "bid5vol"]].max(axis=1)
features["maxask"] = features[["ask1vol", "ask2vol", "ask3vol", "ask4vol", "ask5vol"]].max(axis=1)
features["bid345"] = (features["bid3"] + features["bid4"] + features["bid5"]) / 3
features["ask345"] = (features["ask3"] + features["ask4"] + features["ask5"]) / 3

features["bid345vol"] = (features["bid3vol"] + features["bid4vol"] + features["bid5vol"]) / 3
features["ask345vol"] = (features["ask3vol"] + features["ask4vol"] + features["ask5vol"]) / 3
features = features[["transacted_qty", "d_open_interest", "bid1", "bid2", "bid345", "ask1", "ask2",
                     "ask345", "bid1vol", "bid2vol", "bid345vol", "ask1vol", "ask2vol",
                     "ask345vol", "amountbid1", "amountask1", "maxask", "maxbid"]]

# features = features.fillna(0)
for column in list(features.columns[features.isnull().sum() > 0]):
    mean_val = features[column].mean()
    features[column].fillna(mean_val, inplace=True)

scalar = preprocessing.StandardScaler().fit(features.values)
X = scalar.transform(features.values)
# M = np.size(X, 0)
# N = np.size(X, 1)
# X = X + np.random.normal(0, 0.01, size=(M, N))
# X = features.values
Y = labels.values
Y = Y.reshape(len(Y))

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

score = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
models = [
    LogisticRegression(random_state=1),
    MLPClassifier(random_state=1, hidden_layer_sizes=(30, 15), batch_size=128, activation='tanh'),
    LGBMClassifier(random_state=1, reg_alpha=1, reg_lambda=1, learning_rate=0.1)
]
for model in models:
    score = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        model.fit(X_train, y_train)
        score.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
        print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print("a", sum(score) / 5)
