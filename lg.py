import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import KFold

train_data = pd.read_csv("data/train.csv", index_col=0)
train_data = train_data.fillna(0)
features = train_data.iloc[:, :-1]
labels = train_data.iloc[:, -1:]
features["mid"] = (features["mid"] - features["last_price"]) / features["last_price"]
features["bid1"] = (features["bid1"] - features["last_price"]) / features["last_price"]
features["ask1"] = (features["ask1"] - features["last_price"]) / features["last_price"]
features["bid2"] = (features["bid2"] - features["last_price"]) / features["last_price"]
features["ask2"] = (features["ask2"] - features["last_price"]) / features["last_price"]

features = features[
    ["transacted_qty", "d_open_interest", "mid",
     "bid1", "bid2",
     "ask1", "ask2",
     "bid1vol", "bid2vol", "bid3vol", "bid4vol", "bid5vol",
     "ask1vol", "ask2vol", "ask3vol", "ask4vol", "ask5vol"]]

features["bidcross1"] = features["bid1"] * features["bid1vol"]
features["bidcross2"] = features["bid2"] * features["bid2vol"]
features["askcross1"] = features["ask1"] * features["ask1vol"]
features["askcross2"] = features["ask2"] * features["ask2vol"]

scalar = preprocessing.StandardScaler().fit(features.values)
X = scalar.transform(features.values)
Y = labels.values
Y = Y.reshape(len(Y))



from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
score = []
kf = KFold(n_splits=10, random_state=1, shuffle=True)

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
    #model = LGBMClassifier(random_state=1, reg_alpha=1, reg_lambda=1, learning_rate=0.1)
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(30, 30), batch_size=128, activation='relu')
    model.fit(X_train, y_train)
    score.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
print(sum(score) / 10)
