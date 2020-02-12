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

features = features[["transacted_qty", "d_open_interest", "mid", "bid1", "bid2", "ask1", "ask2",
                     "bid1vol", "bid2vol", "bid3vol", "bid4vol", "bid5vol", "ask1vol", "ask2vol", "ask3vol", "ask4vol",
                     "ask5vol"]]

features["bidcross1"] = features["bid1"] * features["bid1vol"]
features["bidcross2"] = features["bid2"] * features["bid2vol"]
features["askcross1"] = features["ask1"] * features["ask1vol"]
features["askcross2"] = features["ask2"] * features["ask2vol"]

X = preprocessing.scale(features.values)
Y = labels.values
Y = Y.reshape(len(Y))

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

score = []
kf = KFold(n_splits=10, random_state=1, shuffle=True)
from sklearn.model_selection import GridSearchCV

'''
cv_params = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    'n_estimators': [50, 100, 200, 300, 500],
    # 'min_child_weight': [0, 2, 5, 10, 20],
    # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
    # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
    # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
    # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
}
model = LGBMClassifier(random_state=1, n_jobs=-1)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)
optimized_GBM.fit(X, Y)
print(optimized_GBM.best_estimator_.get_params())
'''

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
    model = LGBMClassifier(random_state=1, reg_alpha=1, reg_lambda=1)
    model.fit(X_train, y_train)
    score.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
print(sum(score) / 10)
