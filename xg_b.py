import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import KFold

train_data = pd.read_csv("data/train.csv", index_col=0)
train_data = train_data.fillna(0)
features = train_data.iloc[:, :-1]
labels = train_data.iloc[:, -1:]

features["bid1"] = (features["bid1"] - features["last_price"]) / features["last_price"]
features["ask1"] = (features["ask1"] - features["last_price"]) / features["last_price"]
features["bid2"] = (features["bid1"] - features["last_price"]) / features["last_price"]
features["ask2"] = (features["ask1"] - features["last_price"]) / features["last_price"]
features["bid3"] = (features["bid1"] - features["last_price"]) / features["last_price"]
features["ask3"] = (features["ask1"] - features["last_price"]) / features["last_price"]
features["bid4"] = (features["bid1"] - features["last_price"]) / features["last_price"]
features["ask4"] = (features["ask1"] - features["last_price"]) / features["last_price"]
features["bid5"] = (features["bid1"] - features["last_price"]) / features["last_price"]
features["ask5"] = (features["ask1"] - features["last_price"]) / features["last_price"]
features = features[["transacted_qty", "d_open_interest", "bid1", "bid2", "bid3", "bid4", "bid5", "ask1", "ask2",
                     "ask3", "ask4", "ask5", "bid1vol", "bid2vol", "bid3vol", "bid4vol", "bid5vol", "ask1vol",
                     "ask2vol",
                     "ask3vol", "ask4vol", "ask5vol"]]

X = preprocessing.scale(features.values)
Y = labels.values
Y = Y.reshape(len(Y))

import xgboost as xgb

from sklearn.metrics import roc_auc_score

score = []
kf = KFold(n_splits=10, random_state=1)
from sklearn.model_selection import GridSearchCV
cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
model = xgb.XGBClassifier(random_state=1, learning_rate=0.1)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X, Y)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
'''
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
    model = xgb.XGBClassifier(random_state=1, learning_rate=0.1)
    model.fit(X_train, y_train)
    score.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
print(sum(score) / 10)
'''