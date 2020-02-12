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

scalar = preprocessing.StandardScaler().fit(features.values)
X = scalar.transform(features.values)
Y = labels.values
Y = Y.reshape(len(Y))

import xgboost as xgb


model = xgb.XGBClassifier(random_state=1, learning_rate=0.1)
model.fit(X, Y)

test_data = pd.read_csv("data/test.csv", index_col=0)
test_data = test_data.fillna(0)
features = test_data
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

X = scalar.transform(features.values)
df_test = pd.read_csv('data/test.csv', index_col=0)
df_test['Predicted'] = model.predict_proba(X)[:, 1]
df_test[['Predicted']].to_csv('submission.csv')
