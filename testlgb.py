import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing

train_data = pd.read_csv("data/train.csv", index_col=0)
#train_data = train_data.fillna(0)
features = train_data.iloc[:, :-1]
labels = train_data.iloc[:, -1:]

features["amountbid1"] = features["bid1"] * features["bid1vol"]
features["amountask1"] = features["bid2"] * features["bid2vol"]
#features["bid1", "ask1", "bid2", "ask2",  "bid3", "ask3",  "bid4", "ask4",  "bid5", "ask5"] \
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

#features = features.fillna(0)
for column in list(features.columns[features.isnull().sum() > 0]):
    mean_val = features[column].mean()
    features[column].fillna(mean_val, inplace=True)

X = preprocessing.scale(features.values)
#X = features.values
Y = labels.values
Y = Y.reshape(len(Y))

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

max_leaves = [127, 255, 511]
num_est = [ 100, 300, 500, 700, 900]
for num in num_est:
    for max_leaf in max_leaves:
        score = []
        print('leaves = ', max_leaf, 'num_est = ', num)
        for i in range(10):
            index = np.random.permutation(len(X))
            cur_x = X[index]
            cur_y = Y[index]
            x_train = cur_x[:530000]
            x_val = cur_x[530000:]
            y_train = cur_y[:530000]
            y_val = cur_y[530000:]
            model = lgb.LGBMClassifier(random_state=1, learning_rate=0.1, num_leaves=max_leaf, n_estimators=num)
            model.fit(x_train, y_train)
            score.append(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))

        print(score)
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