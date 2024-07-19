#import
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb 
from xgboost import XGBClassifier
import lightgbm as lgm
import warnings
warnings.simplefilter('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# データをロード
# load data
path  = '/kaggle/input/playground-series-s3e9/'
pretrain = pd.read_csv(path+'train.csv').drop(['id','FlyAshComponent'],axis=1)
pretest  = pd.read_csv(path+'test.csv').drop(['id','FlyAshComponent'],axis=1)
sub   = pd.read_csv(path +'sample_submission.csv')

# 目的変数
# Objective variable
target = 'Strength'


# 目的変数の列を除いたデータ
# Data excluding columns for the objective variable
train = pretrain.drop(target,axis=1)


# 目的変数列のデータ
# Data for the objective variable column
train_tgt  = pretrain[target]


# split列の作成
# Creating a split column
pretrain['split']= 'train'
pretest['split']= 'test'
data = pd.concat([pretrain,pretest]).reset_index(drop=True)


# 特徴量
# feature value
features = [c for c in train.columns if c not in ['id',target]]

# trainの欠損値情報の把握＆先頭の情報の出力
# Capturing missing value information of train 
# outputting information at the beginning of the train
pretrain.info()
pretrain.head(3)


# 特徴量の分布を図示
# Graphical representation of feature distribution
fig, axs = plt.subplots(7,1, figsize=(14,12))
for f, ax in zip(features,axs.ravel()):
    sns.histplot(data, x=f, hue='split', ax=ax)
plt.show()


# 予測に不要なのでsplitの列を削除
# Remove split columns as they are not needed for forecasting
test = pretest.drop('split',axis=1)


# 訓練・テストデータに分割
# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(train, train_tgt, test_size=0.2, shuffle=True)


# xgboostモデルの仮作成
# Tentative creation of xgboost model
model = xgb.XGBRegressor()

# ここをいじるとハイパーパラメーターを調整できます。
# You can adjust the hyper parameters by tinkering here.
depth     = [2]
estimator = [18]


# ハイパーパラメータ探索
# Hyperparameter Search
model_cv = GridSearchCV(model, {'max_depth': depth, 'n_estimators':estimator},verbose=3)
model_cv.fit(X_train, y_train)
print(model_cv.best_params_, model_cv.best_score_)


# 改めて最適パラメータで学習
# Learning again with optimal parameters
model = xgb.XGBRegressor(**model_cv.best_params_)
model.fit(X_train, y_train)



# 学習モデルの評価・出力（RMSE）
# Evaluate and output the learning model (RMSE)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print('RMSE(train data):',round(np.sqrt(mean_squared_error(y_train, y_pred_train)),3))
print('RMSE(test data):',round(np.sqrt(mean_squared_error(y_test, y_pred_test)),3))


# ミスしていないか確認するための出力
# Output to check for mistakes
print(y_pred_train)
print(y_pred_test)


#テストデータの予測,提出
#Prediction of test data, submission
pred = model.predict(test)
sub[target] = pred
sub.to_csv('submission.csv', index=False)
print(sub)

# ヒストグラム
# histogram
data = np.array(pred)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(data, bins='auto', histtype='barstacked', ec='black')
plt.show()

# 重要度の図示
# Illustration of importance
importances = pd.Series(model.feature_importances_, index =features)
importances = importances.sort_values()
importances.plot(kind = "barh")
plt.title("imporance in the xgboost Model")
plt.show()
