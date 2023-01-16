# General tools
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from IPython.display import Image, HTML
from plotnine import *
import pydot
from plotnine import *
from tqdm import tqdm

# For transformations and predictions
from scipy.optimize import curve_fit
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from six import StringIO

# For scoring
from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import accuracy_score as acc_classification
from sklearn.metrics import explained_variance_score as acc_regression

# For validation
from sklearn.model_selection import train_test_split as split

# sns.set_theme(style="darkgrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

path = "data.csv"
df = pd.read_csv(path,encoding='ISO-8859-1')
df.head()

# data = pd.read_csv("top10s.csv", encoding='ISO-8859-1')

# data.head()
# print(data.head())

# Adding Mean & Count values to each artist
df['mean'] = df.groupby('artists')['popularity'].transform('mean')
df['count'] = df.groupby('artists')['popularity'].transform('count')
# plotting
fig, ax = plt.subplots(figsize = (20, 3))
ax = sns.distplot(df['count'], bins = 600)
ax.set_xlabel('Count of apperances in data', fontsize=12, c='r')
ax.set_ylabel('% of artists', fontsize=12, c='r')
plt.show()

fig, ax = plt.subplots(figsize = (15, 3))
ax = sns.distplot(df['count'], bins=600, kde=False)
ax.set_xlabel('Count of appearances in data', fontsize=12, c='r')
ax.set_ylabel('# of artists', fontsize=12, c='r')
ax.set_xlim(1,20)
ax.set_xticks(range(1,21,1))
ax.axvline(x=3, ymin=0, ymax=1, color='orange', linestyle='dashed', linewidth=3)
font = {'family': 'serif',
        'color':  'red',
        'weight': 'bold',
        'size': 14,
        }
ax.annotate("", xy=(3, 19000), xytext=(4.8, 19000), arrowprops=dict(arrowstyle="->", color='r', linestyle='dashed', linewidth=3))
ax.text(x = 5, y = 19000, s='cutoff = 3', fontdict=font)

plt.show()

fig, ax = plt.subplots(figsize = (15, 3))
stat = df.groupby('count')['mean'].mean().to_frame().reset_index()
ax = stat.plot(x='count', y='mean', marker='.', linestyle = '', ax=ax)
ax.set_xlabel('Count of appearances in data', fontsize=12, c='r')
ax.set_ylabel('Mean Popularity', fontsize=12, c='r')
plt.show()

# Read column names from file
cols = list(pd.read_csv(path, encoding='ISO-8859-1',nrows =1))
df = pd.read_csv(path, encoding='ISO-8859-1',usecols=[i for i in cols if i not in ['id','name','release_date','year']])

# Remove duplicated
df = df[~df.duplicated()==1]
# df = df.sample(frac=0.3)
#Split the data to train and test
X_train, X_test, y_train, y_test = split(df.drop('popularity', axis=1), df['popularity'], test_size = 0.2, random_state = 12345)


class ArtistsTransformer():
    """ This transformer recives a DF with a feature 'artists' of dtype object
        and convert the feature to a float value as follows:
        1. Replace the data with the artists mean popularity
        2. Replace values where artists appear less than MinCnt with y.mean()
        3. Replace values where artists appear more than MaxCnt with 0

        PARAMETERS:
        ----------
        MinCnt (int): Minimal treshold of artisits apear in dataset, default = 3
        MaxCnt (int): Maximal treshold of artisits apear in dataset, default = 600

        RERTURN:
        ----------
        A DataFrame with converted artists str feature to ordinal floats
    """

    def __init__(self, MinCnt=3.0, MaxCnt=600.0):
        self.MinCnt = MinCnt
        self.MaxCnt = MaxCnt
        self.artists_df = None

    def fit(self, X, y):
        self.artists_df = y.groupby(X.artists).agg(['mean', 'count'])
        print(self.artists_df)
        self.artists_df.loc['unknown'] = [y.mean(), 1]
        self.artists_df.loc[self.artists_df['count'] <= self.MinCnt, 'mean'] = y.mean()
        self.artists_df.loc[self.artists_df['count'] >= self.MaxCnt, 'mean'] = 0
        return self

    def transform(self, X, y=None):
        X['artists'] = np.where(X['artists'].isin(self.artists_df.index), X['artists'], 'unknown')
        X['artists'] = X['artists'].map(self.artists_df['mean'])
        return X
# Apply AritistsTransformer on train and test seperatly
artists_transformer = ArtistsTransformer(MinCnt=2)
X_train = artists_transformer.fit(X_train, y_train).transform(X_train, y_train)
X_test = artists_transformer.transform(X_test, y_test)
print(X_train,y_train)

# ohe = OneHotEncoder(categories='auto', drop='first')
#
# # Train
# feature_arr = ohe.fit_transform(X_train[['instrumentalness','key']]).toarray()
# columns_key = ['key_'+str(i) for i in list(set(X_train['key'].values))[1:]]
# instrumentalness_key = ['ins_'+str(i) for i in list(set(X_train['instrumentalness'].values))[1:]]
# feature_labels = columns_key + instrumentalness_key
# feature_labels = np.concatenate((feature_labels), axis=None)
# features = pd.DataFrame(feature_arr, columns = feature_labels, index = X_train.index)
# X_train = pd.concat([X_train, features], axis=1).drop(['key','instrumentalness'], axis=1)
#
# # Test
# feature_arr = ohe.fit_transform(X_test[['instrumentalness','key']]).toarray()
# columns_key = ['key_'+str(i) for i in list(set(X_test['key'].values))[1:]]
# instrumentalness_key = ['ins_'+str(i) for i in list(set(X_test['instrumentalness'].values))[1:]]
# feature_labels = columns_key + instrumentalness_key
# feature_labels = np.concatenate((feature_labels), axis=None)
# features = pd.DataFrame(feature_arr, columns = feature_labels, index = X_test.index)
# X_test = pd.concat([X_test, features], axis=1).drop(['key','instrumentalness'], axis=1)

scaler = MinMaxScaler()
cols = ['artists','duration_ms','loudness','tempo']
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.fit_transform(X_test[cols])

# Divide the popularity by 100
y_train = y_train / 100
y_test = y_test / 100

print(X_train.head(3))
# model 1
LR = LinearRegression()
cols = [col for col in X_train.columns if abs(X_train[col].corr(y_train))>0.2]

# Fit the model and
LR.fit(X_train.drop(columns=cols), y_train)

# Train Predicting with the model
y_train_pred = LR.predict(X_train.drop(columns=cols)).clip(0, 1)

# RMSE Train
LR_rmse = np.sqrt(mse(y_train, y_train_pred))
print(f"RMSE Train = {LR_rmse:.5f}")

#Predicting with the model
y_test_pred = LR.predict(X_test.drop(columns=cols)).clip(0, 1)

# RMSE Test
LR_rmse = np.sqrt(mse(y_test, y_test_pred))
accuracy = acc_regression(y_test, y_test_pred)*100
print(f"RMSE Test = {LR_rmse:.5f}")
print(f"Accuracy Test = {accuracy:.5f}")

# model 1 with all features
LR = LinearRegression()
print('Model = LinearRegression')
# Fit the model and
LR.fit(X_train, y_train)

# Train Predicting with the model
y_train_pred = LR.predict(X_train).clip(0, 1)

# RMSE Train
LR_rmse = np.sqrt(mse(y_train, y_train_pred))
print(f"RMSE Train = {LR_rmse:.6f}")

#Predicting with the model
y_test_pred = LR.predict(X_test).clip(0, 1)

# RMSE Test
LR_rmse = np.sqrt(mse(y_test, y_test_pred))
accuracy = acc_regression(y_test, y_test_pred)*100
accuracy1 = LR.score(X_test, y_test)*100
print(f"RMSE Test = {LR_rmse:.5f}")
print(f"Accuracy Test = {accuracy:.5f}")
print(accuracy1)

# model 2
LR = RandomForestRegressor()
print('Model = RandomForestRegressor')
# Fit the model and
LR.fit(X_train, y_train)

# Train Predicting with the model
y_train_pred = LR.predict(X_train).clip(0, 1)

# RMSE Train
LR_rmse = np.sqrt(mse(y_train, y_train_pred))
print(f"RMSE Train = {LR_rmse:.6f}")

#Predicting with the model
y_test_pred = LR.predict(X_test).clip(0, 1)

# RMSE Test
LR_rmse = np.sqrt(mse(y_test, y_test_pred))
accuracy = acc_regression(y_test, y_test_pred)*100
accuracy1 = LR.score(X_test, y_test)*100
print(f"RMSE Test = {LR_rmse:.5f}")
print(f"Accuracy Test = {accuracy:.5f}")
print(accuracy1)

# model 3
LR = DecisionTreeRegressor()
print('Model = DecisionTreeRegressor')
# Fit the model and
LR.fit(X_train, y_train)

# Train Predicting with the model
y_train_pred = LR.predict(X_train).clip(0, 1)

# RMSE Train
LR_rmse = np.sqrt(mse(y_train, y_train_pred))
print(f"RMSE Train = {LR_rmse:.6f}")

#Predicting with the model
y_test_pred = LR.predict(X_test).clip(0, 1)

# RMSE Test
LR_rmse = np.sqrt(mse(y_test, y_test_pred))
accuracy = acc_regression(y_test, y_test_pred)*100
accuracy1 = LR.score(X_test, y_test)*100
print(f"RMSE Test = {LR_rmse:.5f}")
print(f"Accuracy Test = {accuracy:.5f}")
print(accuracy1)



