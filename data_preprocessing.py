import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def KbestFeatures(X_train,y_train,k):
    # Create an SelectKBest object to select features with two best ANOVA F-Values
    fvalue_selector = SelectKBest(f_regression, k=k)

    # Apply the SelectKBest object to the features and target
    X_kbest = fvalue_selector.fit_transform(X_train, y_train)

    mask = fvalue_selector.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    old_features = [col for col in X_train.columns]
    for bool, feature in zip(mask, old_features):
        if bool:
            new_features.append(feature)

    return new_features

class Target_Encoding():

    def __init__(self, MinCnt=3.0, MaxCnt=600.0):
        self.MinCnt = MinCnt
        self.MaxCnt = MaxCnt
        self.artists_df = None

    def fit(self, X, y):
        self.artists_df = y.groupby(X.artists).agg(['mean', 'count'])
        self.artists_df.loc['unknown'] = [y.mean(), 1]
        return self

    def transform(self, X, y=None):
        X['artists'] = np.where(X['artists'].isin(self.artists_df.index), X['artists'], 'unknown')
        X['artists'] = X['artists'].map(self.artists_df['mean'])
        return X

class Target_Encoding_withthreshold():

    def __init__(self, MinCnt=3.0, MaxCnt=600.0):
        self.MinCnt = MinCnt
        self.MaxCnt = MaxCnt
        self.artists_df = None

    def fit(self, X, y):
        self.artists_df = y.groupby(X.artists).agg(['mean', 'count'])
        self.artists_df.loc['unknown'] = [y.mean(), 1]
        self.artists_df.loc[self.artists_df['count'] <= self.MinCnt, 'mean'] = y.mean()
        self.artists_df.loc[self.artists_df['count'] >= self.MaxCnt, 'mean'] = 0
        return self

    def transform(self, X, y=None):
        X['artists'] = np.where(X['artists'].isin(self.artists_df.index), X['artists'], 'unknown')
        X['artists'] = X['artists'].map(self.artists_df['mean'])
        return X