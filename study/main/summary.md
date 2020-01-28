# How to Use SK-Learn Library

# Preprocessing

- Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
lbe.fit_transform(data['Sex'])
sex = lbe.classes_ # mapping table
```

- One Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
ohe.fit_transform(data[categorical]).toarray()
# if col-name needed, use pandas get_dummy()
for c in categorical:
    data = pd.concat([data, pd.get_dummies(data[c], prefix=c)], axis=1)
    data = data.drop(c, axis=1)
```



## Clustering

- k-means

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, max_iter=300, init='k-means++', random_state=2019)
# init = ['k-means++', 'random']
```

- DBSCAN

```python
from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=0.5, min_samples=4, metric='euclidean')
# metric = [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
# from scipy = [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
```

- GMM

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=n, random_state=2019).fit(data)
# covariance_type = ['full', 'tied', 'diag', 'spherical']
# init_params = ['kmeans', 'random']
```



## Classifiaction

- Logistic

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=2019).fit(train_X, train_y)
# penalty = ['l1', 'l2', 'elasticnet']
```

- SVM

```python
from sklearn.svm import SVC

svm = svm = SVC(kernel='rbf', C=1, gamma=0.1).fit(train_X, train_y)
# kernel = ['poly', 'linear', 'rbf', 'sigmoid']
# gamma = ['scale', 'auto']
```

- Tree

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=2019).fit(train_X, train_y)
```

- RF

```python
from sklearn.ensemble import RandomForestClassifier

rclf = RandomForestClassifier(random_state=2019, max_depth=9).fit(train_X, train_y)
```

- xgboost/lightgbm params

```python
# xgboost
params= {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'n_estimators': 400,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1,
    'reg_lambda': 2
    'random_state': 0
}
# lgbm
params= {
    'boosting': 'gbdt',
    'n_estimators': 400,
    'learning_rate': 0.1,
    'max_depth': 9,
    'min_child_samples': 20, # 결정노드가 되기 위한 최소 자료 수
    'num_leaves': 31, # 트리가 가질 수 있는 최대 리프 수
    'subsample': 0.7, # bagging_fraction
    'colsample_bytree': 0.7, # feature_fraction
    'random_state': 0,
}

early_stopping_rounds 조정 가능
```

