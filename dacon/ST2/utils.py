import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import gc
from bayes_opt import BayesianOptimization
from time import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
# np.random.seed(42)

bounds_LGB = {
    'num_leaves': (100, 800), 
    'min_data_in_leaf': (0, 150),
    'bagging_fraction' : (0.3, 0.9),
    'feature_fraction' : (0.3, 0.9),
#     'learning_rate': (0.01, 1),
    'min_child_weight': (0.01, 3),   
    'reg_alpha': (0.1, 3), 
    'reg_lambda': (0.1, 3),
    'max_depth':(6, 29),
    'n_estimators': (64, 512)
}

def build_lgb(x, y, init_points=15, n_iter=0, param=True, verbose=2):
    train_X, test_X, train_y, test_y = train_test_split(x.values, y.values, test_size=0.3, random_state=12, shuffle=True)
    def LGB_bayesian(
        #learning_rate,
        num_leaves, 
        bagging_fraction,
        feature_fraction,
        min_child_weight, 
        min_data_in_leaf,
        max_depth,
        reg_alpha,
        reg_lambda,
        n_estimators
         ):
        # LightGBM expects next three parameters need to be integer. 
        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)

        assert type(num_leaves) == int
        assert type(min_data_in_leaf) == int
        assert type(max_depth) == int


        params = {
                  'num_leaves': num_leaves, 
                  'min_data_in_leaf': min_data_in_leaf,
                  'min_child_weight': min_child_weight,
                  'bagging_fraction' : bagging_fraction,
                  'feature_fraction' : feature_fraction,
                  'learning_rate' : 0.05,
                  'max_depth': max_depth,
                  'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda,
                  'objective': 'cross_entropy',
                  'save_binary': True,
                  'seed': 12,
                  'feature_fraction_seed': 12,
                  'bagging_seed': 12,
                  'drop_seed': 12,
                  'data_random_seed': 12,
                  'boosting': 'gbdt', ## some get better result using 'dart'
                  'verbose': 1,
                  'is_unbalance': False,
                  'boost_from_average': True,
                  'metric':'auc',
                  'n_estimators': int(n_estimators),
                  'tree_learner ': 'voting'
        }    

        ## set clf options
        clf = lgb.LGBMClassifier(**params).fit(train_X, train_y)
    #     score = roc_auc_score(test_y, clf.predict(test_X))
        score = roc_auc_score(test_y, clf.predict_proba(test_X)[:,1])

        return score
    
    optimizer = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42, verbose=verbose)
    init_points = init_points
    n_iter = n_iter

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    param_lgb = {
        'min_data_in_leaf': int(optimizer.max['params']['min_data_in_leaf']), 
        'num_leaves': int(optimizer.max['params']['num_leaves']), 
        'learning_rate': 0.05,
        'min_child_weight': optimizer.max['params']['min_child_weight'],
        'bagging_fraction': optimizer.max['params']['bagging_fraction'], 
        'feature_fraction': optimizer.max['params']['feature_fraction'],
        'reg_lambda': optimizer.max['params']['reg_lambda'],
        'reg_alpha': optimizer.max['params']['reg_alpha'],
        'max_depth': int(optimizer.max['params']['max_depth']), 
        'objective': 'binary',
        'save_binary': True,
        'seed': 12,
        'feature_fraction_seed': 12,
        'bagging_seed': 12,
        'drop_seed': 12,
        'data_random_seed': 12,
        'boosting_type': 'gbdt',  # also consider 'dart'
        'verbose': 1,
        'is_unbalance': False,
        'boost_from_average': True,
        'metric':'auc',
        'n_estimators': int(optimizer.max['params']['n_estimators']),
        'tree_learner ': 'voting'
    }

    params = param_lgb.copy()
    
    lgb_clf = lgb.LGBMClassifier(**params)
    lgb_clf.fit(x.values, y.values)
    
    if param:
        return lgb_clf, params
    else:
        return lgb_clf


class EDA:
    def __init__(self, x, y, criterion, models=[]):
        self.x = x
        self.y = y
        self.criterion = criterion
        self.models = models
        self.score = None
        self.params = None

    def featureSelect(self):
        start = time()
        n = 1
        while 1:
            gc.collect()
            clf, params = build_lgb(self.x, self.y, n_iter=15, init_points=5, verbose=1)
            self.params = params
            ctd = []
            for i in range(len(clf.feature_importances_)):
                if clf.feature_importances_[i] <= self.criterion :
                    ctd.append(self.x.columns[i])

            if len(ctd) <= 10:
                print('complete')
                break
            else:
                self.x = self.x.drop(ctd, axis=1)
                print('iter:', n, 'ctd:', len(ctd), round(time()-start, 3),'sec')
                n += 1

    def fit(self, score=True):
        if self.params == None:
            _, self.params = build_lgb(self.x, self.y, n_iter=15, init_points=5, verbose=1)

        params_fx = {'min_data_in_leaf': self.params['min_data_in_leaf'],
                    'num_leaves': self.params['num_leaves'],
                    'min_child_weight': self.params['min_child_weight'],
                    'bagging_fraction': self.params['bagging_fraction'],
                    'feature_fraction': self.params['feature_fraction'],
                    'reg_lambda': self.params['reg_lambda'],
                    'reg_alpha': self.params['reg_alpha'],
                    'max_depth': self.params['max_depth'],
                    'n_estimators': self.params['n_estimators'],
        }
        self.models.append(lgb.LGBMClassifier(**self.params))
        self.models.append(xgb.XGBClassifier(**params_fx, tree_method = 'hist', booster = 'gbtree'))

        for i, m in enumerate(self.models):
            m.fit(self.x.values, self.y.values)
        if score:
            print('getting score')
            self._score()

    def predict(self, test_X):
        res = []
        for m in self.models:
            res.append(m.predict_proba(test_X.values)[:,1])
        return res

        
    def _score(self, cv=4):
        score = []
        for m in tqdm(self.models):
            score.append(np.mean(cross_val_score(m, self.x.values, self.y.values, scoring='roc_auc', cv=cv)))

        self.score = score


def wr(a, p):
    x = a.copy()
    res = np.zeros(len(x))
    if 'p1' not in x.columns:
        x = pd.concat([x, p], axis=1)
        
    if 'wr' not in x.columns:
        for i in range(len(x)):
            if x['p1'].values[i] == 0 and x['p2'].values[i] == 1:
                res[i] = round(50.12-49.88, 3)
            elif x['p1'].values[i] == 1 and x['p2'].values[i] == 0:
                res[i] = round(49.88-50.12, 3)
            elif x['p1'].values[i] == 0 and x['p2'].values[i] == 2:
                res[i] = round(48.79-51.21, 3)
            elif x['p1'].values[i] == 2 and x['p2'].values[i] == 0:
                res[i] = round(51.21-48.79, 3)
            elif x['p1'].values[i] == 1 and x['p2'].values[i] == 2:
                res[i] = round(49.35-50.65, 3)
            elif x['p1'].values[i] == 2 and x['p2'].values[i] == 1:
                res[i] = round(50.65-49.35, 3)
            
    if 'match' not in x.columns:
        x['match'] = (x['p1'] == x['p2']).map(lambda x: int(x))

    x['wr'] = res
            
    return x.drop(['p1', 'p2'], axis=1)