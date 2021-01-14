import copy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class TransferStacking():
    def __init__(self,
                 expert_model,
                 meta_model = LinearRegression(),
                 search_params = None,
                 base_estimator = None,
                 fold = 5,
                 random_state = np.random.mtrand._rand):
        
        self.expert_model = expert_model
        self.meta_model = meta_model
        self.search_params = search_params
        self.fold = fold
        self.random_state = random_state

        assert 'estimators_' in vars(self.expert_model),\
            'expert_model has not "estimators_"'

        # 弱学習器の定義(指定しない場合はexpert_modelと同じ)
        if base_estimator is not None:
            self.base_estimator = base_estimator
        else:
            assert 'base_estimator' in vars(self.expert_model),\
                'expert_model has not "base_estimator"'
            self.base_estimator = self.expert_model.base_estimator
            if self.base_estimator is None:
                raise TypeError("base_estimator is None.")

    
    def fit(self, X, y):
        # 1層目の出力=2層目の入力を作成
        self.n_estimators = len(self.expert_model.estimators_) + 1
        valid_preds = np.full((y.shape[0], self.n_estimators), np.nan, dtype=float)

        # 新たに作成する弱学習器の学習
        self.model_new = copy.deepcopy(self.base_estimator)
        self.model_new.fit(X, y)
        # cvでyの予測値を計算
        kf = KFold(n_splits=self.fold)
        for train_idx, valid_idx in kf.split(X):
            train_X = X[train_idx]
            train_y = y[train_idx]
            valid_X = X[valid_idx]
            model = copy.deepcopy(self.base_estimator)
            model.fit(train_X, train_y)
            valid_preds[valid_idx, -1] = model.predict(valid_X)

        # expert_modelでもyの予測値を計算
        for idx, model in enumerate(self.expert_model.estimators_):
            valid_preds[:,idx] = model.predict(X)

        # 2層目のモデル学習
        # 2層目のモデルのパラメータを探索する場合
        if self.search_params is not None:
            self.gscv = GridSearchCV(copy.deepcopy(self.meta_model), 
                    self.search_params, cv=self.fold)
            self.gscv.fit(valid_preds, y)
            self.meta_model.set_params(**self.gscv.best_params_)
            self.meta_model.fit(valid_preds, y)
        # 2層目のモデルのパラメータを探索しない場合
        else:
            self.meta_model.fit(valid_preds, y)


    def predict(self, X):
        # test予測
        test_preds = np.empty(shape=(X.shape[0], self.n_estimators), dtype=float)
        for idx, model in enumerate(self.expert_model.estimators_):
            test_preds[:,idx] = model.predict(X)
        test_preds[:,-1] = self.model_new.predict(X)
        
        return self.meta_model.predict(test_preds)




        





if __name__ == "__main__":
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.metrics import mean_squared_error

    # パラメータ
    max_depth = 3
    n_estimators = 30
    fold = 10
    random_state = np.random.RandomState(1)
    max_depth = max_depth
    base_estimator = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    stack_estimator = LinearRegression()

    # データ生成
    X = np.random.rand(1000,10)
    y = X.mean(axis=1)
    domain_col = np.arange(0, 10, 0.01).astype(np.int8)
    # targetは10サンプルだけ
    X = X[90:]
    y = y[90:]
    domain_col = domain_col[90:]
    X = np.concatenate([X[10:], X[:10]])
    y = np.concatenate([y[10:], y[:10]])
    domain_col = np.concatenate([domain_col[10:], domain_col[:10]])
    y[domain_col==0] = X[domain_col==0].sum(axis=1)
    effective_source = np.random.random_integers(1,9,3)
    print("effective source : {}".format(effective_source))
    y[np.isin(domain_col, effective_source)] = X[np.isin(domain_col, effective_source)].sum(axis=1)
    y[domain_col==0] = X[domain_col==0].sum(axis=1)*np.random.rand(X[domain_col==0].shape[0])
    test_X = np.random.rand(1000,10)
    test_y = test_X.sum(axis=1)
    sample_size = [X.shape[0]-10, 10]

    # sourceだけで学習
    ada_source = AdaBoostRegressor(random_state=0, base_estimator=base_estimator, n_estimators=100)
    ada_source.fit(X[domain_col!=0], y[domain_col!=0])
    ada_y_predict = ada_source.predict(test_X)
    ada_r = mean_squared_error(ada_y_predict, test_y)**0.5
    print("all adaboost: {}".format(ada_r))


    transfer_stacking = TransferStacking(ada_source, 
                            meta_model=stack_estimator, fold=fold)
    transfer_stacking.fit(X[domain_col==0], y[domain_col==0])
    stacked_y_predict = transfer_stacking.predict(test_X)
    stacked_r = mean_squared_error(stacked_y_predict, test_y)**0.5
    print("Transfer Stacking: {}".format(stacked_r))