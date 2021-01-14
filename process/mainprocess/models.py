import os
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, \
                             GradientBoostingRegressor, GradientBoostingClassifier, \
                             TrAdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV

from tl_algs.trbag import TrBag, sc_trbag_filter, mvv_filter
from tl_algs import voter

os.environ['TransferMethods'] = './transfer_methods'
# from transfer_methods.TrAdaBoost import TrAdaBoostClassifier
from transfer_methods.TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 as TwoStageTrAdaBoostRegressor
from transfer_methods.TransferStacking import TransferStacking as TransferStackingRegressor
from transfer_methods.MultipleSourceTwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 as MultipleSourceTwoStageTrAdaBoostRegressor


class Model(metaclass = ABCMeta):
    def __init__(self, options, model_name):
        self.info_datasets = options['datasets']
        # self.info_experiments = options['experiments']
        self.model_name = model_name
        self.task = options["datasets"]["task"]
        assert self.task in ["regression", "classification"], f'{self.task} task is illegal.'

    @abstractmethod
    def fit(self, X, y, domain_col=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass



class AdaBoost(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['AdaBoost']
        base_estimator_params = model_params['base_estimator']
        self.n_estimators = model_params['n_estimators']
        self.learning_rate = model_params["learning_rate"]
        if base_estimator_params['name']=="DecisionTree":
            if self.task=="regression":
                self.base_estimator = DecisionTreeRegressor(
                    max_depth=base_estimator_params['max_depth'])
            elif self.task=="classification":
                self.base_estimator = DecisionTreeClassifier(
                    max_depth=base_estimator_params['max_depth'])
        else:
            msg = 'Base estimator of AdaBoost "{}" is invalid'.format(
                base_estimator_params['name']
            )
            raise TypeError(msg)
        
        if self.task=="regression":
            self.model = AdaBoostRegressor(self.base_estimator,
                                n_estimators = self.n_estimators, 
                                learning_rate=self.learning_rate,
                                random_state = np.random.RandomState(1))
        elif self.task=="classification":
            self.model = AdaBoostClassifier(self.base_estimator,
                                n_estimators = self.n_estimators, 
                                learning_rate=self.learning_rate,
                                random_state = np.random.RandomState(1))


    def fit(self, X, y, domain_col=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        self.label_class = self.model.classes_
        return self.model.predict_proba(X)


class GradientBoostingDecisionTree(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['GradientBoostingDecisionTree']
        self.n_estimators = model_params['n_estimators']
        self.learning_rate = model_params['learning_rate']
        self.max_depth = model_params['max_depth']
        self.validation_fraction = model_params['validation_fraction']
        self.n_iter_no_change = model_params['n_iter_no_change']
        self.tol = model_params['tol']

        if self.task=="regression":
            self.model = GradientBoostingRegressor(n_estimators = self.n_estimators, 
                                learning_rate=self.learning_rate,
                                max_depth = self.max_depth,
                                validation_fraction = self.validation_fraction,
                                n_iter_no_change = self.n_iter_no_change,
                                tol = self.tol,
                                random_state = np.random.RandomState(1))
        elif self.task=="classification":
            self.model = GradientBoostingClassifier(n_estimators = self.n_estimators, 
                                learning_rate=self.learning_rate,
                                max_depth = self.max_depth,
                                validation_fraction = self.validation_fraction,
                                n_iter_no_change = self.n_iter_no_change,
                                tol = self.tol,
                                random_state = np.random.RandomState(1))

    def fit(self, X, y, domain_col=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        self.label_class = self.model.classes_
        return self.model.predict_proba(X)



class TrAdaBoost(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['TrAdaBoost']
        base_estimator_params = model_params['base_estimator']
        self.n_estimators = model_params['n_estimators']
        self.learning_rate = model_params["learning_rate"]
        if base_estimator_params['name']=="DecisionTree":
            self.base_estimator = DecisionTreeClassifier(
                max_depth=base_estimator_params['max_depth'])
        else:
            msg = 'Base estimator of TrAdaBoost "{}" is invalid'.format(
                base_estimator_params['name']
            )
            raise TypeError(msg)

        if self.task=="regression":
            msg = f'TrAdaBoost is not suitable for {self.task} task.'
            raise TypeError(msg)

        elif self.task=="classification":
            self.model = TrAdaBoostClassifier(
                            base_estimator=self.base_estimator,
                            n_estimators= self.n_estimators,
                            learning_rate= self.learning_rate,
                            random_state= 1
                        )
        
    def fit(self, X, y, domain_col, target_name, init_weights=None):
        # assert len(np.unique(y))==2, "TrAdaBoost only supports binary classification."
        # self.y_convert_base = np.unique(y)
        # flag_y = np.where(y==self.y_convert_base[0], 1, 0)
        # flag_domain = np.where(domain_col==target_name, 1, 0)
        # X_domain_col = np.concatenate((X, flag_domain.reshape(-1,1)), axis=1)
        # df_X_domain_col = pd.DataFrame(X_domain_col)
        # domain_col_name = df_X_domain_col.columns[-1]
        # df_X_domain_col.rename(columns={domain_col_name: "domain"}, inplace=True)
        # self.model.fit(df_X_domain_col, flag_y, 1, init_weights=init_weights)

        self.model.fit(X, y, domain_col, target_name, sample_weight=init_weights)

    def predict(self, X):
        # domain = np.full(X.shape[0], 1).reshape((-1,1))
        # X_domain_col = np.concatenate((X, domain), axis=1)
        # pred = self.model.predict(X_domain_col)
        # return np.where(pred==1, self.y_convert_base[0], self.y_convert_base[1])

        return self.model.predict(X)
    
    def predict_proba(self, X):
        # domain = np.full(X.shape[0], 1).reshape((-1,1))
        # X_domain_col = np.concatenate((X, domain), axis=1)
        # return self.model.predict(X_domain_col)
        self.label_class = self.model.classes_

        return self.model.predict_proba(X)
            


class TwoStageTrAdaBoostR2(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['two-stage TrAdaBoost']
        base_estimator_params = model_params['base_estimator']
        self.n_estimators = model_params['n_estimators']
        self.learning_rate = model_params["learning_rate"]
        self.steps = model_params['steps']
        self.fold = model_params['fold']
        if base_estimator_params['name']=="DecisionTree":
            self.base_estimator = DecisionTreeRegressor(
                max_depth=base_estimator_params['max_depth'])
        else:
            msg = 'Base estimator of TwoStageTrAdaBoost "{}" is invalid'.format(
                base_estimator_params['name']
            )
            raise TypeError(msg)

        if self.task=="regression":
            self.model = TwoStageTrAdaBoostRegressor(self.base_estimator,
                                n_estimators = self.n_estimators, 
                                steps = self.steps,
                                fold = self.fold,
                                learning_rate=self.learning_rate,
                                random_state = np.random.RandomState(1))
        elif self.task=="classification":
            msg = f'TwoStageTrAdaBoostR2 is not suitable for {self.task} task.'
            raise TypeError(msg)

    def fit(self, X, y, domain_col, target_name):
        self.model.fit(X, y, domain_col, target_name)

    def predict(self, X):
        return self.model.predict(X)


class MultipleSourceTwoStageTrAdaBoostR2(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['MS two-stage TrAdaBoost']
        base_estimator_params = model_params['base_estimator']
        self.n_estimators = model_params['n_estimators']
        self.learning_rate = model_params["learning_rate"]
        self.steps = model_params['steps']
        self.fold = model_params['fold']
        if base_estimator_params['name']=="DecisionTree":
            self.base_estimator = DecisionTreeRegressor(
                max_depth=base_estimator_params['max_depth'])
        else:
            msg = 'Base estimator of MultipleSourceTwoStageTrAdaBoost "{}" is invalid'.format(
                base_estimator_params['name']
            )
            raise TypeError(msg)
        if self.task=="regression":
            self.model = MultipleSourceTwoStageTrAdaBoostRegressor(
                                self.base_estimator,
                                n_estimators = self.n_estimators, 
                                steps = self.steps,
                                fold = self.fold,
                                learning_rate=self.learning_rate,
                                random_state = np.random.RandomState(1))
        elif self.task=="classification":
            msg = f'MultipleSourceTwoStageTrAdaBoostR2 is not suitable for {self.task} task.'
            raise TypeError(msg)
    
    def fit(self, X, y, domain_col, target_name):
        self.model.fit(X, y, domain_col, target_name)

    def predict(self, X):
        return self.model.predict(X)


class Trbagg(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['Trbagg']
        base_estimator_params = model_params['base_estimator']
        self.n_estimators = model_params['n_estimators']
        if model_params['filter_func']=="SC":
            self.filter_func = sc_trbag_filter
            self.validate_proportion = None
        elif model_params['filter_func']=="MVT":
            self.filter_func = mvv_filter
            self.validate_proportion = None
        elif model_params['filter_func']=="MVV":
            self.filter_func = mvv_filter
            self.validate_proportion = 0.5
        else:
            msg = 'filter_func of Trbagg "{}" is invalid'.format(
                model_params['filter_func']
            )
            raise TypeError(msg)
        if base_estimator_params['name']=="DecisionTree":
            if self.task=="classification":
                self.base_estimator = DecisionTreeClassifier
            elif self.task=="regression":
                self.base_estimator = DecisionTreeRegressor
            self.base_params = {'max_depth':base_estimator_params['max_depth']}
        else:
            msg = 'Base estimator of Trbagg "{}" is invalid'.format(
                base_estimator_params['name']
            )
            raise TypeError(msg)

    def fit(self, X_train, y_train, domain_col, target_name, X_test):
        # 分類問題はlabelが文字列になっていることもあるのでencoding
        if self.task=="classification":
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
        # domainが文字列になっていることもあるのでencoding
        domain_encoder = LabelEncoder()
        domain_col = domain_encoder.fit_transform(domain_col)
        target_name = domain_encoder.transform(np.array([target_name]))[0]
        # DataFrame型に直す
        X_train = pd.DataFrame(X_train)
        y_train = pd.Series(y_train)
        domain_col = pd.Series(domain_col)
        # モデルと入力データを用意
        if self.task=='classification':
            filter_metric = f1_score
            vote_func = voter.count_vote
        elif self.task=='regression':
            filter_metric = mean_squared_error
            vote_func = voter.count_vote_regression
        self.model = TrBag(test_set_X=X_test, 
                           test_set_domain=target_name, 
                           train_pool_X=X_train, 
                           train_pool_y=y_train, 
                           train_pool_domain=domain_col, 
                           sample_size=y_train.shape[0],
                           Base_Classifier=self.base_estimator,
                           classifier_params=self.base_params,
                           filter_metric=filter_metric,
                           filter_func=self.filter_func,
                           validate_proportion=self.validate_proportion,
                           T=self.n_estimators,
                           vote_func=vote_func,
                           rand_seed=1)
        # fitとpredict両方実行
        # print(self.model.train_filter_test())
        self.confidences, self.predictions = self.model.train_filter_test()

    def predict(self, X):
        return self.predictions
    
    def predict_proba(self, X):
        self.label_class = self.label_encoder.classes_
        predictions_proba = np.vstack([1-self.confidences, self.confidences]).T
        return predictions_proba



class TransferStacking(Model):
    def __init__(self, options, model_name, expert_model):
        super().__init__(options, model_name)
        model_params = options['model_params']['TransferStacking']
        self.fold = model_params['fold']
        self.base_estimator = model_params['base_estimator']
        meta_model_params = model_params['meta_model']
        if meta_model_params['name']=="LinearRegression":
            self.meta_model = LinearRegression()
        elif meta_model_params['name']=="Lasso":
            self.meta_model = Lasso(alpha=meta_model_params['alpha'])
        else:
            msg = 'Meta model of TransferStacking "{}" is invalid'.format(
                meta_model_params['name']
            )
            raise TypeError(msg)
        self.search_params = meta_model_params['search_params']
        self.expert_model = expert_model
        if self.task=="regression":
            self.model = TransferStackingRegressor(self.expert_model,
                                meta_model = self.meta_model,
                                search_params = self.search_params,
                                base_estimator = self.base_estimator,
                                fold = self.fold,
                                random_state = np.random.mtrand._rand)
        elif self.task=="classification":
            msg = f'TransferStacking is not suitable for {self.task} task.'
            raise TypeError(msg)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)



class DartForDomainAdaptation(Model):
    def __init__(self, options, model_name):
        super().__init__(options, model_name)
        model_params = options['model_params']['DartForDomainAdaptation']
        self.n_estimators = model_params['n_estimators']
        self.learning_rate = model_params['learning_rate']
        self.max_depth = model_params['max_depth']
        self.validation_fraction = model_params['validation_fraction']
        self.n_iter_no_change = model_params['n_iter_no_change']
        self.tol = model_params['tol']
        self.drop_rate = model_params['drop_rate']
        # self.is_drop_dynamically = model_params['is_drop_dynamically']
        self.is_best_drop = model_params['is_best_drop']
        self.n_estimators_add = model_params['n_estimators_add']
        self.dart_rate = model_params['dart_rate']
        # if model_name=="DART for DA":
        #     self.drop_mode = model_params['drop_mode']
        # elif model_name=="DART for DA (worse)":
        #     self.drop_mode = "worse"
        # elif model_name=="DART for DA (latest)":
        # #     self.drop_mode = "latest"
        # else:
        #     raise TypeError(f"model_name param:{model_name} is illegal.")
        self.drop_mode = model_params['drop_mode']
        if model_name=="only Dropout":
            self.n_estimators_add = 0
        elif model_name=="target DART":
            self.is_best_drop = False
            self.drop_rate = 1.



        if self.task=="regression":
            self.model = GradientBoostingRegressor(n_estimators = self.n_estimators, 
                                learning_rate=self.learning_rate,
                                max_depth = self.max_depth,
                                validation_fraction = self.validation_fraction,
                                n_iter_no_change = self.n_iter_no_change,
                                tol = self.tol,
                                random_state = 1)
        elif self.task=="classification":
            self.model = GradientBoostingClassifier(n_estimators = self.n_estimators, 
                                learning_rate=self.learning_rate,
                                max_depth = self.max_depth,
                                validation_fraction = self.validation_fraction,
                                n_iter_no_change = self.n_iter_no_change,
                                tol = self.tol,
                                random_state = 1)

    def fit_source(self, X, y, domain_col=None):
        self.model.fit(X, y)

    def fit(self, X, y, domain_col=None):
        # self.model.fit_domain_adaptation(X, y, drop_rate=self.drop_rate, 
        #                                  is_drop_dynamically=self.is_drop_dynamically, 
        #                                  is_best_drop=self.is_best_drop, 
        #                                  n_estimators_add=self.n_estimators_add,
        #                                  drop_mode=self.drop_mode)
        self.model.fit_domain_adaptation(X, y, drop_rate=self.drop_rate, 
                                         is_best_drop=self.is_best_drop, 
                                         n_estimators_add=self.n_estimators_add,
                                         drop_mode=self.drop_mode,
                                         dart_rate=self.dart_rate)

    def predict(self, X):
        return self.model.predict_domain_adaptation(X)

    def predict_proba(self, X):
        self.label_class = self.model.classes_
        return self.model.predict_proba_domain_adaptation(X)