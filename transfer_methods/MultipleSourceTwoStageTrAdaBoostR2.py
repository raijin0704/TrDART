"""
TwoStageTrAdaBoostR2 algorithm

based on algorithm 3 in paper "Boosting for Regression Transfer".

"""
import warnings
import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

################################################################################
## the second stage
################################################################################
class Multiple_Stage2_TrAdaBoostR2:
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth=4),
                 sample_size = None,
                 n_estimators = 50,
                 learning_rate = 1.,
                 loss = 'linear',
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_sources = len(self.sample_size)-1
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None):
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")
        
        self.sample_weight = np.zeros((self.n_estimators, X.shape[0]))

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        # Clear any previous fit results
        self.estimators_ = []
        self.accept_sources = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        for iboost in range(self.n_estimators): # this for loop is sequential and does not support parallel(revison is needed if making parallel)
            # Boosting step
            # 有効なsourceを選択
            accept_sources = self._select_min_sources(X, y, sample_weight)
            self.accept_sources.append(accept_sources)


            self.sample_weight[iboost, :] = sample_weight
            sample_weight, estimator_weight, estimator_error = self._stage2_adaboostR2(
                    iboost,
                    X, y,
                    sample_weight,
                    accept_sources)
            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self


    # 誤差が最小になるsource１つだけを選択
    def _select_min_sources(self, X, y, sample_weight):
        errors = np.full(len(self.sample_size[:-1]), np.inf)
        sample_weight_target = sample_weight[-self.sample_size[-1]:].copy()
        sample_weight_target /= sample_weight_target.sum()
        before_idx = 0
        for source, length in enumerate(self.sample_size[:-1]):
            after_idx = before_idx + length
            sample_weight_source = sample_weight[before_idx:after_idx].copy()
            sample_weight_source /= sample_weight_source.sum()
            errors[source] = self._error_by_one_source(
                                X[before_idx:after_idx,:], y[before_idx:after_idx],
                                sample_weight_source,
                                X[-self.sample_size[-1]:], y[-self.sample_size[-1]:],
                                sample_weight_target
                            )


        return np.array(errors.argmin())
    
    # 1つのソースで学習&test予測&評価
    def _error_by_one_source(self, X_source, y_source, sample_weight_source, 
                                X_target, y_target, sample_weight_target):

        estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight_source)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X_source.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X_source[bootstrap_idx], y_source[bootstrap_idx])
        y_predict = estimator.predict(X_target)

        error_vect = np.abs(y_predict - y_target)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight_target * error_vect).sum()
        return estimator_error



    def _stage2_adaboostR2(self, iboost, X, y, sample_weight, accept_sources):

        estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        # 採用されなかったsourceのweightを0にして学習
        sample_weight_train = sample_weight.copy()
        before_idx = 0
        for source, length in enumerate(self.sample_size[:-1]):
            after_idx = before_idx + length
            if source not in accept_sources:
                sample_weight_train[before_idx:after_idx] = 0
            before_idx = after_idx
        # 総和を1に
        sample_weight_train /=sample_weight_train.sum()


        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight_train)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        self.estimators_.append(estimator)  # add the fitted estimator

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # avoid overflow of np.log(1. / beta)
        if beta < 1e-308:
            beta = 1e-308
        estimator_weight = self.learning_rate * np.log(1. / beta)

        # Boost weight using AdaBoost.R2 alg except the weight of the source data
        # the weight of the source data are remained
        source_weight_sum= np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
        target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)

        # targetのみsample_weightを更新している
        if not iboost == self.n_estimators - 1:
            sample_weight[-self.sample_size[-1]:] *= np.power(
                    beta,
                    (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)
            # make the sum weight of the source data not changing
            source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
                sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new

        return sample_weight, estimator_weight, estimator_error


    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([
                est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]


################################################################################
## the whole two stages
################################################################################
class TwoStageTrAdaBoostR2:
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth=4),
                 sample_size = None,
                 n_estimators = 50,
                 steps = 10,
                 fold = 5,
                 learning_rate = 1.,
                 loss = 'linear',
                 random_state = np.random.mtrand._rand,
                 binary_search_step = 1e-30):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_estimators = n_estimators
        self.steps = steps
        self.fold = fold
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.binary_search_step = binary_search_step


    def fit(self, X_ori, y_ori, domain_col_ori, target_name, sample_weight=None):
        assert target_name in domain_col_ori, 'There are no target domains'+domain_col_ori
        # change into numpy
        if type(X_ori) in (pd.DataFrame, pd.Series):
            X_ori = X_ori.values
        if type(y_ori) == pd.Series:
            y_ori = y_ori.values
        if type(domain_col_ori) == pd.Series:
            domain_col_ori = domain_col_ori.values
        self.target_name = target_name
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        # sort by domain_col (target domain is tail)
        concat_data = np.concatenate([X_ori, domain_col_ori.reshape((-1,1)), y_ori.reshape((-1,1))], axis=1)
        source_ori = concat_data[concat_data[:,-2]!=self.target_name]
        source = source_ori[np.argsort(source_ori[:,-2])]
        target = concat_data[concat_data[:,-2]==self.target_name]
        X = np.concatenate([source[:,:-2], target[:,:-2]]).astype(X_ori.dtype) # not include domain_col & y
        y = np.concatenate([source[:,-1], target[:,-1]]).astype(y_ori.dtype)
        # domain_col = np.concatenate([source[:,-2], target[:,-2]]).astype(domain_col_ori.dtype)

        # ドメインごと・source全体のサンプルサイズを取得
        domain = np.concatenate([source[:,-2], target[:,-2]])
        domain_names, domain_index, domain_counts =  np.unique(domain, return_index=True, return_counts=True)
        self.domain_names = domain_names[np.argsort(domain_index)]
        self.sample_size = domain_counts[np.argsort(domain_index)]
        self.n_sources = len(self.sample_size)-1
        del domain_names, domain_counts

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        # sample_size = [sourceのsize, targetのsize]

        X_source = X[:-self.sample_size[-1]]
        y_source = y[:-self.sample_size[-1]]
        X_target = X[-self.sample_size[-1]:]
        y_target = y[-self.sample_size[-1]:]

        self.models_ = []
        self.errors_ = []
        for istep in range(self.steps):
            # 普通にAdaBoost.R2にかける
            model = Multiple_Stage2_TrAdaBoostR2(self.base_estimator,
                                        sample_size = self.sample_size,
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, loss = self.loss,
                                        random_state = self.random_state)
            model.fit(X, y, sample_weight = sample_weight)
            self.models_.append(model)
            # cv training
            kf = KFold(n_splits = self.fold)
            # error = []
            valid_pred = np.zeros(shape=y_target.shape)
            target_weight = sample_weight[-self.sample_size[-1]:]
            source_weight = sample_weight[:-self.sample_size[-1]]
            for train, test in kf.split(X_target):
                sample_size = self.sample_size[:-1].copy()
                sample_size.append(len(train))
                cv_model = Multiple_Stage2_TrAdaBoostR2(self.base_estimator,
                                        sample_size = sample_size,
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, loss = self.loss,
                                        random_state = self.random_state)
                X_train = np.concatenate((X_source, X_target[train]))
                y_train = np.concatenate((y_source, y_target[train]))
                X_test = X_target[test]
                y_test = y_target[test]
                # make sure the sum weight of the target data do not change with CV's split sampling
                target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])
                cv_model.fit(X_train, y_train, sample_weight = np.concatenate((source_weight, target_weight_train)))
                valid_pred[test] = cv_model.predict(X_test)
                # error.append(mean_squared_error(y_predict, y_test))
            error = mean_squared_error(y_target, valid_pred)
            self.errors_.append(error)

            sample_weight = self._twostage_adaboostR2(istep, X, y, sample_weight, model)

            if sample_weight is None:
                break
            if error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if istep < self.steps - 1:
                # Normalize
                sample_weight /= sample_weight_sum
        return self


    def _twostage_adaboostR2(self, istep, X, y, sample_weight, model):

        # estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        # ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # # Weighted sampling of the training set with replacement
        # cdf = np.cumsum(sample_weight)
        # cdf /= cdf[-1]
        # uniform_samples = self.random_state.random_sample(X.shape[0])
        # bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # # searchsorted returns a scalar
        # bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # # Fit on the bootstrapped sample and obtain a prediction
        # # for all samples in the training set
        # estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        # y_predict = estimator.predict(X)

        y_predict = model.predict(X)


        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Update the weight vector
        beta = self._beta_binary_search(istep, sample_weight, error_vect, self.binary_search_step)

        if not istep == self.steps - 1:
            sample_weight[:-self.sample_size[-1]] *= np.power(
                    beta,
                    (error_vect[:-self.sample_size[-1]]) * self.learning_rate)
        return sample_weight


    def _beta_binary_search(self, istep, sample_weight, error_vect, stp):
        # calculate the specified sum of weight for the target data
        n_target = self.sample_size[-1]
        n_source = np.array(self.sample_size).sum() - n_target
        theoretical_sum = n_target/(n_source+n_target) + istep/(self.steps-1)*(1-n_target/(n_source+n_target))
        # for the last iteration step, beta is 0.
        if istep == self.steps - 1:
            beta = 0.
            return beta
        # binary search for beta
        L = 0.
        R = 1.
        beta = (L+R)/2
        sample_weight_ = copy.deepcopy(sample_weight)
        sample_weight_[:-n_target] *= np.power(
                    beta,
                    (error_vect[:-n_target]) * self.learning_rate)
        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)

        while np.abs(updated_weight_sum - theoretical_sum) > 0.01:
            if updated_weight_sum < theoretical_sum:
                R = beta - stp
                if R > L:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    msg = f"At step:{istep+1}\n"
                    msg += "Binary search's goal not meeted! Value is set to be the available best!\n"
                    msg += f"Try reducing the search interval. Current stp interval:{stp}"
                    warnings.warn(msg)
                    break

            elif updated_weight_sum > theoretical_sum:
                L = beta + stp
                if L < R:
                    beta = (L+R)/2
                    sample_weight_ = copy.deepcopy(sample_weight)
                    sample_weight_[:-n_target] *= np.power(
                                beta,
                                (error_vect[:-n_target]) * self.learning_rate)
                    sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                    updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                else:
                    msg = f"At step:{istep+1}\n"
                    msg += "Binary search's goal not meeted! Value is set to be the available best!\n"
                    msg += f"Try reducing the search interval. Current stp interval:{stp}"
                    warnings.warn(msg)
                    break
        return beta


    def predict(self, X):
        # select the model with the least CV error
        fmodel = self.models_[np.array(self.errors_).argmin()]
        predictions = fmodel.predict(X)
        return predictions


if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import AdaBoostRegressor

    from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 as Previous
 
    X = np.random.rand(10000,10)
    y = X.mean(axis=1)
    domain_col = np.arange(0, 10, 0.001).astype(np.int8)
    # targetは100サンプルだけ
    X = X[900:]
    y = y[900:]
    domain_col = domain_col[900:]
    X = np.concatenate([X[100:], X[:100]])
    y = np.concatenate([y[100:], y[:100]])
    domain_col = np.concatenate([domain_col[100:], domain_col[:100]])
    y[domain_col==0] = X[domain_col==0].sum(axis=1)
    effective_source = np.random.random_integers(1,9,3)
    print("effective source : {}".format(effective_source))
    y[np.isin(domain_col, effective_source)] = X[np.isin(domain_col, effective_source)].sum(axis=1)
    y[domain_col==0] = X[domain_col==0].sum(axis=1)*np.random.rand(X[domain_col==0].shape[0])
    test_X = np.random.rand(10000,10)
    test_y = test_X.sum(axis=1)
    # y[domain_col==10] = X[domain_col==10].sum(axis=1)*np.random.rand(X[domain_col==10].sum(axis=1).shape[0])
    # print(y[domain_col==0])
    sample_size = [X.shape[0]-100, 100]

    # all adaboost
    adaregr = AdaBoostRegressor(random_state=0, n_estimators=100)
    adaregr.fit(X, y)
    ada_y_predict = adaregr.predict(test_X)
    ada_r = mean_squared_error(ada_y_predict, test_y)**0.5
    print("all adaboost: {}".format(ada_r))

    # target adaboost
    adaregr_t = AdaBoostRegressor(random_state=0, n_estimators=100)
    adaregr_t.fit(X[domain_col==0], y[domain_col==0])
    ada_y_predict = adaregr_t.predict(test_X)
    ada_r_t = mean_squared_error(ada_y_predict, test_y)**0.5
    print("target adaboost: {}".format(ada_r_t))

    # only effective source adaboost
    adaregr_only = AdaBoostRegressor(random_state=0, n_estimators=100)
    adaregr_only.fit(pd.DataFrame(X[(domain_col==0)+(np.isin(domain_col, effective_source))]),y[(domain_col==0)+(np.isin(domain_col, effective_source))])
    ada_y_predict = adaregr_only.predict(test_X)
    ada_r_only = mean_squared_error(ada_y_predict, test_y)**0.5
    print("only effective adaboost: {}".format(ada_r_only))

    # # normal twostage
    # regr = Previous(random_state=np.random.RandomState(0), n_estimators=100, sample_size=sample_size)
    # regr.fit(X, y)
    # y_predict = regr.predict(test_X)
    # n_r = mean_squared_error(y_predict, test_y)**0.5
    # print("normal twostage: {}".format(n_r))

    # proposal
    sample_size_multi = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100]
    regr = TwoStageTrAdaBoostR2(random_state=np.random.RandomState(0), n_estimators=100, sample_size=sample_size_multi)
    regr.fit(X, y)
    y_predict = regr.predict(test_X)
    n_r = mean_squared_error(y_predict, test_y)**0.5
    print("multiple twostage: {}".format(n_r))