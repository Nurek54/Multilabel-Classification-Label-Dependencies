import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._utils import default_estimator, fit_binary, positive_proba


class ClassifierChain(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None, order=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = Y.astype(int)
        base = default_estimator(self.base_estimator)
        self.n_labels_ = Y.shape[1]

        if self.order is None:
            self.order_ = np.arange(self.n_labels_)
        elif isinstance(self.order, str) and self.order == "random":
            self.order_ = check_random_state(self.random_state).permutation(self.n_labels_)
        else:
            self.order_ = np.asarray(self.order)

        self.estimators_ = []
        for k, label in enumerate(self.order_):
            X_aug = np.hstack([X, Y[:, self.order_[:k]]])
            self.estimators_.append(fit_binary(base, X_aug, Y[:, label]))
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        proba = np.zeros((X.shape[0], self.n_labels_))
        Y_pred = np.zeros((X.shape[0], self.n_labels_), dtype=int)
        for k, (label, est) in enumerate(zip(self.order_, self.estimators_)):
            X_aug = np.hstack([X, Y_pred[:, self.order_[:k]]])
            proba[:, label] = positive_proba(est, X_aug)
            Y_pred[:, label] = proba[:, label] >= 0.5
        return proba

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
