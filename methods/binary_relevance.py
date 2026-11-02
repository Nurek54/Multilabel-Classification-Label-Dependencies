import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._utils import default_estimator, fit_binary, positive_proba


class BinaryRelevance(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y, multi_output=True)
        Y = Y.astype(int)
        base = default_estimator(self.base_estimator)
        self.n_labels_ = Y.shape[1]
        self.estimators_ = [fit_binary(base, X, Y[:, j]) for j in range(self.n_labels_)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.column_stack([positive_proba(est, X) for est in self.estimators_])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
