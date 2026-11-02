import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .classifier_chain import ClassifierChain


class EnsembleClassifierChains(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None, n_chains=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_chains = n_chains
        self.random_state = random_state

    def fit(self, X, Y):
        X, Y = check_X_y(X, Y, multi_output=True)
        rng = check_random_state(self.random_state)
        self.chains_ = [
            ClassifierChain(self.base_estimator, order="random",
                            random_state=rng.randint(np.iinfo(np.int32).max)).fit(X, Y)
            for _ in range(self.n_chains)
        ]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.mean([chain.predict_proba(X) for chain in self.chains_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
