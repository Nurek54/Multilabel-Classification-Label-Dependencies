import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def default_estimator(base_estimator):
    # Zwraca klasyfikator bazowy (domyślnie regresję logistyczną)
    if base_estimator is None:
        return LogisticRegression(max_iter=1000)
    return base_estimator


def fit_binary(base_estimator, X, y):
    # Uczy kopię klasyfikatora bazowego. Gdy etykieta ma w danych tylko jedną klasę, używa stałego predyktora
    if np.unique(y).size < 2:
        return DummyClassifier(strategy="prior").fit(X, y)
    return clone(base_estimator).fit(X, y)


def positive_proba(estimator, X):
    # Prawdopodobieństwo klasy 1, także gdy w danych treningowych wystąpiła tylko jedna klasa
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0])
    return estimator.predict_proba(X)[:, classes.index(1)]
