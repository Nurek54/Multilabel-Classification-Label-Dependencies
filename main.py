import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from methods import BinaryRelevance, ClassifierChain, EnsembleClassifierChains

RANDOM_STATE = 42

base_estimators = {
    "LogReg": LogisticRegression(max_iter=1000),
    "Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(),
}


def make_methods(base):
    return {
        "BR": BinaryRelevance(base),
        "CC": ClassifierChain(base, order="random", random_state=RANDOM_STATE),
        "ECC": EnsembleClassifierChains(base, n_chains=10, random_state=RANDOM_STATE),
    }


def evaluate(model, X, Y, n_splits=5):
    hl, sa = [], []
    for train, test in KFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(X):
        Y_pred = clone(model).fit(X[train], Y[train]).predict(X[test])
        hl.append(hamming_loss(Y[test], Y_pred))
        sa.append(accuracy_score(Y[test], Y_pred))
    return np.mean(hl), np.mean(sa)


if __name__ == "__main__":
    X, Y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=6,
                                          random_state=RANDOM_STATE)
    print(f"Dane: {X.shape[0]} próbek, {X.shape[1]} cech, {Y.shape[1]} etykiet\n")
    print(f"{'Bazowy':<8}{'Metoda':<6}{'Hamming loss':>14}{'Subset acc.':>14}")
    print("-" * 42)
    for base_name, base in base_estimators.items():
        for method_name, model in make_methods(base).items():
            hl, sa = evaluate(model, X, Y)
            print(f"{base_name:<8}{method_name:<6}{hl:>14.4f}{sa:>14.4f}")
        print()
