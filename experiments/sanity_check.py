# Sprawdzenie poprawności: porównanie własnych implementacji z gotowymi klasami ze scikit-learn.
# Uruchomienie z katalogu głównego repozytorium:  python -m experiments.sanity_check

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain as SkClassifierChain
from sklearn.multioutput import MultiOutputClassifier

from methods import BinaryRelevance, ClassifierChain, EnsembleClassifierChains

X, Y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=6, random_state=42)
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.3, random_state=42)
base = LogisticRegression(max_iter=1000)


def report(name, Y_pred):
    print(f"{name:<32} Hamming loss = {hamming_loss(Y_te, Y_pred):.4f}   "
          f"subset accuracy = {accuracy_score(Y_te, Y_pred):.4f}")


# Binary Relevance
br = BinaryRelevance(base).fit(X_tr, Y_tr).predict(X_te)
sk_br = MultiOutputClassifier(base).fit(X_tr, Y_tr).predict(X_te)
report("BR (nasza)", br)
report("BR (sklearn MultiOutput)", sk_br)
print("  identyczne predykcje:", np.array_equal(br, sk_br))

# Classifier Chains z tą samą losową kolejnością
cc_model = ClassifierChain(base, order="random", random_state=0).fit(X_tr, Y_tr)
cc = cc_model.predict(X_te)
sk_cc = SkClassifierChain(base, order=cc_model.order_).fit(X_tr, Y_tr).predict(X_te)
report("CC (nasza)", cc)
report("CC (sklearn ClassifierChain)", sk_cc)
print("  identyczne predykcje:", np.array_equal(cc, sk_cc.astype(int)))

# Ensemble of Classifier Chains
ecc = EnsembleClassifierChains(base, n_chains=10, random_state=0).fit(X_tr, Y_tr).predict(X_te)
report("ECC (nasza, 10 łańcuchów)", ecc)
