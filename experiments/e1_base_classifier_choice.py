from data import DATASETS, load_dataset
from .generate_splits import load_splits
from sklearn.linear_model import LogisticRegression
from methods import BinaryRelevance, ClassifierChain, EnsembleClassifierChains
import os
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn.base import clone
import pandas as pd

# mute FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

RANDOM_SEED = 26600
RESULTS_DIR = "results/e1"
os.makedirs(RESULTS_DIR, exist_ok=True)

multilabel_methods = {
    "BR":  lambda base: BinaryRelevance(base_estimator=base),
    "CC":  lambda base: ClassifierChain(base_estimator=base, random_state=RANDOM_SEED),
    "ECC": lambda base: EnsembleClassifierChains(base_estimator=base, n_chains=10,
                                                 random_state=RANDOM_SEED),
}

base_classifiers = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
}

metrics = {
    "hamming_loss":    lambda y, p: hamming_loss(y, p),
    "subset_accuracy": lambda y, p: accuracy_score(y, p),
    "macro_F1":        lambda y, p: f1_score(y, p, average="macro", zero_division=0),
    "micro_F1":        lambda y, p: f1_score(y, p, average="micro", zero_division=0),
}

for dataset_name in DATASETS:
    X, Y = load_dataset(dataset_name)
    splits = load_splits(dataset_name)
    rows = []

    for split_id, s in enumerate(splits):
        X_train, Y_train = X[s["train_inner"]], Y[s["train_inner"]]
        X_val,   Y_val   = X[s["val"]],         Y[s["val"]]
        X_test,  Y_test  = X[s["test"]],        Y[s["test"]]

        for method_name, make_method in multilabel_methods.items():
            for clf_name, base_clf in base_classifiers.items():
                model = make_method(clone(base_clf))
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                row = {
                    "dataset": dataset_name,
                    "split": split_id,
                    "method": method_name,
                    "base_classifier": clf_name,
                }
                for metric_name, metric_fn in metrics.items():
                    row[metric_name] = metric_fn(Y_test, Y_pred)
                rows.append(row)
                print(f"{dataset_name} | split {split_id} | {method_name} | {clf_name} done")

    pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/{dataset_name}.csv", index=False)