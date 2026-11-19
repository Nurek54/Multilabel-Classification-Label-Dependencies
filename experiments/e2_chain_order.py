from data import DATASETS, load_dataset
from .generate_splits import load_splits
from methods import ClassifierChain
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from sklearn.base import clone
import pandas as pd
from scipy.special import factorial
import itertools
import numpy as np
import time

# mute FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

def make_permutations(n_labels, n_perms, seed):
    if n_perms >= factorial(n_labels):
        return [list(p) for p in itertools.permutations(range(n_labels))]

    rng = np.random.default_rng(seed)
    seen, perms = set(), []
    while len(perms) < n_perms:
        p = tuple(int(i) for i in rng.permutation(n_labels))
        if p not in seen:
            seen.add(p)
            perms.append(list(p))
    return perms

def main():
    RANDOM_SEED = 26600
    RESULTS_DIR = "results/e2"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED))
    metrics = {
        "hamming_loss":    lambda y, p: hamming_loss(y, p),
        "subset_accuracy": lambda y, p: accuracy_score(y, p),
        "macro_F1":        lambda y, p: f1_score(y, p, average="macro", zero_division=0),
        "micro_F1":        lambda y, p: f1_score(y, p, average="micro", zero_division=0),
    }

    for dataset_name in ["enron"]:
        X, Y = load_dataset(dataset_name)
        splits = load_splits(dataset_name)
        rows = []

        permutations_count = 100
        if dataset_name in ("emotions", "scene"):
            permutations_count = factorial(6)

        permutations = make_permutations(Y.shape[1], permutations_count, RANDOM_SEED)

        for split_id, s in enumerate(splits):
            X_train, Y_train = X[s["train_inner"]], Y[s["train_inner"]]
            X_val,   Y_val   = X[s["val"]],         Y[s["val"]]
            X_test,  Y_test  = X[s["test"]],        Y[s["test"]]

            for perm_id, order in enumerate(permutations):
                model = ClassifierChain(base_estimator=clone(svm), order=order, random_state=RANDOM_SEED)
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                Y_val_pred = model.predict(X_val)
                row = {
                    "dataset": dataset_name,
                    "split": split_id,
                    "perm": perm_id,
                    "order": "-".join(map(str, order)),
                    "method": "CC",
                    "base_classifier": "SVM",
                }
                for metric_name, metric_fn in metrics.items():
                    row[f"test_{metric_name}"] = metric_fn(Y_test, Y_pred)
                    row[f"val_{metric_name}"] = metric_fn(Y_val, Y_val_pred)
                rows.append(row)
                print(f"{dataset_name} | split {split_id} | perm {perm_id} done")

            pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/{dataset_name}.csv", index=False)

if __name__ == "__main__":
    main()