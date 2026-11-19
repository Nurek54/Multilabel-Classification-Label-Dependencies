from data import DATASETS, load_dataset
from .generate_splits import load_splits
from sklearn.linear_model import LogisticRegression
from methods import BinaryRelevance, ClassifierChain, EnsembleClassifierChains


multilabel_methods = {
    "BR": BinaryRelevance(),
    "CC": ClassifierChain(),
    "ECC": EnsembleClassifierChains(),
}

# TODO: use base_classifiers from library. what other base_classifiers could be used here?
base_classifiers = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": "",
    "kNN": "",
    "SVM": "",
}

# TODO: use metric from library
metrics = {
    "hamming_loss": "",
    "subset_accuracy": "",
    "macro_F1": "",
    "micro_F1": ""
}

for dataset_name in DATASETS:
    X, Y = load_dataset(dataset_name)
    splits = load_splits(dataset_name)

    for s in splits:
        X_train, Y_train = X[s["train_inner"]], Y[s["train_inner"]]
        X_val,   Y_val   = X[s["val"]],         Y[s["val"]]
        X_test,  Y_test  = X[s["test"]],        Y[s["test"]]

        # TODO: train multilabel methods with given base_classifiers. evaluate on all metrics. save scores to files