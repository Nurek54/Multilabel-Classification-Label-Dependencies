import pickle
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from data import DATASETS, load_dataset
import os

RANDOM_SEED = 26600

def generate_splits(X, Y, n_repeats=5, n_splits=5, val_fraction=0.2):
    splits = []

    for repeat in range(n_repeats):
        cv = MultilabelStratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_SEED+repeat
        )

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            inner = MultilabelStratifiedKFold(
                n_splits=round(1 / val_fraction), shuffle=True,
                random_state=RANDOM_SEED + 1000 + repeat * n_splits + fold,
            )
            tr_in, val = next(inner.split(X[train_idx], Y[train_idx]))
            splits.append({
                "repeat": repeat,
                "fold": fold,
                "train": train_idx,
                "train_inner": train_idx[tr_in],
                "val": train_idx[val],
                "test": test_idx,
            })

    return splits

def main():    
    os.makedirs("experiments/dataset_splits", exist_ok=True)

    for name in DATASETS:
        X, Y = load_dataset(name)
        splits = generate_splits(X, Y)

        with open(f"experiments/dataset_splits/{name}.pkl", "wb") as f:
            pickle.dump(splits, f)

if __name__ == "__main__":
    main()