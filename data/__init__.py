"""
Wczytywanie zbiorów wieloetykietowych z repozytorium MULAN (zapisanych lokalnie jako .npz).

Użycie:
    from data import load_dataset, DATASETS
    X, Y = load_dataset("emotions")
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
DATASETS = ["emotions", "scene", "yeast", "birds", "medical", "enron"]


def load_dataset(name, return_names=False):
    """
    Zwraca X (próbki × cechy, float) oraz Y (próbki × etykiety, macierz 0/1).
    Przy return_names=True dodatkowo nazwy cech i nazwy etykiet.
    """
    if name not in DATASETS:
        raise ValueError(f"Nieznany zbiór '{name}'. Dostępne: {DATASETS}")
    d = np.load(DATA_DIR / f"{name}.npz")
    X = d["X"].astype(np.float64)
    Y = d["Y"].astype(int)
    if return_names:
        return X, Y, list(d["feature_names"]), list(d["label_names"])
    return X, Y


def summary():
    """Wypisuje podstawowe statystyki wszystkich zbiorów."""
    print(f"{'Zbiór':<10}{'Próbki':>8}{'Cechy':>8}{'Etykiety':>10}{'Cardinality':>13}{'Density':>10}")
    for name in DATASETS:
        X, Y = load_dataset(name)
        card = Y.sum(axis=1).mean()
        print(f"{name:<10}{X.shape[0]:>8}{X.shape[1]:>8}{Y.shape[1]:>10}{card:>13.3f}{card / Y.shape[1]:>10.3f}")

