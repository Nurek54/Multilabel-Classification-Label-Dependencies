import argparse
import numpy as np
from . import DATASETS, load_dataset

def analyze(name):
    X, Y, _, label_names = load_dataset(name, return_names=True)
    n, q = Y.shape
    counts = Y.sum(axis=0)
    card = Y.sum(axis=1).mean()
    density = card / q

    print("=" * 78)
    print(f"{name}: {n} próbek, {X.shape[1]} cech, {q} etykiet, "
          f"cardinality={card:.3f}, density={density:.3f}")
    print(f"Pozytywów na etykietę: min={counts.min()}  "
          f"Q1={np.percentile(counts, 25):.0f}  mediana={np.median(counts):.0f}  "
          f"Q3={np.percentile(counts, 75):.0f}  max={counts.max()}  "
          f"średnia={counts.mean():.1f}")

    for idx, label in enumerate(label_names):
        print(str(label), "\tpos: ", counts[idx], "\tneg: ", len(Y) - counts[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()

    for name in args.datasets:
        analyze(name)


if __name__ == "__main__":
    main()