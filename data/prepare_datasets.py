import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parent
OPENML_IDS = {"emotions": 40589, "scene": 40595, "yeast": 40597, "birds": 40588, "enron": 40590}


def to_numeric(df):
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            vals = out[c].astype(str).str.strip().str.upper()
            if set(vals.unique()) <= {"0", "1", "TRUE", "FALSE", "YES", "NO"}:
                out[c] = vals.isin(["1", "TRUE", "YES"]).astype(int)
            else:
                out[c] = out[c].astype("category").cat.codes
    return out.astype(float)


def save(name, X, Y, feature_names, label_names):
    np.savez_compressed(DATA_DIR / f"{name}.npz", X=X.astype(np.float32), Y=Y.astype(np.int8),
                        feature_names=np.array(feature_names), label_names=np.array(label_names))
    print(f"{name:<9} X={X.shape}  Y={Y.shape}")


def prepare_openml():
    for name, data_id in OPENML_IDS.items():
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto",
                             data_home=str(DATA_DIR / "raw" / "openml"))
        X, Y = to_numeric(bunch.data), to_numeric(bunch.target)
        save(name, X.values, Y.values.astype(int), list(X.columns), list(Y.columns))


def prepare_medical(folder):
    import arff  # liac-arff
    folder = Path(folder)
    labels = [l.attrib["name"] for l in ET.parse(folder / "medical.xml").getroot()]
    with open(folder / "medical.arff") as f:
        d = arff.load(f)
    cols = [a[0] for a in d["attributes"]]
    df = pd.DataFrame(d["data"], columns=cols)
    Y = to_numeric(df[labels])
    X = to_numeric(df[[c for c in cols if c not in labels]])
    save("medical", X.values, Y.values.astype(int), list(X.columns), labels)


if __name__ == "__main__":
    prepare_openml()
    medical_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "raw" / "medical"
    if (Path(medical_dir) / "medical.arff").exists():
        prepare_medical(medical_dir)
    else:
        print(f"Pominięto medical: brak plików w {medical_dir}")
