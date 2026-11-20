"""
CLIF: Collaborative Learning of label semantIcs and deep label-specific Features.
J.-Y. Hang, M.-L. Zhang, IEEE TPAMI 44(12), 2022, s. 9860-9871.

Jak działa (w skrócie):
  1. Z danych treningowych liczymy, które etykiety często występują razem -> graf etykiet.
  2. Sieć grafowa (GIN) zamienia graf na wektory etykiet: podobne etykiety -> podobne wektory.
  3. Wektor etykiety mówi, które cechy próbki są ważne dla tej etykiety.
  4. Dla każdej etykiety osobny mały klasyfikator mówi "tak" albo "nie".
Wszystko uczy się naraz (jedna funkcja straty).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

NACHYLENIE = 0.1


def graf_wspolwystepowania(Y):
    # A[i][j] = (P(j | i) + P(i | j)) / 2
    liczba_etykiet = Y.shape[1]
    ile_razy = Y.sum(axis=0)
    razem = Y.T @ Y

    P = np.zeros((liczba_etykiet, liczba_etykiet))
    for i in range(liczba_etykiet):
        if ile_razy[i] > 0:
            P[i] = razem[i] / ile_razy[i]

    A = (P + P.T) / 2
    np.fill_diagonal(A, 0)
    return A


def graf_korelacji(Y):
    # modyfikacja: korelacja ze znakiem, uwzględnia też wykluczanie się etykiet
    A = np.corrcoef(Y, rowvar=False)
    A = np.nan_to_num(A)
    np.fill_diagonal(A, 0)
    return A


def zbuduj_graf(Y, rodzaj):
    if rodzaj == "cooccurrence":
        return graf_wspolwystepowania(Y)
    if rodzaj == "correlation":
        return graf_korelacji(Y)
    if rodzaj == "none":
        return np.zeros((Y.shape[1], Y.shape[1]))
    raise ValueError("adjacency musi być: 'cooccurrence', 'correlation' albo 'none'")


class WarstwaGIN(nn.Module):

    def __init__(self, wymiar_wej, wymiar_wyj):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.siec = nn.Sequential(
            nn.Linear(wymiar_wej, wymiar_wyj),
            nn.BatchNorm1d(wymiar_wyj),
            nn.LeakyReLU(NACHYLENIE),
            nn.Linear(wymiar_wyj, wymiar_wyj),
            nn.BatchNorm1d(wymiar_wyj),
            nn.LeakyReLU(NACHYLENIE),
        )

    def forward(self, H, A):
        return self.siec((1 + self.eps) * H + A @ H)


class SiecCLIF(nn.Module):

    def __init__(self, liczba_cech, liczba_etykiet, d_e, d_z):
        super().__init__()
        self.liczba_etykiet = liczba_etykiet
        self.H0 = torch.randn(liczba_etykiet, liczba_etykiet)

        self.gin1 = WarstwaGIN(liczba_etykiet, d_e)
        self.gin2 = WarstwaGIN(d_e, d_e)

        self.warstwa_cech = nn.Linear(liczba_cech, d_z)
        self.warstwa_waznosci = nn.Linear(d_e, d_z)
        self.warstwa_wyjsciowa = nn.Linear(d_z, d_z)
        self.klasyfikatory = nn.ModuleList([nn.Linear(d_z, 1) for _ in range(liczba_etykiet)])

        for warstwa in self.modules():
            if isinstance(warstwa, nn.Linear):
                nn.init.kaiming_normal_(warstwa.weight, a=NACHYLENIE)
                nn.init.zeros_(warstwa.bias)

    def wektory_etykiet(self, A):
        H1 = self.gin1(self.H0, A)
        H2 = self.gin2(H1, A)
        return H2 + H1

    def forward(self, x, A):
        E = self.wektory_etykiet(A)
        x_z = nn.functional.leaky_relu(self.warstwa_cech(x), NACHYLENIE)

        wyniki = []
        for k in range(self.liczba_etykiet):
            waznosc = torch.sigmoid(self.warstwa_waznosci(E[k]))
            z_k = nn.functional.leaky_relu(self.warstwa_wyjsciowa(x_z * waznosc), NACHYLENIE)
            wyniki.append(self.klasyfikatory[k](z_k))

        return torch.cat(wyniki, dim=1), E


def strata_wektorow_etykiet(E, A):
    # podobieństwo cosinusowe wektorów etykiet powinno odpowiadać grafowi A + I
    E_norm = E / E.norm(dim=1, keepdim=True)
    podobienstwo = E_norm @ E_norm.T
    cel = A + torch.eye(A.shape[0])
    return ((podobienstwo - cel) ** 2).mean()


class CLIF(BaseEstimator, ClassifierMixin):
    """
    adjacency: "cooccurrence" (artykuł), "correlation" (modyfikacja), "none" (ablacja)
    standardize: "auto" standaryzuje tylko cechy, które nie są binarne
    """

    def __init__(self, adjacency="cooccurrence", lambda_le=1.0, d_e=128, d_z=512,
                 n_epochs=500, patience=20, validation_fraction=0.1,
                 batch_size=1000, lr=1e-3, weight_decay=1e-5,
                 standardize="auto", random_state=None, verbose=False):
        self.adjacency = adjacency
        self.lambda_le = lambda_le
        self.d_e = d_e
        self.d_z = d_z
        self.n_epochs = n_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose

    def _przygotuj_X(self, X):
        X = (X - self.srednia_) / self.odchylenie_
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=int)

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(int(rng.integers(1_000_000)))

        # część danych treningowych odkładamy na walidację do early stopping
        kolejnosc = rng.permutation(len(X))
        ile_walidacja = int(self.validation_fraction * len(X))
        idx_wal = kolejnosc[:ile_walidacja]
        idx_tren = kolejnosc[ile_walidacja:]

        self.srednia_ = X[idx_tren].mean(axis=0)
        self.odchylenie_ = X[idx_tren].std(axis=0)
        self.odchylenie_[self.odchylenie_ == 0] = 1
        if self.standardize == "auto":
            # standaryzacja rzadkich cech binarnych (medical, enron) pogarszała wyniki
            cechy_binarne = np.all((X == 0) | (X == 1), axis=0)
            self.srednia_[cechy_binarne] = 0
            self.odchylenie_[cechy_binarne] = 1
        elif not self.standardize:
            self.srednia_[:] = 0
            self.odchylenie_[:] = 1

        X_tren = self._przygotuj_X(X[idx_tren])
        Y_tren = torch.tensor(Y[idx_tren], dtype=torch.float32)
        X_wal = self._przygotuj_X(X[idx_wal])
        Y_wal = torch.tensor(Y[idx_wal], dtype=torch.float32)

        self.A_ = torch.tensor(zbuduj_graf(Y[idx_tren], self.adjacency), dtype=torch.float32)

        self.siec_ = SiecCLIF(X.shape[1], Y.shape[1], self.d_e, self.d_z)
        optymalizator = torch.optim.Adam(self.siec_.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        najlepsza_strata = float("inf")
        najlepsze_wagi = copy.deepcopy(self.siec_.state_dict())
        epoki_bez_poprawy = 0

        for epoka in range(self.n_epochs):
            self.siec_.train()
            pomieszane = rng.permutation(len(X_tren))
            for start in range(0, len(X_tren), self.batch_size):
                batch = pomieszane[start:start + self.batch_size]
                logity, E = self.siec_(X_tren[batch], self.A_)
                strata = bce(logity, Y_tren[batch]) + self.lambda_le * strata_wektorow_etykiet(E, self.A_)

                optymalizator.zero_grad()
                strata.backward()
                optymalizator.step()

            if ile_walidacja == 0:
                continue
            self.siec_.eval()
            with torch.no_grad():
                logity_wal, _ = self.siec_(X_wal, self.A_)
                strata_wal = bce(logity_wal, Y_wal).item()

            if strata_wal < najlepsza_strata:
                najlepsza_strata = strata_wal
                najlepsze_wagi = copy.deepcopy(self.siec_.state_dict())
                epoki_bez_poprawy = 0
            else:
                epoki_bez_poprawy += 1

            if self.verbose:
                print(f"epoka {epoka + 1}: strata walidacyjna = {strata_wal:.4f}")
            if epoki_bez_poprawy >= self.patience:
                break

        if ile_walidacja > 0:
            self.siec_.load_state_dict(najlepsze_wagi)
        self.n_epochs_ = epoka + 1
        return self

    def predict_proba(self, X):
        X = self._przygotuj_X(np.asarray(X, dtype=float))
        self.siec_.eval()
        with torch.no_grad():
            logity, _ = self.siec_(X, self.A_)
        return torch.sigmoid(logity).numpy()

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def label_embeddings(self):
        self.siec_.eval()
        with torch.no_grad():
            return self.siec_.wektory_etykiet(self.A_).numpy()
