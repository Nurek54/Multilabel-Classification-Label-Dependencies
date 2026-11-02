# Multilabel-Classification-Label-Dependencies

Projekt z kursu **Uczenie Maszyn** (Politechnika Wrocławska, studia II stopnia, Informatyka Techniczna).
Ścieżka: **projekt badawczy**.
Autorzy: Robert Pytel, Grzegorz Sienkiewicz.

## Temat

**Klasyfikacja wieloetykietowa: kiedy opłaca się modelować zależności między etykietami?**

W klasyfikacji wieloetykietowej jedna próbka może mieć kilka etykiet jednocześnie (np. zdjęcie zawiera „plażę”, „morze” i „zachód słońca”). Etykiety często są ze sobą powiązane, a pytanie brzmi, czy jawne uwzględnienie tych zależności poprawia jakość klasyfikacji.

## Metody

### Metody bazowe (zaimplementowane)

- **Binary Relevance (BR):** osobny klasyfikator binarny dla każdej etykiety. Proste i szybkie, ale ignoruje zależności między etykietami.
- **Classifier Chains (CC):** klasyfikatory ustawione w łańcuch. Klasyfikator dla etykiety *k* dostaje cechy oraz etykiety 1, ..., *k-1*, dzięki czemu uczy się zależności między nimi.
- **Ensemble of Classifier Chains (ECC):** kilka łańcuchów o losowej kolejności z uśrednionymi prawdopodobieństwami.

### Metoda z literatury (do wyboru przez prowadzącego)

Metody bazowe zostaną uzupełnione o jedną metodę neuronową z literatury, niedostępną w gotowych bibliotekach:

- **Propozycja A: Classifier Chain Networks (CCN) [1], 2025.** Łańcuch klasyfikatorów uczony jak sieć neuronowa. Każda etykieta dostaje cechy próbki oraz prawdopodobieństwa wcześniejszych etykiet, a wszystkie etykiety uczą się naraz z jedną funkcją straty. Siłę zależności między etykietami opisują wyuczone wagi. **Modyfikacja:** warstwy ukryte (Deep CCN).
- **Propozycja B: CLIF [2], 2022.** Z danych treningowych budowany jest graf współwystępowania etykiet. Sieć grafowa (GIN) uczy się na nim reprezentacji etykiet, które następnie wybierają z cech próbki cechy ważne dla danej etykiety. **Modyfikacja:** inny sposób budowania grafu, np. z korelacji uwzględniającej wykluczanie się etykiet.

Wszystkie metody implementujemy samodzielnie (numpy / PyTorch), zgodnie z API scikit-learn.

## Pytania badawcze

1. Jak wybór klasyfikatora bazowego wpływa na jakość klasyfikacji?
2. Jak kolejność przetwarzania etykiet w łańcuchu wpływa na jakość klasyfikacji?
3. Jak liczba łańcuchów w metodzie Ensemble of Classifier Chains wpływa na wyniki klasyfikacji?
4. W zależności od wybranej metody z literatury:
    - **CCN:** ile wnoszą połączenia między etykietami i czy warstwy ukryte (Deep CCN) poprawiają jakość klasyfikacji?
    - **CLIF:** ile wnosi graf etykiet (porównanie z wariantem bez informacji o etykietach) i czy alternatywna konstrukcja grafu poprawia jakość klasyfikacji?

## Zbiory danych

Standardowe zbiory wieloetykietowe z repozytorium [MULAN](https://mulan.sourceforge.net/datasets-mlc.html):

| Zbiór | Dziedzina | Przykłady | Cechy | Etykiety |
|---|---|---|---|---|
| emotions | muzyka | 593 | 72 | 6 |
| scene | obrazy | 2407 | 294 | 6 |
| yeast | biologia | 2417 | 103 | 14 |
| birds | dźwięk | 645 | 260 | 19 |
| medical | tekst medyczny | 978 | 1449 | 45 |
| enron | tekst (e-maile) | 1702 | 1001 | 53 |

## Stan implementacji

| Element | Status |
|---|---|
| Binary Relevance | ✔ gotowe, predykcje identyczne z `MultiOutputClassifier` |
| Classifier Chains | ✔ gotowe, predykcje identyczne z `ClassifierChain` ze sklearn |
| Ensemble of Classifier Chains | ✔ gotowe |
| Metoda z literatury (CCN / CLIF) | ⏳ czeka na decyzję prowadzącego |
| Wczytywanie zbiorów MULAN | ⏳ do zrobienia |
| Eksperymenty i analiza statystyczna | ⏳ do zrobienia |

## Struktura repozytorium

```
Multilabel-Classification-Label-Dependencies/
├── main.py                              # demo: BR, CC, ECC z 3 klasyfikatorami bazowymi
├── methods/
│   ├── __init__.py
│   ├── _utils.py                        # klasyfikator domyślny, obsługa etykiet z jedną klasą
│   ├── binary_relevance.py              # BinaryRelevance
│   ├── classifier_chain.py              # ClassifierChain
│   └── ensemble_classifier_chains.py    # EnsembleClassifierChains
├── experiments/
│   └── sanity_check.py                  # porównanie własnych implementacji ze sklearn
├── requirements.txt
└── README.md
```

## Użycie

Każda metoda dziedziczy po `BaseEstimator` i `ClassifierMixin` i ma metody `fit(X, Y)`, `predict(X)` oraz `predict_proba(X)`. `Y` to macierz 0/1 o wymiarze (liczba próbek × liczba etykiet).

```python
from sklearn.tree import DecisionTreeClassifier
from methods import BinaryRelevance, ClassifierChain, EnsembleClassifierChains

br = BinaryRelevance(base_estimator=DecisionTreeClassifier())
cc = ClassifierChain(base_estimator=DecisionTreeClassifier(), order="random", random_state=0)
ecc = EnsembleClassifierChains(base_estimator=DecisionTreeClassifier(), n_chains=10, random_state=0)

Y_pred = ecc.fit(X_train, Y_train).predict(X_test)
```

- `base_estimator`: dowolny klasyfikator binarny ze sklearn (domyślnie regresja logistyczna).
- `order` (CC): `None` (kolejność 0, 1, ..., L-1), `"random"` albo lista indeksów etykiet.
- `n_chains` (ECC): liczba łańcuchów.

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python main.py                          # demo na danych syntetycznych
python -m experiments.sanity_check      # porównanie ze sklearn
```

## TODO

**Metoda z literatury**
- [ ] Zaimplementować wybraną metodę w PyTorch, zgodnie z API scikit-learn
- [ ] Zaimplementować modyfikację (Deep CCN albo alternatywny graf etykiet)
- [ ] Dodać `torch` do `requirements.txt`

**Dane**
- [ ] Wczytywanie zbiorów MULAN (np. `skmultilearn.dataset.load_dataset`)
- [ ] Podział danych ze stratyfikacją iteracyjną (`IterativeStratification`)

**Eksperymenty (osobny skrypt dla każdego)**
- [ ] Plan eksperymentu: protokół (np. 5x2 CV), metryki (Hamming loss, subset accuracy, macro F1), parametry
- [ ] E1: wpływ klasyfikatora bazowego (regresja logistyczna, drzewo, kNN)
- [ ] E2: wpływ kolejności etykiet w łańcuchu
- [ ] E3: wpływ liczby łańcuchów w ECC
- [ ] E4: ablacja metody z literatury
- [ ] Zapis surowych wyników do `results/` (.npy / .csv)

**Analiza i raport**
- [ ] Testy statystyczne: Wilcoxon (parami), Friedman + Nemenyi (ranking globalny)
- [ ] Tabele i wykresy (matplotlib), diagram różnic krytycznych
- [ ] Raport w szablonie Springer LNCS lub IEEE
- [ ] Deklaracja użycia narzędzi AI w raporcie

**Opcjonalnie**
- [ ] Bootstrap w ECC (jak w oryginalnym artykule Read i in.)

## Bibliografia

1. D. J. W. Touw, M. Van de Velden, *Classifier chain networks for multi-label classification*, Expert Systems with Applications, vol. 286, art. 128048, 2025. DOI: [10.1016/j.eswa.2025.128048](https://doi.org/10.1016/j.eswa.2025.128048)
2. J.-Y. Hang, M.-L. Zhang, *Collaborative learning of label semantics and deep label-specific features for multi-label classification*, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 12, s. 9860–9871, 2022. DOI: [10.1109/TPAMI.2021.3136592](https://doi.org/10.1109/TPAMI.2021.3136592)
3. W. Liu, H. Wang, X. Shen, I. W. Tsang, *The emerging trends of multi-label learning*, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, s. 7955–7974, 2022. DOI: [10.1109/TPAMI.2021.3119334](https://doi.org/10.1109/TPAMI.2021.3119334)
4. J. Bogatinovski, L. Todorovski, S. Džeroski, D. Kocev, *Comprehensive comparative study of multi-label classification methods*, Expert Systems with Applications, vol. 203, art. 117215, 2022. DOI: [10.1016/j.eswa.2022.117215](https://doi.org/10.1016/j.eswa.2022.117215)
