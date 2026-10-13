# Multilabel-Classification-Label-Dependencies

Projekt z kursu **Uczenie Maszyn** (Politechnika Wrocławska, studia II stopnia, Informatyka Techniczna).
Ścieżka: **projekt badawczy**.

## Temat

**Klasyfikacja wieloetykietowa: kiedy opłaca się modelować zależności między etykietami?**

W klasyfikacji wieloetykietowej jedna próbka może mieć kilka etykiet jednocześnie (np. utwór muzyczny jest „spokojny” i „smutny”, zdjęcie zawiera „plażę” i „zachód słońca”). Etykiety często są ze sobą powiązane, a pytanie brzmi, czy jawne uwzględnienie tych zależności poprawia jakość klasyfikacji.

Projekt porównuje dwa podejścia:

- **Binary Relevance (BR):** osobny klasyfikator binarny dla każdej etykiety. Proste i szybkie, ale ignoruje zależności między etykietami.
- **Classifier Chains (CC):** klasyfikatory ustawione w łańcuch. Klasyfikator dla etykiety *k* dostaje cechy oraz etykiety 1, ..., *k-1*, dzięki czemu uczy się zależności między nimi. Wadą jest wrażliwość na kolejność łańcucha i propagacja błędów.
- **Ensemble of Classifier Chains (ECC):** kilka łańcuchów o losowej kolejności z uśrednionymi predykcjami, jako sposób na ograniczenie wpływu kolejności.

## Pytania badawcze

1. Czy Classifier Chains osiąga statystycznie lepsze wyniki niż Binary Relevance?
2. Jak bardzo wynik Classifier Chains zależy od kolejności etykiet w łańcuchu?
3. Czy zespół łańcuchów (ECC) eliminuje problem kolejności i od ilu łańcuchów zysk przestaje rosnąć?
4. Czy przewaga CC nad BR rośnie wraz z siłą zależności między etykietami w zbiorze danych?

## Powiązane prace

Kompletne wprowadzenie do uczenia wieloetykietowego, obejmujące podział metod na transformacyjne i adaptacyjne, miary jakości oraz standardowe zbiory danych, zawiera tutorial [1]. Na nim oparte są definicje i metryki używane w projekcie.

Classifier Chains zaproponowali Read i in. jako rozszerzenie Binary Relevance, które uwzględnia zależności między etykietami. Przegląd [2] podsumowuje kilkanaście lat badań nad tą metodą: wyjaśnia, dlaczego łańcuchy działają, jak na wynik wpływają kolejność etykiet i propagacja błędów, oraz omawia warianty, w tym Ensemble of Classifier Chains.

Najszersze dotąd porównanie eksperymentalne metod wieloetykietowych [3] obejmuje 26 metod, 42 zbiory danych i 18 miar jakości, z analizą testem Friedmana i post-hoc Nemenyiego. ECC znalazło się wśród metod rekomendowanych jako punkt odniesienia. Na tej pracy wzorowany jest protokół eksperymentalny projektu. Ci sami autorzy w [4] wiążą wyniki metod z właściwościami zbiorów danych, co jest punktem wyjścia dla pytania badawczego 4.

Niniejszy projekt nie proponuje nowej metody. Skupia się na kontrolowanym porównaniu BR, CC i ECC, aby sprawdzić, w jakich warunkach modelowanie zależności między etykietami faktycznie się opłaca.

## Metody

### Implementacja własna

Metody zaimplementowane od podstaw (Python + numpy), zgodnie z API scikit-learn (`BaseEstimator`, `ClassifierMixin`, `fit`, `predict`, `predict_proba`):

- `BinaryRelevance`
- `ClassifierChain` (losowa lub zadana kolejność etykiet)
- `EnsembleClassifierChains`

### Metody referencyjne

- `MultiOutputClassifier` (scikit-learn), odpowiednik BR
- `ClassifierChain` (scikit-learn)

### Klasyfikatory bazowe

- Regresja logistyczna
- Drzewo decyzyjne
- k-NN

## Dane

Standardowe zbiory wieloetykietowe z repozytorium MULAN / OpenML, m.in.:

| Zbiór | Dziedzina |
|---|---|
| emotions | muzyka |
| scene | obrazy |
| yeast | biologia |
| birds | dźwięk |
| medical | tekst |
| enron | tekst |

## Protokół eksperymentalny

- Walidacja: 5x2 CV (lub powtarzana 5-krotna walidacja krzyżowa)
- Metryki główne: **Hamming loss**, **macro F1**
- Metryki dodatkowe: subset accuracy, micro F1
- Analiza statystyczna:
  - test Wilcoxona (porównania parowe),
  - test Friedmana z testem post-hoc Nemenyiego (ranking globalny, diagram różnic krytycznych)
- Stały `random_state` dla powtarzalności wyników

## Eksperymenty

| ID | Eksperyment | Co jest badane |
|---|---|---|
| E0 | Porównanie bazowe | Implementacja własna vs scikit-learn, BR vs CC na wszystkich zbiorach |
| E1 | Kolejność łańcucha | Rozrzut wyników CC dla 20 losowych kolejności etykiet |
| E2 | Ensemble of Classifier Chains | Liczba łańcuchów: 1, 5, 10, 20 |
| E3 | Kolejność przemyślana | Sortowanie etykiet wg częstości lub korelacji vs kolejność losowa |
| E4 | Ablacja uczenia łańcucha | Prawdziwe etykiety vs predykcje poprzednich klasyfikatorów jako cechy |
| E5 | Zależności etykiet | Związek między siłą korelacji etykiet w zbiorze a zyskiem CC względem BR |

## Struktura repozytorium

```
Multilabel-Classification-Label-Dependencies/
├── methods/          # implementacje BR, CC, ECC
├── experiments/      # osobny skrypt dla każdego eksperymentu
├── analysis/         # testy statystyczne, tabele, wykresy
├── data/             # zbiory danych
├── results/          # surowe wyniki (.npy / .csv)
├── figures/          # wygenerowane wykresy
├── requirements.txt
└── README.md
```

## Uruchomienie

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python experiments/e0_baseline.py
python analysis/statistics.py
```

## Wymagania

- Python 3.10+
- numpy, scipy, pandas
- scikit-learn
- matplotlib
- scikit-multilearn (wczytywanie zbiorów MULAN)

## Literatura

1. E. Gibaja, S. Ventura, *A Tutorial on Multilabel Learning*, ACM Computing Surveys, vol. 47, nr 3, art. 52, 2015. DOI: [10.1145/2716262](https://doi.org/10.1145/2716262)
2. J. Read, B. Pfahringer, G. Holmes, E. Frank, *Classifier Chains: A Review and Perspectives*, Journal of Artificial Intelligence Research, vol. 70, s. 683–718, 2021. DOI: [10.1613/jair.1.12376](https://doi.org/10.1613/jair.1.12376)
3. J. Bogatinovski, L. Todorovski, S. Džeroski, D. Kocev, *Comprehensive comparative study of multi-label classification methods*, Expert Systems with Applications, vol. 203, 117215, 2022. DOI: [10.1016/j.eswa.2022.117215](https://doi.org/10.1016/j.eswa.2022.117215) ([arXiv](https://arxiv.org/abs/2102.07113))
4. J. Bogatinovski, L. Todorovski, S. Džeroski, D. Kocev, *Explaining the performance of multilabel classification methods with data set properties*, International Journal of Intelligent Systems, 2022. DOI: [10.1002/int.22835](https://doi.org/10.1002/int.22835)
