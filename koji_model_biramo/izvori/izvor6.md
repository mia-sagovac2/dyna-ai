# IZVOR 6 - istrazivanje XGBoost-a

### Naslov: 
XGBoost: A Scalable Tree Boosting System (2016)

### Autor:
Tianqi Chen i Carlos Guestrin

### Link:
https://dl.acm.org/doi/epdf/10.1145/2939672.2939785

## Opis
Rad predstavlja XGBoost (Extreme Gradient Boosting) – optimizirani i skalabilni sistem za gradient boosting nad stablima odlučivanja.

XGBoost je brzo postao state-of-the-art metoda za tabularne podatke i standard u industriji i natjecanjima.

### Glavna ideja
Model gradi ansambl stabla:
$F(x) = \sum_{m=1}^{M} f_m(x)$

gdje svako novo stablo uci na gradijentima (greskama) prethodnog modela i minimizira funckiju gubitka


### Kljucni doprinosi rada
1. Skalabilan sistem - moze raditi na jednoj masini kao i distribuiranim sistemima, obraduje milijarde primjera uz manju potrosnju resursa
2. Sparsity aware algoritam - efikasno rukuje missing vrijednostima i sparse feature-ima i koristi default direction u stablima
3. Weighted Quantile Sketch - koristi se za priblizno trazenje optimalnih splitovai i omogucava rad sa tezinama uzoraka i efikasno trniranje velikih datasetova
4. Podaci se spremaju u sortirane kolonske blokove sto omogucava brze trazenje splitova

### Regularizacija 
XGBoost uvodi regularizaciju stabala:

$L = \sum l(y_i, \hat{y}_i) + \sum \Omega(f_k)$

- gdje  $\Omega(f_k)$  penalizira kompleksnost stabla
- rezultat je manje overfittinga i stabilniji modeli


### Prednosti
- visoka tocnost na tabularnim podacima
- podrzava klasifikaciju, regresiju i ranking
- ugradjene opcije su regularizacija, subsampling i feature importance

### Nedostatci
- mnogo hiperparametara --> teze tuniranje
- moze overfitati bez regularizacije
- manje interpretabilan od pojedinacnih stabala

