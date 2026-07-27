# IZVOR 8 - pregledni/uvodni clanak o Naive Bayesu

### Naslov:
A Simple Explanation of Naive Bayes Classification

### Autor:
Baeldung (online edukacijski portal)

### Link:
https://www.baeldung.com/cs/naive-bayes-classification

## Opis
Pregledni edukativni članak koji uvodi Naive Bayes klasifikator kroz Bayesov teorem i
postupno ga pojednostavljuje do praktičnog modela za klasifikaciju. Fokus je na intuiciji:
kako model kombinira prethodno znanje (prior) i podatke (likelihood) za izračun posterior
vjerojatnosti klase.

Glavna ideja članka je da se kompleksni problem procjene zajedničke distribucije značajki
pojednostavljuje pretpostavkom njihove međusobne nezavisnosti, čime se izračun svodi na
umnožak pojedinačnih vjerojatnosti. Time model postaje računski vrlo jednostavan i skalabilan.

## Zasto je bitno za nas?
1. Daje jasnu intuiciju iza formule
   - pomaže objasniti što model zapravo radi u našem audio klasifikacijskom zadatku
2. Objašnjava uloge:
   - prior - raspodjela klasa (npr. dominacija CAR u datasetu)
   - likelihood - kako značajke izgledaju unutar klase
   - posterior - konačna odluka modela
3. Direktno se može povezati s našim pipelineom:
   - značajke (MFCC, spectral...) - ulaz X
   - klase (BIC, CAR, HEAVY, MC) - C
4. Dobar temelj za objašnjenje zašto koristimo GaussianNB
   - jer su naše značajke kontinuirane

## Glavni zakljucci
1. Naive Bayes računa najvjerojatniju klasu koristeći Bayesov teorem i pojednostavljenje
nezavisnosti značajki
2. Pretpostavka nezavisnosti značajno smanjuje kompleksnost modela
3. Model je vrlo brz i skalabilan, čak i za velike skupove podataka
4. Unatoč pojednostavljenjima, često daje dobre rezultate u praksi
5. Ključna snaga modela je u jednostavnosti i interpretabilnosti, a ne u maksimalnoj točnosti
