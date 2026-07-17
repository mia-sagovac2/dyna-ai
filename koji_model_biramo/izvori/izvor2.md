# IZVOR 2 - istraživanje algoritma Random Forest

### Naslov: 
Random Forest (2001)

### Autor:
Leo Breiman

### Link:
https://link.springer.com/article/10.1023/A:1010933404324

## Opis
Rad Lea Breimana (2001) uvodi algoritam Random Forest, jedan od najvažnijih modela u machine learningu.

Osnovna ideja:
- Random Forest = ansambl (ensemble) decision tree modela
- svaki tree:
    - trenira se na slučajnom uzorku podataka (bootstrap)
    - koristi slučajni podskup feature-a pri splitu

Intuicija: umjesto jednog stabla koje je nestabilno gradimo mnogo razlicitih stabala i kombiniramo njihove predikcije (glasanje/prosjek)


## Zasto je bitan za nas?
1. Rjesaca kljucni problem decision tree-a(visoki variance i nestabilni a sada smanjuje variance kroz averaging)
2. Uvodi kontrolu korelacije modela (randomizacija featurea, razlikuje ga od bagginga koji koristi samo bootstrap)
3. Teorijska analiza ensemble metoda (rad daje formalnu analizu generalization errora i vezu izmedju snage modela i korelacije)


## Glavni zakljucci
1. Random Forest = poboljsanje decision tree-a
2. Randomizaija je bitna (bootstrap + feature sampling)
3. Performanse ovise o jacini stabala i njihovoj medjusobnoj korelaciji
4. Vise stabala znaci stabilniji model