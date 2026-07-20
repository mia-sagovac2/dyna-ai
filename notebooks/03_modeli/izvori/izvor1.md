# IZVOR 1 - istraživanje vaznosti CART-a

### Naslov: 
Classiﬁcation and Regression Trees (2011)

### Autor:
Wel-Yin Loh

### Link:
https://www.researchgate.net/publication/227658748_Classification_and_Regression_Trees

## Opis
Članak o Classification and Regression Trees (CART) predstavlja jedan od temeljnih radova u području statističkog učenja i machine learninga. CART metoda (Breiman et al., 1984) definira način kako izgraditi decision tree modele za:
- klasifikaciju (diskretni izlazi)
- regresiju (kontinuirani izlazi)

Osnovna ideja:
- Podaci se rekurzivno dijele (partitioning) u manje skupove
- U svakom čvoru bira se najbolji split (feature + threshold)
- Cilj je dobiti homogene grupe (što sličnije vrijednosti ciljne varijable)

Rezultat je:
- hijerarhijska struktura (stablo)
- model koji se može interpretirati kao niz if–then pravila


## Zasto je bitno za nas?
1. Standardizira decision tree metodologiju (definira formalni framework, uvodi: splitting kriterije; pruning; evaluaciju)
2. Temelj je za moderne algoritme (CART je osnova za Random Forest, Gradient Boosting i XGBoost)
3. Most izmedju statistike i ML-a (spaja neparametarske metode i algoritamski pristup)
4. Interpretabilnost modela (decision tree je jedan od rijetkih transparentnih i lako objasnjivih modela)


## Glavni zakljucci
1. Decision trees = fleksibilna aproksimacija funkcije
2. Kljuc uspjeha su dobro definirani split kriteriji
3. Overfitting je centralni problem
4. Jednostavnost nije nuzno slabost
5. CART je temelj moderne ML prakse