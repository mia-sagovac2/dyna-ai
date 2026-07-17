# IZVOR 3 - istraživanje bagging predictora

### Naslov: 
Bagging Predictors (1996)

### Autor:
Leo Breiman

### Link:
https://link.springer.com/article/10.1007/BF00058655

## Opis
Rad uvodi metodu: Bagging (Bootstrap Aggregating)

Osnovna ideja:
- generira se više verzija istog modela
- treniraju se na različitim bootstrap uzorcima
- predikcije se agregiraju (prosjek / glasanje)

Intuicija je da jedan model moze jako varirati ovisno o podacima i da je rjesenje trenirati puno modela na razlicitim podacima i kombinirati ih


## Zasto je bitno za nas
1. Prvi uspješan ensemble algoritam (uvodi ideju da je vise modela bolje od jednog)
2. Rješava ključni problem decision tree-a (da su nestabilni i imaju veliku varijancu a bagging stabilizira model)
3. Temelj za Random Forest (RF = bagging + dodatna randomizacija featurea)


## Glavni zakljucci rada
1. Variance je ključni problem ML modela
2. Bootstrap + averaging = snažna kombinacija
3. Nestabilni modeli najviše profitiraju
4. Ensemble pristup poboljšava generalizaciju
5. Bagging je univerzalna metoda (moze se primijeniti na stabla, regresiju i druge modele)
