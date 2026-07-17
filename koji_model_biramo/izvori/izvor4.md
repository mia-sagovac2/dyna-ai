# IZVOR 4 - isticanje negativnih strana Random Forest-a

### Naslov: 
Bias in random forest variable importance measures: Illustrations, sources and a solution (2007)

### Autor:
Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis, Torsten Hothorn

### Link:
https://link.springer.com/article/10.1186/1471-2105-8-25

## Opis
Ovaj rad je dobar za razumijevanje ogranicenja Random Forest-a, posebno u kontekstu feature importancea.
Rad analizira koliko su pouzdane mjere vaznosti varijabli u Random Forest-u.

Osnovna ideja- Random Forest se često koristi za feature selection i interpretaciju modela

Problem je da te mjere mogu biti pristrane (biased)

Glavni zaključak rada: važnost varijabli nije pouzdana kada varijable imaju različit broj kategorija ili različitu skalu


## Zasto je bitan za nas?
1. Razbija mit o interpretabilnosti RF-a (prije se mislilo da RF daje pouzdanu vaznost feature-a, nakon ovog rada to vrijedi samo pod odredjenim uvjetima)
2. Kljucan za Explainable AI - pokazuje da interpretacija modela moze biti pogresna 
3. Nadogradnja Breimanovog Random Forest-a- pokazuje njegova slabosti


## Glavni zakljucci
1. Variable importance nije uvijek pouzdan - posebno kod mejsovitih tipova podataka
2. Bias dolazi iz strukture stabla - ne samo iz RF-a nego i iz CART-a
3. Bootstrap dodatno pojacava pristranost
4. Potrebne su modificirane metode (unbiased trees, subsampling)
5. Ne smije se vjerovati na slijepo importance mjerama