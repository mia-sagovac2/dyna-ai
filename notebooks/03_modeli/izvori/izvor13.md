# IZVOR 13 - teoretski dokaz da MLP moze aproksimirati bilo koju funkciju

### Naslov:
Approximation by Superpositions of a Sigmoidal Function (1989)

### Autor:
George Cybenko

### Link:
https://link.springer.com/article/10.1007/BF02551274

## Opis
Poznat kao "Universal Approximation Theorem". Cybenko matematicki dokazuje da mreza s **jednim**
skrivenim slojem (dovoljno velikim) i sigmoidalnom aktivacijom moze aproksimirati bilo koju
kontinuiranu funkciju na proizvoljnu tocnost. Ovo je cisto teoretski rezultat o postojanju
takve mreze - ne govori kako je efikasno trenirati (za to treba backprop, izvor12).

## Zasto je bitno za nas?
1. Daje teoretsko opravdanje zasto uopce koristimo neuronske mreze - u principu mogu
   modelirati bilo koji odnos izmedju ulaznih znacajki i izlaza
2. Objasnjava zasto dodavanje skrivenog sloja rjesava probleme koje perceptron ne moze (XOR)
3. Vazno je za razumijevanje granica: teorem kaze da mreza *moze* postojati, ne da ce je
   gradient descent nuzno pronaci - u praksi treba dovoljno podataka i dobra arhitektura

## Glavni zakljucci
1. Jedan dovoljno "sirok" skriveni sloj je teoretski dovoljan za aproksimaciju bilo koje funkcije
2. Teorem ne govori nista o broju neurona potrebnih u praksi niti o treniranju
3. Motivira zasto dublje/sire mreze rjesavaju kompleksnije probleme bolje u praksi (efikasnost,
   ne teoretska mogucnost)
