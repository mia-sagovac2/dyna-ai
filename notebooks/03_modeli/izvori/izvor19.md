# IZVOR 19 - zasto je duboke mreze tesko trenirati i kako to popraviti

### Naslov:
Understanding the difficulty of training deep feedforward neural networks (2010)

### Autor:
Xavier Glorot, Yoshua Bengio

### Link:
https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

## Opis
Rad koji sustavno analizira zasto duboke mreze sa sigmoid/tanh aktivacijama i standardnom
(nasumicnom) inicijalizacijom tezina lose treniraju - gradijenti se pri prolasku unatrag kroz
mnogo slojeva eksponencijalno smanjuju (vanishing gradient), isti problem koji smo vidjeli kod
RNN-a kroz vrijeme (sekcija 5), ali ovdje kroz **dubinu** mreze. Autori predlazu "Xavier/Glorot"
inicijalizaciju tezina koja skalira pocetne vrijednosti prema broju ulaznih/izlaznih neurona sloja,
cime se gradijent odrzava priblizno konstantne velicine kroz slojeve.

## Zasto je bitno za nas?
1. Direktno objasnjava fenomen koji demonstriramo u sintetickom primjeru - duboka mreza sa sigmoid
   aktivacijom uci sporo/nikako, dok ReLU i/ili bolja inicijalizacija to popravljaju
2. Motivira zasto se danas gotovo iskljucivo koristi ReLU (i njene varijante) i pazljiva
   inicijalizacija tezina umjesto sigmoid/tanh u skrivenim slojevima
3. Isto nacelo (odrzavanje stabilnog gradijenta kroz dubinu) lezi u pozadini modernih tehnika
   poput batch normalizacije i rezidualnih (skip) veza

## Glavni zakljucci
1. Standardna inicijalizacija + sigmoid/tanh aktivacije uzrokuju vanishing gradient u dubokim mrezama
2. Velicina gradijenta ovisi o skaliranju tezina i izlaznoj varijanci svakog sloja
3. Pazljiva inicijalizacija (Xavier/Glorot) i ReLU aktivacije bitno olaksavaju treniranje dubine
