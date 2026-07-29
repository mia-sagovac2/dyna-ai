# IZVOR 12 - backpropagation, algoritam koji omogucava treniranje visesojnih mreza

### Naslov:
Learning representations by back-propagating errors (1986)

### Autor:
David Rumelhart, Geoffrey Hinton, Ronald Williams

### Link:
https://www.nature.com/articles/323533a0

## Opis
Rad koji je popularizirao backpropagation - algoritam koji omogucava treniranje mreza s
skrivenim slojevima (visesojne, feedforward neuronske mreze / MLP). Prije ovoga nije postojao
prakticni nacin da se azuriraju tezine slojeva koji nisu direktno povezani s izlazom. Backprop
racuna gradijent greske unatrag kroz mrezu (chain rule), sloj po sloj, i azurira sve tezine
odjednom pomocu gradient descenta.

## Zasto je bitno za nas?
1. Rjesava ogranicenje perceptrona (izvor11) - skriveni slojevi omogucavaju nelinearne granice
   odluke (npr. XOR), ali ih je bez backpropa nemoguce trenirati
2. Temelj je za apsolutno sve moderne neuronske mreze (MLP, RNN, autoencoderi...) koje koristimo
   dalje
3. Direktno objasnjava kako Keras/TensorFlow interno trenira nase modele (model.fit poziva
   backprop ispod haube)

## Glavni zakljucci
1. Backprop = chain rule primijenjen unatrag kroz slojeve mreze
2. Omogucava treniranje proizvoljno dubokih mreza (u teoriji)
3. Skriveni slojevi uce interne reprezentacije podataka koje covjek ne definira rucno
