# IZVOR 7 - zasto naivna pretpostavka nezavisnosti ipak cesto radi

### Naslov:
On the Optimality of the Simple Bayesian Classifier under Zero-One Loss (1997)

### Autor:
Pedro Domingos i Michael Pazzani

### Link:
https://doi.org/10.1023/A:1007413511361

## Opis
Rad se bavi ocitim paradoksom oko Naive Bayes klasifikatora: pretpostavka o uvjetnoj
nezavisnosti znacajki (dan razred) je gotovo uvijek krsena u stvarnim podacima (znacajke su
korelirane), a klasifikator ipak cesto postize konkurentnu tocnost u odnosu na puno
slozenije modele koji tu pretpostavku ne rade.

Autori formalno pokazuju da tocnost klasifikacije (zero-one loss, tj. "je li predvidjeni
razred tocan") ovisi samo o tome je li **argmax** posteriorne vjerojatnosti ispravan, a ne o
tome koliko su same procijenjene vjerojatnosti (posteriori) tocne. Drugim rijecima - model
moze imati sasvim pogresne (iskrivljene) procjene $P(y|x)$ zbog krive pretpostavke o
nezavisnosti, a svejedno uvijek birati ispravan razred kao maksimum, sve dok se ta
distorzija "jednako" odrazava na sve klase.

## Zasto je bitno za nas?
1. Direktno objasnjava zasto smijemo koristiti Naive Bayes na nasem datasetu iako smo u
   korelacijskoj matrici (VIDJETI notebook) vidjeli da znacajke poput `spectral_centroid` i
   `spectral_bandwidth` ili parovi MFCC-a nisu nezavisne
2. Razdvaja dva razlicita cilja: dobra procjena vjerojatnosti (kalibracija) vs dobra
   klasifikacijska odluka - NB moze biti los u prvom, a dobar u drugom
3. Daje teorijski okvir za interpretaciju zasto je NB "naivan" u imenu, ali ne nuzno naivan
   u praksi

## Glavni zakljucci
1. Tocnost klasifikacije != tocnost procijenjenih vjerojatnosti
2. Krsenje pretpostavke nezavisnosti ne mora smanjiti tocnost ako ne mijenja koji razred ima
   najvecu posteriornu vjerojatnost
3. NB moze biti optimalan (u smislu 0-1 gubitka) i na domenama gdje pretpostavka ocito ne
   vrijedi
4. Slozeniji modeli (koji ispravno modeliraju zavisnosti) i dalje mogu pobijediti kad je
   uzorak dovoljno velik da opravda dodatne parametre (usporedivo s QDA vs Naive Bayes
   raspravom u nasem notebooku)
