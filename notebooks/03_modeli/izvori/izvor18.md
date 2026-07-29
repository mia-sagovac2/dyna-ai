# IZVOR 18 - pregledni rad koji je popularizirao pojam "duboko ucenje"

### Naslov:
Deep Learning (2015)

### Autor:
Yann LeCun, Yoshua Bengio, Geoffrey Hinton

### Link:
https://www.nature.com/articles/nature14539

## Opis
Pregledni (review) rad u casopisu Nature koji je napisala "sveta trojka" dubokog ucenja. Objasnjava
zasto slaganje vise slojeva (dubina) omogucava mrezi da uci **hijerarhiju reprezentacija** - rani
slojevi uce jednostavne obrasce (npr. rubove), a dublji slojevi ih kombiniraju u sve apstraktnije
koncepte. Ova hijerarhijska kompozicija je kljucna razlika izmedju "duboke" mreze (DNN, vise
skrivenih slojeva) i "plitke" mreze (MLP s jednim skrivenim slojem iz sekcije 2).

## Zasto je bitno za nas?
1. Objasnjava zasto dodavanje dubine (a ne samo sirine) mrezi daje bolju reprezentacijsku
   ucinkovitost - isti problem moze zahtijevati eksponencijalno manje neurona ako je mreza dublja
   umjesto sira
2. Postavlja temelj za razumijevanje CNN-a i autoencodera (sekcije 4 i 6) kao specificnih dubokih
   arhitektura izgradjenih na istoj ideji hijerarhijskog ucenja znacajki
3. Uvodi kontekst za probleme koje dubina donosi (vanishing/exploding gradient), koje demonstriramo
   u sintetickom primjeru ove sekcije

## Glavni zakljucci
1. Dubina omogucava hijerarhijsko ucenje znacajki - svaki sloj gradi na apstrakcijama proslog
2. Duboke mreze su reprezentacijski ucinkovitije od plitkih, sirokih mreza za isti broj parametara
3. Vecina moderne primjene strojnog ucenja (govor, slika, tekst) oslanja se na duboke arhitekture
