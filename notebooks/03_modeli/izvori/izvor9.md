# IZVOR 9 - empirijska studija Naive Bayesa i utjecaj korelacije znacajki

### Naslov:
An Empirical Study of the Naive Bayes Classifier (2001)

### Autor:
Irina Rish

### Link:
https://www.researchgate.net/publication/228845263_An_Empirical_Study_of_the_Naive_Bayes_Classifier

## Opis
Empirijski rad koji sustavno testira Naive Bayes na sintetickim i stvarnim skupovima
podataka s razlicitim stupnjem korelacije izmedju znacajki, kako bi se utvrdilo kada
pretpostavka nezavisnosti najvise steti, a kada gotovo uopce ne utjece na performanse.

Glavni eksperimentalni nalaz: NB radi najbolje na dva suprotna ekstrema - kad su znacajke
gotovo potpuno nezavisne (pretpostavka tocna) ILI kad su ekstremno funkcionalno zavisne
(npr. gotovo duplicirane znacajke), a najgore u "srednjem" podrucju umjerene korelacije -
sto je zanimljivo jer to znaci da sama jacina korelacije nije direktan prediktor koliko ce
NB izgubiti u odnosu na model koji modelira zavisnosti.

## Zasto je bitno za nas?
1. Nas dataset ima bas to "srednje" podrucje - znacajke poput `spectral_bandwidth` i
   `spectral_rolloff` ili pojedini MFCC parovi su umjereno (ne ekstremno) korelirane, sto je
   prema ovom radu tocno scenarij gdje bi razlika NB vs QDA trebala biti najvidljivija
2. Daje konkretnu metodologiju (mjerenje korelacije + usporedba tocnosti) koju smo iskoristili
   u notebooku kad smo napravili korelacijsku matricu prije treniranja modela
3. Empirijski komplement teorijskom radu u izvor7 - pokazuje da teorijska moguca optimalnost
   ne znaci da ce NB uvijek biti blizu optimalnom u praksi

## Glavni zaključci
1. Stupanj korelacije znacajki nije linearno povezan s gubitkom performansi NB-a
2. Ekstremi (skoro nezavisno ili skoro potpuno zavisno) su "sigurniji" za NB nego umjerena
   korelacija
3. Empirijsko testiranje (kao u nasem notebooku - NB vs QDA na istom datasetu) je nuzno jer
   se stvarna steta ne moze pouzdano predvidjeti samo gledanjem korelacijske matrice
4. Dataset-specificna evaluacija bitnija je od opcenitih pravila o "koliko nezavisnosti je
   dovoljno"
