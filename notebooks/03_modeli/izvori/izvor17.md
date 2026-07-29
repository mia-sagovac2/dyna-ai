# IZVOR 17 - koristenje autoencodera za detekciju anomalija (nas binarni klasifikator)

### Naslov:
Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction (2014)

### Autor:
Mayu Sakurada, Takehisa Yairi

### Link:
https://www.researchgate.net/publication/288492549_Anomaly_Detection_Using_Autoencoders_with_Nonlinear_Dimensionality_Reduction

## Opis
Rad direktno primjenjuje ideju autoencodera (izvor16) na detekciju anomalija: autoencoder se
trenira **samo na "normalnim" podacima**. Kad mu se kasnije da anomalan uzorak (koji mreza
nikad nije vidjela u treningu), rekonstrukcija ce biti losa jer mreza nije naucila kako
rekonstruirati taj obrazac - greska rekonstrukcije (MSE izmedju ulaza i izlaza) sluzi kao
"anomaly score". Prag (threshold) na toj gresci pretvara autoencoder u binarni klasifikator:
ispod praga = normalno, iznad praga = anomalija.

## Zasto je bitno za nas?
1. Ovo je direktan template za nas finalni model: autoencoder treniran iskljucivo na
   `raw/background` (normalno/sum), a `raw/vehicle` tretiramo kao "anomaliju" u odnosu na sum -
   klasifikacija "je li sum ili nije" postaje pitanje "je li greska rekonstrukcije ispod praga"
2. Objasnjava zasto ne treniramo autoencoder na obje klase odjednom (to bi bio drugaciji,
   supervised pristup, blizi CNN klasifikatoru iz `scripts/legacy/Classify_Data_2.py`) - bit
   pristupa je da mreza nikad ne vidi "vozilo" primjere u treningu
3. Daje opravdanje za nacin biranja praga (threshold) - najcesce se koristi percentil greske
   rekonstrukcije na treniranom (normalnom) skupu, sto koristimo i mi

## Glavni zakljucci
1. Autoencoder treniran samo na "normalnoj" klasi prirodno lose rekonstruira sve sto odstupa
2. Greska rekonstrukcije (MSE) = anomaly score, threshold na njoj daje binarnu odluku
3. Pristup ne treba labelirane primjere anomalije u treningu (samo za evaluaciju/threshold)
