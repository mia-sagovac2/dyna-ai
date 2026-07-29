# IZVOR 16 - autoencoder, ucenje kompresije podataka bez oznaka

### Naslov:
Reducing the Dimensionality of Data with Neural Networks (2006)

### Autor:
Geoffrey Hinton, Ruslan Salakhutdinov

### Link:
https://www.cs.toronto.edu/~hinton/absps/science.pdf

## Opis
Kljucni rad koji pokazuje da se duboka neuronska mreza moze trenirati da sama sebe reproducira
(autoencoder): ulaz se propusta kroz "encoder" koji ga komprimira u nisko-dimenzionalni
"bottleneck" (latentni prostor), a zatim "decoder" pokusava iz te komprimirane reprezentacije
rekonstruirati original. Mreza se trenira da minimizira razliku izmedju ulaza i rekonstrukcije
(reconstruction loss), bez potrebe za oznakama (unsupervised learning). Autori pokazuju da ovako
naucena reprezentacija bolje cuva strukturu podataka nego klasicni PCA.

## Zasto je bitno za nas?
1. Ovo je temeljni koncept iza naseg finalnog modela - autoencoder koji uci "izgled" pozadinske
   buke (background/noise klasa) i onda koristi gresku rekonstrukcije za detekciju odstupanja
2. Kljucna ideja: mreza koja dobro rekonstruira samo ono na cemu je trenirana - ako naucimo
   autoencoder samo na sumu (noise), on ce lose rekonstruirati zvuk vozila (jer nikad nije vidio
   takav uzorak), sto nam daje signal za klasifikaciju bez potrebe da "vozilo" bude labelirana
   klasa u treningu
3. Bottleneck sloj prisiljava mrezu da nauci samo najbitnije znacajke ulaza - konceptualno slicno
   PCA/dimenzionalnoj redukciji koju smo vec spominjali kod ostalih modela u ovoj mapi

## Glavni zakljucci
1. Autoencoder = encoder (kompresija) + bottleneck (latentni prostor) + decoder (rekonstrukcija)
2. Trenira se unsupervised - cilj je izlaz = ulaz, ne neka vanjska labela
3. Naucena reprezentacija u bottlenecku moze biti korisnija od PCA jer moze uhvatiti nelinearne
   odnose u podacima
