# IZVOR 20 - rad koji je uveo konvolucijsku arhitekturu (LeNet-5)

### Naslov:
Gradient-Based Learning Applied to Document Recognition (1998)

### Autor:
Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner

### Link:
http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

## Opis
Temeljni rad koji uvodi konvolucijsku neuronsku mrezu (CNN) - LeNet-5, primijenjenu na
prepoznavanje rukom pisanih znamenki (MNIST). Umjesto da svaki neuron gleda cijelu sliku (kao gusto
povezani DNN iz sekcije 3), konvolucijski sloj koristi mali **filter (kernel)** koji "klizi" preko
slike i uci lokalne obrasce (rubove, kutove). Isti filter (iste tezine) se dijeli na svim
pozicijama slike - **dijeljenje parametara (weight sharing)** - sto mrezi daje **translacijsku
invarijantnost**: nauceni obrazac prepoznaje bez obzira gdje se u slici pojavljuje. Pooling slojevi
zatim smanjuju prostornu rezoluciju i grade robusnost na male pomake.

## Zasto je bitno za nas?
1. Objasnjava zasto CNN treba manje parametara od gusto povezane DNN mreze za obradu slika (jedan
   filter se ponovno koristi na svakoj poziciji, umjesto zasebne tezine za svaki piksel)
2. Direktno relevantno za nas projekt - `scripts/legacy/Classify_Data_2.py` koristi CNN nad
   spektrogramima (slikovni prikaz zvuka) za BG/Veh klasifikaciju, isti princip demonstriramo u
   sintetickom primjeru ove sekcije
3. Kljucno za usporedbu s autoencoderom (sekcija 6) - oba pristupa mogu raditi nad slikovnim/
   spektrogramskim ulazom, ali CNN klasifikator uci supervizirano na oba razreda, dok autoencoder
   uci samo iz jedne klase (bez oznaka)

## Glavni zakljucci
1. Konvolucija + dijeljenje parametara drasticno smanjuje broj tezina u odnosu na gustu mrezu
2. Translacijska invarijantnost - naucen obrazac prepoznaje se neovisno o poziciji u ulazu
3. Pooling smanjuje prostornu dimenziju i cuva samo najbitnije aktivacije, poboljsavajuci robusnost
