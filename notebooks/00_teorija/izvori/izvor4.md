# IZVOR 4 - popis značajki zvuka

### Naslov: 
A large set of audio features for sound description (similarity and classification) in the CUIDADO project

### Autor:
Geoffroy Peeters

### Link:
http://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf

## Opis
Ovaj rad je jedan od temeljnih pregleda audio značajki koji sustavno prikuplja i organizira veliki broj različitih feature-a za opis zvuka.
</br>
Cilj rada je:
- pronaći način kako opisati zvuk numerički
- omogućiti usporedbu, klasifikaciju i pretraživanje zvuka pomoću ML-a

## Zašto je bitan za nas?
Ovo je svojevrsni most izmedju prve sekcije (osnovne znacajke zvuka) i druge sekcije (prevodenje njih u ML jezik)
</br>
</br>
Osnovne značajke:
- frekvencija
- amplituda
- boja zvuka
- envelope

Peeters pokazuje kako te stvari pretvoriti u konkretne feature koje ce model razumijeti...

## Glavni zaključci iz rada

### 1. Ne postoji jedan "feature" nego veliki skup različitih
- zvuk je kompleksan, treba više različitih opisa
- rad pokazuje preko 70 audio značajki koje se koriste za različite zadatke

### 2. Featurei se mogu organizirati u grupe
Glavne grupe:
- **Temporal (vremenski):** kako se zvuk mijenja kroz vrijeme (ADSR, trajanje, onset)
- **Spectral (frekvencijski):** kako je energija rasporedjena po frekvencijama, dobiva se preko FFT-a (spectral centroid, spectral bandwidth, spectral rolloff...)
- **Energetski:** koliko je zvuk jak (RMS energy, loudness), povezano sa amplitudom
- **Harmonic (tonal):** struktura harmonika (pitch, hermonicity, inharmonicity), povezano sa timbre i frekvencijom
- **Perceptual (kako ljudi čuju zvuk):** model ljudskog zvuka (Mel skala), npr. MFCC

### 3. Featurei dolaze iz različitih reprezentacija zvuka
Rad naglašava da se featurei ne računaju iz sirovog signala nego iz različitih reprezentacija/transformacija:
- vreemski signal
- spektar (FFT)
- spektogram

### 4. Isti featurei mogu se koristiti u više domena
- feturei nisu vezani uz jednu primjenu
- npr. MFCC se koristi i za glazbu i za enviromental sound
- oni su generalni opis zvuka, ne samo specificni za govor ili glazbu

### 5. Feturei opisuju percepciju, ne samo fiziku
- velik dio rada povezuje fizikalna svojstva zvuka sa ljudskom percepcijom
- npr. loudness nije samo aplituda, pitch nije samo frekvencija, timbre je kompleksna kombinacija spktralnih svojstava

## Summa Summarum
Zvuk (fizika) -->
Reprezentacija (FFT, spektrogram) -->
Feature-i (MFCC, RMS, spectral features) -->
ML model