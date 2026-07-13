# IZVOR 1 - istraživanje značajke pitch-a (frekvencija?)

### Naslov: 
Pitch perception

### Autor:
William A. Yost

### Link:
https://www.researchgate.net/publication/40026915_Pitch_Perception

## Opis
Rad je pregled koji objašnjava kako ljudi percipiraju pitch (visinu tona) i kako je povezan s fizičkim signalom
Ključna ideja je da pitch nije isto što i frekvencija nego perceptualna interpretacija signala.
</br>
Povezan je sa periodičnošću signala i često s fundamentalnom frekvencijom.
</br>
</br>
Dvije glavne ideje su:
- Spectral teorija - pitch dolazi iz raspodjele frekvencijai koristi harmonike i spektar
- Temporal teorija - pitch dolazi iz periodičnosti u vremenu, koristi autocorrelation i ponavljanje signala

</br>

Kako radi sluh?
1. signal ulazi u uho
2. cochlea radi frekvencijsku analizu (filter bank)
3. mozak koristi vrijeme (period) i strukturu harmonika kako bi raspoznao zvuk


## Zašto je bitno za nas?
1. Objašnjava zašto FFT nije dovoljan - pitch nije samo spektar nego i vremenska struktura, treba nam spektogram uz FFT
2. Temporal informacije su bitne - pitch dolazi iz periodičnosti
3. Imamo direktnu implikaciju za ML feature - trebamo koristiti autokorelaciju, temporal envelope, periodične feature, spektograme

## Glavni zaključci
1. Pitch != frekvencija
2. Periodicnost je najvazniji faktor pitcha
3. Temporal modeli su dominantni odnosno bolje objašnjavaju stvarnost
