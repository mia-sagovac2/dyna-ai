# HELLO

## Struktura projekta
scripts/    -->     sve relevantne skripte
eda_analiza/    --> analiza cijelog skupa podataka (svih vozila, cijelog dataseta, to je ona stara analiza)
eda_analiza_solo_vs_solo_comparison/   --> eda analiza, usporeduje cisti bicikl i bicikl sa sumom
eda_analiza_solo_vs_dir_comparison/ --> eda analiza, usporedcuje cisti bicikl i cijeli dataset bicikala
fourier_analiza/    --> fourier za pojedini audio file (unijeti path filea na pocetak)
fourier_separated_audio --> konkretni odvojeni zvuk, bicikl + sum i cisti bicikl i spektogrami za oba
original_methods/   --> nebitno, dosle sa gitom s kojega sam skidala dataset, mozda bude korisno kasnije za treniranje modela

## Upute
1. Skini sa Sharepoint-a cijeli direktorij all_sorted
2. napravi mapu data/ i unutar nje stavi all_sorted/
3. aktiviraj .venv
4. instaliraj requirementse
5. Pokreci skripte i modificiraj path-ove ako je potrebno

## Interpretacija analiza

### Solo vs solo analiza
U analizi se uspoređuju:
* **A = čisti (ili bolje reći čišći) signal bicikla**
* **B = signal s prisutnim šumom**


**Waveform**

* Clean signal:
    - tanji signal, vidi se da nema puno šuma
    - RMS je manji, dakle manja energija, ima manje spikeova (tj. dogadaja)
    - ZCR je čudan...trebao biti biti nizi ovdje jer bi bicikl trebao biti vise tonalan, ali ovdje je suprotno
* Noisy signal:
    - deblji signal, ima dosta šuma (logično)
    - RMS veci, veca energija, manje spikeova (dogadaja)
    - ZCR opet cudan (to trebamo prokomnetirat)
Zaključak:
RMS i sami waveform odgovarajuci, ZCR cudan


**Spektrogram**

* Clean: dominantne su nize frekvencije, zuto na grafu, vise je jasnih zutih linija
* Noisy: rasprsena je energija, ima dosta suma razlicitih frekvencija

Zaključak:
malo jest cudno jer se na bicik+sum doima kao da smo izgubili dio podataka...


**FFT analiza**

* Clean: ...glavni pikovi...
* Noisy: ...širi spektar...

Razlika:
...upiši gdje je najveća razlika...


**Feature analiza**

* Spectral centroid: ...A vs B...
* Bandwidth: ...A vs B...
* ZCR: ...A vs B...

Zaključak:
...koji feature najbolje razlikuje signal i šum...


**PDF / CDF**

* Distribucija A: ...opis...
* Distribucija B: ...opis...

Zaključak:
...upiši...

---

### Solo vs dataset analiza

U analizi se uspoređuju:

* jednog **clean signal-a**
* **cijelog dataset-a (više noisy uzoraka)**


**Waveform**

* Dataset signal: ...opis...

Zaključak:
...upiši...


**Spektrogram**

* Stabilne frekvencije: ...upiši...
* Varijabilne/noise regije: ...upiši...

Zaključak:
...upiši...


**FFT analiza**

* Dominantne frekvencije dataset-a: ...upiši...
* Noise floor: ...upiši...

Zaključak:
...upiši...


**Feature analiza**

* Mean vrijednosti: ...upiši...
* Varijanca: ...upiši...

Zaključak:
...upiši...


**PDF / CDF**

* Distribucija dataset-a: ...opis...

Zaključak:
...upiši...

---

**Frekvencijske zone**

| Zona      | Opis           | Observacija |
| --------- | -------------- | ----------- |
| 20–250 Hz | niski tonovi   | ...upiši...     |
| 250–2k Hz | mehanički zvuk | ...upiši...     |
| 2k–20k Hz | šum            | ...upiši...     |

---

## GLAVNI ZAKLJUČCI

* Bicikl dominira u: **...upiši frekvencijski raspon...**

* Šum dominira u: **...upiši raspon...**

* Najkorisniji feature-i:

  * ...feature 1...
  * ...feature 2...

* Glavne razlike između clean i noisy:

  * ...upiši...


