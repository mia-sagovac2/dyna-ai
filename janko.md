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

Aplituda vs time - prikazuje overall shape signala, peakove i strukturu tisine

RMS - root mean square amplitude
RMS vs time - short-term glasnoca/energija, kako jacina evoluira kroz vrijeme

ZCR - zero crossing rate
ZCR vs time - mjeri koliko cesto val prelazi nulu, veci ZCR obicno znaci vise suma

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

Mel vs time - boja signalizira energiju u svakom mel bandu, koje su frekvencijske zone aktivne kada
MFCC vs time - svaki redak je jedan MFCC coeff over time, MFCC mjeri svojevrsnu boju zvuka

* Clean: dominantne su nize frekvencije, zuto na grafu, vise je jasnih zutih linija
* Noisy: rasprsena je energija, ima dosta suma razlicitih frekvencija

Zaključak:
malo jest cudno jer se na bicik+sum doima kao da smo izgubili dio podataka...


**FFT analiza**

Fourier spektar za ta dva signala. tamo di ja visi, taj signal ima vise energije

Razlika:
zanimljivo je to sto je cisti bicikl trebao biti samo izvedenica iz ovog drugog, tj. bic + sum
bi u pravilu uvijek trebo biti veci nego sami bicikl, jelda? a ovdje, na kraju ovog grafa imamo 
jedan veliki peak gdje se prikazuje da ocito bicikl ima vise energije...


**Feature analiza**

ovo su dosl sveee znacajke sta mi je klaudija mogla izvuc van:
* MFCC mean: Prosječne Mel-frekventne kepstralne koeficijente , opisuju boju/tembr zvuka, oblik spektralnog omotača
* MFCC std: Standardna devijacija MFCC-a, mjeri vremensku varijabilnost tembra
* Spectral centroid: "Težište" spektra, visoke vrijednosti = zvuk bogatiji visokim frekvencijama (svjetliji zvuk)
* Spectral bandwidth: Širina spektra oko centroida, koliko su frekvencije raspršene
* Spectral rolloff: Frekvencija ispod koje se nalazi 85% ukupne energije spektra
* ZCR mean: Zero Crossing Rate, koliko puta signal prelazi nulu u sekundi; visoko = šum/perkusivni zvuk
* RMS mean: Root Mean Square energija — glasnoća / jačina signala
* Spectral flatness: Koliko je spektar "ravan" (bijeli šum = 1, čisti ton = 0)
* Spectral contrast: Razlika između spektralnih vrhova i dolina u pojasima — tekstura zvuka
* Onset strength: Jačina naglih promjena energije — udarci, ataci, ritmički eventi
* Onset std: Varijabilnost tih naglih promjena kroz vrijeme
* Chroma mean: Raspodjela energije po 12 tonskih klasa (C, C#, D...) — harmonijski sadržaj



Zaključak:
Ociti najbolji kandidat je RMS mean, eh sad drugi su li la, jos mozda uzet ove: Spectral flatness,
ZCR mean, MFCC mean, i mozebitno ove iznad 50%: Spectral rolloff, Onset std, Spectral centroid


**PDF / CDF**

PDF - probability density fuction, koliko je cesta svaka vrijednost, ako se dvije krivulje jako razlikuju 
onda im se i ta znacajka jako razlikuju 

CDF - cumulative distribution function, korisno za usporedjivanju medijana ili percentila, pokazuje samo 
frakciju nekog framea

Napomena: ovdje nemamo sve iste znacajke kao u prethodnom grafu, ali vecinu imamo

Zaključak:
RMS ispada opet najbolja znacajka za razlikovanje dva zvuka, Spectral rolloff, Spectral centroid,
Spectral bandwidth, MFCC i ZCR takodjer

**Finalni zakljucak**
RMS se u svim analizama pokazao kao najbolja znacajka za prepoznavanje, ostali su tako tako, ali
dalo bi se izvuci 4-5 sigurnih znacajki
Cudno mi je ovo brisanje dijelova zvuka i to sto nekada cisti bicikl predvodi u FFT-u. Molim te ako mozes 
to protumaciti bila bih ti jako zahvalna :).
Ovi konkretni zvukovi imaju dosta cudnoga suma (tipa pjesma), taj sum nije cisti white noise,
ako cemo ici klasificirati sva vozila trebamo vidjeti koliko sve tipova sumova imamo, ne znam koliko
je korisno imati ovoliko zasicene tj. unclear zvukove, pogotovo kada oni cini vecinu dataseta...

---

### Solo vs dataset analiza

U analizi se uspoređuju:

* jednog **clean signal-a**
* **cijelog dataset-a (više noisy uzoraka)** - ovdje su samo svi zvukovi spojeni u jedan, 
mozda bi bilo ispravnije uzeti prosjecan (mogu to napraviti u sljedocoj iteraciji, ako
zelis isprobati samo otkomentiraj onu liniju u kodu gdje opsiujem drugi nacin usporedivanja
linija 81-85), nisu nimalo slicni rezultati...


**Waveform**

Aplituda vs time - prikazuje overall shape signala, peakove i strukturu tisine

RMS - root mean square amplitude
RMS vs time - short-term glasnoca/energija, kako jacina evoluira kroz vrijeme

ZCR - zero crossing rate
ZCR vs time - mjeri koliko cesto val prelazi nulu, veci ZCR obicno znaci vise suma

* One signal:
    - manje spieakova, tisi signal, to ima smisla posto je samo jedan
    - manji RMS, opet ima smisla jer je samo jedan zvuk, u cijelom datasetu ima puno vise zvukova sa puno vecom energijom
    - nizi je, to ima smisla, cijeli dataset naravno da ima vise sumova
* All signal:
    - deblji signal, ima dosta šuma (logično)
    - RMS veci, veca energija, naravno kada imamo veci uzorak zvukova
    - ZCR isto ko rms, ima smisla cisto zvog kolicine zvuka koji je ukljucen
Zaključak:
sve odgovara :)


**Spektrogram**

Mel vs time - boja signalizira energiju u svakom mel bandu, koje su frekvencijske zone aktivne kada
MFCC vs time - svaki redak je jedan MFCC coeff over time, MFCC mjeri svojevrsnu boju zvuka

* Clean: bogat, ravnomjeran signla kroz sve frekvencije
* Noisy: rpretezno niskofrekventni signal s puno tisine

Zaključak:
MFCC1 i MFCC2 dosta slicni, samo sto all dataset ima malo vise varijabilnost, MFCC 3-13 oboje su dosta ravnomjerno rasporedjeni, najbolje kroistiti 1 i 2 koji se relativno zestoko razilikuju


**FFT analiza**

Fourier spektar za ta dva signala. tamo di ja visi, taj signal ima vise energije

Razlika:
ovo je dosta slicno kao i na prethodnoj analizi, na istom mjestu cisti bicikl ima vecu energiju, ovo nam ne govori puno jer ustvari ne znamo koji su zvukovi na tom dijelu gdje cisti bicikl uzma prednost...


**Feature analiza**

ovo su dosl sveee znacajke sta mi je klaudija mogla izvuc van:
* MFCC mean: Prosječne Mel-frekventne kepstralne koeficijente , opisuju boju/tembr zvuka, oblik spektralnog omotača
* MFCC std: Standardna devijacija MFCC-a, mjeri vremensku varijabilnost tembra
* Spectral centroid: "Težište" spektra, visoke vrijednosti = zvuk bogatiji visokim frekvencijama (svjetliji zvuk)
* Spectral bandwidth: Širina spektra oko centroida, koliko su frekvencije raspršene
* Spectral rolloff: Frekvencija ispod koje se nalazi 85% ukupne energije spektra
* ZCR mean: Zero Crossing Rate, koliko puta signal prelazi nulu u sekundi; visoko = šum/perkusivni zvuk
* RMS mean: Root Mean Square energija — glasnoća / jačina signala
* Spectral flatness: Koliko je spektar "ravan" (bijeli šum = 1, čisti ton = 0)
* Spectral contrast: Razlika između spektralnih vrhova i dolina u pojasima — tekstura zvuka
* Onset strength: Jačina naglih promjena energije — udarci, ataci, ritmički eventi
* Onset std: Varijabilnost tih naglih promjena kroz vrijeme
* Chroma mean: Raspodjela energije po 12 tonskih klasa (C, C#, D...) — harmonijski sadržaj



Zaključak:
Opet RMS mean, onda ZCR mean, spectral flatness, spectral centroid, mfcc mean i eventualno onset std, spctral rolloff... 


**PDF / CDF**

PDF - probability density fuction, koliko je cesta svaka vrijednost, ako se dvije krivulje jako razlikuju 
onda im se i ta znacajka jako razlikuju 

CDF - cumulative distribution function, korisno za usporedjivanju medijana ili percentila, pokazuje samo 
frakciju nekog framea

Napomena: ovdje nemamo sve iste znacajke kao u prethodnom grafu, ali vecinu imamo

Zaključak:
RMS nam oept ispada najbolji, ZCR isto nije los, spectral flatness, MFCC 1 i 3 mi izgledaju obecavajuce...

---

**Zasto nisam opisivala jos ova preostala dva grafa?**
Ostala su nam jso dva grafa - mfcc usporedba i frekvencijske zone, mfcc mislim da se dovoljno dobro vidi iz grafa o spektogramima i ostalim analizama, a frekvencijske zone nam samo govore u kojim frekevencijama zvukovi dominiraju, to mi se ne cini od presudne vaznosti, ali ostvila sam grafove neka budu tu, akoce ih naknadno trebati komentirati. Frekvencije bibikla cu doduse prokomentirati u donjem zakljucku kako bise dalo zakljuciti otp gjde je bicikl

---

## GLAVNI ZAKLJUČCI

* Bicikl dominira u: visim frekvencijama (mid, high-mid i high)

* Šum dominira u: nizim frekvencijama u oba slucaju usporedbe (sub bass i bass)

* Najkorisniji feature-i:

  * RMS
  * Spectral flatness
  * ZCR mean
  * MFCC mean

* Glavne razlike između clean i noisy:

  * Frekvencijska polja
  * boja zvuka (mfcc)
  * jacina zvuka (rms)

## Komentari za kraj:
1. Bi li bilo korisnije gledati prosjecni zvuk a ne cijelid dataset bicikala spojen u jedan zvuk?
2. Jesu li znacajke koje sam uzela kao najkorisnije stvarno dobre?
3. Kako protumaciti spektograme i je li moje tumacenje na dobrom tragu? (gubi li se stvarno neki zvuk u slucaju suma, kao da se zvuk stisa tj. gubi)
4. Kako napraviti da se raspoznavaju sumovi svih oblika (kao sto ovdje vidimo zvuk je pjesma), a ne samo bijeli sum?
5. Koji su sljedeci koraci?

