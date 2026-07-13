# IZVOR 6 - Najbitnije značajke zvuka za ML model

### Naslov: 
Environmental Sound Recognition With Time–Frequency Audio Features

### Autor:
Selina Chu et al.

### Link:
https://sail.usc.edu/publications/files/selinachu-taslp2009.pdf

## Opis
Rad istražuje problem prepoznavanja okolišnih zvukova (environmental sound recognition), odnosno kako iz audio signala izvući značajke koje omogućuju računalu da prepozna različite vrste zvukova iz stvarnog svijeta (npr. kiša, insekti, promet, životinje, različiti ambijentalni zvukovi).

Autori naglašavaju da se većina tadašnjih sustava oslanjala na MFCC (Mel-frequency cepstral coefficients) značajke, koje dobro opisuju spektralni oblik zvuka, ali nisu uvijek dovoljne za okolišne zvukove. Razlog je što mnogi takvi zvukovi nemaju jasnu harmonijsku strukturu kao govor ili glazba, nego sadrže važne vremenske obrasce (npr. ritam kiše, ponavljanje zvuka kukaca, promjene intenziteta).

Zbog toga autori predlažu kombinaciju:
- MFCC značajki: opisuju frekvencijski/spektralni sadržaj
- Time-frequency značajki dobivenih pomoću Matching Pursuit (MP) algoritma: hvataju lokalne promjene u vremenu i frekvenciji

Cilj rada je pokazati da kombiniranje različitih tipova značajki daje bolju klasifikaciju zvukova nego korištenje samo jedne grupe značajki

## Zašto je bitan za nas?
Ovaj rad je vazan jer pokazuje da ne psotoji jedna najbolja značajka za sve vrste zvukova.

Glavna poruka bitna za nas: Dobar ML model ne ovisi samo o algoritmu klasifikacije, nego prvenstveno o tome koliko dobro značajke predstavljaju informaciju sadrzanu u zvuku.

Autori pokazuju da:
- MFCC same po sebi mogu izgubiti informacije o vremenskoj dinamici zvuka
- vremenske značajke mogu biti jednako važne kao frekvencijske
- kombinacija više reprezentacija zvuka daje bolje rezultate

Za naš slučaj to znači da kod ekstrakcije značajki ne bismo trebali gledati samo što postoji u frekvenciji (spectral domain) nego i kako se zvuk mijenja kroz vrijeme (temporal domain).

### Najvažnije kategorije koje ovaj rad sugerira
**Spektralne značajke (boja zvuka):**
- MFCC - najcesce koristena audio znacajka, distribucija energije kroz frekvencije
- Spectral centorid - pokazuje gdje je centar energije spektra, povezan sa percepcijom svjetline zvuka
- Spectral bandwidth - koliko je sirok frekvencijski raspon tj. prosjecno odstupanje energije od centroida
- Spectral roll-off - frekvencija ispod koje se nalazi odredjeni postotak energije (obicno 95%)

**Vremenske značajke (poseban naglasak):**
- Zero Crossing Rate (ZCR) - koliko cesto signal mijenja predznak, razlikuje šumne i tonalne zvukove
- Energy (RMS amplituda) - jacina zvuka kroz vrijeme
- Temporal envelope - oblik promjene energije
- Attack/Decay karakteristike - koliko brzo zvuk počinje i završava

**Time-frequency značajke:**
- Zvuk nije samo "koje frekvencije postoje" nego "koje frekvencije postoje u kojem trenutku"
- autori koriste Matching Pursuit (MP) koji pronalazi vazne lokalne strukture u vremensko-frekvencijskom prostoru
- za moderne ML sustave to bi bilo: spektogram kao ulaz CNN modelu, mel-spektogram, log-mel spektogram

## Glavni zaključci iz rada

### 1. MFCC nisu dovoljne za sve vrste zvuka
- MFCC dobro opisuju spektar, ali okolisni zvukovi cesto imaju vaznu vremensku strukturu

### 2. Kombinacija frekvencijskih i vremenskih značajki daje bolje rezultate
- autori pokazuju da dodavanje MP time-frequency značajki uz MFCC povećava preciznost klasifikacije

### 3. Najbolje značajke su one koje imaju fizičko značenje
- umjesto velikog broja značajki bolje je koristiti značajke koje odgovaraju načinu na koji ljudi percipiraju zvuk, dakle  frekvencija, energija, promjena kroz vrijeme, struktura signala

### 4. Za ML klasifikaciju zvuka potrebno je kombinirati više pogleda na isti signal
- Audio = Frequency + Time + Time-Frequency
