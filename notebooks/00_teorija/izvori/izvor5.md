# IZVOR 5 - Fourier Transform (DFT & FFT)
### Naslov: 
The Scientist and Engineer's Guide to DSP (Fourier Transform chapters)

### Autor:
Smith, Steven W.

### Link:
https://img.anfulai.cn/bbs/97312/The%20Scientist%20and%20Engineer%20Guide%20to%20DSP.pdf

## Opis

### DFT 
- matematicka metoda koja pretvara signal iz vremenske domene u frekvencijsku domenu
- ulaz: signal kroz vrijeme
- izlaz: spektar frekvencija (koje frekvencije postoje i kolika im je amplituda)
- **Ideja:** svaki signal moze se rastaviti na sumu sinusnih i kosinusnih komponenti razlicitih frekvencija
- DFT zapravo:
    - uzima diskretni signal (digitalni)
    - razlaze ga na osnovne frekvencijske komponente
    - daje: amplitudu (koliko je jaka frekvencija) i fazu
- dobivamo spektralni prikaz signala

### FFT
- brzi algoritam za izracun DFT-a
- daje isti rezultat kao DFT samo je stotinama put brzi
- DFT: O(N^2)
- FFT: O(N logN)
- FFT razbija veliki problem na vise manjih DFT-ova i tako dramaticno ubrzava racunanje


## Zašto je ovo bitno za nas?

### DFT i FFT omogućuju:
- Analizu frekvencija - vidimo koje frekvencije u signalu
- Filtriranje - mozemo ukloniti nezeljene frekvencije
- Obradu signala - kompresija, detekcija, klasifikacija

### Primjena na naš problem:
- imamo audio snimku iz stvarnog prometa
- trebamo izvući čisti zvuk bicikla
- **Kako DFT/FFT pomaze?**:
    - pretvorba u frekvencijsku domenu
    - identifikacija šuma
    - filtriranje (za bicikl konkretno high pass filter)
    - povrat u vremensku domenu (rekontruira signal bez šuma)
- **primjer:** VIDJETI FFT_real_primjer.ipynb

### Glavni zaključci:
1. DFT je temelj DSP-a (Digital Signal proessing)
2. FFT čini DSP praktičnim
3. Frekvencijska domena je često "prirodnija"
4. Obrada signala = rad u dvije domene
5. DFT ima tri glavne primjene: analiza spektra, analiza sustava, ubrzavanje drugih algoritama
