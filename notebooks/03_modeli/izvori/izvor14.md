# IZVOR 14 - jedan od prvih opisa rekurentnih neuronskih mreza

### Naslov:
Finding Structure in Time (1990)

### Autor:
Jeffrey L. Elman

### Link:
https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1

## Opis
Elman uvodi jednostavnu rekurentnu arhitekturu (danas poznatu kao "Elman network" ili Simple
RNN) - mreza ima "context layer" koji pamti skriveno stanje iz prethodnog vremenskog koraka i
vraca ga natrag kao dodatni ulaz u sljedecem koraku. Time mreza dobiva neku vrstu memorije i
moze obradjivati sekvence promjenjive duljine (npr. recenice, vremenske nizove), umjesto samo
fiksne ulaze kao obicni MLP (izvor12).

## Zasto je bitno za nas?
1. Zvuk (i nas audio signal) je sekvenca kroz vrijeme - obican MLP tretira svaki uzorak
   nezavisno i gubi informaciju o redoslijedu, RNN to rjesava
2. Uvodi koncept skrivenog stanja (hidden state) koje se prenosi kroz vrijeme - temelj za
   LSTM/GRU (izvor15) koji rjesavaju probleme koje ovaj jednostavni model ima
3. Direktno motivira zasto bismo za vremenske serije (audio, senzori) razmisljali drugacije
   nego za staticke znacajke (feature vektor koji koristimo kod ostalih modela u ovoj mapi)

## Glavni zakljucci
1. Rekurzija (feedback skrivenog stanja) omogucava mrezi da "pamti" kontekst kroz vrijeme
2. Jednostavni RNN ima problem s dugim sekvencama (vanishing gradient) - vidi izvor15
3. Uveo temelj za sve kasnije sekvencijalne arhitekture (LSTM, GRU, pa i transformer)
