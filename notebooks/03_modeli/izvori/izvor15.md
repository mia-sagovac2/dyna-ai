# IZVOR 15 - LSTM, rjesenje za vanishing gradient problem kod RNN-a

### Naslov:
Long Short-Term Memory (1997)

### Autor:
Sepp Hochreiter, Jurgen Schmidhuber

### Link:
https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory

## Opis
Obicni RNN (izvor14) ima problem: kad se gradijent propagira unatrag kroz mnogo vremenskih
koraka, on eksponencijalno iscezava (vanishing gradient) ili eksplodira, pa mreza prakticki ne
moze nauciti dugorocne ovisnosti (npr. povezati pocetak i kraj duge recenice/signala). LSTM
rjesava to uvodjenjem "cell state" - odvojene memorijske linije kroz koju informacija moze teci
gotovo nepromijenjena - i tri "gate" mehanizma (forget, input, output gate) koji kontroliraju
sto se pamti, sto se zaboravlja i sto se propusta na izlaz.

## Zasto je bitno za nas?
1. Objasnjava zasto se u praksi gotovo nikad ne koristi "goli" Simple RNN (izvor14) nego
   LSTM ili GRU - vanishing gradient je stvaran problem kod duljih audio sekvenci
2. Gate mehanizam (forget/input/output) je koncept koji se pojavljuje i u modernijim
   arhitekturama (npr. attention mehanizmi su konceptualni nasljednik ideje "selektivnog
   pamcenja")
3. Relevantno za nasu odluku o arhitekturi - u ovoj biljeznici koristimo autoencoder na
   feature-vektorima (ne sirovom signalu), pa LSTM ne koristimo direktno, ali je vazno znati
   zasto bi bio izbor da smo modelirali sirovi audio kao sekvencu

## Glavni zakljucci
1. Cell state + gate mehanizmi rjesavaju vanishing/exploding gradient problem kod RNN-a
2. LSTM moze nauciti ovisnosti kroz stotine/tisuce vremenskih koraka, obicni RNN ne moze
3. Standard za sekvencijalne probleme prije pojave transformera i self-attentiona
