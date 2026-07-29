# IZVOR 11 - originalni rad o perceptronu

### Naslov:
The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain (1958)

### Autor:
Frank Rosenblatt

### Link:
https://pdfs.semanticscholar.org/865f/b2cfe6fdb7af2c663ef346ea05889f237108.pdf

## Opis
Rosenblatt uvodi perceptron - prvi model umjetnog neurona koji **uci** tezine iz podataka
(za razliku od McCulloch-Pittsovog modela, izvor10, gdje su tezine fiksne/rucno postavljene).
Perceptron uzima vektor ulaza, mnozi ga s tezinama, zbraja i propusta kroz step funkciju
(aktivacija: 1 ako je suma > prag, inace 0). Tezine se azuriraju iterativno na temelju greske
izmedju predvidjenog i stvarnog izlaza (perceptron learning rule).

## Zasto je bitno za nas?
1. Prvi algoritam koji dokazano konvergira za linearno separabilne probleme (perceptron
   convergence theorem)
2. Uvodi ideju ucenja iz greske (error-driven update) - preteca gradient descenta i backpropa
3. Ogranicenje (ne moze rijesiti XOR - linearno neseparabilan problem) izravno motivira potrebu
   za visesojnim mrezama (izvor12)

## Glavni zakljucci
1. Perceptron = tezinska suma + step aktivacija + iterativno azuriranje tezina
2. Garantirano konvergira samo za linearno separabilne klase
3. Jedan neuron ne moze modelirati nelinearne granice odluke (npr. XOR)
