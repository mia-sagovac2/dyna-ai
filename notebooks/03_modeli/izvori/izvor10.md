# IZVOR 10 - matematicki temelj umjetnog neurona (prije perceptrona)

### Naslov:
A Logical Calculus of the Ideas Immanent in Nervous Activity (1943)

### Autor:
Warren McCulloch, Walter Pitts

### Link:
https://link.springer.com/content/pdf/10.1007/BF02478259.pdf

## Opis
Prvi formalni matematicki model umjetnog neurona - prije Rosenblattovog perceptrona. McCulloch
i Pitts su pokazali da se bioloski neuron moze pojednostaviti u logicku jedinicu: prima vise
binarnih ulaza, zbraja ih (tezinski) i ako suma prijedje odredjeni prag, "puca" (izlaz = 1),
inace ne (izlaz = 0). Pokazali su da mreza ovakvih jedinica moze izracunati bilo koju logicku
funkciju (AND, OR, NOT...), sto ih direktno povezuje s teorijom automata i digitalnim krugovima.

## Zasto je bitno za nas?
1. Ovo je "nulta tocka" cijelog polja - prije ovoga ne postoji formalni model neurona
2. Direktno motivira Rosenblattov perceptron (izvor11) - on samo doda mogucnost ucenja tezina
3. Uvodi ideju praga (threshold) i tezinske sume koja se provlaci kroz sve kasnije arhitekture

## Glavni zakljucci
1. Neuron = tezinska suma ulaza + prag (threshold aktivacija)
2. Mreza jednostavnih jedinica moze simulirati proizvoljno slozenu logiku
3. Model je staticki - nema mehanizma ucenja tezina, to dolazi tek s perceptronom
