# IZVOR 21 - AlexNet, rad koji je pokrenuo "deep learning boom"

### Naslov:
ImageNet Classification with Deep Convolutional Neural Networks (2012)

### Autor:
Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton

### Link:
https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

## Opis
Rad koji pokazuje da duboka konvolucijska mreza (AlexNet), trenirana na velikom skupu oznacenih
slika (ImageNet) uz ReLU aktivacije, GPU treniranje i dropout, drasticno nadmasuje sve dotadasnje
pristupe klasifikaciji slika. Ovo je prakticna potvrda ideja iz LeNet-5 (izvor20) i teorije o
dubini (izvor18/19) na velikoj skali, i "iskra" koja je pokrenula siroko usvajanje dubokog ucenja
u industriji nakon 2012.

## Zasto je bitno za nas?
1. Pokazuje zasto se CNN arhitektura (konvolucija + pooling + gusti slojevi na kraju) postala
   standard za sve zadatke nad slikovnim ulazom, ukljucujuci spektrograme zvuka u nasem projektu
2. Kombinacija ReLU aktivacija (rjesava vanishing gradient iz izvor19) i dubine je upravo ono sto
   demonstriramo u sintetickom primjeru DNN sekcije (sekcija 3)
3. Popularizirala je danas standardnu CNN "recepturu" (konvolucija -> ReLU -> pooling, ponovljeno,
   pa gusti slojevi na kraju) koju koristimo u sintetickom primjeru CNN sekcije

## Glavni zakljucci
1. Dubina + konvolucija + ReLU + dovoljno podataka daje bitno bolju tocnost od plitkih pristupa
2. GPU treniranje omogucilo je prakticno treniranje puno dubljih mreza nego ranije
3. Regularizacijske tehnike (dropout, augmentacija podataka) potrebne su da duboka mreza ne
   pretrenira na ogranicenom skupu podataka
