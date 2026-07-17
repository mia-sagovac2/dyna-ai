# IZVOR 5 - istrazivanje gradient boostinga

### Naslov: 
Greedy Function Approximation: A Gradient Boosting Machine (1999)

### Autor:
Jerome H. Friedman

### Link:
https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faos%2F1013203451

## Opis
Ovaj rad uvodi gradient boosting - jedan od najvaznijih algoritama u modernom machine learningu (na kojem se bazira i XGBoost)

Glavna ideja je da se model gradi postepeno (iterativno)i svaki model pokusava ispraviti pogreske prethodnih, za razliku
od bagginga (Breiman) ovdje modeli nisu nezavisni vec zavise jedan od drugog.

Algoritam:
1. Pocnemo sa jednsotavnim modelom
2. Izracunamo gresku (loss)
3. Izracunamo gradijent te pogreske
4. Naucimo novi model da aproksimira taj gradijent
5. Dodamo ga postojecem modelu

To je zapravo optimizacija funkcije pogreske koristeci gradijent descent u funkcijskom prostoru

### Formula
$F_m(x) = F_{m-1}(x) + \gamma_m \, h_m(x)$

$F_m(x)$ - model u iteraciji u m
</br>
$h_m(x)$ - novi "slabi ucenik" (najcesce stablo)
</br>
$\gamma_m $ - tezina (learning rate)


### Sto znaci greedy u naslovu?
Greedy ili pohlepno znaci da se u svakoj iteraciji bira najbolji moguci korak lokalno i ne gleda se 
globalni optimum odmah, to cini algoritam efikasnim ali sklonim prenaucavanju

### Zasto su decision trees vazni?
Friedman pokazuje da su mala stabla (weak learners) idealna baza i jer mogu modelirati nelinearnosti i interakcije

### Razlika u odnosu na Bagging (Breiman)
**Bagging** - modeli nezavisni, smanjuje varijancu, paralleno treniranje i koristi bootstrap

**Boosting** - modeli zavise jedan od drugog, smanjuje bias, sekvencijalno, koriste gradijent

## Zasto je bitno za nas?
- temelj za XGBoost, CatBoost i LightGBM