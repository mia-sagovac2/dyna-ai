# Evaluacijske metrike za klasifikacijske modele

## 1. Uvod

Evaluacija modela strojnog učenja ključan je korak u razvoju sustava za klasifikaciju, jer omogućuje kvantitativnu procjenu sposobnosti modela da generalizira na neviđene podatke. U kontekstu ovog rada, gdje se razmatra problem višeklasne klasifikacije zvukova vozila (automobili, teška vozila, bicikli i motocikli), odabir odgovarajućih metrika posebno je važan zbog neuravnoteženosti skupa podataka i različite složenosti pojedinih klasa.

Različite metrike mjere različite aspekte performansi modela, poput točnosti klasifikacije, sposobnosti razlikovanja klasa i kvalitete procijenjenih vjerojatnosti. Stoga je potrebno koristiti kombinaciju metrika kako bi se dobila cjelovita slika performansi modela.

## 2. Accuracy

Točnost (engl. *accuracy*) definira se kao omjer ispravno klasificiranih uzoraka i ukupnog broja uzoraka:


$ Accuracy = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_i = \hat{y}_i) $


gdje je:

* $ y_i $ stvarna oznaka klase,
* $ \hat{y}_i $ predviđena oznaka,
* $ \mathbb{1} $ indikator funkcija.

Iako je accuracy intuitivna i jednostavna metrika, ona može biti varljiva u slučaju neuravnoteženih skupova podataka. U takvim slučajevima model može postići visoku točnost favorizirajući dominantnu klasu, dok zanemaruje slabije zastupljene klase. Zbog toga accuracy nije dovoljna kao jedina metrika u ovom radu.


## 3. Precision, Recall i F1-score

Za detaljniju analizu performansi koriste se metrike temeljene na matrici zabune (*confusion matrix*).

### 3.1 Precision

$ Precision =  \frac{TP}{TP + FP} $


Precision mjeri udio točnih pozitivnih predikcija među svim predikcijama određene klase. Visoka precision znači da model rijetko pogrešno klasificira druge klase kao promatranu klasu.

### 3.2 Recall

$ Recall =  \frac{TP}{TP + FN} $

Recall mjeri sposobnost modela da pronađe sve stvarne primjere određene klase. Nizak recall ukazuje na to da model propušta velik broj stvarnih uzoraka.

### 3.3 F1-score


$ F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} $


F1-score predstavlja harmonijsku sredinu precision i recall metrike te omogućuje balansiranu procjenu modela.


### 3.4 Macro i Weighted prosjeci

U višeklasnim problemima metrike se agregiraju:

* **Macro prosjek**: računa se kao aritmetička sredina metrike po klasama

    $  F1_{macro} = \frac{1}{K} \sum_{k=1}^{K} F1_k $

* **Weighted prosjek**: ponderiran brojem uzoraka po klasi

Macro F1-score daje jednaku važnost svim klasama i posebno je prikladan za neuravnotežene skupove podataka, što je slučaj u ovom radu.


## 4. Confusion Matrix

Matrica zabune (*confusion matrix*) predstavlja tablični prikaz stvarnih i predviđenih klasa te omogućuje detaljnu analizu pogrešaka modela. Svaki element matrice prikazuje broj uzoraka koji pripadaju određenoj stvarnoj klasi, a klasificirani su kao neka druga klasa.

U ovom radu matrica zabune omogućuje identifikaciju specifičnih pogrešaka, primjerice zamjene između akustički sličnih klasa poput motora i bicikala ili auta i teških vozila.


## 5. Log Loss (Cross-Entropy Loss)

Logaritamski gubitak (*log loss*) definiran je kao:

$ LogLoss = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(p_{i,k}) $


gdje je:

* $ y_{i,k} $ indikator pripadnosti klase,
* $ p_{i,k} $ predviđena vjerojatnost za klasu ( k ).

Log loss uzima u obzir ne samo točnost klasifikacije, već i sigurnost modela u svoje predikcije. Model koji daje visoku vjerojatnost pogrešnoj klasi bit će strogo penaliziran. Ova metrika je posebno važna kod modela koji generiraju probabilističke izlaze, poput logističke regresije i XGBoost-a.


## 6. MSE i MAE

### 6.1 Mean Squared Error (MSE)

$ MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $

### 6.2 Mean Absolute Error (MAE)

$ MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $


Iako su MSE i MAE standardne metrike u regresijskim problemima, njihova primjena u klasifikaciji nije prirodna jer pretpostavljaju numerički odnos između klasa. U ovom radu klase nemaju inherentan redoslijed, pa ove metrike nemaju jasnu interpretaciju i koriste se isključivo u eksperimentalne ili usporedne svrhe.


## 7. ROC-AUC

ROC krivulja (*Receiver Operating Characteristic*) prikazuje odnos između stope istinskih pozitivnih (TPR) i lažno pozitivnih (FPR) predikcija. Površina ispod krivulje (*AUC*) kvantificira sposobnost modela da razlikuje klase.

U višeklasnim problemima ROC-AUC se najčešće računa metodom *one-vs-rest*. Iako pruža korisne informacije o separabilnosti klasa, ova metrika može biti manje informativna u slučaju neuravnoteženih skupova podataka.


## 8. Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation nije evaluacijska metrika, već metoda za procjenu parametara modela. Temelji se na maksimizaciji vjerojatnosti opaženih podataka:

$ \hat{\theta} = \arg\max_{\theta} P(D \mid \theta) $


U klasifikacijskim modelima optimizacija MLE-a često vodi do minimizacije log loss funkcije, čime se uspostavlja direktna veza između metode treniranja i evaluacijske metrike.


## 9. Odabir metrika za ovaj rad

S obzirom na karakteristike problema (višeklasna klasifikacija i neuravnotežen skup podataka), kao glavna metrika koristi se:

* **Macro F1-score** – jer jednako tretira sve klase i daje realnu sliku performansi na manjinskim klasama.

Dodatno se koriste:

* **Confusion matrix** – za analizu pogrešaka,
* **Log loss** – za evaluaciju kvalitete predikcijskih vjerojatnosti,
* **Accuracy** – kao pomoćna metrika za opći pregled performansi.

Ovakav skup metrika omogućuje sveobuhvatnu evaluaciju modela, uzimajući u obzir i klasifikacijsku točnost i ponašanje modela na razini pojedinih klasa.
