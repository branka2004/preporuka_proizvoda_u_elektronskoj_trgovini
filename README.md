# **Primena Deep Learning Modela za Preporuku Proizvoda u Elektronskoj Trgovini**
# 1. Opis problema

## Uvod

Savremene platforme elektronske trgovine svakodnevno obrađuju ogromne količine podataka o korisnicima, proizvodima i njihovim interakcijama. Zbog velikog broja dostupnih proizvoda, korisnicima je često teško da pronađu artikle koji najbolje odgovaraju njihovim interesovanjima i potrebama.

Iz tog razloga sistemi za preporuku proizvoda (Recommendation Systems) predstavljaju jednu od najvažnijih oblasti primene mašinskog i dubokog učenja u elektronskoj trgovini.

Cilj recommendation sistema jeste analiza prethodnog ponašanja korisnika kako bi se predvideli proizvodi koje bi korisnik mogao da kupi, oceni ili pregleda u budućnosti.

Takvi sistemi danas se koriste na platformama kao što su:

* Amazon
* Netflix
* YouTube
* Spotify
* eBay
* AliExpress

---

## Problem koji se rešava

Osnovni problem u recommendation sistemima predstavlja predviđanje nedostajućih interakcija između korisnika i proizvoda.

U okviru ovog projekta analiziran je problem preporuke proizvoda u elektronskoj trgovini korišćenjem tehnika dubokog učenja i autoenkoderskih neuronskih mreža.

Na osnovu istorije ocenjivanja proizvoda od strane korisnika, cilj modela jeste da nauči obrasce korisničkog ponašanja i rekonstruiše user-item matricu, odnosno da predvidi koje bi proizvode korisnik mogao pozitivno oceniti u budućnosti.

Drugim rečima, model pokušava da odgovori na pitanje:

> „Koje proizvode treba preporučiti korisniku na osnovu njegovih prethodnih ocena i ponašanja sličnih korisnika?“

---

## User-Item matrica

Za potrebe recommendation sistema korišćena je tzv. user-item matrica.

U ovoj matrici:

* redovi predstavljaju korisnike,
* kolone predstavljaju proizvode,
* vrednosti predstavljaju ocene koje su korisnici dali proizvodima.

Međutim, u realnim sistemima većina korisnika ocenjuje veoma mali broj proizvoda, zbog čega matrica sadrži veliki broj nedostajućih vrednosti.

Ovakva matrica naziva se sparse matrica.

Velika sparsnost predstavlja značajan problem za klasične recommendation algoritme, jer model raspolaže ograničenim brojem informacija o preferencijama korisnika.

---

## Primena autoenkodera

Kako bi se rešio problem sparsnosti i pronalaženja skrivenih obrazaca u podacima, u projektu su korišćeni autoenkoderi.

Autoenkoder predstavlja specijalizovanu neuronsku mrežu koja pokušava da:

1. kompresuje ulazne podatke u latentni prostor,
2. nauči skrivene reprezentacije korisničkih preferencija,
3. rekonstruiše originalne podatke sa što manjom greškom.

Na osnovu naučenih latentnih reprezentacija model može da predvidi nedostajuće ocene proizvoda i generiše preporuke za korisnike.

U projektu su implementirane i analizirane različite varijante autoenkoderskih arhitektura:

* Baseline Autoencoder
* Improved Autoencoder
* Regularized Autoencoder
* Denoising Autoencoder

---

## Cilj projekta

Glavni cilj projekta jeste razvoj recommendation sistema zasnovanog na dubokom učenju koji može uspešno da modeluje korisničke preferencije i generiše preporuke proizvoda na osnovu istorijskih podataka.

---

## Očekivani rezultati

Očekuje se da modeli uspešno nauče latentne reprezentacije korisničkih preferencija i omoguće rekonstrukciju nedostajućih ocena proizvoda.

Na osnovu dobijenih rezultata moguće je:

* identifikovati proizvode relevantne za određenog korisnika,
* generisati personalizovane preporuke,
* analizirati performanse različitih neuronskih mreža,
* proceniti uticaj hiperparametara na kvalitet recommendation sistema.

# 2. Podaci (izvor, struktura, analiza i preprocesiranje)

## Opis skupa podataka

Za potrebe razvoja recommendation sistema korišćen je javno dostupan skup podataka sa platforme Amazon Product Reviews.

Dataset sadrži veliki broj korisničkih recenzija i ocena proizvoda iz oblasti elektronske trgovine. Svaka recenzija predstavlja interakciju između korisnika i proizvoda, što ovaj skup podataka čini pogodnim za razvoj sistema za preporuku proizvoda.

Podaci obuhvataju:

* identifikatore korisnika,
* identifikatore proizvoda,
* numeričke ocene proizvoda,
* tekstualne recenzije,
* informacije o korisnosti recenzije,
* vremenske oznake.

Dataset predstavlja realan primer recommendation problema sa velikim brojem korisnika i proizvoda, kao i izraženom sparsnošću podataka.

---

## Struktura podataka

Originalni dataset sadrži ukupno:

* **568454 redova**
* **10 kolona**

Kolone u datasetu:

| Kolona                 | Opis                                            |
| ---------------------- | ----------------------------------------------- |
| Id                     | Jedinstveni identifikator recenzije             |
| ProductId              | Identifikator proizvoda                         |
| UserId                 | Identifikator korisnika                         |
| ProfileName            | Ime korisničkog profila                         |
| HelpfulnessNumerator   | Broj korisnika koji smatraju recenziju korisnom |
| HelpfulnessDenominator | Ukupan broj glasova za korisnost                |
| Score                  | Ocena proizvoda                                 |
| Time                   | Vremenska oznaka recenzije                      |
| Summary                | Kratak opis recenzije                           |
| Text                   | Tekst kompletne recenzije                       |

---

## Osnovna analiza podataka

Nakon učitavanja podataka izvršena je osnovna analiza skupa podataka kako bi se stekao uvid u njegovu strukturu i kvalitet.

Analiza je pokazala:

* broj jedinstvenih korisnika: **256059**
* broj jedinstvenih proizvoda: **74258**

Prilikom analize uočene su i određene nedostajuće vrednosti:

| Kolona      | Broj nedostajućih vrednosti |
| ----------- | --------------------------- |
| ProfileName | 26                          |
| Summary     | 27                          |

S obzirom na veoma mali broj nedostajućih vrednosti u odnosu na ukupan broj podataka, ove kolone nisu predstavljale značajan problem za recommendation sistem.

---

## Problem sparsnosti podataka

Recommendation sistemi se u praksi suočavaju sa problemom sparsnosti podataka.

To znači da većina korisnika ocenjuje samo mali broj proizvoda, dok veliki broj proizvoda nikada nije ocenjen od strane određenog korisnika.

Zbog toga user-item matrica sadrži veliki broj nedostajućih vrednosti.

Velika sparsnost otežava proces učenja modela jer neuronska mreža raspolaže ograničenim brojem poznatih interakcija između korisnika i proizvoda.

Iz tog razloga bilo je neophodno izvršiti dodatnu filtraciju podataka pre treniranja modela.

---

## Preprocesiranje podataka

Kako bi se omogućilo efikasnije treniranje modela, izvršeno je filtriranje korisnika i proizvoda sa veoma malim brojem interakcija.

Zadržani su:

* korisnici sa minimum 20 recenzija,
* proizvodi sa minimum 20 recenzija.

Ovakvim pristupom uklonjeni su retki korisnici i proizvodi koji ne sadrže dovoljno informacija za kvalitetno treniranje recommendation sistema.

Nakon filtracije dimenzije dataseta postale su:

| Skup podataka      | Dimenzije    |
| ------------------ | ------------ |
| Originalni dataset | (568454, 10) |
| Filtrirani dataset | (47380, 10)  |

Filtracija podataka omogućila je smanjenje sparsnosti i stabilnije treniranje neuronskih mreža.

---

## Kreiranje user-item matrice

Nakon preprocesiranja podataka kreirana je user-item matrica.

U ovoj matrici:

* redovi predstavljaju korisnike,
* kolone predstavljaju proizvode,
* vrednosti predstavljaju ocene proizvoda.

Za kreiranje matrice korišćena je pivot tabela nad kolonama:

* UserId
* ProductId
* Score

Nedostajuće vrednosti popunjene su nulama, pri čemu vrednost 0 označava da korisnik nije ocenio određeni proizvod.

Dimenzije kreirane user-item matrice iznosile su:

* **1643 korisnika**
* **4308 proizvoda**

---

## Analiza sparsnosti matrice

Nakon kreiranja user-item matrice izvršena je analiza sparsnosti podataka.

Dobijena sparsnost matrice iznosila je:

* **0.9947**

To znači da više od 99% matrice čine nedostajuće interakcije između korisnika i proizvoda.

Ovakva sparsnost predstavlja tipičan problem recommendation sistema i jedan od glavnih razloga zbog kojih su autoenkoderi pogodni za ovu vrstu zadatka.

Autoenkoderi omogućavaju modelu da nauči latentne obrasce korisničkog ponašanja i predvidi nedostajuće ocene proizvoda čak i u veoma sparse okruženjima.

---

## Normalizacija podataka

Pre treniranja neuronskih mreža izvršena je normalizacija podataka korišćenjem MinMaxScaler metode.

Ocene proizvoda skalirane su na opseg između 0 i 1 kako bi:

* treniranje modela bilo stabilnije,
* ubrzala konvergencija neuronske mreže,
* smanjio uticaj velikih numeričkih vrednosti.

Normalizacija predstavlja standardni korak u deep learning projektima i posebno je važna kod neuronskih mreža sa Sigmoid aktivacionom funkcijom na izlazu.

# 3. Arhitektura modela

## Uvod u arhitekturu recommendation sistema

U okviru projekta implementirano je više varijanti autoenkoderskih neuronskih mreža sa ciljem modelovanja korisničkih preferencija i generisanja preporuka proizvoda.

Svi modeli zasnovani su na principu rekonstrukcije user-item matrice, pri čemu neuronska mreža pokušava da nauči latentne obrasce ponašanja korisnika i predvidi nedostajuće ocene proizvoda.

Za implementaciju modela korišćen je PyTorch framework zbog njegove fleksibilnosti i efikasnosti u radu sa dubokim neuronskim mrežama.

---

# Model 1 — Baseline Autoencoder

## Opis modela

Prvi implementirani model predstavlja osnovnu verziju autoenkoderske neuronske mreže (Baseline Autoencoder).

Cilj ovog modela bio je kreiranje početne reference performansi recommendation sistema.

Model se sastoji iz dve osnovne celine:

* Encoder dela
* Decoder dela

Encoder vrši kompresiju ulaznih podataka i ekstrakciju latentnih reprezentacija korisničkih preferencija, dok decoder pokušava da rekonstruiše originalnu user-item matricu.

---

## Aktivacione funkcije

U modelu su korišćene sledeće aktivacione funkcije:

### ReLU (Rectified Linear Unit)

ReLU funkcija korišćena je u skrivenim slojevima zbog:

* jednostavne implementacije,
* brzog treniranja,
* efikasnog propagiranja gradijenata,
* smanjenja problema nestajanja gradijenata.

### Sigmoid funkcija

Sigmoid funkcija korišćena je na izlaznom sloju jer su podaci prethodno normalizovani na opseg između 0 i 1.

Na taj način model generiše izlazne vrednosti kompatibilne sa skaliranim ocenama proizvoda.

---

# Model 2 — Improved Autoencoder

## Motivacija unapređenja modela

Nakon implementacije osnovnog autoenkodera izvršeno je unapređenje arhitekture kako bi se omogućilo:

* bolje modelovanje korisničkih preferencija,
* stabilnije treniranje,
* preciznija rekonstrukcija user-item matrice,
* smanjenje greške rekonstrukcije.

---

## Arhitektura unapređenog modela

Drugi model koristi dublju neuronsku mrežu sa većim brojem neurona i slojeva.

Dublja arhitektura omogućava modelu da nauči kompleksnije obrasce ponašanja korisnika.

---

## Prednosti dublje arhitekture

Dublja mreža omogućava:

* učenje složenijih relacija između korisnika i proizvoda,
* efikasnije pronalaženje latentnih obrazaca,
* precizniju rekonstrukciju nedostajućih ocena,
* bolje recommendation performanse.

Rezultati evaluacije pokazali su da je upravo ovaj model ostvario najbolje RMSE performanse među implementiranim arhitekturama.

---

# Model 3 — Regularized Autoencoder

## Problem overfitting-a

Duboke neuronske mreže često pate od problema overfitting-a.

Overfitting nastaje kada model previše dobro zapamti trening podatke, ali loše generalizuje na novim podacima.

Kako bi se smanjio ovaj problem implementiran je regularizovani autoenkoder sa Dropout slojevima.

---

## Dropout regularizacija

Dropout predstavlja tehniku regularizacije u kojoj se tokom treniranja nasumično deaktivira određeni procenat neurona.

Na taj način model:

* ne zavisi previše od pojedinačnih neurona,
* bolje generalizuje podatke,
* postaje otporniji na overfitting.

---

## Arhitektura modela

Regularizovani autoenkoder koristi Dropout slojeve između linearnih transformacija.

---

## Prednosti regularizacije

Iako ovaj model nije ostvario najbolje RMSE performanse, pokazao je stabilne rezultate i dobru sposobnost generalizacije.

---

# Model 4 — Denoising Autoencoder

## Ideja denoising autoenkodera

Denoising autoencoder predstavlja specijalnu varijantu autoenkodera koja tokom treniranja koristi namerno oštećene (noisy) ulazne podatke.

Cilj modela jeste:

* uklanjanje šuma iz podataka,
* rekonstrukcija originalnih korisničkih preferencija,
* povećanje robusnosti recommendation sistema.

---

## Dodavanje šuma

Tokom treniranja deo ulaznih vrednosti nasumično je menjan ili maskiran.

---

## Prednosti denoising pristupa

Denoising autoenkoder omogućava:

* veću otpornost na sparse podatke,
* stabilnije recommendation performanse,
* bolju generalizaciju,
* robusnije predikcije korisničkih preferencija.

Ovakvi modeli često se koriste u recommendation sistemima velikih platformi zbog njihove sposobnosti rada sa nepotpunim i šumnim podacima.

---

# 4. Trening

## Podela podataka

Pre početka treniranja neuronskih mreža izvršena je podela podataka na:

* trening skup (train set)
* test skup (test set)

Trening skup korišćen je za obučavanje modela i prilagođavanje težina neuronske mreže, dok je test skup korišćen za evaluaciju performansi recommendation sistema nad podacima koje model prethodno nije video.

Ovakav pristup omogućava realniju procenu sposobnosti generalizacije modela.

---

## Konverzija podataka u tenzore

Pošto su modeli implementirani korišćenjem PyTorch framework-a, bilo je neophodno konvertovati podatke u PyTorch tenzore.

User-item matrica transformisana je u numerički format pogodan za rad neuronskih mreža i GPU akceleraciju.

Na taj način omogućena je efikasna obrada velikih matrica tokom treniranja modela.

---

## Mini-batch treniranje

Za treniranje modela korišćen je mini-batch pristup.

Umesto obrade celog dataseta odjednom, podaci su podeljeni u manje grupe (batch-eve).

Za sve modele korišćena je veličina batch-a od 256 uzoraka.

---

## Funkcija greške

Kao funkcija greške korišćena je:

* Mean Squared Error (MSE) loss funkcija.

Cilj treniranja jeste minimizacija funkcije greške i što preciznija rekonstrukcija user-item matrice.

---

## Optimizacioni algoritam

Za optimizaciju neuronskih mreža korišćen je:

* Adam optimizer.

Za većinu modela korišćen je learning rate od 0.0005 i 0.001 u zavisnosti od arhitekture modela.

---

## Proces treniranja

Tokom svake epohe izvršavani su sledeći koraci:

1. učitavanje batch-a podataka,
2. forward propagation kroz mrežu,
3. računanje funkcije greške,
4. backpropagation algoritam,
5. ažuriranje težina neuronske mreže.

Nakon svake epohe računata je prosečna vrednost loss funkcije kako bi se pratila stabilnost procesa treniranja.

---
# 5. Analiza osetljivosti i hiperparametarska optimizacija

## Značaj learning rate parametra

Learning rate predstavlja jedan od najvažnijih hiperparametara u dubokom učenju.

Ovaj parametar određuje brzinu prilagođavanja težina neuronske mreže tokom optimizacije.

---

## Eksperimentalna analiza

Radi analize osetljivosti modela izvršeno je treniranje unapređenog autoenkodera sa različitim learning rate vrednostima.

Testirane vrednosti:

| Learning Rate |
| ------------- |
| 0.01          |
| 0.001         |
| 0.0005        |

Za svaku vrednost izvršen je kraći trening modela, nakon čega je analizirana finalna vrednost funkcije greške.

---

## Rezultati hiperparametarske analize

Dobijeni rezultati pokazali su sledeće:

| Learning Rate | Final Loss |
| ------------- | ---------- |
| 0.01          | 0.003173   |
| 0.001         | 0.002849   |
| 0.0005        | 0.002933   |

Na osnovu rezultata može se uočiti da je najbolji rezultat ostvaren za learning rate vrednost:

* **0.001**

Ova vrednost omogućila je:

* najstabilniji proces treniranja,
* najnižu vrednost funkcije greške,
* najbolju konvergenciju modela.

---


# Zaključak hiperparametarske analize

Rezultati sprovedene analize potvrđuju da hiperparametri imaju veoma značajan uticaj na recommendation performanse neuronskih mreža.

Najbolje ukupne rezultate ostvario je unapređeni autoencoder sa learning rate vrednošću 0.001.

# 6. Rezultati evaluacije

## Uvod

Nakon treniranja svih implementiranih modela izvršena je evaluacija recommendation sistema sa ciljem analize kvaliteta rekonstrukcije korisničkih preferencija i poređenja performansi različitih arhitektura autoenkodera.

Evaluacija modela predstavlja jedan od najvažnijih koraka u razvoju recommendation sistema, budući da omogućava procenu uspešnosti modela nad podacima koje neuronska mreža prethodno nije videla tokom treniranja.

Performanse modela procenjene su korišćenjem:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

Manje vrednosti ovih metrika ukazuju na precizniju rekonstrukciju korisničkih preferencija i kvalitetnije recommendation performanse.

---

# Evaluacione metrike

## RMSE — Root Mean Squared Error

RMSE predstavlja standardnu metriku za evaluaciju recommendation sistema.

Ova metrika meri prosečno kvadratno odstupanje između:

* originalnih korisničkih ocena,
* rekonstruisanih vrednosti koje generiše model.

Veće greške imaju značajno veći uticaj na konačnu RMSE vrednost, zbog čega ova metrika posebno kažnjava velika odstupanja modela.

Manja RMSE vrednost označava precizniji recommendation sistem.

---

## MAE — Mean Absolute Error

MAE predstavlja prosečnu apsolutnu grešku između originalnih i rekonstruisanih vrednosti.

Za razliku od RMSE metrike, MAE ne kažnjava velike greške kvadratno, već ravnomerno tretira sva odstupanja.

Ova metrika omogućava dodatni uvid u stabilnost recommendation sistema i prosečnu preciznost rekonstrukcije korisničkih preferencija.

---

# Rezultati evaluacije modela

Nakon evaluacije svih implementiranih modela dobijeni su sledeći rezultati:

| Model                   | RMSE     | MAE      |
| ----------------------- | -------- | -------- |
| Baseline Autoencoder    | 0.080830 | 0.004911 |
| Improved Autoencoder    | 0.069254 | 0.010436 |
| Regularized Autoencoder | 0.072605 | 0.008189 |
| Denoising Autoencoder   | 0.072386 | 0.009402 |

---

# Poređenje modela

Na osnovu sprovedene evaluacije može se uočiti značajan uticaj arhitekture neuronske mreže i regularizacionih tehnika na recommendation performanse sistema.

Glavni zaključci evaluacije:

* dublje arhitekture ostvaruju bolje rekonstrukcione performanse,
* regularizacija doprinosi stabilnijem treniranju,
* denoising pristup povećava robusnost modela,
* latentni prostor ima značajan uticaj na recommendation kvalitet.

Unapređeni autoencoder ostvario je najbolji RMSE rezultat zahvaljujući većem kapacitetu mreže i uspešnijem modelovanju latentnih korisničkih preferencija.

---

# Generisanje preporuka proizvoda

Na osnovu najboljeg modela izvršeno je generisanje preporuka proizvoda za korisnike.

Recommendation sistem predviđa ocene proizvoda koje korisnik prethodno nije ocenio i na osnovu tih predikcija generiše listu preporučenih proizvoda.

Dobijeni rezultati potvrđuju da implementirani autoenkoder modeli mogu uspešno da:

* modeluju korisničke preference,
* rekonstruišu user-item matricu,
* predviđaju nedostajuće interakcije,
* generišu personalizovane preporuke proizvoda.

---
# 7. Diskusija

Rezultati projekta pokazali su da autoenkoder modeli predstavljaju efikasan pristup recommendation sistemima zasnovanim na dubokom učenju.

Najveći izazov tokom rada predstavljala je izrazita sparsnost user-item matrice, budući da više od 99% vrednosti predstavlja nedostajuće interakcije između korisnika i proizvoda. Zbog toga modeli raspolažu veoma ograničenim brojem poznatih korisničkih preferencija, što otežava proces učenja latentnih obrazaca ponašanja.

Uprkos tome, implementirani modeli uspešno su naučili osnovne latentne reprezentacije korisnika i proizvoda i omogućili generisanje personalizovanih preporuka.

Najbolje performanse ostvario je unapređeni autoencoder zahvaljujući dubljoj arhitekturi i većem latentnom prostoru, što potvrđuje značaj kapaciteta mreže u recommendation sistemima.

Regularizacija i denoising tehnike doprinele su stabilnijem treniranju i boljoj generalizaciji modela, ali zbog velike sparsnosti podataka nisu dovele do značajno boljih evaluacionih rezultata.

Jedno od glavnih ograničenja projekta jeste činjenica da recommendation sistem koristi isključivo numeričke ocene proizvoda, bez dodatnih informacija o korisnicima, proizvodima i tekstualnim recenzijama.

Moguća unapređenja sistema uključuju:

* korišćenje embeddings reprezentacija,
* analizu tekstualnih recenzija pomoću NLP tehnika,
* implementaciju transformers arhitektura,
* sekvencijalne recommendation modele,
* hibridne recommendation sisteme.

Dobijeni rezultati potvrđuju da deep learning recommendation sistemi imaju veliki potencijal za primenu u elektronskoj trgovini i personalizaciji korisničkog iskustva.

---

# 8. Zaključak

U okviru projekta implementiran je recommendation sistem za preporuku proizvoda u elektronskoj trgovini korišćenjem deep learning pristupa i autoenkoderskih neuronskih mreža.

Kroz analizu i preprocesiranje podataka kreirana je sparse user-item matrica koja predstavlja osnovu recommendation sistema. Implementirane su četiri različite arhitekture autoenkodera sa ciljem modelovanja korisničkih preferencija i rekonstrukcije nedostajućih interakcija između korisnika i proizvoda.

Rezultati evaluacije pokazali su da unapređeni autoencoder ostvaruje najbolje recommendation performanse zahvaljujući dubljoj arhitekturi i većem latentnom prostoru.

Regularizovani i denoising modeli pokazali su stabilnije ponašanje i bolju generalizaciju, dok je analiza hiperparametara potvrdila značajan uticaj learning rate parametra na stabilnost i uspešnost treniranja modela.

Pored evaluacije modela, implementiran je i sistem za generisanje personalizovanih preporuka proizvoda, čime je demonstrirana praktična primena deep learning recommendation sistema u elektronskoj trgovini.

Dobijeni rezultati potvrđuju da autoenkoder modeli mogu uspešno da modeluju korisničke preference i generišu preporuke čak i u uslovima izrazito sparse podataka.

Iako recommendation sistem poseduje određena ograničenja zbog nedostatka dodatnih informacija o korisnicima i proizvodima, projekat uspešno demonstrira potencijal dubokog učenja u oblasti recommendation sistema i personalizacije korisničkog iskustva.

