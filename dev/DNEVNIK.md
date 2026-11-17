# Dnevnik poskusov (IBB A2, razpoznavanje šarenice z LBP)

Vsaka iteracija ima: **vprašanje**, **kaj smo izmerili**, **odločitev in zakaj**.
Surove številke vseh meritev so v `dev/results.jsonl` (ena vrstica na poskus).

## Kako merimo

- **Podatki:** 348 slik iz `dataset_downloaded/`, 80 identitet (oseba + oko; levo
  in desno oko sta različna razreda). Vsi pari slik: 60 378, od tega pravih
  (genuine) parov nekaj sto, ostalo so lažni (impostor).
- **Predobdelava** je enaka kot v `hdbif.py` (rdeči kanal, `fix_image`,
  segmentacija, odvijanje v 64×512). Naredi se enkrat (`dev/build_cache.py`).
- **Metrike** (`dev/evallib.py`), vse nad razdaljo (manjše = bolj podobno):
  - **EER**: točka, kjer je delež napačno sprejetih (FMR) enak deležu napačno
    zavrnjenih (FNMR). Glavna metrika.
  - **AUC**: verjetnost, da ima naključen pravi par manjšo razdaljo kot lažni.
  - **d'**: razmik med porazdelitvama pravih in lažnih parov v enotah std. odklona.
  - **FNMR@FMR=1 %**: koliko pravih parov zavrnemo, če dovolimo 1 % lažnih sprejemov.
- **Preverjanje prenaučenosti:** osebe razdelimo na dve disjunktni polovici
  (A = sode, B = lihe). Parametre izbiramo po skupnem EER, a gledamo, da
  izboljšava velja na obeh polovicah.
- **Raziskovalna koda** (`dev/research.py`) računa vse pare naenkrat; končna
  koda v `modules/irisRecognition.py` se na koncu izmeri posebej z
  `dev/eval_class.py`, ki kliče prave metode razreda.

## Izhodišče: HDBIF

| Metoda | EER | AUC | d' |
|---|---|---|---|
| HDBIF (`extractCode` + `matchCodesEfficient`) | 0,14 % | 1,0000 | 5,22 |

Hitrost: 50 ms/sliko, 60 ms/par. To je referenca, ne cilj.

## Iteracija 1: klasični LBP s histogrami (`exp01_baselines.py`)

**Vprašanje:** kako dobro deluje LBP točno po Ojali (histogram LBP oznak,
primerjava s chi-kvadrat)?

| Različica | EER (A / B) |
|---|---|
| P=8, R=1, 256 košev, en globalni histogram | 25,84 % (22,2 / 28,7) |
| isto, uniformni vzorci (59 košev) | 25,80 % (22,5 / 28,8) |
| uniformni, bloki 16×32 | 10,84 % (8,2 / 13,6) |
| uniformni, bloki 8×16, zamik ±16 px | 8,25 % (5,4 / 10,1) |
| uniformni, bloki 8×16, zamik ±16 px, z masko | 7,33 % (5,4 / 8,5) |

**Ugotovitve:**
- Globalni histogram je slab, ker zavrže položaj: pove, *katere* mikrostrukture
  so v šarenici, ne pa *kje*. Šarenice se ločijo ravno po razporeditvi.
- Uniformni vzorci dajo enak rezultat s 4× manj koši (59 namesto 256), torej
  neuniformni vzorci ne nosijo uporabne informacije.
- Manjši bloki = več informacije o položaju = boljši rezultat. Skrajni primer
  tega trenda je blok velikosti 1 piksel, kar vodi v iteracijo 2.

## Iteracija 2: bitna koda LBP + Hammingova razdalja (`exp02_bitcode.py`)

**Vprašanje:** kaj če LBP bitov ne histogramiramo, ampak jih pustimo na mestu
in dve šarenici primerjamo bit za bitom (kot Daugmanova koda šarenice)?

Vsak piksel da P bitov: bit p = 1, če je sosed p svetlejši od središča. Razdalja
= delež bitov, ki se razlikujejo (Hamming). Rotacijo očesa kompenziramo s
krožnim zamikom kode po kotni osi in vzamemo najmanjšo razdaljo.

| Različica | EER (A / B) |
|---|---|
| P=8, R=1, brez zamika, brez maske | 12,67 % (13,2 / 12,0) |
| P=8, R=1, zamik ±16, brez maske | 7,33 % (6,1 / 8,2) |
| P=8, R=1, zamik ±16, z masko | 3,21 % (2,9 / 3,4) |
| P=8, R=2, σ=0, z masko | 2,60 % |
| P=8, R=2, σ=2, z masko | 0,78 % |
| P=8, R=3, σ=2, z masko | 0,76 % |
| P=8, R=3, σ=2, brez maske | 3,37 % |

(σ = standardni odklon Gaussovega glajenja pred LBP.)

**Ugotovitve:**
- Bitna koda je bistveno boljša od histogramov (0,8 % proti 7,3 %).
- **Zamik** prepolovi napako (12,7 → 7,3 %): glava ni vedno enako nagnjena.
- **Maska** zmanjša napako še za 2- do 4-krat: brez nje primerjamo tudi veke in
  trepalnice, ki niso del šarenice.
- **Glajenje** je največji posamezni dobitek (2,6 → 0,8 %): LBP na surovih
  pikslih primerja šum senzorja in JPEG artefakte; po glajenju primerja
  dejansko teksturo šarenice.
- Večji radij pomaga manj kot glajenje.

**Odločitev:** nadaljujemo z bitno kodo. Odprto: natančnejše iskanje
parametrov (R, σ, P, eliptična soseščina), izločanje nestabilnih bitov,
normalizacija rezultata, lastna ocena maske za primer, ko maske ne dobimo.

## Iteracija 3: vpliv posameznih parametrov (`exp03_sweeps.py`)

**Vprašanje:** kateri parametri bitne kode sploh vplivajo? Izhodišče P=8, R=3,
σ=2, zamik ±16, z masko (EER 0,76 %); vsakič spremenimo en parameter.

| Sprememba | EER | Sklep |
|---|---|---|
| P=4 / 6 / 12 / 16 | 0,46 / 0,61 / 0,77 / 0,78 % | več točk na majhnem krogu ne pomaga, biti so odvečni |
| CS-LBP (primerja nasprotna soseda), P=8 | 0,59 % | nekoliko bolje, a manj kot eliptična soseščina |
| elipsa Rx=6, Ry=2 (široka kotno) | **0,29 %** | velik dobitek |
| elipsa Rx=2, Ry=4 (široka radialno) | 1,07 % | slabše od kroga |
| trdo izločanje nestabilnih bitov, 10–50 % | 0,89–1,98 % | škodi |
| erozija prave maske za 1–7 px | 0,77–0,92 % | ne pomaga |
| zamik 0 / ±4 / ±8 / ±16 / ±32 | 4,89 / 0,88 / 0,73 / 0,76 / 0,88 % | zamik je nujen; 8–16 je plato |
| Daugmanova normalizacija z √(n) | 1,07 % | škodi |
| vsaka 2. / 4. vrstica kode | 0,76 / 0,76 % | brez izgube, koda je 2–4× manjša |

**Ugotovitve:**
- **Smer soseščine je ključna.** Tekstura šarenice so pretežno radialne proge
  (kripte, brazde), zato se svetlost najhitreje spreminja v *kotni* smeri. Široka
  elipsa v kotni smeri zajame to spremembo; v radialni smeri primerja skoraj
  enake vrednosti (šumni biti).
- Prevelik zamik škodi, ker minimum čez več zamikov zniža tudi razdalje lažnih
  parov. ±16 stolpcev = ±11° pokrije realne nagibe glave.
- Daugmanova normalizacija kaznuje pare z malo skupnimi biti; pri nas je mask
  dovolj (povprečno 80 % veljavnih pikslov), zato samo doda šum.

## Iteracija 4: skupno iskanje po mreži (`exp04_grid.py`)

**Vprašanje:** parametri so med seboj odvisni (večji radij ↔ manj glajenja),
zato jih preiščemo skupaj: P ∈ {4, 8}, Rx ∈ {4…16}, Ry ∈ {1…4}, σ ∈ {1,5; 2; 3},
144 kombinacij, z masko.

| Konfiguracija | EER (A / B) | d' |
|---|---|---|
| P=4, Rx=12, Ry=1, σ=2 | 0,29 % (0,42 / 0,02) | 5,47 |
| P=4, Rx=10, Ry=2, σ=2 | 0,31 % | 5,48 |
| **P=8, Rx=12, Ry=1, σ=2** | **0,02 %** (0,01 / 0,01) | 5,91 |
| P=8, Rx=10, Ry=1, σ=2 | 0,12 % | 5,86 |
| P=8, Rx=16, Ry=1, σ=1,5 | 0,11 % | 5,80 |
| P=8, Rx=6, Ry=2, σ=2 | 0,29 % | 5,47 |

**Ugotovitve:**
- Pri zelo sploščeni elipsi je P=8 boljši od P=4. Diagonalni sosedje ležijo na
  (±8,5; ±0,7), torej skoraj na isti vrstici, a na *drugi razdalji* kot glavna
  (±12; 0). P=8 tako primerja kotno smer na dveh merilih hkrati.
- Rx 10–14 je plato (EER 0,02–0,13 %), izbira Rx=12 je na sredini platoja in ne
  na ostrem vrhu.
- **Opozorilo o šumu:** pravih parov je 655, torej en sam napačen par pomeni
  0,15 % EER. Razlike pod ~0,15 % niso statistično pomembne; zato poleg EER
  gledamo d' (uporablja vse pare, ne le repov) in obe podmnožici A/B.

## Iteracija 5 in 8: kaj, če maske ne dobimo? (`exp05_selfmask.py`, `exp08_selfmask2.py`)

**Vprašanje:** predloga kliče `extractIBBCode(im_polar)` brez maske. Brez maske
je EER 2,0 % (z masko 0,02 %). Ali lahko masko ocenimo sami iz odvite slike?

| Maska (P=8, Rx=12, Ry=1, σ=2, brez uteži) | EER | d' |
|---|---|---|
| brez maske | 1,98 % | 4,25 |
| prior: povprečna maska učne množice > 0,9 (tabela 64×512) | 0,31 % | 5,53 |
| prior kot dve kosinusni kupoli (6 parametrov) | 0,46 % | 5,41 |
| samo svetlost (\|z\| < 2) | 0,77 % *(pri Rx=6)* | 5,02 |
| kupoli ∧ svetlost \|z\| < 2 | 0,19 % | 5,95 |
| **kupoli ∧ svetlost \|z\| < 2, erozija 2 px** | **0,15 %** | **6,14** |
| kupoli ∧ svetlost \|z\| < 2,5 / 3, erozija 2 | 0,31 % | 5,98 / 5,86 |
| prava maska (zgornja meja) | 0,02 % | 5,90 |

**Kako deluje lastna maska:**
1. **Prior.** Veki prekrivata šarenico vedno na istem mestu odvite slike:
   spodnja pri 90°, zgornja pri 270°, od zunanjega roba navznoter. Obliko
   opišemo s kupolo `h·cos(π/2 · d/w)`; parametre smo prilagodili povprečni
   maski (spodnja: središče 0,26, polširina 0,17, višina 0,47; zgornja: 0,74,
   0,25, 0,875, vse kot delež širine/višine).
2. **Svetlost.** Znotraj priorja izračunamo mediano in MAD svetlosti; piksel
   zavržemo, če je od mediane oddaljen več kot 2·MAD (koža in odsevi so
   svetlejši, trepalnice temnejše).
3. **Erozija za 2 px**, ker LBP na robu maske primerja šarenico z veko.

**Odločitev:** koda sprejme masko kot neobvezen argument. Če jo dobi, jo
uporabi; sicer si jo oceni sama. Tako deluje v obeh primerih, brez maske pa je
napaka 13× manjša kot brez tega koraka.

## Iteracija 6 in 7: dodatki (`exp06_extras.py`, `exp07_soft.py`)

**Vprašanje:** kaj še lahko izboljšamo nad najboljšo konfiguracijo?
(Merjeno na P=4, Rx=10, Ry=2, σ=2, z masko; izhodišče EER 0,31 %, d' 5,48.)

| Dodatek | EER | d' | Sklep |
|---|---|---|---|
| radialni zamik ±1 / ±2 vrstici | 0,28 % | 5,61 / 5,66 | majhen dobitek, 3–5× počasneje → ne |
| več meril (Rx 5+10, 6+12, 4+8+16) | 0,31–0,32 % | 5,44 | nič (P=8 na elipsi to že naredi) |
| mehke uteži bitov, t = 0,25 / 0,5 / 1 / 2 | 0,27 / 0,27 / 0,15 / 0,14 % | 5,66 / 5,86 / 6,16 / 6,44 | velik dobitek |
| mehke uteži, t = 4 / 6 / 10 / ∞ | 0,18 / 0,48 / 0,76 / 1,22 % | 6,23 → 5,25 | preveč: nekaj bitov prevlada |
| mehke uteži t=2 **brez maske** | 5,46 % | 3,64 | brez maske močno škodijo |

**Mehke uteži:** utež bita = min(1, |razlika| / (t · mediana |razlik|)). Bit, pri
katerem sta sosed in središče skoraj enako svetla, se ob najmanjšem šumu
obrne; tak bit šteje manj. Trdo izločanje (iteracija 3) je škodilo, ker vrže
stran tudi uporabne bite; mehka utež jih le oslabi.

**Zakaj brez maske škodijo:** najmočnejše razlike v sliki so robovi vek in
odsevi. Če jih maska ne odstrani, dobijo največjo utež.

## Iteracija 9: končna izbira v vseh načinih (`exp09_final.py`)

P=8, σ=2, vsaka 2. vrstica; M = podana maska, S = lastna maska.

| Način | Rx, Ry | t=0 | t=0,5 | t=1 | t=2 |
|---|---|---|---|---|---|
| M | 12, 1 | 0,02 % / d' 5,90 | **0,03 % / 6,32** | 0,03 % / 6,62 | 0,15 % / 6,82 |
| M | 10, 1 | 0,04 % / 5,86 | 0,13 % / 6,28 | 0,13 % / 6,62 | 0,15 % / 6,94 |
| M | 14, 1 | 0,02 % / 5,90 | 0,02 % / 6,29 | 0,12 % / 6,54 | 0,15 % / 6,62 |
| M | 12, 2 | 0,02 % / 5,80 | 0,03 % / 6,19 | 0,03 % / 6,50 | 0,12 % / 6,60 |
| S | 12, 1 | **0,15 % / 6,14** | 0,31 % / 6,60 | 0,31 % / 6,91 | 0,61 % / 7,08 |
| S | 10, 1 | 0,15 % / 6,07 | 0,19 % / 6,52 | 0,31 % / 6,86 | 0,61 % / 7,12 |
| S | 14, 1 | 0,28 % / 6,17 | 0,31 % / 6,62 | 0,32 % / 6,89 | 0,62 % / 6,95 |

Dodatno: podana maska ∧ svetlost ne prinese zanesljive izboljšave (0,03–0,15 %),
zato pri podani maski svetlosti ne preverjamo (enostavnejša koda).

**Končna odločitev:**

| Parameter | Vrednost | Zakaj |
|---|---|---|
| P | 8 | na elipsi da dve merili v kotni smeri (iter. 4) |
| Rx, Ry | 12, 1 | sredina platoja Rx 10–14; radialna smer ne nosi informacije (iter. 3, 4) |
| σ | 2 | odstrani šum/JPEG; σ=3 že briše teksturo (iter. 2, 7) |
| zamik | ±16 stolpcev (±11°) | plato 8–16; večji doseg je varnejši za neznane slike (iter. 3) |
| vrstice | vsaka 2. | brez izgube, pol manjša predloga (iter. 3, 6) |
| uteži | t=0,5 s podano masko, brez uteži z lastno masko | z masko: isti EER, d' +0,4; brez maske uteži škodijo (iter. 7, 9) |
| lastna maska | kupoli ∧ \|z\|<2, erozija 2 | iter. 8 |

Pri uteži smo namenoma izbrali zmerno vrednost (t=0,5) in ne tiste z največjim
d' (t=2): pri t≥1 EER na nekaterih konfiguracijah že naraste, torej repi
porazdelitve trpijo.


## Končni rezultat (prava koda razreda, `dev/eval_class.py`)

Merjeno s klici `irisRec.extractIBBCode` / `irisRec.matchIBBCodes` na vseh
60 378 parih (655 pravih, 59 723 lažnih).

| Način klica | EER (A / B) | d' | FNMR@FMR=0,1 % |
|---|---|---|---|
| HDBIF (izhodišče) | 0,14 % (0,31 / 0,01) | 5,22 | – |
| `extractIBBCode(polar, mask)` + `matchIBBCodes(c1, c2)` | 0,03 % (0,00 / 0,02) | 6,32 | 0,0 % |
| `extractIBBCode(polar)` + `matchIBBCodes(c1, c2)` (lastna maska) | 0,15 % (0,36 / 0,00) | 6,14 | 0,3 % |
| `extractIBBCode(polar)` + `matchIBBCodes(c1, c2, m1, m2)` | 0,00 % (0,00 / 0,00) | 6,46 | 0,0 % |

Številke se ujemajo z raziskovalno kodo (iteracija 9), torej je prenos v razred
pravilen. Hitrost: ekstrakcija 17 ms/sliko, ujemanje 3,4 ms/par (HDBIF 15 ms/par).
Predloga: int8 [8, 32, 512] = 128 kB.

**Preverjanje pravilnosti** (`dev/test_ibb.py`): FFT ujemanje da isto kot
preprosta zanka z `np.roll` (razlika < 1e-7); ista slika → razdalja 0; krožno
zamaknjena slika (5 in 16 stolpcev) → 0, zamik 40 (izven dosega) → 0,40;
`None`, prazna koda, maska kot bool / 0-1 / 0-255 delujejo.

**Omejitve, ki jih je treba pošteno povedati:**
- Parametri so izbrani na istih 348 slikah, na katerih poročamo rezultat. Delitev
  A/B (disjunktne osebe) kaže, da izbira ni vezana na posamezne osebe, ni pa to
  neodvisna testna množica. Na skritem setu pričakujemo nekoliko slabši rezultat.
- Pri 655 pravih parih en napačen par pomeni 0,15 % EER. Razlike med 0,00 %,
  0,03 % in 0,15 % so torej 0–1 par; trditev "boljše od HDBIF" statistično ne
  drži, drži pa "primerljivo s HDBIF".
- Prior lege vek je prilagojen slikam s pogledom naravnost (`_s_`). Pri pogledu
  gor/levo so veke drugje; tam pomaga le preverjanje svetlosti.


## Pregled kode in dodatna evalvacija

**Preureditev** (brez spremembe obnašanja): `extractIBBCode` razdeljena na manjše
funkcije, konstante za izbiro vrstic in za vrednost "ni primerjave", preverjanje
oblike maske, komentarji v angleščini. Po preureditvi so razdalje na vseh 60 378
parih **identične** (največja razlika 0,0) v vseh treh načinih klica.

**Intervali zaupanja** (`eval_bootstrap.py`, bootstrap po osebah, 1000 ponovitev):

| Metoda | EER | 95 % interval | d' | 95 % interval |
|---|---|---|---|---|
| HDBIF | 0,14 % | 0,00–0,31 % | 5,22 | 4,46–6,34 |
| LBP, maska ob ekstrakciji | 0,03 % | 0,00–0,09 % | 6,32 | 5,52–7,52 |
| LBP, brez maske | 0,15 % | 0,00–0,77 % | 6,14 | 5,05–7,80 |
| LBP, maska samo ob primerjavi | 0,00 % | 0,00–0,00 % | 6,46 | 5,57–7,86 |

Intervali se prekrivajo: z LBP smo primerljivi s HDBIF, razlika ni statistično
pomembna. Način brez maske ima najširši interval, torej je najmanj zanesljiv.

**Dodatni testi** (`test_ibb.py`): vzorčenje sosedov se ujema z ročno bilinearno
interpolacijo; predznak kode je natanko LBP bit; biti s sosedom izven slike so
izločeni; maska napačne oblike preklopi na lastno masko; lastna maska ima IoU
0,53 s pravo in 1,2 % napačno obdržanih pikslov (namenoma previdna).
