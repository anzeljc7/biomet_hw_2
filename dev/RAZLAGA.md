# Razlaga in utemeljitev rešitve (IBB A2, razpoznavanje šarenice z LBP)

Dokument razloži, **kaj koda dela**, **zakaj je tako narejena** in **kako vemo, da
deluje**. Vsaka trditev ima za sabo meritev; surove tabele so v `DNEVNIK.md`,
kratki odgovori na vprašanja pa v `ZAGOVOR.md`.

Oddana datoteka: `modules/irisRecognition.py` (blok »IBB: LBP-based iris code«,
vrstice 260–520).

---

## 1. Rezultat

| Metoda | EER | 95 % interval | d' |
|---|---|---|---|
| HDBIF (izhodišče iz predloge) | 0,14 % | 0,00–0,31 % | 5,22 |
| najin LBP, maska podana | 0,03 % | 0,00–0,09 % | 6,32 |
| najin LBP, brez maske (lastna ocena) | 0,15 % | 0,00–0,77 % | 6,14 |
| klasični LBP po Ojali (histogrami po blokih) | 7,3 % | – | 3,0 |

348 slik, 80 identitet (oseba + oko), 655 pravih in 59 723 lažnih parov.
Intervali so iz bootstrapa po osebah (1000 ponovitev, `eval_bootstrap.py`).

**Kako to brati:** intervali se prekrivajo, zato je pravilna trditev *»najin LBP
je na teh podatkih primerljiv s HDBIF«*, ne »boljši«. Od klasičnega LBP je
boljši zanesljivo (7,3 % proti 0,03 %).

Hitrost: ekstrakcija 15 ms na sliko, primerjava 3 ms na par (HDBIF 15 ms na par).
Predloga je ena matrika `int8 [8, 32, 512]`, 128 kB.

---

## 2. Ideja v treh stavkih

1. LBP za vsak piksel pove, kateri sosedje so svetlejši od njega; to je opis
   lokalne teksture, neodvisen od osvetlitve.
2. Članek (Ojala) te opise *prešteje* v histogram. Pri šarenici pa ni pomembno,
   *katere* strukture so prisotne, ampak *kje* so, zato bite pustim na mestu in
   dve šarenici primerjam bit za bitom (Hammingova razdalja), kot Daugman.
3. Vse ostalo (glajenje, eliptična soseščina, zamik, maska, uteži) so
   prilagoditve temu, kakšne so slike šarenic v resnici.

---

## 3. Cevovod

```
odvita šarenica 64×512 (iz predloge)          maska 64×512 (neobvezno)
        │                                              │
        ▼                                              ▼
 _ibbBlur            Gauss σ=2, vodoravno krožno   _ibbToBool   ali   _ibbEstimateMask
        │                                              │        (če maske ni)
        ▼                                              │
 _ibbLbpDifferences  8 sosedov na elipsi 12×1;         │
        │            razlika sosed − središče          │
        ▼                                              ▼
 extractIBBCode      predznak = bit, |razlika| → utež, maska → veljavnost
        │            vsaka 2. vrstica, zapis v int8
        ▼
   koda int8 [8, 32, 512]     >0 bit 1,  <0 bit 0,  |x|/127 utež,  0 = ne upoštevaj

 matchIBBCodes(koda1, koda2)
        │  _ibbSignedWeights          int8 → [−1, 1]
        │  _ibbCircularCorrelation    ujemanje in število primerjanih bitov za vse zamike (FFT)
        ▼
   razdalja = min čez zamike ±16 stolpcev  [ (n − c) / 2n ]
```

---

## 4. Koda po korakih, z utemeljitvijo

### 4.1 `_ibbBlur`: glajenje pred LBP

Gaussov filter s σ=2. Slika se pred filtriranjem razširi: vodoravno **krožno**
(`BORDER_WRAP`), ker sta levi in desni rob odvite slike na šarenici soseda (kot
0° = 360°); navpično s ponovitvijo roba, ker onkraj zenice oziroma zunanjega roba
ni ničesar.

**Zakaj:** slike so s telefona, v vidni svetlobi, stisnjene z JPEG. LBP primerja
posamezne piksle, torej na surovi sliki primerja šum senzorja in JPEG bloke, ki
so pri vsakem posnetku drugačni. Glajenje pusti teksturo šarenice in odstrani
šum.

| | EER |
|---|---|
| brez glajenja (R=2) | 2,60 % |
| σ=1 | 1,49 % |
| σ=2 | 0,78 % |
| σ=3 (na eliptični soseščini z utežmi) | slabše kot σ=2: 0,15 → 0,31 %, d' 6,4 → 5,9 |

### 4.2 `_ibbSampleNeighbour`: vrednost soseda z bilinearno interpolacijo

Sosed leži na odmiku (dy, dx), ki praviloma ni celo število (npr. dx = 8,49,
dy = −0,71). Vrednost je utežena vsota 4 najbližjih pikslov; uteži so necela
dela odmika. Funkcija to naredi za vse piksle hkrati: `np.roll` premakne sliko
vodoravno (krožno), indeksiranje z `np.clip` navpično.

Vrne tudi `inside`: za katere vrstice sosed res leži v sliki. Vrstica 0 nima
soseda nad sabo; tak bit bi primerjal piksel s samim sabo, zato ga izločim.

**Preverjeno:** 1000 naključnih pikslov pri 5 odmikih se ujema z ročnim izračunom
po definiciji (`test_ibb.py`, točka 6).

### 4.3 `_ibbLbpDifferences`: jedro LBP

Za p = 0…7 izračuna odmik soseda na elipsi,
`dx = Rx·cos(2πp/P)`, `dy = −Ry·sin(2πp/P)`, in shrani razliko sosed − središče.
LBP bit je predznak te razlike. Razliko (ne samo bita) obdržim, ker iz njene
velikosti kasneje dobim utež.

**Zakaj elipsa 12×1 in ne krog:** tekstura šarenice so pretežno radialne proge
(kripte, brazde). V odviti sliki so to navpične črte, zato se svetlost spreminja
predvsem vodoravno. Sosed levo/desno da informativen bit, sosed gor/dol skoraj
enako vrednost, torej bit, ki ga določa šum.

| Soseščina (P=8, σ=2, z masko) | EER |
|---|---|
| krog R=3 | 0,76 % |
| elipsa 6×2 (široka kotno) | 0,29 % |
| elipsa 12×1 | 0,02 % |
| elipsa 2×4 (široka radialno) | 1,07 % |

**Zakaj P=8, če je radialna smer nepomembna:** na tako sploščeni elipsi
diagonalni sosedje ležijo na (±8,5; ±0,7), skoraj na isti vrstici kot glavna
(±12; 0), a na drugi razdalji. P=8 torej primerja kotno smer na dveh merilih.
P=4 da 0,29 %, P=8 da 0,02 %.

**Zakaj Rx=12:** Rx 10, 12 in 14 dajo 0,04 %, 0,02 % in 0,02 %. Izbral sem
sredino platoja, ne ostrega vrha, ker je to bolj verjetno stabilno na drugih
slikah.

### 4.4 Maska: `_ibbToBool` in `_ibbEstimateMask`

Bit sme šteti samo, če piksel pripada šarenici. Veke, trepalnice in odsevi niso
del identitete, so pa pri vsakem posnetku drugje.

| (P=8, 12×1, σ=2) | EER |
|---|---|
| brez maske | 1,98 % |
| prava (segmentacijska) maska | 0,02 % |

Težava: predloga kliče `extractIBBCode(im_polar)` **brez** maske in iz navodil ni
jasno, ali jo ocenjevalna skripta poda. Zato je `mask` neobvezen argument; če ga
ni (ali je napačne oblike), masko ocenim iz slike:

1. **Prior lege vek.** V odviti sliki sta veki vedno na istem mestu: spodnja pri
   90°, zgornja pri 270°, od zunanjega roba navznoter. Obliko opišem s kosinusno
   kupolo `h·cos(π/2 · d/w)`. Šest parametrov sem prilagodil povprečju pravih
   mask (območje, kjer je šarenica vidna v več kot 90 % slik).
2. **Svetlost.** Znotraj priorja izračunam mediano in MAD (mediano absolutnih
   odstopanj). Piksel obdržim, če od mediane odstopa manj kot 2·MAD. Koža,
   beločnica in odsevi so svetlejši, trepalnice temnejše. Mediana in MAD zato,
   ker nanju ne vplivajo ravno tisti osamelci, ki jih iščem.
3. **Erozija za 2 px**, da LBP na robu ne primerja šarenice z veko.

| Lastna maska | EER |
|---|---|
| brez maske | 1,98 % |
| samo prior | 0,46 % |
| prior + svetlost | 0,19 % |
| prior + svetlost + erozija | 0,15 % |

Lastna maska je namenoma previdna: obdrži približno polovico prave šarenice
(IoU 0,53), a med obdržanimi piksli je le 1,2 % napačnih. Izgubiti nekaj dobrih
bitov je ceneje kot primerjati veko.

### 4.5 `extractIBBCode`: sestava kode

Pet oštevilčenih korakov v kodi:

1. **Kateri piksli so šarenica:** podana maska ali lastna ocena.
2. **LBP razlike** in veljavnost bita = piksel je šarenica **in** sosed je v sliki.
3. **Vsaka 2. vrstica.** Po glajenju s σ=2 sta sosednji vrstici skoraj enaki;
   EER ostane 0,76 % pri vsaki 2. in vsaki 4. vrstici, koda je pol manjša.
4. **Mehke uteži** (samo s podano masko):
   `utež = min(1, |razlika| / (0,5 · mediana |razlik|))`.
   Bit, pri katerem sta sosed in središče skoraj enako svetla, se ob najmanjšem
   šumu obrne. Tak bit šteje manj.

   | | EER | d' |
   |---|---|---|
   | brez uteži | 0,02 % | 5,90 |
   | t=0,5 (izbrano) | 0,03 % | 6,32 |
   | t=2 | 0,15 % | 6,82 |
   | trdo izločanje 10 % najšibkejših bitov | slabše (0,76 → 0,89 %) | |

   Izbral sem zmerno vrednost: d' zraste, EER ostane. Pri t=2 d' še raste, a EER
   se že poslabša (nekaj močnih bitov prevlada).

   **Z lastno masko uteži izklopim.** Najmočnejše razlike v sliki so robovi vek;
   če jih lastna maska zgreši, bi dobili največjo utež (EER 0,15 % → 0,31 %).
5. **Zapis v int8:** `round(127 · predznak · utež · veljavnost)`. Ena matrika
   nosi bit, utež in masko; 0 pomeni »ne upoštevaj«. Ker je koda navaden
   `ndarray`, deluje tudi `np.savez_compressed` v `hdbif.py`.

### 4.6 `matchIBBCodes`: primerjava

Naj bo x = predznak × utež. Pri zamiku s:

- `c(s) = Σ x1·x2` = utež ujemajočih bitov − utež neujemajočih,
- `n(s) = Σ |x1|·|x2|` = skupna utež bitov, ki jih primerjamo,
- `HD(s) = (n − c) / (2n)` = utežen delež bitov, ki se razlikujejo.

Brez uteži in maske je to natanko Hammingova razdalja. Ista šarenica da okoli
0,2, različni okoli 0,42, in 1,0 vrnem, če primerjava ni mogoča (koda je `None`
ali je skupnih veljavnih bitov manj kot 5 %).

**Rotacija očesa.** Nagib glave pomeni krožni zamik odvite slike po stolpcih.
Vzamem najmanjšo razdaljo čez zamike ±16 stolpcev (±11,25°).

| Največji zamik | 0 | ±4 | ±8 | ±16 | ±32 |
|---|---|---|---|---|---|
| EER (R=3) | 4,89 % | 0,88 % | 0,73 % | 0,76 % | 0,88 % |

Prevelik doseg škodi, ker minimum čez več zamikov zniža tudi razdalje lažnih
parov. 8–16 je plato; izbral sem 16, ker je za neznane slike varnejši.

**Zakaj FFT (`_ibbCircularCorrelation`).** `c(s)` in `n(s)` sta križni
korelaciji. Po korelacijskem izreku je korelacija za vse zamike hkrati enaka
`IFFT(FFT(a) · konj(FFT(b)))`. Rezultat je enak zanki z `np.roll` čez zamike
(razlika pod 1e-7, `test_ibb.py`, točka 1), le da je primerjava 5-krat hitrejša
od HDBIF.

**Česa namenoma ni:** Daugmanove normalizacije z √n (poslabša: 0,76 → 1,07 %),
radialnega zamika (d' +0,15, a 3-krat počasneje) in več meril z različnimi Rx
(brez učinka).

---

## 5. Parametri na enem mestu

| Konstanta | Vrednost | Utemeljitev | Meritev |
|---|---|---|---|
| `IBB_P` | 8 | dve merili v kotni smeri | P=4: 0,29 %, P=8: 0,02 % |
| `IBB_RX`, `IBB_RY` | 12, 1 | tekstura je radialna; sredina platoja | krog R=3: 0,76 % |
| `IBB_SIGMA` | 2 | odstrani šum in JPEG | brez: 2,60 % |
| `IBB_ROW_STEP` | 2 | sosednje vrstice so odvečne | brez izgube |
| `IBB_WEIGHT_T` | 0,5 | nestabilni biti štejejo manj | d' 5,90 → 6,32 |
| `IBB_MAX_SHIFT` | 16 | nagib glave do ±11° | brez: 4,89 % |
| `IBB_MIN_OVERLAP` | 0,05 | premalo skupnih bitov = ni primerjave | varovalo |
| `IBB_Z_MAX` | 2 | prag svetlosti v enotah MAD | 2,5: 0,31 % |
| `IBB_ERODE` | 2 | odmik od roba maske | 0,19 → 0,15 % |
| `IBB_EYELIDS` | 2 kupoli | tipična lega vek | samo prior: 0,46 % |

---

## 6. Evalvacija kode

### 6.1 Kako je merjeno

- **Podatki:** vseh 348 slik, predobdelava enaka kot v `hdbif.py`.
- **Protokol:** vsi pari (60 378). Identiteta = oseba + oko.
- **Metrike:** EER (glavna), d', AUC, FNMR pri FMR = 1 % in 0,1 %.
- **Proti prenaučenosti:** osebe razdeljene na dve disjunktni polovici A in B;
  izboljšava mora veljati na obeh. Končna koda: A 0,00 %, B 0,02 %.
- **Končna meritev kliče prave metode razreda** (`eval_class.py`), ne
  raziskovalne kode. Številke obeh se ujemajo.

### 6.2 Trije načini klica

Ker ne vem, kako ocenjevalna skripta kliče funkciji, sem izmeril vse možnosti:

| Klic | EER | d' |
|---|---|---|
| `extractIBBCode(polar, mask)` + `matchIBBCodes(c1, c2)` | 0,03 % | 6,32 |
| `extractIBBCode(polar)` + `matchIBBCodes(c1, c2)` | 0,15 % | 6,14 |
| `extractIBBCode(polar)` + `matchIBBCodes(c1, c2, m1, m2)` | 0,00 % | 6,46 |

### 6.3 Testi pravilnosti (`test_ibb.py`)

| Test | Kaj dokaže |
|---|---|
| FFT primerjava = zanka z `np.roll` (razlika < 1e-7) | FFT je samo pospešitev, ne druga metoda |
| ista slika → razdalja 0; d(a,b) = d(b,a) | razdalja je smiselna |
| slika, zamaknjena za 5 in 16 stolpcev → 0; za 40 → 0,40 | kompenzacija rotacije deluje do meje |
| vzorčenje soseda = ročna bilinearna interpolacija | LBP vzorči prave točke |
| predznak kode = (sosed ≥ središče) | koda je res LBP |
| biti s sosedom izven slike so izločeni | ni lažnih bitov na robu |
| `None`, prazna koda, maska bool / 0-1 / 0-255, maska napačne oblike | koda ne pade pri nenavadnem klicu |
| lastna maska: 1,2 % napačno obdržanih pikslov | lastna maska je previdna |

### 6.4 Pregled kode (code review)

Kaj sem pri pregledu našel in popravil; rezultati so po popravkih **bit za bitom
enaki** kot prej (največja razlika razdalj 0,0 na vseh 60 378 parih).

| Ugotovitev | Popravek |
|---|---|
| `extractIBBCode` je delala preveč stvari naenkrat | razdeljena na `_ibbLbpDifferences` in pet oštevilčenih korakov |
| izbira vrstic `1::2` je bila zapisana na dveh mestih | ena funkcija `_ibbCodeRows`, konstanti `IBB_ROW_START/STEP` |
| maska napačne oblike bi sprožila izjemo | preverjanje oblike, sicer lastna maska |
| vrednost 1,0 za »ni primerjave« je bila raztresena po kodi | konstanta `IBB_NO_MATCH` |
| FFT je bil skrit v vgnezdeni funkciji | samostojna `_ibbCircularCorrelation` z razlago |
| komentarji v slovenščini brez šumnikov, ostala datoteka je angleška | komentarji v angleščini, enotno z datoteko |
| `np.fft` je računal v dvojni natančnosti (6 ms na par) | `torch.fft` v float32 (3 ms na par), enak rezultat |

Česar nisem spreminjal: preostanek datoteke (HDBIF, segmentacija) je nedotaknjen;
dodan je samo blok IBB in dva neobvezna argumenta (`mask=None`).

---

## 7. Omejitve

- **Ni neodvisne testne množice.** Parametri so izbrani na istih slikah, na
  katerih poročam. Delitev A/B to omili, ne odpravi. Na skritem setu pričakujem
  nekoliko slabši rezultat.
- **Malo pravih parov.** En napačen par pomeni 0,15 % EER, zato so intervali
  zaupanja široki in razlike med najboljšimi konfiguracijami niso pomembne.
- **Prior vek velja za pogled naravnost** (slike `_s_`). Pri pogledu gor ali
  vstran so veke drugje; tam ostane le preverjanje svetlosti.
- **Parametri v pikslih** (Rx, σ, zamik) predpostavljajo odvito sliko 64×512,
  kot jo določa `cfg.yaml`.
- **Smer rezultata:** `matchIBBCodes` vrača razdaljo (manjše = bolj podobno),
  enako kot HDBIF. Če ocenjevalna skripta pričakuje podobnost, je treba vrniti
  `1 − razdalja`.

---

## 8. Kako vse ponoviti

```
.venv\Scripts\python.exe dev\build_cache.py                 # segmentacija, enkrat (~3 min)
.venv\Scripts\python.exe dev\test_ibb.py                    # testi pravilnosti
.venv\Scripts\python.exe dev\eval_class.py --method hdbif --save dev\cache\D_hdbif.npy   # izhodišče
.venv\Scripts\python.exe dev\eval_class.py --method ibb --masks --save dev\cache\D_ibb_mask.npy
.venv\Scripts\python.exe dev\eval_class.py --method ibb --save dev\cache\D_ibb_nomask.npy
.venv\Scripts\python.exe dev\eval_class.py --method ibb --match-masks --save dev\cache\D_ibb_matchmask.npy
.venv\Scripts\python.exe dev\eval_bootstrap.py              # intervali zaupanja (bere vse štiri matrike D_*.npy)
.venv\Scripts\python.exe hdbif.py                           # originalna skripta na data/
```

Poskusi po iteracijah so v `exp01_…` do `exp09_…`, vsak z opisom na vrhu datoteke.
