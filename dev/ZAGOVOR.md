# Priprava na zagovor

Oddaš samo `modules/irisRecognition.py`. Vse številke spodaj so iz
`dev/DNEVNIK.md` (podrobnosti in tabele po iteracijah).

## Rešitev v enem odstavku

Odvito šarenico (64×512) zgladim z Gaussom (σ=2), nato za vsak piksel izračunam
LBP z 8 sosedi na sploščeni elipsi (12 px v kotni, 1 px v radialni smeri). LBP
bitov ne histogramiram, ampak jih pustim na mestu: koda je matrika bitov
[8, 32, 512]. Dve kodi primerjam z uteženo Hammingovo razdaljo (delež bitov, ki
se razlikujejo) in vzamem minimum čez krožne zamike ±16 stolpcev, da kompenziram
nagib glave. Bite na vekah izločim z masko; če maske ne dobim, jo ocenim sam.

## Rezultati

| | EER | d' |
|---|---|---|
| HDBIF (izhodišče) | 0,14 % | 5,22 |
| moj LBP, s podano masko | 0,03 % | 6,32 |
| moj LBP, brez maske (lastna ocena) | 0,15 % | 6,14 |
| klasični LBP po Ojali (histogrami po blokih) | 7,3 % | 3,0 |

348 slik, 80 identitet, 655 pravih in 59 723 lažnih parov.

## Pot do rešitve (kaj je koliko prineslo)

| Korak | EER |
|---|---|
| LBP P=8, R=1, en globalni histogram | 25,8 % |
| histogrami po blokih 8×16 + zamik + maska | 7,3 % |
| bitna koda + Hamming, brez zamika, brez maske | 12,7 % |
| + krožni zamik ±16 | 7,3 % |
| + maska | 3,2 % |
| + Gaussovo glajenje σ=2 (R=3) | 0,76 % |
| + eliptična soseščina Rx=12, Ry=1 | 0,02 % |
| + mehke uteži bitov (t=0,5) | 0,03 % (d' 5,9 → 6,3) |
| brez maske: 1,98 % → z lastno masko | 0,15 % |

## Vprašanja, ki jih lahko pričakuješ

**Kaj je LBP?**
Za vsak piksel pogledam P sosedov na krogu s polmerom R. Vsak sosed da en bit:
1, če je svetlejši ali enak središču, sicer 0. Dobim P-bitno oznako, ki opisuje
lokalno mikrostrukturo (rob, kot, pega, ravnina). Ker gledam samo *predznak*
razlike, je LBP neodvisen od monotone spremembe osvetlitve (svetlejša ali
kontrastnejša slika da iste bite).

**Zakaj bilinearna interpolacija?**
Sosedje na krogu/elipsi ne padejo na cele koordinate piksla (npr. 8,49; −0,71).
Vrednost sestavim iz 4 najbližjih pikslov, uteženih z razdaljo (`_ibbSampleNeighbour`).

**Zakaj ne histogrami, kot v članku?**
Ojala rešuje klasifikacijo tekstur: zanima ga, *katere* mikrostrukture so v
sliki, ne *kje*. Pri šarenici je identiteta ravno v razporeditvi. Histogram
položaj zavrže. Izmerjeno: globalni histogram 25,8 %, manjši ko so bloki, boljše
je (10,8 % → 7,3 %); bitna koda je skrajni primer (blok = 1 piksel) in da 0,8 %.
Ista ideja kot Daugmanova koda šarenice, le da so biti iz LBP in ne iz Gaborja.

**Zakaj uniformni vzorci niso v končni rešitvi?**
Uniformni vzorci (≤ 2 prehoda 0/1) zmanjšajo histogram z 256 na 59 košev. Pri
histogramih sem jih preizkusil (enak EER, 4× manjši histogram). Pri bitni kodi
histograma ni, zato niso potrebni.

**Zakaj glajenje?**
Slike so iz telefona, v vidni svetlobi, JPEG. Na surovih pikslih LBP primerja
šum senzorja in JPEG bloke, ki so pri vsakem posnetku drugačni. Po glajenju
primerja dejansko teksturo. EER 2,6 % → 0,8 %. σ=3 je že preveč (briše teksturo).

**Zakaj elipsa 12×1 in ne krog?**
Tekstura šarenice so pretežno radialne proge. V odviti sliki so to navpične
črte, torej se svetlost spreminja predvsem vodoravno (v kotni smeri). Sosedje
levo/desno dajo informativne bite, sosedje gor/dol skoraj enake vrednosti (šum).
Izmerjeno: krog R=3 0,76 %, elipsa 6×2 0,29 %, elipsa 12×1 0,02 %; obratna
elipsa 2×4 je slabša od kroga (1,07 %).

**Zakaj P=8, če radialna smer ni pomembna?**
Na sploščeni elipsi diagonalni sosedje ležijo na (±8,5; ±0,7): skoraj ista
vrstica, a druga razdalja kot glavna soseda (±12; 0). Dobim dve merili v kotni
smeri. P=4 da 0,29 %, P=8 da 0,02 %.

**Kako kompenziraš rotacijo očesa?**
Nagib glave pomeni krožni zamik odvite slike po vodoravni osi. Kodo zamaknem za
−16…+16 stolpcev in vzamem najmanjšo razdaljo. ±16 od 512 stolpcev = ±11,25°.
Brez zamika 12,7 %, z zamikom 7,3 % (že pri osnovnem LBP). Večji doseg škodi,
ker minimum čez več zamikov zniža tudi razdalje lažnih parov.

**Zakaj FFT pri primerjavi?**
Razdaljo pri zamiku s dobim iz dveh vsot: c(s) = Σ x1·x2(zamaknjen) in
n(s) = Σ |x1|·|x2(zamaknjen)|. To sta križni korelaciji. Korelacijski izrek:
korelacija za *vse* zamike naenkrat = IFFT(FFT(a) · konj(FFT(b))). Rezultat je
identičen zanki z `np.roll` (preverjeno v `dev/test_ibb.py`, razlika < 1e-7),
le hitrejši: 3 ms/par (HDBIF 15 ms/par).

**Kako točno izračunaš razdaljo?**
x = predznak × utež (v [−1, 1], 0 = neveljaven bit). Pri zamiku s:
c = utež ujemajočih − utež neujemajočih bitov, n = skupna utež primerjanih.
HD = (n − c) / (2n) = delež (uteži) bitov, ki se razlikujejo. Ista šarenica → ~0,2,
različni → ~0,42, primerjava nemogoča → 1.

**Kaj so mehke uteži?**
Bit, pri katerem sta sosed in središče skoraj enako svetla, se ob najmanjšem
šumu obrne (nestabilen bit). Utež = min(1, |razlika| / (0,5 · mediana |razlik|)).
Trdo izločanje takih bitov je škodilo (0,76 → 0,89 %), mehka utež pomaga
(d' 5,9 → 6,3 pri istem EER). Uporabim jih samo s podano masko: brez nje bi
robovi vek, ki jih lastna maska zgreši, dobili največjo utež (0,15 → 0,31 %).

**Kako deluje lastna maska?**
(1) Prior: veki sta v odviti sliki vedno pri 90° in 270°, od zunanjega roba
navznoter; obliko opišem s kosinusno kupolo. (2) Svetlost: znotraj priorja
izračunam mediano in MAD; piksel zavržem, če odstopa za več kot 2·MAD (koža in
odsevi svetlejši, trepalnice temnejše). (3) Erozija 2 px. EER 1,98 % → 0,15 %.

**Zakaj MAD in ne standardni odklon?**
Mediana in MAD sta robustna: nanju ne vplivajo ravno tisti osamelci (veka,
odsev), ki jih želim odkriti.

**Kaj je EER, d', FMR, FNMR?**
FMR = delež lažnih parov, ki jih sistem sprejme. FNMR = delež pravih parov, ki
jih zavrne. EER = napaka pri pragu, kjer sta enaka. d' = razdalja med povprečjema
porazdelitev pravih in lažnih parov, deljena s skupnim standardnim odklonom.

**Kako si optimiziral parametre in kako veš, da nisi prenaučil?**
Vsak parameter najprej posamično (iteracija 3), nato skupna mreža 144
kombinacij (iteracija 4), nato končna primerjava sosednjih konfiguracij
(iteracija 9). Izbral sem sredino platoja (Rx 10–14 dajo podobno), ne ostrega
vrha. Osebe sem razdelil na dve disjunktni polovici in preverjal, da izboljšava
velja na obeh. Pošteno: to ni neodvisna testna množica, in pri 655 pravih parih
en par pomeni 0,15 % EER, zato razlik pod to mejo ne jemljem resno.

**Si boljši od HDBIF?**
Na teh podatkih primerljiv: 0,03 % proti 0,14 % je razlika enega para, 95 % intervala
zaupanja (0,00–0,09 % in 0,00–0,31 %) se prekrivata. HDBIF
uporablja naučene filtre BSIF, jaz ročno izbran LBP, prilagojen tej bazi.

**Kaj bi še izboljšal?**
Radialni zamik (toleranca na napako radija zenice; izmerjeno d' +0,15, a 3×
počasneje), prior vek za druge smeri pogleda, validacija na neodvisni bazi.

## Kaj je kje v kodi

| Funkcija | Naloga |
|---|---|
| `IBB_*` konstante | vsi parametri na enem mestu |
| `_ibbBlur` | Gaussovo glajenje, vodoravno krožno |
| `_ibbSampleNeighbour` | vrednost soseda z bilinearno interpolacijo |
| `_ibbLbpDifferences` | jedro LBP: razlika sosed − središče za 8 sosedov na elipsi |
| `_ibbToBool` | maska 0/255, 0/1 ali bool → bool |
| `_ibbEstimateMask` | lastna ocena maske (prior + svetlost + erozija) |
| `_ibbCodeRows` | izbira vrstic, ki gredo v kodo (vsaka 2.) |
| `extractIBBCode` | glajenje → maska → LBP razlike → uteži → int8 koda |
| `_ibbSignedWeights` | int8 koda → [−1, 1], po želji maska ob primerjavi |
| `_ibbCircularCorrelation` | križna korelacija za vse zamike (FFT) |
| `matchIBBCodes` | utežen Hamming, minimum čez zamike ±16 |

Podrobna razlaga vsake funkcije z utemeljitvijo in meritvami je v `RAZLAGA.md`.

## Odprto vprašanje za asistenta

Iz navodil ni jasno, kako ocenjevalna skripta kliče funkciji. Koda podpira vse
tri možnosti (`mask` je neobvezen), vendar je s podano masko rezultat boljši
(0,03 % proti 0,15 %). Vredno vprašati: **ali se maska poda, in ali `matchIBBCodes`
vrača razdaljo (manjše = bolj podobno), kot HDBIF?** Trenutno vrača razdaljo.
