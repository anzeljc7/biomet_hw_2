"""Skupna orodja za evalvacijo: nalaganje predpomnilnika, oznake identitet, metrike.

Vse metrike predpostavljajo RAZDALJO (manjse = bolj podobno), enako kot HDBIF.
"""
import json
import os
import time

import numpy as np

CACHE_PATH = os.path.join("dev", "cache", "polar.npz")
RESULTS_PATH = os.path.join("dev", "results.jsonl")


def load_cache(path=CACHE_PATH):
    """Vrne (names, polar[N,64,512] uint8, mask[N,64,512] uint8, labels[N], persons[N]).

    Ime datoteke je '<oseba>_<oseba><L|R>_<pogled>_<zaporedna>.jpg' (pogled: s = naravnost).
    Identiteta (razred) je oseba + oko, ker sta levo in desno oko razlicni sarenici. `persons`
    je samo oseba in ga rabimo za delitev na disjunktni podmnozici.
    """
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    # Prvi dve polji imena. Delitev po '_s_' bi sliko z drugo smerjo pogleda ('_u_', '_l_', '_r_')
    # tiho razglasila za svojo identiteto, njeni pravi pari pa bi steli kot lazni.
    eye_ids = ["_".join(n.split("_")[:2]) for n in names]   # npr. '10_10L'
    person_ids = [n.split("_")[0] for n in names]           # npr. '10'
    labels = np.unique(eye_ids, return_inverse=True)[1]
    persons = np.unique(person_ids, return_inverse=True)[1]
    return names, z["polar"], z["mask"], labels, persons


def split_pairs(D, labels, subset=None):
    """Iz matrike razdalj D[N,N] vrne (genuine, impostor) razdalje za pare i<j.

    `subset` je logicna maska slik; par se uposteva samo, ce sta obe sliki v njej.
    """
    n = len(labels)
    iu, ju = np.triu_indices(n, k=1)
    if subset is not None:
        keep = subset[iu] & subset[ju]
        iu, ju = iu[keep], ju[keep]
    d = D[iu, ju]
    same = labels[iu] == labels[ju]
    return d[same], d[~same]


def compute_metrics(gen, imp):
    """EER, AUC, d' in FNMR pri fiksnem FMR iz genuine/impostor razdalj."""
    gen = np.sort(np.asarray(gen, dtype=np.float64))
    imp = np.sort(np.asarray(imp, dtype=np.float64))
    # Kandidati za prag so vse opazene razdalje. Par sprejmemo, ce je razdalja <= prag.
    thr = np.unique(np.concatenate([gen, imp]))
    fmr = np.searchsorted(imp, thr, side="right") / len(imp)        # impostorji, ki so sprejeti
    fnmr = 1.0 - np.searchsorted(gen, thr, side="right") / len(gen)  # genuine, ki so zavrnjeni
    k = np.argmin(np.abs(fmr - fnmr))
    eer = (fmr[k] + fnmr[k]) / 2.0

    def fnmr_at(target_fmr):
        ok = np.where(fmr <= target_fmr)[0]
        return 1.0 if len(ok) == 0 else fnmr[ok[-1]]

    # AUC = verjetnost, da ima nakljucen genuine par manjso razdaljo kot nakljucen impostor.
    less = np.searchsorted(imp, gen, side="left")     # impostorji strogo manjsi od genuine
    leq = np.searchsorted(imp, gen, side="right")
    auc = 1.0 - (less.sum() + 0.5 * (leq - less).sum()) / (len(gen) * len(imp))
    dprime = abs(gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()))
    return {
        "eer": float(eer),
        "auc": float(auc),
        "dprime": float(dprime),
        "fnmr@fmr1%": float(fnmr_at(0.01)),
        "fnmr@fmr0.1%": float(fnmr_at(0.001)),
        "thr_eer": float(thr[k]),
        "n_gen": int(len(gen)),
        "n_imp": int(len(imp)),
    }


def finite_distances(D):
    """Nadomesti inf/nan ("primerjava ni mogoca", npr. HDBIF pri premajhni maski) z vrednostjo,
    ki je vecja od vseh pravih razdalj, da tak par steje kot zavrnjen.

    Uporabljata jo evaluate() in eval_bootstrap.py, da obe tabeli racunata iz istih stevil.
    """
    finite = np.isfinite(D)
    return np.where(finite, D, np.nanmax(D[finite]) + 1.0)


def evaluate(D, labels, persons):
    """Metrike na vseh parih ter na dveh podmnozicah z disjunktnimi osebami.

    Podmnozici (soda/liha oseba) sluzita preverjanju prenaucenosti: parametre
    izbiramo na A, potrdimo na B.
    """
    D = finite_distances(D)
    fold_a = persons % 2 == 0
    out = {"all": compute_metrics(*split_pairs(D, labels))}
    out["A"] = compute_metrics(*split_pairs(D, labels, fold_a))
    out["B"] = compute_metrics(*split_pairs(D, labels, ~fold_a))
    return out


def report(name, D, labels, persons, extra=None, log=True):
    """Izpise eno vrstico z rezultati in jo po zelji doda v dev/results.jsonl."""
    m = evaluate(D, labels, persons)
    a = m["all"]
    print(
        f"{name:<58s} EER {100 * a['eer']:6.2f}%  (A {100 * m['A']['eer']:5.2f} | B {100 * m['B']['eer']:5.2f})"
        f"  AUC {a['auc']:.4f}  d' {a['dprime']:.2f}  FNMR@1% {100 * a['fnmr@fmr1%']:5.1f}%"
        f"  FNMR@0.1% {100 * a['fnmr@fmr0.1%']:5.1f}%",
        flush=True,
    )
    if log:
        rec = {"name": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": m}
        if extra:
            rec["params"] = extra
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return m
