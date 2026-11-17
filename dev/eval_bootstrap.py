"""Intervali zaupanja za EER in d' z bootstrapom po osebah.

EER 0,03 % proti 0,14 % zveni kot velika razlika, a pravih parov je samo 655.
Bootstrap pove, koliko bi se rezultat spremenil, ce bi imeli druge osebe:
1000-krat nakljucno (s ponavljanjem) izberemo osebe in ponovno izracunamo metrike.
Vzorcimo OSEBE in ne parov, ker pari iste osebe niso neodvisni.

Zagon:  .venv\\Scripts\\python.exe dev\\eval_bootstrap.py
(potrebuje matrike razdalj, ki jih shrani dev/eval_class.py --save)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.evallib import compute_metrics, finite_distances, load_cache  # noqa: E402

RUNS = [
    ("HDBIF (izhodisce)", "dev/cache/D_hdbif.npy"),
    ("LBP, maska ob ekstrakciji", "dev/cache/D_ibb_mask.npy"),
    ("LBP, brez maske (lastna)", "dev/cache/D_ibb_nomask.npy"),
    ("LBP, maska samo ob primerjavi", "dev/cache/D_ibb_matchmask.npy"),
]
N_BOOT = 1000

names, polar, mask, labels, persons = load_cache()
n = len(names)
iu, ju = np.triu_indices(n, k=1)
same = labels[iu] == labels[ju]
person_ids = np.unique(persons)
rng = np.random.default_rng(0)
# vsak bootstrap vzorec: kolikokrat je bila izbrana vsaka oseba
draws = [np.bincount(rng.choice(len(person_ids), len(person_ids)), minlength=len(person_ids))
         for _ in range(N_BOOT)]

print(f"{'metoda':<32s} {'EER':>7s}  {'95 % interval':>17s}   {'d-prime':>7s}  {'95 % interval':>13s}")
for name, path in RUNS:
    if not os.path.exists(path):
        continue
    D = finite_distances(np.load(path))
    d = D[iu, ju]
    point = compute_metrics(d[same], d[~same])
    eers, dps = [], []
    for cnt in draws:
        # par (i, j) se v vzorcu pojavi cnt[oseba_i] * cnt[oseba_j]-krat
        w = cnt[persons[iu]] * cnt[persons[ju]]
        g = np.repeat(d[same], w[same])
        i = np.repeat(d[~same], w[~same])
        m = compute_metrics(g, i)
        eers.append(m["eer"])
        dps.append(m["dprime"])
    lo, hi = np.percentile(eers, [2.5, 97.5])
    dlo, dhi = np.percentile(dps, [2.5, 97.5])
    print(f"{name:<32s} {100 * point['eer']:6.2f}%  [{100 * lo:5.2f}, {100 * hi:5.2f}] %   "
          f"{point['dprime']:7.2f}  [{dlo:.2f}, {dhi:.2f}]", flush=True)
