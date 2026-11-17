"""Iteracija 3: posamicni vplivi parametrov bitne kode (z masko).

Izhodisce: P=8, R=3, sigma=2, zamik +-16. Vsak blok spremeni en parameter.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import research as R  # noqa: E402
from dev.evallib import load_cache, report  # noqa: E402

names, polar_u8, mask_u8, labels, persons = load_cache()
polar = torch.from_numpy(polar_u8).float()
mask = torch.from_numpy(mask_u8 > 127).float()


def run(tag, **kw):
    t0 = time.time()
    D = R.run_bits(polar, mask, **kw)
    report(f"{tag} [{time.time() - t0:.0f}s]", D, labels, persons, extra=kw)


blocks = sys.argv[1:] or ["base", "rs", "P", "aniso", "fragile", "erode", "shift", "norm", "rowstep"]

if "base" in blocks:
    print("== izhodisce ==")
    run("base P8 R3 s2")
if "rs" in blocks:
    print("== radij x glajenje ==")
    for sg in [1.5, 2, 2.5, 3, 4]:
        for r in [2, 3, 4, 5, 6, 8]:
            run(f"R{r} sigma{sg}", Rx=r, Ry=r, sigma=sg)
if "P" in blocks:
    print("== stevilo tock P / CS-LBP ==")
    for P in [4, 6, 12, 16]:
        run(f"P{P}", P=P)
    for P in [8, 16]:
        run(f"CS-LBP P{P}", P=P, cs=True)
if "aniso" in blocks:
    print("== elipticna sosescina (Rx kotno, Ry radialno) ==")
    for rx, ry in [(4, 2), (6, 2), (6, 3), (8, 3), (8, 4), (2, 4), (3, 6)]:
        run(f"Rx{rx} Ry{ry}", Rx=rx, Ry=ry)
if "fragile" in blocks:
    print("== izlocanje nestabilnih bitov ==")
    for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
        run(f"fragile q={q}", fragile_q=q)
if "erode" in blocks:
    print("== erozija maske ==")
    for e in [1, 2, 3, 5, 7]:
        run(f"erode {e}", erode=e)
if "shift" in blocks:
    print("== najvecji zamik ==")
    for s in [0, 4, 8, 12, 20, 24, 32]:
        run(f"max_shift {s}", max_shift=s)
if "norm" in blocks:
    print("== Daugmanova normalizacija rezultata ==")
    for f in [0.5, 0.8, 1.0]:
        run(f"n_ref = {f} * vsi biti", n_ref=f * 8 * 64 * 512)
if "rowstep" in blocks:
    print("== podvzorcenje vrstic kode ==")
    for rs in [2, 4]:
        run(f"row_step {rs}", row_step=rs)
