"""Iteracija 5: kaj ce maske NE dobimo? Lastna ocena maske iz same slike.

Predloga klice extractIBBCode(polar) brez maske, zato mora resitev delati tudi
tako. Primerjamo:
  - brez maske
  - statisticni prior: kje je sarenica obicajno prekrita (veki sta pri kotih
    90 in 270 stopinj, od zunanjega roba navznoter)
  - svetlost: koza veke je v rdecem kanalu precej svetlejsa od sarenice
  - oboje skupaj
  - prava (segmentacijska) maska kot zgornja meja
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import research as R  # noqa: E402
from dev.evallib import load_cache, report  # noqa: E402

names, polar_u8, mask_u8, labels, persons = load_cache()
polar = torch.from_numpy(polar_u8).float()
mask = torch.from_numpy(mask_u8 > 127).float()
N, H, W = polar.shape
CFG = dict(P=8, Rx=6, Ry=2, sigma=2)
mean_mask = mask.mean(0)


def bright_mask(ref, z_hi, z_lo=None, sigma=2.0):
    """Veljavni so piksli, ki po svetlosti niso dalec od tipicne sarenice.

    Referenca (mediana, MAD) se izracuna v obmocju `ref`, ki je skoraj vedno sarenica.
    """
    img = R.gaussian_blur(polar, sigma)
    out = torch.empty_like(img)
    for i in range(N):
        vals = img[i][ref]
        med = vals.median()
        mad = (vals - med).abs().median() + 1e-6
        z = (img[i] - med) / mad
        ok = z < z_hi
        if z_lo is not None:
            ok = ok & (z > -z_lo)
        out[i] = ok.float()
    return out


def run(tag, m, **kw):
    cfg = dict(CFG)
    cfg.update(kw)
    report(tag, R.run_bits(polar, m, **cfg), labels, persons, extra=dict(exp="selfmask"))


run("brez maske", None)
run("prava maska (zgornja meja)", mask)
for tau in [0.5, 0.7, 0.8, 0.9, 0.95]:
    prior = (mean_mask > tau).float().expand(N, H, W)
    run(f"prior: povprecna maska > {tau}", prior)
ref = mean_mask > 0.9
for z_hi in [2.0, 2.5, 3.0, 4.0]:
    run(f"svetlost z < {z_hi}", bright_mask(ref, z_hi))
    run(f"svetlost |z| < {z_hi}", bright_mask(ref, z_hi, z_hi))
for tau in [0.5, 0.8]:
    prior = (mean_mask > tau).float().expand(N, H, W)
    for z_hi in [2.5, 3.0, 4.0]:
        run(f"prior > {tau} IN svetlost z < {z_hi}", prior * bright_mask(ref, z_hi))
        run(f"prior > {tau} IN svetlost z < {z_hi}, erozija 2", prior * bright_mask(ref, z_hi), erode=2)
