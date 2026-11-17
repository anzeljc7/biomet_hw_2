"""Iteracija 8: parametricna lastna maska (brez podatkov iz ucne mnozice v kodi).

Iteracija 5: najboljsi nadomestek prave maske je strog statisticni prior
(povprecna maska > 0.9). Povprecne maske ne moremo oddati kot tabelo, zato jo
opisemo z dvema "kupolama" (spodnja veka pri ~90 st., zgornja pri ~270 st.):

    odrezano(c) = h * cos(pi/2 * d / w)   za |d| < w,   d = krozna razdalja stolpca c do sredisca kupole
    piksel (r, c) je veljaven, ce  r0 <= r < H - odrezano(c)

Skripta (1) prilagodi parametre kupol povprecni maski, (2) izmeri EER brez maske.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import research as R  # noqa: E402
from dev.evallib import load_cache, report  # noqa: E402
from dev.exp08_helpers import bright_ok, dome_mask  # noqa: E402

names, polar_u8, mask_u8, labels, persons = load_cache()
polar = torch.from_numpy(polar_u8).float()
mask = torch.from_numpy(mask_u8 > 127).float()
N, H, W = polar.shape
mean_mask = mask.mean(0)
X = dict(exp="selfmask8")


def fit(tau):
    target = (mean_mask > tau).float()
    best = []
    for lo, hi, cs in [(0, 256, range(96, 176, 4)), (256, 512, range(352, 432, 4))]:
        b = (1e9, None)
        for c, w, h in itertools.product(cs, range(40, 200, 8), range(4, 64, 2)):
            m = dome_mask([(c, w, h)], r0=0)
            err = (m[1:, lo:hi] != target[1:, lo:hi]).float().mean().item()
            if err < b[0]:
                b = (err, (c, w, h))
        best.append(b)
    return [b[1] for b in best], [b[0] for b in best]


def run(tag, m, t, erode=0):
    diffs, vrows = R.lbp_diffs(polar, P=8, Rx=12, Ry=1, sigma=2)
    rows = slice(1, None, 2)
    c, v = R.bit_code(diffs, vrows, mask=m, erode=erode, rows=rows)
    if t:
        mag = diffs.abs()[:, :, rows]
        med = mag.reshape(N, -1).median(dim=1).values.view(N, 1, 1, 1)
        v = v * (mag / (t * med)).clamp(max=1.0).reshape(v.shape)
    report(f"{tag} | t={t} erozija={erode}", R.hamming_all_pairs(c, v, 16), labels, persons, extra=X)


for t in [0, 2]:
    run("prava maska", mask, t)
    run("brez maske", None, t)

for tau in [0.8, 0.9, 0.95]:
    domes, errs = fit(tau)
    print(f"tau={tau}: kupoli (sredisce, polsirina, visina) = {domes}, napaka prileganja = {np.round(errs, 3)}")
    pm = dome_mask(domes)
    emp = (mean_mask > tau).float()
    for t in [0, 2]:
        run(f"empiricni prior > {tau}", emp.expand(N, H, W), t)
        run(f"kupoli (tau {tau})", pm.expand(N, H, W), t)
        for z in [2.0, 2.5, 3.0]:
            for e in [0, 2]:
                run(f"kupoli (tau {tau}) IN |z| < {z}", pm.expand(N, H, W) * bright_ok(polar, pm, z), t, erode=e)
