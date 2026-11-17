"""Iteracija 4: skupno iskanje (grid) po elipticni sosescini, glajenju in P (z masko).

Iteracija 3 je pokazala, da je siroka sosescina v kotni smeri (Rx > Ry) in
manj tock (P=4) boljse. Tu parametre preiscemo skupaj, ker so med seboj odvisni
(vecji radij potrebuje manj glajenja in obratno).
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import research as R  # noqa: E402
from dev.evallib import load_cache, report  # noqa: E402

names, polar_u8, mask_u8, labels, persons = load_cache()
polar = torch.from_numpy(polar_u8).float()
mask = torch.from_numpy(mask_u8 > 127).float()

Ps = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4, 8]
for P, sg, ry, rx in itertools.product(Ps, [1.5, 2, 3], [1, 2, 3, 4], [4, 6, 8, 10, 12, 16]):
    D = R.run_bits(polar, mask, P=P, Rx=rx, Ry=ry, sigma=sg)
    report(f"grid P{P} Rx{rx} Ry{ry} sigma{sg}", D, labels, persons,
           extra=dict(P=P, Rx=rx, Ry=ry, sigma=sg, exp="grid4"))
