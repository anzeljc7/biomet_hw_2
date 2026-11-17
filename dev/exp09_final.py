"""Iteracija 9: izbira koncne konfiguracije v vseh nacinih uporabe.

Nacini:  M = podana prava maska,  S = lastna maska (kupoli IN svetlost, erozija 2),
         MB = prava maska IN svetlost
Za vsak nacin: utezi t in nekaj sosednjih konfiguracij (preverjamo, da je izbira
na platoju in ne na ostrem vrhu). Vsaka 2. vrstica kode.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import research as R  # noqa: E402
from dev.evallib import load_cache, report  # noqa: E402
from dev.exp08_helpers import bright_ok, dome_mask  # noqa: E402

names, polar_u8, mask_u8, labels, persons = load_cache()
polar = torch.from_numpy(polar_u8).float()
mask = torch.from_numpy(mask_u8 > 127).float()
N, H, W = polar.shape
X = dict(exp="final9")
DOMES = [(132, 88, 30), (380, 128, 56)]
pm = dome_mask(DOMES)
self_mask = R.erode_mask(pm.expand(N, H, W) * bright_ok(polar, pm, 2.0), 2)
mask_bright = mask * bright_ok(polar, pm, 2.0)
MODES = {"M": mask, "S": self_mask, "MB": mask_bright}

cfgs = [(12, 1, 2), (10, 1, 2), (14, 1, 2), (12, 2, 2), (12, 1, 1.5)]
for rx, ry, sg in cfgs:
    diffs, vrows = R.lbp_diffs(polar, P=8, Rx=rx, Ry=ry, sigma=sg)
    rows = slice(1, None, 2)
    mag = diffs.abs()[:, :, rows]
    med = mag.reshape(N, -1).median(dim=1).values.view(N, 1, 1, 1)
    for mode, m in MODES.items():
        c, v = R.bit_code(diffs, vrows, mask=m, rows=rows)
        for t in [0, 0.5, 1, 2]:
            vv = v if t == 0 else v * (mag / (t * med)).clamp(max=1.0).reshape(v.shape)
            report(f"[{mode:>2s}] Rx{rx} Ry{ry} s{sg} t={t}", R.hamming_all_pairs(c, vv, 16), labels, persons,
                   extra=dict(X, mode=mode, Rx=rx, Ry=ry, sigma=sg, t=t))
