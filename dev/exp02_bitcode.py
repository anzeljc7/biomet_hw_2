"""Iteracija 2: bitna koda LBP + Hamming. Osnova, zamik, maska, radij, glajenje."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import research as R  # noqa: E402
from dev.evallib import load_cache, report  # noqa: E402

names, polar_u8, mask_u8, labels, persons = load_cache()
polar = torch.from_numpy(polar_u8).float()
mask = torch.from_numpy(mask_u8 > 127).float()

print("== 2a: osnova, zamik, maska ==")
diffs, vrows = R.lbp_diffs(polar, P=8, Rx=1, Ry=1)
c, v = R.bit_code(diffs, vrows)
report("bit P8 R1 brez zamika, brez maske", R.hamming_all_pairs(c, v, 0), labels, persons)
report("bit P8 R1 zamik +-16, brez maske", R.hamming_all_pairs(c, v, 16), labels, persons)
c, v = R.bit_code(diffs, vrows, mask=mask)
report("bit P8 R1 zamik +-16, +maska", R.hamming_all_pairs(c, v, 16), labels, persons)

print("== 2b: radij in glajenje (zamik +-16), brez maske / z masko ==")
for Rr in [2, 3, 4, 6, 8]:
    for sigma in [0, 1, 2, 3]:
        diffs, vrows = R.lbp_diffs(polar, P=8, Rx=Rr, Ry=Rr, sigma=sigma)
        c, v = R.bit_code(diffs, vrows)
        report(f"bit P8 R{Rr} sigma{sigma} brez maske", R.hamming_all_pairs(c, v, 16), labels, persons)
        c, v = R.bit_code(diffs, vrows, mask=mask)
        report(f"bit P8 R{Rr} sigma{sigma} +maska", R.hamming_all_pairs(c, v, 16), labels, persons)
