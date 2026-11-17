"""Iteracija 1: klasicni LBP (histogrami) proti bitni kodi LBP.

Vprasanje: ali LBP oznake histogramiramo (Ojala) ali bite pustimo na mestu in
primerjamo s Hammingovo razdaljo (kot Daugmanova koda sarenice)?
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

print("== 1a: klasicni LBP, histogrami po blokih + chi2 ==")
diffs, vrows = R.lbp_diffs(polar, P=8, Rx=1, Ry=1)
lab, nb = R.lbp_labels(diffs, uniform=False)
h = R.block_histograms(lab, nb, 64, 512, 512)
report("hist P8 R1 full-256bin, 1 globalni histogram", R.chi2_all_pairs(h, 0), labels, persons)
lab, nb = R.lbp_labels(diffs, uniform=True)
h = R.block_histograms(lab, nb, 64, 512, 512)
report("hist P8 R1 uniform, 1 globalni histogram", R.chi2_all_pairs(h, 0), labels, persons)
for (P, Rr) in [(8, 1), (8, 2), (8, 3)]:
    diffs, vrows = R.lbp_diffs(polar, P=P, Rx=Rr, Ry=Rr)
    lab, nb = R.lbp_labels(diffs, uniform=True)
    for (bh, bw, step, ms) in [(16, 32, 32, 0), (16, 32, 8, 2), (8, 16, 8, 2), (8, 16, 4, 4), (16, 16, 4, 4)]:
        h = R.block_histograms(lab, nb, bh, bw, step)
        report(f"hist P{P} R{Rr} u2 blok {bh}x{bw} korak {step} zamik +-{ms * step}px",
               R.chi2_all_pairs(h, ms), labels, persons)
        h = R.block_histograms(lab, nb, bh, bw, step, mask=mask)
        report(f"hist P{P} R{Rr} u2 blok {bh}x{bw} korak {step} zamik +-{ms * step}px +maska",
               R.chi2_all_pairs(h, ms), labels, persons)

print("== 1b: bitna koda LBP + Hamming ==")
for (P, Rr) in [(8, 1), (8, 2), (8, 3), (8, 4)]:
    diffs, vrows = R.lbp_diffs(polar, P=P, Rx=Rr, Ry=Rr)
    c, v = R.bit_code(diffs, vrows)
    report(f"bit  P{P} R{Rr} brez zamika, brez maske", R.hamming_all_pairs(c, v, 0), labels, persons)
    report(f"bit  P{P} R{Rr} zamik +-16, brez maske", R.hamming_all_pairs(c, v, 16), labels, persons)
    c, v = R.bit_code(diffs, vrows, mask=mask)
    report(f"bit  P{P} R{Rr} zamik +-16, +maska", R.hamming_all_pairs(c, v, 16), labels, persons)
