"""Iteracija 7: natancneje o mehkih utezeh bitov in kombinacije z ostalimi dodatki.

Utez bita = min(1, |razlika| / (t * mediana |razlik| v sliki)).
t -> 0 pomeni navadni Hamming (vse utezi 1), t -> inf pomeni utez sorazmerno |razliki|.
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
X = dict(exp="soft7")
blocks = sys.argv[1:] or ["t", "combo", "params"]


def soft_codes(P=4, Rx=10, Ry=2, sigma=2, t=2.0, rows=None, use_mask=True, med_from_mask=False):
    diffs, vrows = R.lbp_diffs(polar, P=P, Rx=Rx, Ry=Ry, sigma=sigma)
    c, v = R.bit_code(diffs, vrows, mask=mask if use_mask else None, rows=rows)
    mag = diffs.abs()
    if rows is not None:
        mag = mag[:, :, rows]
    K, Hn = mag.shape[1], mag.shape[2]
    if t is not None:
        flat = mag.reshape(N, -1)
        if med_from_mask:       # mediana samo cez veljavne bite
            med = torch.stack([flat[i][v.reshape(N, -1)[i] > 0].median() for i in range(N)])
        else:
            med = flat.median(dim=1).values
        w = (mag / (t * med.view(N, 1, 1, 1))).clamp(max=1.0)
    else:
        w = mag / mag.reshape(N, -1).mean(dim=1).view(N, 1, 1, 1)
    return c, v * w.reshape(N, K * Hn, W), K, Hn


if "t" in blocks:
    for t in [2, 3, 4, 6, 10, None]:
        c, v, K, Hn = soft_codes(t=t)
        report(f"mehke utezi t={t}", R.hamming_all_pairs(c, v, 16), labels, persons, extra=X)
    c, v, K, Hn = soft_codes(t=3, med_from_mask=True)
    report("mehke utezi t=3, mediana samo znotraj maske", R.hamming_all_pairs(c, v, 16), labels, persons, extra=X)

if "combo" in blocks:
    for t in [2, 4]:
        c, v, K, Hn = soft_codes(t=t)
        report(f"t={t} + radialni zamik +-1", R.hamming_radial(c, v, K, Hn, 16, 1), labels, persons, extra=X)
        report(f"t={t} + max_shift 8", R.hamming_all_pairs(c, v, 8), labels, persons, extra=X)
        report(f"t={t} + max_shift 12", R.hamming_all_pairs(c, v, 12), labels, persons, extra=X)
        report(f"t={t} + max_shift 24", R.hamming_all_pairs(c, v, 24), labels, persons, extra=X)
        for rs in [2, 4]:
            c2, v2, K, Hn = soft_codes(t=t, rows=slice(rs // 2, None, rs))
            report(f"t={t} + vsaka {rs}. vrstica", R.hamming_all_pairs(c2, v2, 16), labels, persons, extra=X)
        c, v, K, Hn = soft_codes(t=t, use_mask=False)
        report(f"t={t} BREZ maske", R.hamming_all_pairs(c, v, 16), labels, persons, extra=X)

if "params" in blocks:
    # ali se optimum P / Rx / Ry / sigma premakne, ko so utezi vklopljene?
    for kw in [dict(P=8), dict(P=2), dict(Rx=6), dict(Rx=8), dict(Rx=12), dict(Rx=16), dict(Ry=1), dict(Ry=3),
               dict(Ry=4), dict(sigma=1), dict(sigma=1.5), dict(sigma=3), dict(Rx=3, Ry=3, P=8)]:
        c, v, K, Hn = soft_codes(t=3, **kw)
        report(f"t=3 + {kw}", R.hamming_all_pairs(c, v, 16), labels, persons, extra=X)
