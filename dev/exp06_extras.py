"""Iteracija 6: dodatki nad najboljso konfiguracijo iz grida (P4, Rx10, Ry2, sigma2, z masko).

  a) najvecji kotni zamik
  b) radialni zamik (toleranca na napako radija zenice/sarenice)
  c) vec meril hkrati (zdruzitev kod z razlicnim Rx)
  d) mehke utezi bitov (namesto trdega izlocanja nestabilnih bitov)
  e) podvzorcenje vrstic (manjsa predloga, hitrejse ujemanje)
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
BASE = dict(P=4, Rx=10, Ry=2, sigma=2)
X = dict(exp="extras6")
blocks = sys.argv[1:] or ["shift", "radial", "multi", "soft", "rowstep"]


def codes(rows=None, **kw):
    cfg = dict(BASE)
    cfg.update(kw)
    diffs, vrows = R.lbp_diffs(polar, P=cfg["P"], Rx=cfg["Rx"], Ry=cfg["Ry"], sigma=cfg["sigma"])
    c, v = R.bit_code(diffs, vrows, mask=mask, rows=rows)
    return c, v, diffs


c0, v0, d0 = codes()
report("base P4 Rx10 Ry2 s2", R.hamming_all_pairs(c0, v0, 16), labels, persons, extra=X)

if "shift" in blocks:
    for s in [6, 8, 10, 12, 20, 24]:
        report(f"max_shift {s}", R.hamming_all_pairs(c0, v0, s), labels, persons, extra=X)
if "radial" in blocks:
    for dy in [1, 2, 3]:
        report(f"radialni zamik +-{dy}", R.hamming_radial(c0, v0, 4, 64, 16, dy), labels, persons, extra=X)
if "multi" in blocks:
    for rxs in [(5, 10), (6, 12), (4, 8, 16), (10, 20)]:
        cs, vs = zip(*[codes(Rx=rx)[:2] for rx in rxs])
        report(f"vec meril Rx={rxs}", R.hamming_all_pairs(torch.cat(cs, 1), torch.cat(vs, 1), 16),
               labels, persons, extra=X)
    cs, vs = zip(*[codes(Rx=10, sigma=sg)[:2] for sg in (1.5, 3)])
    report("vec meril sigma=(1.5,3)", R.hamming_all_pairs(torch.cat(cs, 1), torch.cat(vs, 1), 16),
           labels, persons, extra=X)
if "soft" in blocks:
    N = polar.shape[0]
    mag = d0.abs()
    med = mag.reshape(N, -1).median(dim=1).values.view(N, 1, 1, 1)
    for t in [0.25, 0.5, 1.0, 2.0]:
        w = (mag / (t * med)).clamp(max=1.0).reshape(N, -1, 512)
        report(f"mehke utezi |d| / ({t} * mediana)", R.hamming_all_pairs(c0, v0 * w, 16), labels, persons, extra=X)
if "rowstep" in blocks:
    for rs in [2, 4]:
        c, v, _ = codes(rows=slice(rs // 2, None, rs))
        report(f"vsaka {rs}. vrstica", R.hamming_all_pairs(c, v, 16), labels, persons, extra=X)
