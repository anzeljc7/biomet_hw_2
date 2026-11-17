"""Preverjanje pravilnosti koncne kode (extractIBBCode / matchIBBCodes).

Zagon iz korena projekta:  .venv\\Scripts\\python.exe dev\\test_ibb.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.evallib import load_cache  # noqa: E402
from modules.irisRecognition import irisRecognition  # noqa: E402
from modules.utils import get_cfg  # noqa: E402

names, polar, mask, labels, persons = load_cache()
ir = irisRecognition(get_cfg("cfg.yaml"))


def naive_match(code1, code2, max_shift=16, min_overlap=0.05):
    """Ista razdalja kot matchIBBCodes, a z ocitno zanko cez zamike (brez FFT)."""
    x1 = code1.astype(np.float64) / 127
    x2 = code2.astype(np.float64) / 127
    best = 1.0
    for s in range(-max_shift, max_shift + 1):
        r = np.roll(x2, s, axis=-1)
        n = np.sum(np.abs(x1) * np.abs(r))
        if n > min_overlap * x1.size:
            best = min(best, (n - np.sum(x1 * r)) / (2 * n))
    return best


# 1) FFT == zanka z np.roll
idx = [0, 1, 5, 40, 41, 100, 200, 300]
for use_mask in (True, False):
    codes = [ir.extractIBBCode(polar[i], mask[i]) if use_mask else ir.extractIBBCode(polar[i]) for i in idx]
    worst = 0.0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            worst = max(worst, abs(ir.matchIBBCodes(codes[a], codes[b]) - naive_match(codes[a], codes[b])))
    assert worst < 1e-4, worst
    print(f"FFT proti zanki (maska={use_mask}): najvecja razlika {worst:.2e}  OK")

# 2) oblika in tip predloge
c = ir.extractIBBCode(polar[0], mask[0])
assert c.dtype == np.int8 and c.shape == (8, 32, 512), (c.dtype, c.shape)
print(f"predloga: {c.dtype} {c.shape}, {c.nbytes / 1024:.0f} kB, veljavnih bitov {np.mean(c != 0):.2f}  OK")

# 3) ista slika -> razdalja 0 (zaradi zaokrozevanja FFT ne sme biti niti rahlo negativna); simetricnost
for i in range(0, len(polar), 6):
    for ci in (ir.extractIBBCode(polar[i], mask[i]), ir.extractIBBCode(polar[i])):
        assert 0.0 <= ir.matchIBBCodes(ci, ci) < 1e-6
c2 = ir.extractIBBCode(polar[1], mask[1])
assert abs(ir.matchIBBCodes(c, c2) - ir.matchIBBCodes(c2, c)) < 1e-5
print("ista slika -> 0, simetricnost  OK")

# 4) rotacija ocesa: krozno zamaknjena slika mora dati (skoraj) 0 znotraj dosega zamika
for shift in (5, 16):
    cr = ir.extractIBBCode(np.roll(polar[0], shift, axis=1), np.roll(mask[0], shift, axis=1))
    d = ir.matchIBBCodes(c, cr)
    assert d < 1e-6, d
cr = ir.extractIBBCode(np.roll(polar[0], 40, axis=1), np.roll(mask[0], 40, axis=1))
print(f"zamik 5 in 16 stolpcev -> 0; zamik 40 (izven dosega) -> {ir.matchIBBCodes(c, cr):.3f}  OK")

# 5) robni primeri in razlicni nacini klica
assert ir.extractIBBCode(None) is None
assert ir.matchIBBCodes(None, c) == 1.0 and ir.matchIBBCodes(c, None) == 1.0
assert ir.matchIBBCodes(c, np.zeros_like(c)) == 1.0                      # nic veljavnih bitov
cb = ir.extractIBBCode(polar[0], mask[0] > 127)                          # bool maska
c01 = ir.extractIBBCode(polar[0], (mask[0] > 127).astype(np.uint8))      # maska 0/1
assert np.array_equal(c, cb) and np.array_equal(c, c01)
assert isinstance(ir.matchIBBCodes(c, c2), float)
d_all = ir.matchIBBCodes(c, c2, mask[0], mask[1])                        # maska tudi ob primerjavi
assert abs(d_all - ir.matchIBBCodes(c, c2)) < 1e-6
cn, cn2 = ir.extractIBBCode(polar[0]), ir.extractIBBCode(polar[1])       # brez maske ob ekstrakciji ...
print(f"brez maske: {ir.matchIBBCodes(cn, cn2):.3f}, maska samo ob primerjavi: "
      f"{ir.matchIBBCodes(cn, cn2, mask[0], mask[1]):.3f}")
assert np.array_equal(ir.extractIBBCode(polar[0], mask[0][:10]), cn)     # maska napacne oblike -> lastna maska
for bad in (mask[0][0], mask[0][:0], mask[0][None]):                     # 1-D, prazna in 3-D maska ob primerjavi
    assert abs(ir.matchIBBCodes(c, c2, bad, bad) - ir.matchIBBCodes(c, c2)) < 1e-6   # se ne uposteva, brez izjeme
for attr in ("IBB_SIGMA", "IBB_ERODE"):                                  # 0 = izklop glajenja / erozije
    setattr(ir, attr, 0)                                                 # atribut objekta prekrije konstanto razreda
    assert ir.extractIBBCode(polar[0]).shape == c.shape
    delattr(ir, attr)                                                    # spet velja konstanta razreda
print("robni primeri  OK")

# 6) vzorcenje soseda: vektorizirana bilinearna interpolacija == rocni izracun po definiciji
rng = np.random.default_rng(0)
img = polar[3].astype(np.float32)
H, W = img.shape
for dy, dx in [(0.0, 12.0), (-0.7071, 8.4853), (-1.0, 0.0), (0.7071, -8.4853), (0.3, -2.6)]:
    val, inside = ir._ibbSampleNeighbour(img, dy, dx)
    for _ in range(200):
        row, col = int(rng.integers(H)), int(rng.integers(W))            # ne r / c: c je zgoraj predloga
        y, x = row + dy, col + dx
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - y0, x - x0
        px = lambda yy, xx: img[min(max(yy, 0), H - 1), xx % W]          # noqa: E731
        ref = ((1 - fy) * (1 - fx) * px(y0, x0) + (1 - fy) * fx * px(y0, x0 + 1)
               + fy * (1 - fx) * px(y0 + 1, x0) + fy * fx * px(y0 + 1, x0 + 1))
        assert abs(val[row, col] - ref) < 1e-3, (dy, dx, row, col, val[row, col], ref)
        assert inside[row] == (np.floor(y) >= 0 and np.ceil(y) <= H - 1)
print("bilinearno vzorcenje sosedov == rocni izracun  OK")

# 7) LBP bit po definiciji: bit p = (sosed p >= sredisce) na zglajeni sliki
blurred = ir._ibbBlur(polar[0].astype(np.float32))
diffs, inside = ir._ibbLbpDifferences(blurred)
code = ir.extractIBBCode(polar[0], np.full_like(mask[0], 255))            # vse veljavno
rows = slice(ir.IBB_ROW_START, None, ir.IBB_ROW_STEP)
expected_bit = diffs[:, rows] >= 0
used = code != 0
assert np.array_equal((code > 0)[used], expected_bit[used])
assert not np.any(used & ~inside[:, rows][:, :, None])                   # sosed izven slike -> bit ni uporabljen
assert inside[0].all() and inside[ir.IBB_P // 2].all()                   # soseda pri 0 in 180 st. sta v isti vrstici
print("predznak kode == LBP bit; biti s sosedom izven slike so izloceni  OK")

# 8) lastna maska: prekrivanje s pravo segmentacijsko masko
iou, kept_bad = [], []
for i in range(0, len(polar), 6):
    est = ir._ibbEstimateMask(ir._ibbBlur(polar[i].astype(np.float32)))
    true = mask[i] > 127
    iou.append((est & true).sum() / max(1, (est | true).sum()))
    kept_bad.append((est & ~true).sum() / max(1, est.sum()))
print(f"lastna maska: IoU s pravo {np.mean(iou):.2f}, delez napacno obdrzanih pikslov {np.mean(kept_bad):.3f}")
print("VSE OK")
