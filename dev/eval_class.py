"""Koncna ("resnicna") evalvacija: poklice prave metode razreda irisRecognition.

Tako merimo natanko tisto kodo, ki se odda, in ne raziskovalne razlicice.

  --method hdbif   izhodisce: extractCode + matchCodesEfficient
  --method ibb     najina resitev: extractIBBCode + matchIBBCodes
  --masks          najinima funkcijama poda tudi segmentacijske maske
                   (brez tega jih klice tocno tako kot predloga: samo polar / samo kodi)

Zagon iz korena projekta:  .venv\\Scripts\\python.exe dev\\eval_class.py --method ibb
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.evallib import load_cache, report  # noqa: E402
from modules.irisRecognition import irisRecognition  # noqa: E402
from modules.utils import get_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["hdbif", "ibb"], default="ibb")
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--match-masks", action="store_true",
                    help="maske poda samo ob primerjavi (extractIBBCode(polar), matchIBBCodes(c1, c2, m1, m2))")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save", default=None, help="pot za shranjevanje matrike razdalj (.npy)")
    args = ap.parse_args()

    names, polar, mask, labels, persons = load_cache()
    irisRec = irisRecognition(get_cfg("cfg.yaml"))
    n = len(names)

    t0 = time.time()
    if args.method == "hdbif":
        codes = [irisRec.extractCode(p) for p in polar]
    elif args.masks:
        codes = [irisRec.extractIBBCode(p, m) for p, m in zip(polar, mask)]
    else:
        codes = [irisRec.extractIBBCode(p) for p in polar]
    t_extract = (time.time() - t0) / n

    D = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            if args.method == "hdbif":
                score, _ = irisRec.matchCodesEfficient(codes[i], codes[j], mask[i], mask[j])
            elif args.masks or args.match_masks:
                score = irisRec.matchIBBCodes(codes[i], codes[j], mask[i], mask[j])
            else:
                score = irisRec.matchIBBCodes(codes[i], codes[j])
            D[i, j] = D[j, i] = score
    t_match = (time.time() - t0) / (n * (n - 1) / 2)

    mode = "+masks" if args.masks else "+match-masks" if args.match_masks else ""
    name = args.name or f"class:{args.method}{mode}"
    report(name, D, labels, persons, extra={"ms_extract": 1e3 * t_extract, "ms_match": 1e3 * t_match})
    print(f"extract {1e3 * t_extract:.2f} ms/image, match {1e3 * t_match:.3f} ms/pair")
    if args.save:
        np.save(args.save, D)


if __name__ == "__main__":
    main()
