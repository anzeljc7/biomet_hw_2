"""Segmentira in normalizira vse slike enkrat ter shrani rezultat na disk.

Segmentacija (dve nevronski mrezi) je pocasna in se med poskusi z LBP ne
spreminja, zato jo naredimo enkrat. Vsi nadaljnji poskusi berejo samo
dev/cache/polar.npz (odvite sarenice 64x512 in maske).

Predobdelava je namenoma identicna tisti v hdbif.py (rdeci kanal, fix_image,
segment_and_circApprox, cartToPol_torch), da merimo isto, kar bo pognala
ocenjevalna skripta.

Zagon iz korena projekta:  .venv\\Scripts\\python.exe dev\\build_cache.py
"""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.irisRecognition import irisRecognition  # noqa: E402
from modules.utils import get_cfg  # noqa: E402

DATA_DIR = "dataset_downloaded"
OUT_PATH = os.path.join("dev", "cache", "polar.npz")


def main():
    irisRec = irisRecognition(get_cfg("cfg.yaml"))
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    names, polars, masks, pupils, irises = [], [], [], [], []
    t0 = time.time()
    for k, path in enumerate(paths):
        im = Image.fromarray(np.array(Image.open(path).convert("RGB"))[:, :, 0], "L")
        im = irisRec.fix_image(im)
        mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
        im_polar, mask_polar = irisRec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)
        names.append(os.path.basename(path))
        polars.append(im_polar)
        masks.append(mask_polar)
        pupils.append(pupil_xyr)
        irises.append(iris_xyr)
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        names=np.array(names),
        polar=np.stack(polars),
        mask=np.stack(masks),
        pupil=np.stack(pupils),
        iris=np.stack(irises),
    )
    print(f"saved {len(names)} images -> {OUT_PATH}")


if __name__ == "__main__":
    main()
