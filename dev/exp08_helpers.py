"""Lastna maska (raziskovalna razlicica): parametricni kupoli + preverjanje svetlosti."""
import math

import torch

from dev import research as R


def dome_mask(domes, r0=1, H=64, W=512):
    """domes: seznam (sredisce_stolpec, polsirina, visina) v pikslih. Vrne [H,W] 0/1.

    Piksel (r, c) je veljaven, ce r0 <= r < H - h*cos(pi/2 * d/w), kjer je d krozna
    razdalja stolpca c do sredisca kupole (veka sega od zunanjega roba navznoter).
    """
    cols = torch.arange(W).float()
    cut = torch.zeros(W)
    for c, w, h in domes:
        d = (cols - c + W / 2) % W - W / 2
        cut = torch.maximum(cut, torch.where(d.abs() < w, h * torch.cos(math.pi / 2 * d / w), torch.zeros(W)))
    rows = torch.arange(H).float().view(H, 1)
    return ((rows >= r0) & (rows < H - cut.view(1, W))).float()


def bright_ok(polar, ref, z_hi, sigma=2.0):
    """|svetlost - mediana| < z_hi * MAD; mediana in MAD iz obmocja ref [H,W]."""
    img = R.gaussian_blur(polar, sigma)
    out = torch.empty_like(img)
    for i in range(polar.shape[0]):
        vals = img[i][ref > 0]
        med = vals.median()
        mad = (vals - med).abs().median() + 1e-6
        out[i] = (((img[i] - med) / mad).abs() < z_hi).float()
    return out
