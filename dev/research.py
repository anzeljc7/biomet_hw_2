"""Raziskovalna (vektorizirana) razlicica LBP za hitro preizkusanje parametrov.

Koncna koda v modules/irisRecognition.py dela isto za en par slik; tukaj
racunamo vseh ~60 000 parov naenkrat, da en poskus traja nekaj sekund.

Dve druzini znacilk:
  1. bitna koda  - LBP biti (predznak razlike sosed - sredisce) ostanejo na
                   svojem mestu, primerjava = Hammingova razdalja z zamiki
  2. histogrami  - klasicni LBP po Ojali: histogram LBP oznak po blokih,
                   primerjava = chi-kvadrat
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)


# --------------------------------------------------------------------------
# Predobdelava
# --------------------------------------------------------------------------
def gaussian_blur(img, sigma):
    """Gaussovo glajenje; vodoravno krozno (kot je krozen kot), navpicno s ponovitvijo roba."""
    if sigma <= 0:
        return img
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, dtype=img.dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    t = img.unsqueeze(1)                                             # N,1,H,W
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="circular"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return t.squeeze(1)


def sample_offset(img, dy, dx):
    """Vrednost slike na (y+dy, x+dx) z bilinearno interpolacijo.

    Vodoravno krozno (roll), navpicno z obrezovanjem na rob. Vrne tudi masko
    vrstic [H], kjer je vzorec dejansko znotraj slike.
    """
    H = img.shape[-2]
    y0, x0 = math.floor(dy), math.floor(dx)
    fy, fx = dy - y0, dx - x0
    rows = torch.arange(H)
    out = 0
    for yy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
        if wy == 0:
            continue
        src = img[..., (rows + yy).clamp(0, H - 1), :]
        for xx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
            if wx == 0:
                continue
            out = out + wy * wx * torch.roll(src, shifts=-xx, dims=-1)
    lo, hi = rows + math.floor(dy), rows + math.ceil(dy)
    valid = (lo >= 0) & (hi <= H - 1)
    return out, valid


def lbp_diffs(polar, P=8, Rx=2.0, Ry=2.0, sigma=0.0, center_symmetric=False):
    """Razlike sosed - sredisce za P tock na elipsi (Rx vodoravno, Ry navpicno).

    Vrne diffs [N,K,H,W] in valid [K,H] (vrstice, kjer sosed ne pade iz slike).
    Pri center_symmetric=True primerjamo nasprotna soseda (CS-LBP), K = P/2.
    """
    img = gaussian_blur(polar, sigma)
    samples, valids = [], []
    for p in range(P):
        a = 2 * math.pi * p / P
        s, v = sample_offset(img, -Ry * math.sin(a), Rx * math.cos(a))
        samples.append(s)
        valids.append(v)
    if center_symmetric:
        h = P // 2
        diffs = [samples[p] - samples[p + h] for p in range(h)]
        valid = [valids[p] & valids[p + h] for p in range(h)]
    else:
        diffs = [s - img for s in samples]
        valid = valids
    return torch.stack(diffs, 1), torch.stack(valid, 0)


# --------------------------------------------------------------------------
# Bitna koda + Hamming z zamiki (vsi pari naenkrat prek FFT)
# --------------------------------------------------------------------------
def hamming_radial(code, valid, K, H, max_shift=16, max_dy=1):
    """Kot hamming_all_pairs, a dodatno minimum cez radialne zamike -max_dy..max_dy vrstic.

    code/valid [N,K*H,W]. Radialni zamik ni krozen: vrstice, ki padejo cez rob,
    oznacimo kot neveljavne. Pokrije napake v ocenjenem radiju zenice/sarenice.
    """
    N, _, W = code.shape
    c4, v4 = code.view(N, K, H, W), valid.view(N, K, H, W)
    best = None
    for dy in range(-max_dy, max_dy + 1):
        cs, vs = torch.roll(c4, dy, dims=2), torch.roll(v4, dy, dims=2).clone()
        if dy > 0:
            vs[:, :, :dy] = 0
        elif dy < 0:
            vs[:, :, dy:] = 0
        D = hamming_all_pairs(code, valid, max_shift, other=(cs.reshape(N, K * H, W), vs.reshape(N, K * H, W)))
        best = D if best is None else np.minimum(best, D)
    return np.minimum(best, best.T)


def hamming_all_pairs(code, valid, max_shift=16, n_ref=None, chunk=64, other=None):
    """Maskirana Hammingova razdalja, minimum cez krozne zamike, za vse pare.

    code  [N,K,W]  +1/-1 (biti);  valid [N,K,W] 0/1 (kateri biti stejejo)
    Za zamik s:  c(s) = sum x1*x2 = (#enakih - #razlicnih) med skupno veljavnimi,
                 n(s) = sum m1*m2 = #skupno veljavnih
                 HD(s) = (n - c) / (2n)
    Krizno korelacijo za vse zamike dobimo z enim FFT (korelacijski izrek).
    n_ref != None vklopi Daugmanovo normalizacijo: 0.5 - (0.5-HD)*sqrt(n/n_ref).
    """
    N, K, W = code.shape
    X = torch.fft.rfft(code * valid, dim=-1)                 # N,K,Wf
    M = torch.fft.rfft(valid, dim=-1)
    shifts = torch.cat([torch.arange(0, max_shift + 1), torch.arange(W - max_shift, W)])
    D = torch.empty(N, N)
    if other is not None:                                    # nesimetricno: vrstice = code, stolpci = other
        X2 = torch.fft.rfft(other[0] * other[1], dim=-1)
        M2 = torch.fft.rfft(other[1], dim=-1)
    else:
        X2, M2 = X, M
    Xc, Mc = X2.conj().permute(2, 1, 0), M2.conj().permute(2, 1, 0)  # Wf,K,N
    for i0 in range(0, N, chunk):
        xi = X[i0:i0 + chunk].permute(2, 0, 1)               # Wf,n,K
        mi = M[i0:i0 + chunk].permute(2, 0, 1)
        c = torch.fft.irfft(torch.matmul(xi, Xc).permute(1, 2, 0), n=W, dim=-1)[..., shifts]
        n = torch.fft.irfft(torch.matmul(mi, Mc).permute(1, 2, 0), n=W, dim=-1)[..., shifts]
        n = n.clamp(min=1.0)
        hd = (n - c) / (2 * n)
        if n_ref is not None:
            hd = 0.5 - (0.5 - hd) * torch.sqrt(n / n_ref)
        D[i0:i0 + chunk] = hd.min(dim=-1).values
    if other is not None:
        return D.numpy()
    return (0.5 * (D + D.T)).numpy()


def erode_mask(mask, e):
    """Maska ostane veljavna le, ce je veljavna cela okolica (2e+1)x(2e+1).

    Tako bit ne uporablja sosedov, ki lezijo ze na veki. Vodoravno krozno.
    """
    inv = (1.0 - mask).unsqueeze(1)
    inv = F.pad(inv, (e, e, 0, 0), mode="circular")
    inv = F.pad(inv, (0, 0, e, e), mode="replicate")
    return 1.0 - F.max_pool2d(inv, kernel_size=2 * e + 1, stride=1).squeeze(1)


def run_bits(polar, mask, P=8, Rx=3.0, Ry=3.0, sigma=2.0, cs=False, erode=0, fragile_q=0.0,
             rows=None, max_shift=16, n_ref=None, row_step=1):
    """Cel cevovod bitne kode za en nabor parametrov; vrne matriko razdalj."""
    diffs, vrows = lbp_diffs(polar, P=P, Rx=Rx, Ry=Ry, sigma=sigma, center_symmetric=cs)
    if row_step > 1:
        rows = slice(0, None, row_step)
    c, v = bit_code(diffs, vrows, mask=mask, erode=erode, fragile_q=fragile_q, rows=rows)
    return hamming_all_pairs(c, v, max_shift=max_shift, n_ref=n_ref)


def bit_code(diffs, valid_rows, mask=None, erode=None, fragile_q=0.0, rows=None):
    """Iz razlik naredi (code, valid) za hamming_all_pairs.

    mask       [N,H,W] 0/1 veljavni piksli (None = vsi)
    fragile_q  delez najsibkejsih bitov (|razlika| blizu 0), ki jih oznacimo kot
               neveljavne ("fragile bits"), posebej za vsako sliko
    rows       rezina vrstic, ki jih obdrzimo (npr. slice(0, 48))
    """
    N, K, H, W = diffs.shape
    code = torch.where(diffs >= 0, 1.0, -1.0)
    valid = valid_rows.view(1, K, H, 1).expand(N, K, H, W).float()
    if mask is not None:
        if erode:
            mask = erode_mask(mask, erode)
        valid = valid * mask.unsqueeze(1)
    if fragile_q > 0:
        mag = diffs.abs()
        thr = torch.quantile(mag.reshape(N, -1)[:, ::7], fragile_q, dim=1).view(N, 1, 1, 1)
        valid = valid * (mag > thr)
    if rows is not None:
        code, valid = code[:, :, rows], valid[:, :, rows]
    Hn = code.shape[2]
    return code.reshape(N, K * Hn, W), valid.reshape(N, K * Hn, W)


# --------------------------------------------------------------------------
# Klasicni LBP: histogrami po blokih + chi-kvadrat
# --------------------------------------------------------------------------
def uniform_lut(P):
    """Preslikava LBP oznake -> kos histograma za 'uniform' vzorce (<=2 prehoda 0/1).

    Vsak uniformni vzorec dobi svoj kos, vsi neuniformni skupnega: P*(P-1)+3 kosov.
    """
    lut = np.zeros(2 ** P, dtype=np.int64)
    nxt = 0
    for v in range(2 ** P):
        bits = [(v >> i) & 1 for i in range(P)]
        trans = sum(bits[i] != bits[(i + 1) % P] for i in range(P))
        if trans <= 2:
            lut[v] = nxt
            nxt += 1
        else:
            lut[v] = -1
    lut[lut < 0] = nxt
    return torch.from_numpy(lut), nxt + 1


def lbp_labels(diffs, uniform=True):
    N, P, H, W = diffs.shape
    weights = (2 ** torch.arange(P)).view(1, P, 1, 1)
    lab = ((diffs >= 0).long() * weights).sum(1)
    if uniform:
        lut, nb = uniform_lut(P)
        return lut[lab], nb
    return lab, 2 ** P


def block_histograms(labels, nbins, bh, bw, step, mask=None):
    """Histogrami v drsecih blokih bh x bw; vodoravni korak `step` (krozno).

    Vrne [N, n_vrstic_blokov, W/step, nbins], normalizirano na vsoto 1.
    Zamik za `step` pikslov = premik indeksa bloka za 1.
    """
    out = []
    for i0 in range(0, labels.shape[0], 8):          # po kosih, sicer one-hot ne gre v pomnilnik
        onehot = F.one_hot(labels[i0:i0 + 8], nbins).permute(0, 3, 1, 2).float()   # n,B,H,W
        if mask is not None:
            onehot = onehot * mask[i0:i0 + 8].unsqueeze(1)
        x = F.pad(onehot, (0, bw - 1, 0, 0), mode="circular")
        h = F.avg_pool2d(x, kernel_size=(bh, bw), stride=(bh, step)) * (bh * bw)
        h = h / h.sum(1, keepdim=True).clamp(min=1.0)
        out.append(h.permute(0, 2, 3, 1))
    return torch.cat(out)


def chi2_all_pairs(hist, max_shift_blocks=2, chunk=8):
    """Chi-kvadrat razdalja med histogrami, minimum cez krozne zamike blokov."""
    N = hist.shape[0]
    D = torch.full((N, N), float("inf"))
    for s in range(-max_shift_blocks, max_shift_blocks + 1):
        hs = torch.roll(hist, shifts=s, dims=2).reshape(1, N, -1)
        for i0 in range(0, N, chunk):
            a = hist[i0:i0 + chunk].reshape(-1, 1, hs.shape[-1])
            d = (((a - hs) ** 2) / (a + hs + 1e-9)).sum(-1)
            D[i0:i0 + chunk] = torch.minimum(D[i0:i0 + chunk], d)
    return (0.5 * (D + D.T)).numpy()
