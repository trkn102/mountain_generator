#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""atmos.py — сквозной HDR-рендер атмосферы, облаков, объёмных лучей и шейдинга."""
import math
import numpy as np
from PIL import Image

F32 = np.float32
CAM_H = 1200.0                    # высота камеры, м
CL_LO, CL_HI = 2400.0, 4600.0     # слой облаков, м
SPAN = 45000.0                    # полуширина облачного домена, м

CLOUD_Q = {                       # ширина буфера, шагов марша, шагов к солнцу, объём (y,z,x)
    "low":   (480,  28, 3, (48,  64, 128)),
    "med":   (800,  44, 4, (64,  96, 192)),
    "high":  (1200, 64, 5, (80, 128, 256)),
    "ultra": (1800, 96, 6, (96, 160, 320)),
}

# ---------------------------- цвет / тонмаппинг ----------------------------
def srgb2lin(c):
    c = np.asarray(c, F32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(F32)

def lin2srgb(c):
    c = np.clip(np.asarray(c, F32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055) * 255.0

def aces(x):
    x = np.maximum(np.asarray(x, F32), 0.0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)

def hdr_postprocess_and_tonemap(hdr, seed, bloom=0.25, vignette=0.16, grain=1.3, exposure=1.0):
    """HDR-bloom -> ACES тонмаппинг -> виньетка -> зерно -> uint8 sRGB."""
    H, W = hdr.shape[:2]
    img = hdr * F32(exposure)

    if bloom > 0.01:
        excess = np.maximum(img - 0.85, 0.0)
        excess = excess / (1.0 + 0.15 * excess)
        bw, bh = max(2, W // 8), max(2, H // 8)
        b_small = resize_rgb(excess, (bh, bw))
        for _ in range(3):
            b_small = resize_rgb(
                resize_rgb(b_small, (max(2, b_small.shape[0] // 2), max(2, b_small.shape[1] // 2))),
                (bh, bw)
            )
        img = img + resize_rgb(b_small, (H, W)) * F32(bloom * 0.65)

    srgb = np.clip(lin2srgb(aces(img)) / 255.0, 0.0, 1.0)

    yy = np.linspace(-1.0, 1.0, H, dtype=F32)[:, None]
    xx = np.linspace(-1.0, 1.0, W, dtype=F32)[None, :]
    r2 = np.clip((xx * xx + yy * yy) * 0.5, 0.0, 1.0)
    srgb *= (1.0 - F32(vignette) * (r2 ** 1.6))[..., None]

    if grain > 0:
        srgb += np.random.default_rng(seed + 999).normal(0.0, grain / 255.0, srgb.shape).astype(F32)

    return Image.fromarray(np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8), "RGB")

# ---------------------------- ресайз / шум ---------------------------------
def _resize_axis(a, axis, n, smooth):
    m = a.shape[axis]
    if m == n:
        return a
    t = np.clip((np.arange(n, dtype=F32) + 0.5) * m / n - 0.5, 0.0, m - 1.0)
    i0 = t.astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    f = t - i0
    if smooth:
        f = f * f * (3.0 - 2.0 * f)
    shp = [1] * a.ndim
    shp[axis] = n
    f = f.reshape(shp)
    return np.take(a, i0, axis=axis) * (1.0 - f) + np.take(a, i1, axis=axis) * f

def resize2(a, shape):
    a = _resize_axis(np.asarray(a, F32), 0, shape[0], False)
    return _resize_axis(a, 1, shape[1], False).astype(F32)

def resize_rgb(a, shape):
    a = _resize_axis(np.asarray(a, F32), 0, shape[0], False)
    return _resize_axis(a, 1, shape[1], False).astype(F32)

def resize3(a, shape):
    for ax, n in enumerate(shape):
        a = _resize_axis(np.asarray(a, F32), ax, n, True)
    return a.astype(F32)

def fbm2_np(np_rng, w, h, octaves=4, stretch=3.0):
    w, h = max(2, w), max(2, h)
    total = np.zeros((h, w), F32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        gw = max(2, int(6 * stretch * (2 ** o)))
        gh = max(2, int(6 * (2 ** o)))
        g = np_rng.random((gh, gw), dtype=F32)
        total += resize2(g, (h, w)) * amp
        norm += amp
        amp *= 0.55
    return total / norm

def fbm3(rng, shape, octaves=5, grid=(3, 2, 4), gain=0.55):
    tot = np.zeros(shape, F32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        g = tuple(max(2, int(b * (2 ** o))) for b in grid)
        tot += resize3(rng.random(g, dtype=F32), shape) * amp
        norm += amp
        amp *= gain
    return tot / norm

def sample3(vol, cy, cz, cx):
    ny, nz, nx = vol.shape
    inside = (cy >= 0.0) & (cy <= 1.0)
    y = np.clip(cy, 0.0, 1.0) * (ny - 1)
    z = (cz % 1.0) * (nz - 1)
    x = (cx % 1.0) * (nx - 1)
    y0 = y.astype(np.int32); z0 = z.astype(np.int32); x0 = x.astype(np.int32)
    y1 = np.minimum(y0 + 1, ny - 1); z1 = (z0 + 1) % nz; x1 = (x0 + 1) % nx
    fy, fz, fx = y - y0, z - z0, x - x0
    v = vol.ravel()
    def G(iy, iz, ix):
        return v[(iy * nz + iz) * nx + ix]
    a = G(y0, z0, x0); b = G(y0, z0, x1); c = G(y0, z1, x0); d = G(y0, z1, x1)
    e = G(y1, z0, x0); f = G(y1, z0, x1); g = G(y1, z1, x0); h = G(y1, z1, x1)
    p = (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fz
    q = (e + (f - e) * fx) + ((g + (h - g) * fx) - (e + (f - e) * fx)) * fz
    return (p + (q - p) * fy) * inside

def hg(mu, g):
    return (1.0 - g * g) / (4.0 * math.pi * np.power(np.maximum(1.0 + g * g - 2.0 * g * mu, 1e-4), 1.5))

def blur1(a, r):
    if r < 1:
        return a
    k = np.ones(2 * r + 1, F32) / (2 * r + 1)
    return np.convolve(np.pad(a, (r, r), mode="edge"), k, mode="valid").astype(F32)

def _bilerp(a, x, y):
    h, w = a.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int32); y0 = y.astype(np.int32)
    fx, fy = x - x0, y - y0
    return (a[y0, x0] * (1.0 - fx) + a[y0, x0 + 1] * fx) * (1.0 - fy) + \
           (a[y0 + 1, x0] * (1.0 - fx) + a[y0 + 1, x0 + 1] * fx) * fy

# ---------------------------- солнце ---------------------------------------
class SunRig:
    """Единый источник правды: диск, положение относительно гор (behind/above/front), цвет, тени."""
    def __init__(self, sun_x, elev_deg, W, H, horizon=0.62, vfov=80.0,
                 light_pos="behind", front=None, night=False, turbidity=1.2):
        self.W, self.H, self.horizon, self.vfov = W, H, horizon, vfov
        self.night = night
        self.light_pos = light_pos
        self.in_front = (light_pos == "front")
        self.hfov = vfov * W / H
        self.az = math.radians((sun_x - 0.5) * self.hfov)
        self.el = math.radians(elev_deg)
        ca, sa = math.cos(self.az), math.sin(self.az)
        ce, se = math.cos(self.el), math.sin(self.el)

        self.dir = np.array([sa * ce, se, -ca * ce], F32)

        if front is None:
            front = {"behind": 0.12, "above": 0.45, "front": 0.88}[light_pos]
        self.front = front

        L = self.dir.copy()
        if self.in_front:
            L = np.array([-sa * ce, max(se, 0.25), ca * ce * (0.4 + 0.6 * front)], F32)
        else:
            L[2] = (-1.0 + 2.0 * front) * abs(L[2])
        self.L = (L / np.linalg.norm(L)).astype(F32)

        self.x = sun_x * W
        self.y = (horizon - elev_deg / vfov) * H
        self.r = (0.030 if night else 0.042) * min(W, H)

        T = self.transmittance(elev_deg, turbidity)
        self.sun_col = (T / max(float(T.max()), 1e-4)).astype(F32)
        lum = float(0.2126 * T[0] + 0.7152 * T[1] + 0.0722 * T[2])
        self.sun_I = 0.35 if night else (0.25 + 2.6 * lum)
        if night:
            self.sun_col = np.array([0.80, 0.87, 1.0], F32)
        self.shadow_len = float(np.clip(0.55 / max(math.tan(max(self.el, 0.06)), 0.05), 0.35, 6.0))

    @staticmethod
    def transmittance(elev_deg, turbidity=1.2):
        z = 90.0 - max(elev_deg, -3.0)
        am = min(1.0 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** (-1.6364)), 40.0)
        tauR = np.array([0.0465, 0.1085, 0.2650], F32)
        tauM = np.full(3, 0.018 * turbidity, F32)
        return np.exp(-(tauR + tauM) * am).astype(F32)

# ---------------------------- физическое небо ------------------------------
def physical_sky(w, h, rig, samples=10, lsamples=4, turbidity=1.2):
    Rp, Ra, HR, HM = 6371e3, 6471e3, 8000.0, 1200.0
    bR = np.array([5.8e-6, 13.5e-6, 33.1e-6], F32)
    bM = np.full(3, 4.0e-6 * turbidity, F32)
    v = np.linspace(0, 1, h, dtype=F32)[:, None]
    u = np.linspace(0, 1, w, dtype=F32)[None, :]
    th = (rig.horizon - v) * math.radians(rig.vfov)
    ph = (u - 0.5) * math.radians(rig.hfov)
    st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
    d = np.stack(np.broadcast_arrays(sp * ct, st, -cp * ct), -1).astype(F32)

    sky_sun_dir = -rig.dir if rig.in_front else rig.dir

    oy = Rp + CAM_H
    b = d[..., 1] * oy
    c0 = oy * oy
    disc = b * b - (c0 - Ra * Ra)
    tmax = -b + np.sqrt(np.maximum(disc, 0.0))
    dp = b * b - (c0 - Rp * Rp)
    tp = -b - np.sqrt(np.maximum(dp, 0.0))
    tmax = np.where((dp > 0) & (tp > 0), tp, tmax)
    tmax = np.maximum(tmax, 1.0)
    ds = tmax / samples
    mu = d[..., 0] * sky_sun_dir[0] + d[..., 1] * sky_sun_dir[1] + d[..., 2] * sky_sun_dir[2]
    sumR = np.zeros((h, w, 3), F32); sumM = np.zeros((h, w, 3), F32)
    odR = np.zeros((h, w), F32); odM = np.zeros((h, w), F32)
    for i in range(samples):
        t = (i + 0.5) * ds
        px, py, pz = d[..., 0] * t, oy + d[..., 1] * t, d[..., 2] * t
        hgt = np.maximum(np.sqrt(px * px + py * py + pz * pz) - Rp, 0.0)
        dR, dM = np.exp(-hgt / HR) * ds, np.exp(-hgt / HM) * ds
        odR += dR; odM += dM
        bl = px * sky_sun_dir[0] + py * sky_sun_dir[1] + pz * sky_sun_dir[2]
        cl = px * px + py * py + pz * pz - Ra * Ra
        dsl = (-bl + np.sqrt(np.maximum(bl * bl - cl, 0.0))) / lsamples
        lR = np.zeros((h, w), F32); lM = np.zeros((h, w), F32)
        for j in range(lsamples):
            tt = (j + 0.5) * dsl
            qx, qy, qz = px + sky_sun_dir[0] * tt, py + sky_sun_dir[1] * tt, pz + sky_sun_dir[2] * tt
            hh = np.maximum(np.sqrt(qx * qx + qy * qy + qz * qz) - Rp, 0.0)
            lR += np.exp(-hh / HR) * dsl; lM += np.exp(-hh / HM) * dsl
        T = np.exp(-(bR * (odR + lR)[..., None] + bM * 1.1 * (odM + lM)[..., None]))
        sumR += dR[..., None] * T; sumM += dM[..., None] * T
    phR = (3.0 / (16.0 * math.pi)) * (1.0 + mu * mu)
    phM = hg(mu, 0.76)
    return (22.0 * (bR * phR[..., None] * sumR + bM * phM[..., None] * sumM)).astype(F32)

def add_sun(sky, rig, aw, ah):
    if rig.in_front:
        return sky
    yy, xx = np.mgrid[0:ah, 0:aw].astype(F32)
    sx, sy = rig.x * aw / rig.W, rig.y * ah / rig.H
    r = max(2.0, rig.r * aw / rig.W)
    squash = 1.0 + 0.40 * math.exp(-max(rig.el, 0.0) / 0.10)
    d = np.sqrt(((xx - sx) / r) ** 2 + ((yy - sy) / (r / squash)) ** 2)
    col, I = rig.sun_col, rig.sun_I
    glow = I * (0.55 * np.exp(-d * 0.55) + 0.22 * np.exp(-d * 0.13) + 0.07 * np.exp(-d * 0.035))
    sky = sky + glow[..., None] * col[None, None, :]
    a = np.clip((1.0 - d) * r / 1.2 + 0.5, 0, 1)
    ld = 0.70 + 0.30 * np.sqrt(np.clip(1.0 - d * d, 0, 1))
    return sky * (1 - a[..., None]) + (col[None, None, :] * (I * 34.0 * ld[..., None])) * a[..., None]

def render_sky(pal, rig, aw, ah, phys=0.75, turbidity=1.2, night=False):
    ts = np.array([s[0] for s in pal["sky"]], F32)
    cs = np.array([srgb2lin(s[1]) for s in pal["sky"]], F32)
    y = np.linspace(0, 1, ah, dtype=F32)
    base = np.repeat(np.stack([np.interp(y, ts, cs[:, k]) for k in range(3)], 1)[:, None, :], aw, 1)
    if phys > 0.01 and not night:
        p = resize_rgb(physical_sky(max(4, aw // 2), max(4, ah // 2), rig, turbidity=turbidity), (ah, aw))
        lum = lambda a: a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
        tgt = float(np.median(lum(base))); cur = max(float(np.median(lum(p))), 1e-5)
        base = base * (1 - phys) + (p * (tgt / cur)) * phys
    return add_sun(base, rig, aw, ah)

# ---------------------------- облака ---------------------------------------
class CloudField:
    def __init__(self, seed, shape=(64, 96, 192), coverage=0.5, kind="cumulus", warp=0.4, detail=0.42):
        rng = np.random.default_rng(seed)
        ny, nz, nx = shape
        grid = {"cumulus": (3, 3, 5), "stratus": (2, 3, 6), "cirrus": (2, 4, 14)}[kind]
        base = fbm3(rng, shape, 5, grid)
        if warp > 0:
            wf = resize3(fbm3(rng, (max(2, ny // 3), max(2, nz // 3), max(2, nx // 3)), 3, (2, 2, 3)), shape) - 0.5
            Y, Z, X = np.meshgrid(((np.arange(ny) + 0.5) / ny).astype(F32),
                                  ((np.arange(nz) + 0.5) / nz).astype(F32),
                                  ((np.arange(nx) + 0.5) / nx).astype(F32), indexing="ij")
            k = warp * 0.35
            base = sample3(base, Y + wf * k * 0.5, Z + wf * k, X + wf * k)
        hf = (np.arange(ny, dtype=F32) + 0.5) / ny
        prof = {"cumulus": np.clip(hf / 0.26, 0, 1) * np.clip((1 - hf) / 0.42, 0, 1) ** 1.05,
                "stratus": np.clip(hf / 0.18, 0, 1) * np.clip((1 - hf) / 0.30, 0, 1),
                "cirrus":  np.clip(hf / 0.42, 0, 1) * np.clip((1 - hf) / 0.42, 0, 1)}[kind][:, None, None]
        weather = resize3(fbm3(rng, (1, 4, 9), 3, (1, 2, 4)), (1, nz, nx))[0][None]
        cov = np.clip((weather - (1.0 - coverage * 1.15)) * 2.4, 0, 1)
        # Откалиброванный порог: стабильные облака уже при coverage = 0.4..0.5
        d = np.clip((base - (0.54 - 0.34 * cov)) * 3.2, 0, 1) * prof * cov
        er = fbm3(rng, shape, 4, (7, 6, 10), 0.6)
        d = np.clip(d - (1.0 - d) * er * detail, 0, 1)
        if kind == "cirrus":
            d *= 0.62
        self.d = np.ascontiguousarray(d, F32)

    def render(self, rig, cw, amb, haze, steps=44, lsteps=4, sigma=0.013, region=0.66, seed=0):
        vol = self.d
        ch = max(4, int(round(cw * region * rig.H / rig.W)))
        u = (np.arange(cw, dtype=F32) + 0.5) / cw
        v = (np.arange(ch, dtype=F32) + 0.5) / ch * region
        th = (rig.horizon - v[:, None]) * math.radians(rig.vfov)
        ph = (u[None, :] - 0.5) * math.radians(rig.hfov)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        DX = (sp * ct).ravel()
        DY = np.broadcast_to(st, (ch, cw)).ravel().astype(F32)
        DZ = (-cp * ct).ravel()
        dys = np.maximum(DY, 1e-3)
        t0 = (CL_LO - CAM_H) / dys
        t1 = np.minimum((CL_HI - CAM_H) / dys, 120000.0)
        idx = np.flatnonzero((DY > 0.012) & ((t1 - t0) > 60.0))
        if idx.size == 0:
            return np.zeros((ch, cw, 3), F32), np.zeros((ch, cw), F32), region
        ds = np.minimum((t1 - t0) / steps, 900.0)
        rng = np.random.default_rng(seed + 11)
        dx_, dy_, dz_ = DX[idx], DY[idx], DZ[idx]
        t = t0[idx] + rng.random(idx.size, dtype=F32) * ds[idx]
        ds_ = ds[idx]; tend = t1[idx]
        T = np.ones(idx.size, F32); ACC = np.zeros((idx.size, 3), F32); OD = np.zeros(idx.size, F32)

        sun_dir = np.array([rig.L[0], max(rig.L[1], 0.08), -rig.L[2]], F32)
        sun_dir /= np.linalg.norm(sun_dir)

        mu = dx_ * sun_dir[0] + dy_ * sun_dir[1] + dz_ * sun_dir[2]
        PHM = 0.62 * hg(mu, 0.72) + 0.38 * hg(mu, -0.22)
        PHF = hg(mu, 0.94)
        sun = rig.sun_col * rig.sun_I * 3.0

        segs = np.array([140.0 * (1.7 ** j) for j in range(lsteps)], dtype=F32)
        cent = np.cumsum(segs) - 0.5 * segs
        inv_h, inv_s = F32(1.0 / (CL_HI - CL_LO)), F32(1.0 / (2.0 * SPAN))

        for _ in range(steps):
            px, py, pz = dx_ * t, CAM_H + dy_ * t, dz_ * t
            cy = (py - CL_LO) * inv_h
            dens = sample3(vol, cy, -pz * inv_s, px * inv_s + 0.5)
            hit = np.flatnonzero(dens > 0.004)
            if hit.size:
                d_ = dens[hit]
                pxh, pyh, pzh = px[hit, None], py[hit, None], pz[hit, None]
                l_cy = (pyh + sun_dir[1] * cent[None, :] - CL_LO) * inv_h
                l_cz = -(pzh + sun_dir[2] * cent[None, :]) * inv_s
                l_cx = (pxh + sun_dir[0] * cent[None, :]) * inv_s + 0.5
                lod = (sample3(vol, l_cy, l_cz, l_cx) @ segs) * sigma

                le = np.zeros(hit.size, F32)
                for k in range(3):
                    le += (0.55 ** k) * np.exp(-lod * (0.45 ** k)) * \
                          (PHM[hit] * (0.7 ** k) + PHF[hit] * 0.55 * (0.85 ** k))
                pw = 1.0 - 0.65 * (1.0 - np.exp(-2.0 * OD[hit])) * (0.5 + 0.5 * mu[hit])
                energy = sun[None, :] * le[:, None] * pw[:, None] + \
                         amb[None, :] * (0.35 + 0.65 * cy[hit])[:, None]
                astep = 1.0 - np.exp(-d_ * sigma * ds_[hit])
                ACC[hit] += (T[hit] * astep)[:, None] * energy
                T[hit] *= 1.0 - astep
                OD[hit] += d_ * sigma * ds_[hit]
            t += ds_
            keep = (T > 0.02) & (t < tend)
            if not keep.all():
                if not keep.any():
                    break
                idx = idx[keep]; dx_ = dx_[keep]; dy_ = dy_[keep]; dz_ = dz_[keep]
                t = t[keep]; ds_ = ds_[keep]; tend = tend[keep]
                T = T[keep]; ACC = ACC[keep]; OD = OD[keep]
                mu = mu[keep]; PHM = PHM[keep]; PHF = PHF[keep]
        rgb = np.zeros((cw * ch, 3), F32); alp = np.zeros(cw * ch, F32)
        rgb[idx] = ACC; alp[idx] = 1.0 - T
        fade = (1.0 - np.exp(-t0 / 45000.0))[:, None]
        rgb = rgb * (1 - 0.75 * fade) + (haze[None, :] * 1.1) * 0.75 * fade * alp[:, None]
        alp *= (1 - 0.30 * fade[:, 0])
        return rgb.reshape(ch, cw, 3), alp.reshape(ch, cw), region

    def shadow_strip(self, rig, n=256, ground=600.0):
        prof = self.d.mean(axis=(0, 1))
        sx = (0.5 * (CL_LO + CL_HI) - ground) * rig.L[0] / max(rig.L[1], 0.2) / (2 * SPAN)
        u = (np.arange(n, dtype=F32) + 0.5) / n
        X = (np.tan((u - 0.5) * math.radians(rig.hfov)) * 9000.0 / (2 * SPAN) + 0.5 + sx) % 1.0
        pos = X * len(prof)
        i0 = pos.astype(np.int32) % len(prof); i1 = (i0 + 1) % len(prof); f = pos - np.floor(pos)
        s = blur1(prof[i0] * (1 - f) + prof[i1] * f, max(1, n // 64))
        return np.linspace(0, 1, n, dtype=F32), (1.0 - 0.75 * np.clip(s * 2.2, 0, 1)).astype(F32)

def composite_clouds(sky, crgb, calpha, region):
    ah, aw = sky.shape[:2]
    rh = max(2, int(ah * region))
    a = resize2(calpha, (rh, aw))[..., None]
    sky[:rh] = sky[:rh] * (1 - a) + resize_rgb(crgb, (rh, aw))
    return sky

# ---------------------------- лучи и тени ----------------------------------
def god_rays(sky, sil, rig, aw, ah, W, H, strength=0.5, steps=52, decay=0.962, density=0.92):
    if rig.in_front or strength <= 0.001:
        return sky
    hw, hh = max(4, aw // 2), max(4, ah // 2)
    yy, xx = np.mgrid[0:hh, 0:hw].astype(F32)
    sy = resize2(np.asarray(sil, F32)[None, :], (1, hw))[0] * (hh / H)
    occ = (yy < sy[None, :]).astype(F32)
    sxp, syp = rig.x * hw / W, rig.y * hh / H
    ddx, ddy = (xx - sxp) * (density / steps), (yy - syp) * (density / steps)
    cx, cy = xx.copy(), yy.copy()
    acc = np.zeros((hh, hw), F32); ill = 1.0
    for _ in range(steps):
        cx -= ddx; cy -= ddy
        acc += _bilerp(occ, cx, cy) * ill
        ill *= decay
    acc *= strength / steps
    dist = np.sqrt((xx - sxp) ** 2 + (yy - syp) ** 2) / (0.55 * hw)
    acc *= np.exp(-dist * 1.1) * float(np.clip(1.0 - rig.el / 0.9, 0.15, 1.0))
    sky += resize2(acc, (ah, aw))[..., None] * (rig.sun_col * rig.sun_I * 1.8)[None, None, :]
    return sky

def crepuscular_shafts_hdr(hdr, sil_layer, rig, W, H, strength=0.22, steps=36):
    """Объёмные сумеречные лучи В ВОЗДУХЕ МЕЖДУ ГРЯДАМИ (накладываются после гряды i, до гряды i+1)."""
    if rig.in_front or strength <= 0.001:
        return hdr
    hw, hh = max(4, W // 6), max(4, H // 6)
    yy, xx = np.mgrid[0:hh, 0:hw].astype(F32)
    sy = np.interp(np.linspace(0, W - 1, hw), np.arange(W), sil_layer).astype(F32) * (hh / H)
    occ = (yy < sy[None, :]).astype(F32)
    sxp, syp = rig.x * hw / W, rig.y * hh / H
    density, decay = 0.85, 0.955
    ddx, ddy = (xx - sxp) * (density / steps), (yy - syp) * (density / steps)
    cx, cy = xx.copy(), yy.copy()
    acc = np.zeros((hh, hw), F32); ill = 1.0
    for _ in range(steps):
        cx -= ddx; cy -= ddy
        acc += _bilerp(occ, cx, cy) * ill
        ill *= decay
    acc *= strength / steps
    dist = np.sqrt((xx - sxp) ** 2 + (yy - syp) ** 2) / (0.60 * hw)
    # Затухание к низу долины, чтобы лучи не заливали воду
    vert_fade = np.clip((0.78 * hh - yy) / (0.25 * hh), 0.0, 1.0)
    acc *= np.exp(-dist * 1.25) * vert_fade * float(np.clip(1.0 - rig.el / 0.85, 0.12, 1.0))
    shafts = resize2(acc, (H, W))[..., None] * (rig.sun_col * rig.sun_I * 1.15)[None, None, :]
    return hdr + shafts

def ridge_cast_shadow(xs, ys, rig, W, n=384, steps=30, softness=0.30):
    xg = np.linspace(0, W, n, dtype=F32)
    alt = -np.interp(xg, xs, ys).astype(F32)
    slope = math.tan(max(rig.el, 0.05)) * 0.55
    sgn = 1.0 if rig.L[0] >= 0 else -1.0
    occ = np.zeros(n, F32)
    for i in range(1, steps + 1):
        d = ((i / steps) ** 1.6) * 0.55 * W
        a = np.interp(np.clip(xg + sgn * d, 0, W), xg, alt)
        occ = np.maximum(occ, np.clip((a - (alt + slope * d)) / (softness * d * 0.10 + 3.0) + 0.5, 0, 1))
    occ = blur1(occ, max(1, n // 96))
    return (xg / W).astype(F32), (1.0 - 0.85 * occ).astype(F32)

# ---------------------------- HDR-шейдинг граней --------------------------
class Shader:
    """Линейный HDR-шейдер с экспоненциальной воздушной перспективой по высоте."""
    def __init__(self, rig, pal, W, H, ambient=0.5, backlight=0.55, wrap=0.30, shadows=True):
        self.rig, self.W, self.H = rig, W, H
        self.ambient, self.back, self.wrap = ambient, backlight, wrap
        self.shadows = shadows
        self.sky_amb = 0.6 * srgb2lin(pal["sky"][0][1]) + 0.4 * srgb2lin(pal["sky"][-1][1])
        self.bounce = srgb2lin(pal["ground"]) * 0.4
        self.haze = srgb2lin(pal["haze"])
        self.sun = rig.sun_col * rig.sun_I
        self.cloud_x = np.linspace(0, 1, 8, dtype=F32); self.cloud = np.ones(8, F32)
        self.rx = np.linspace(0, 1, 8, dtype=F32); self.rs = np.ones(8, F32)

    def set_layer(self, col_hi, col_lo, haze_amt, y_top, y_base=None, snow_y=None, snow_col=None, ridge_shadow=None):
        self.hi, self.lo = srgb2lin(col_hi), srgb2lin(col_lo)
        self.haze_amt = haze_amt
        self.y_top = y_top
        self.y_base = y_base if y_base is not None else self.H * 0.85
        self.snow_y = snow_y
        self.snow = srgb2lin(snow_col) if snow_col is not None else None
        if ridge_shadow is not None:
            self.rx, self.rs = ridge_shadow

    def color_lin(self, tri, rng, is_ridge=False):
        W, H = self.W, self.H
        (x0, y0), z0 = tri[0]; (x1, y1), z1 = tri[1]; (x2, y2), z2 = tri[2]
        ux, uy, uz = (x1 - x0) / W, (y1 - y0) / H, z1 - z0
        vx, vy, vz = (x2 - x0) / W, (y2 - y0) / H, z2 - z0
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        if nz < 0:
            nx, ny, nz = -nx, -ny, -nz
        up = -ny
        cx, cy = (x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0
        rel = min(max((cy - self.y_top) / max(H * 0.95 - self.y_top, 1.0), 0.0), 1.0)
        base = self.hi + (self.lo - self.hi) * (rel ** 1.2)

        u = min(max(cx / W, 0.0), 1.0)
        sh = float(np.interp(u, self.rx, self.rs)) * float(np.interp(u, self.cloud_x, self.cloud)) if self.shadows else 1.0

        if self.shadows:
            L = self.rig.L
            ndl = nx * L[0] + up * L[1] + nz * L[2]
            dif = max(0.0, (ndl + self.wrap) / (1.0 + self.wrap))
            direct = self.sun * (0.35 * dif + 0.85 * dif * dif) * sh
            amb = self.sky_amb * self.ambient * (0.45 + 0.55 * (0.5 + 0.5 * up)) + \
                  self.bounce * (0.5 - 0.5 * up) * 0.6
            lit = amb + direct
            col = base * lit
            if not self.rig.in_front:
                rim_pow = (max(0.0, -ndl) ** 2.5) * (1.35 if is_ridge else 1.0)
                col += self.sun * rim_pow * self.back * 0.25 * (0.4 + 0.6 * base)
        else:
            dif = 0.5
            lit = self.sky_amb * (self.ambient * 1.15) + self.bounce * 0.25
            col = base * lit

        if self.snow is not None and self.snow_y is not None:
            shade_shift = (0.5 - dif) * 0.045 * H if self.shadows else 0.0
            eff_snow_y = self.snow_y + shade_shift + rng.uniform(-0.012, 0.012) * H
            f = min(max((eff_snow_y - cy) / (0.055 * H) + 0.5, 0.0), 1.0)
            f *= min(max((up - 0.28) / 0.42, 0.0), 1.0)
            if f > 0.001:
                col = col + (self.snow * lit * 1.15 - col) * (0.92 * f)

        # Барометрический профиль дымки: у подножия плотнее, к вершине разреженнее (пики сохраняют контраст!)
        alt_norm = min(max((self.y_base - cy) / max(self.y_base - self.y_top, 1.0), 0.0), 1.0)
        haze_profile = 0.42 + 0.58 * ((1.0 - alt_norm) ** 1.3)
        a = self.haze_amt * haze_profile

        if self.rig.in_front:
            ins = self.haze * (0.75 + 0.25 * sh)
        else:
            fwd = math.exp(-((cx - self.rig.x) / (0.40 * W)) ** 2)
            # В затенённых ущельях дымка холоднее, на просветах подсвечивается солнцем
            sun_scatter = self.sun * 0.24 * fwd * (0.35 + 0.65 * sh)
            ins = self.haze * (1.0 - 0.40 * fwd) * (0.65 + 0.35 * sh) + sun_scatter

        col = (col * (1.0 - a) + ins * a) * (1.0 + rng.uniform(-0.05, 0.05))
        return np.maximum(col, 0.0).astype(F32)