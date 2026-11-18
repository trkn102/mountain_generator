#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Процедурный генератор горного пейзажа v4.2 — единый HDR-пайплайн.

Новое в v4.2:
  * Барометрическая воздушная перспектива и тени в тумане в долинах;
  * Объёмные сумеречные лучи в воздухе между слоями гор;
  * Тень от ближнего хребта на прибрежную полосу земли + контактный AO под деревьями;
  * Анизотропный шум на бликах воды (полное устранение муара).

Зависимости:  pip install pillow numpy
"""

import argparse
import colorsys
import math
import random
import time

import numpy as np
from PIL import Image, ImageDraw
from atmos import (
    F32, SunRig, Shader, CloudField, CLOUD_Q,
    render_sky, composite_clouds, god_rays, crepuscular_shafts_hdr,
    ridge_cast_shadow, hdr_postprocess_and_tonemap, srgb2lin,
    fbm2_np, resize2, resize_rgb
)

PALETTES = {
    "sunset": dict(
        sky=[(0.0, "#1c1142"), (0.35, "#4a2f68"), (0.62, "#a85a60"),
             (0.82, "#ff9a5c"), (1.0, "#ffcf96")],
        sun="#ffe9c0", mount="#4e3f70", mount_low="#35294f", haze="#e08a6a",
        ground="#3f3260", tree="#2e2547", trunk="#241d38", snow="#f5d5c8", stars=False,
        cloud_lit="#ffc9a0", cloud_shadow="#8a5570",
        water_top="#d88a66", water_deep="#2c2350"),
    "day": dict(
        sky=[(0.0, "#2f7fc4"), (0.45, "#7db8e0"), (0.75, "#cfe9f4"), (1.0, "#f2f9fc")],
        sun="#fff6cc", mount="#3e6b57", mount_low="#365f41", haze="#cfe4ee",
        ground="#4f8a5b", tree="#2f5d3a", trunk="#4a3626", snow="#f4f9fc", stars=False,
        cloud_lit="#ffffff", cloud_shadow="#aec6d4",
        water_top="#a8cadb", water_deep="#2e5a70"),
    "night": dict(
        sky=[(0.0, "#04071a"), (0.45, "#0e1738"), (0.8, "#243266"), (1.0, "#3a4a86")],
        sun="#eef2fa", mount="#2a3558", mount_low="#1b2340", haze="#2c3a6e",
        ground="#202a48", tree="#161f36", trunk="#101828", snow="#b8c6e0", stars=True,
        cloud_lit="#4a5878", cloud_shadow="#141c38",
        water_top="#2c3c6a", water_deep="#070c22"),
    "winter": dict(
        sky=[(0.0, "#8fb2cc"), (0.55, "#c8dce8"), (1.0, "#f4f8fb")],
        sun="#ffffff", mount="#6b7f96", mount_low="#57697e", haze="#e4edf3",
        ground="#e8eef4", tree="#33505c", trunk="#3d3230", snow="#ffffff", stars=False,
        cloud_lit="#ffffff", cloud_shadow="#b9c9d4",
        water_top="#c2d4de", water_deep="#466072"),
    "mono": dict(
        sky=[(0.0, "#c9c9c9"), (0.6, "#e8e8e8"), (1.0, "#fafafa")],
        sun="#ffffff", mount="#4d4d4d", mount_low="#3a3a3a", haze="#e0e0e0",
        ground="#383838", tree="#262626", trunk="#1c1c1c", snow="#f2f2f2", stars=False,
        cloud_lit="#ffffff", cloud_shadow="#aaaaaa",
        water_top="#cccccc", water_deep="#333333"),
}

# ------------------------- работа с цветом --------------------------------

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(float(int(h[i:i + 2], 16)) for i in (0, 2, 4))

def scale_rgb(c, f):
    return tuple(max(0.0, min(255.0, v * f)) for v in c)

def adjust(c, hue_deg, sat):
    h, l, s = colorsys.rgb_to_hls(*(v / 255.0 for v in c))
    h = (h + hue_deg / 360.0) % 1.0
    s = max(0.0, min(1.0, s * sat))
    return tuple(v * 255.0 for v in colorsys.hls_to_rgb(h, l, s))

def load_palette(name, hue, sat):
    pal = {}
    for k, v in PALETTES[name].items():
        if k == "sky":
            pal[k] = [(t, adjust(hex2rgb(c), hue, sat)) for t, c in v]
        elif isinstance(v, str):
            pal[k] = adjust(hex2rgb(v), hue, sat)
        else:
            pal[k] = v
    return pal

def grad_rows_lin(stops, n):
    ts = np.array([s[0] for s in stops], F32)
    cs = np.array([srgb2lin(s[1]) for s in stops], F32)
    y = np.linspace(0.0, 1.0, n, dtype=F32)
    return np.stack([np.interp(y, ts, cs[:, ch]) for ch in range(3)], axis=1)

# ------------------------- процедурный шум и рельеф -----------------------

def ridge_line(rng, n=257, roughness=0.55, pointy=1.25):
    h = [0.0] * n
    h[0], h[-1] = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
    step, amp = n - 1, 1.0
    while step > 1:
        half = step // 2
        for i in range(half, n, step):
            h[i] = (h[i - half] + h[i + half]) / 2.0 + rng.uniform(-1, 1) * amp
        amp *= roughness
        step = half
    lo, hi = min(h), max(h)
    base = np.array([((v - lo) / (hi - lo)) ** pointy for v in h], dtype=F32)

    xs = np.linspace(0.0, 1.0, n, dtype=F32)
    ph1, ph2 = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    warp = 0.045 * np.sin(2 * math.pi * xs + ph1) + 0.02 * np.sin(4 * math.pi * xs + ph2)
    warped_xs = np.clip(xs + warp * np.sin(math.pi * xs), 0.0, 1.0)
    return np.interp(warped_xs, xs, base)

class ValueNoise:
    def __init__(self, rng, gw=9, gh=7):
        self.g = [[rng.random() for _ in range(gw)] for _ in range(gh)]
        self.gw, self.gh = gw, gh

    def at(self, u, v):
        u = min(max(u, 0.0), 0.9999) * (self.gw - 1)
        v = min(max(v, 0.0), 0.9999) * (self.gh - 1)
        x0, y0 = int(u), int(v)
        fx, fy = u - x0, v - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        g = self.g
        a = g[y0][x0] + (g[y0][x0 + 1] - g[y0][x0]) * fx
        b = g[y0 + 1][x0] + (g[y0 + 1][x0 + 1] - g[y0 + 1][x0]) * fx
        return a + (b - a) * fy

def scale_envelope(kind, rng, xn):
    if kind == "range":
        return np.ones_like(xn)
    if kind == "massif":
        env = np.zeros_like(xn)
        for _ in range(rng.randint(2, 3)):
            c, w = rng.uniform(0.15, 0.85), rng.uniform(0.13, 0.22)
            env = np.maximum(env, np.exp(-((xn - c) / w) ** 2))
        return 0.15 + 0.85 * env
    c, w = rng.uniform(0.38, 0.62), rng.uniform(0.30, 0.42)
    tent = np.clip(1 - np.abs(xn - c) / w, 0, 1) ** 1.15
    return 0.10 + 0.90 * tent

# ------------------------- HDR-растеризация слоёв -------------------------

def draw_mosaic_layer_hdr(hdr, rng, W, H, xs, ridge_ys, shader, noise, z_amp, cols, rows):
    y_min = max(0, int(math.floor(min(ridge_ys))) - 2)
    y_max = H
    if y_min >= y_max:
        return 0

    bottom = H * 1.02
    pts, zs = [], []
    for r in range(rows + 1):
        t = r / rows
        row, zrow = [], []
        for c in range(cols + 1):
            x, ry = xs[c], ridge_ys[c]
            y = ry + (bottom - ry) * t
            if 0 < r < rows:
                y += rng.uniform(-1, 1) * (bottom - ry) / rows * 0.35
            if 0 < c < cols:
                x += rng.uniform(-1, 1) * (W / cols) * 0.35
            row.append((x, y))
            zrow.append(noise.at(x / W, y / H) * z_amp)
        pts.append(row)
        zs.append(zrow)

    band_h = y_max - y_min
    id_img = Image.new("I", (W, band_h), 0)
    id_draw = ImageDraw.Draw(id_img)

    colors = [np.zeros(3, F32)]
    tid = 1
    for r in range(rows):
        for c in range(cols):
            q = [(pts[r][c], zs[r][c]), (pts[r][c + 1], zs[r][c + 1]),
                 (pts[r + 1][c + 1], zs[r + 1][c + 1]), (pts[r + 1][c], zs[r + 1][c])]
            tris = [((q[0], q[1], q[2]), r == 0), ((q[0], q[2], q[3]), False)] if rng.random() < 0.5 \
                else [((q[0], q[1], q[3]), r == 0), ((q[1], q[2], q[3]), False)]
            for tri, is_ridge in tris:
                col = shader.color_lin(tri, rng, is_ridge=is_ridge)
                colors.append(col)
                poly_shifted = [(p[0][0], p[0][1] - y_min) for p in tri]
                id_draw.polygon(poly_shifted, fill=tid)
                tid += 1

    idx_map = np.asarray(id_img, dtype=np.int32)
    mask = idx_map > 0
    if mask.any():
        lut = np.stack(colors, axis=0)
        hdr[y_min:y_max][mask] = lut[idx_map[mask]]
    return tid - 1

def add_fog_band_hdr(hdr, np_rng, rig, y0, y1, color_srgb, strength, W, shadow_strip=None):
    """Полоса тумана в долине с учётом теней от передних гор и рассеяния солнца."""
    H = hdr.shape[0]
    y0 = max(0, int(y0))
    y1 = min(H, int(y1))
    if strength <= 0 or y1 - y0 < 4:
        return hdr
    bw, bh = max(2, W // 8), max(2, (y1 - y0) // 8)
    n = fbm2_np(np_rng, bw, bh, 3, stretch=6.0)
    t = np.linspace(0.0, 1.0, bh, dtype=F32)[:, None]
    a_small = (np.clip(np.sin(np.pi * t), 0.0, 1.0) ** 1.4) * (0.45 + 0.55 * n) * F32(strength)
    a = resize2(a_small, (y1 - y0, W))[..., None]

    fog_base = srgb2lin(color_srgb)[None, None, :]
    gx = np.arange(W, dtype=F32)[None, :]

    if shadow_strip is not None:
        sx, sv = shadow_strip
        sh = np.interp(gx / W, sx, sv).astype(F32)[..., None]
    else:
        sh = np.ones((1, W, 1), F32)

    if rig.in_front:
        fog_col = fog_base * (0.75 + 0.25 * sh)
    else:
        fwd = np.exp(-((gx - rig.x) / (0.42 * W)) ** 2)[..., None]
        sun_glow = (rig.sun_col * rig.sun_I * 0.28)[None, None, :] * fwd * (0.3 + 0.7 * sh)
        fog_col = fog_base * (0.60 + 0.40 * sh) * (1.0 - 0.35 * fwd) + sun_glow

    hdr[y0:y1] = hdr[y0:y1] * (1.0 - a) + fog_col * a
    return hdr

# ------------------------- HDR-озеро --------------------------------------

def noisy_shore(rng, W, mean_y, amp, n=48):
    xs = np.linspace(0, W, n)
    ys = mean_y + np.array([rng.gauss(0, 1) for _ in range(n)]) * amp
    return np.interp(np.arange(W), xs, ys)

def add_lake_hdr(hdr, seed, pal, rig, W, H, top, bot, sil_full, fog):
    """Озеро в линейном HDR: постолбцовое отражение + окклюзия солнца + шумная рябь без муара."""
    rng = random.Random(seed + 404)
    np_rng = np.random.default_rng(seed + 404)

    lw = max(2, W // 2)
    y0 = max(0, int(top.min()) - 2)
    y1 = min(H, int(bot.max()) + 2)
    lh = y1 - y0
    if lh < 4:
        return hdr
    lh2 = max(2, lh // 2)

    x_half = np.linspace(0, W - 1, lw)
    top2 = np.interp(x_half, np.arange(W), top).astype(F32)
    bot2 = np.interp(x_half, np.arange(W), bot).astype(F32)
    sil2 = np.interp(x_half, np.arange(W), sil_full).astype(F32)
    ys2 = np.linspace(y0, y1 - 1, lh2, dtype=F32)[:, None]

    scene_half = resize_rgb(hdr, (H, lw))

    dy = np.maximum(0.0, ys2 - top2[None, :])
    tt = np.clip(dy / np.maximum(bot2 - top2, 1.0)[None, :], 0.0, 1.0)

    # Анизотропный шум волн (устраняет муар регулярных синусоид)
    wave_noise = fbm2_np(np_rng, max(4, lw // 4), max(4, lh2 // 2), octaves=3, stretch=9.0) - 0.5
    wave_noise = resize2(wave_noise, (lh2, lw))

    ph1, ph2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    wave_y = (np.sin(dy * 0.25 + ph1) * 1.5 + wave_noise * 8.0) * (0.4 + tt)
    src_y = np.clip(np.round(top2[None, :] - dy * 1.15 - np.abs(wave_y)), 0, H - 1).astype(np.int32)

    gx = np.arange(lw, dtype=F32)[None, :]
    ripple_x = (np.sin(dy * 0.22 + ph2) * 1.2 + wave_noise * 11.0) * (0.3 + tt * 1.4) * (lw / 1920.0)
    src_x = np.clip(np.round(gx + ripple_x), 0, lw - 1).astype(np.int32)

    refl = scene_half[src_y, src_x]

    wcol = grad_rows_lin([(0.0, pal["water_top"]), (1.0, pal["water_deep"])], lh2)
    water = np.repeat(wcol[:, None, :], lw, axis=1)

    fresnel = (0.45 + 0.38 * (tt ** 0.7))[..., None]
    out = water * (1.0 - fresnel) + refl * fresnel

    # Солнечная дорожка с проверкой перекрытия диска горами и стохастическим блеском волн
    if not rig.in_front:
        vis_col = np.clip((sil2[None, :] - (rig.y - rig.r * 0.6)) / max(rig.r * 1.4, 2.0), 0.0, 1.0)
        gw = 0.014 * lw * (1.0 + tt * 4.5)
        glint = np.exp(-((gx - rig.x / 2.0) / gw) ** 2) * (0.48 * (1.0 - tt * 0.65)) * vis_col

        sparkle_noise = fbm2_np(np_rng, max(4, lw // 2), lh2, octaves=3, stretch=12.0)
        sparkle_noise = resize2(sparkle_noise, (lh2, lw))
        sp = np.clip((sparkle_noise - 0.32) * 2.1, 0.0, 1.4) ** 1.6
        out += (glint * sp)[..., None] * (rig.sun_col * rig.sun_I * 1.8)[None, None, :]

    out *= (1.0 + 0.06 * wave_noise)[..., None]
    wet_edge = np.clip(dy / (0.004 * H + 1.0), 0.0, 1.0)[..., None]
    out *= (0.68 + 0.32 * wet_edge)

    mask_small = ((ys2 >= top2[None, :]) & (ys2 <= bot2[None, :])).astype(F32)
    water_full = resize_rgb(out, (lh, W))
    mask_full = resize2(mask_small, (lh, W))[..., None]

    hdr[y0:y1] = hdr[y0:y1] * (1.0 - mask_full) + water_full * mask_full

    if fog > 0:
        mirror = float(top.mean())
        hdr = add_fog_band_hdr(hdr, np_rng, rig, mirror - 0.02 * H, mirror + 0.03 * H,
                               pal["haze"], fog * 0.5, W)
    return hdr

# ------------------------- HDR-деревья и контактный AO --------------------

def draw_trees_hdr(hdr, trees, pal, rig, horizon_y, W, H, shadows):
    if not trees:
        return hdr

    y_min = max(0, int(min(y - h * 1.15 for _, y, h, _ in trees)) - 4)
    y_max = H
    if y_min >= y_max:
        return hdr
    band_h = y_max - y_min

    # 1. Перспективные тени + контактный Ambient Occlusion у корней
    if shadows:
        sh_img = Image.new("L", (W, band_h), 0)
        sh_draw = ImageDraw.Draw(sh_img)
        sun_elev = max(0.08, (horizon_y - rig.y) / H)
        len_scale = min(1.5, 0.36 / sun_elev)

        for x, y, h, _ in trees:
            # Контактный AO у основания ствола (приземляет дерево в траву)
            ao_rx, ao_ry = h * 0.14, h * 0.028
            sh_draw.ellipse([
                x - ao_rx, y - ao_ry - y_min,
                x + ao_rx, y + ao_ry - y_min
            ], fill=75)

            if rig.in_front:
                dx = -(x - rig.x)
                dy = -max(15.0, y - horizon_y) * 0.35
            else:
                dx = x - rig.x
                dy = max(15.0, y - horizon_y)
            norm = math.hypot(dx, dy)
            vx, vy = dx / norm, dy / norm
            ln_ = h * 0.45 * len_scale
            w2 = h * 0.055

            for k, val in ((1.45, 42), (1.0, 90)):
                L_, W_ = ln_ * k, w2 * (1.0 + 0.85 * (k - 1.0))
                pxk, pyk = -vy * W_, vx * W_
                sh_draw.polygon([
                    (x - pxk, y - pyk - y_min),
                    (x + pxk, y + pyk - y_min),
                    (x + vx * L_ + pxk * 0.4, y + vy * L_ + pyk * 0.4 - y_min),
                    (x + vx * L_ - pxk * 0.4, y + vy * L_ - pyk * 0.4 - y_min),
                ], fill=val)

        sh_mask = (np.asarray(sh_img, dtype=F32) / 255.0) * 0.48
        hdr[y_min:y_max] *= (1.0 - sh_mask)[..., None]

    # 2. Геометрия крон и стволов
    id_img = Image.new("I", (W, band_h), 0)
    id_draw = ImageDraw.Draw(id_img)

    sky_amb = 0.6 * srgb2lin(pal["sky"][0][1]) + 0.4 * srgb2lin(pal["sky"][-1][1])
    sun_lin = rig.sun_col * rig.sun_I
    trunk_lin = srgb2lin(pal["trunk"]) * (sky_amb * 0.7 + sun_lin * 0.25)

    colors = [np.zeros(3, F32), trunk_lin]
    tid = 2

    for x, y, h, tseed in trees:
        trng = random.Random(tseed)
        base_lin = srgb2lin(scale_rgb(pal["tree"], trng.uniform(0.85, 1.15)))

        if shadows:
            front_boost = 1.25 if rig.in_front else 0.85
            lit_lin = base_lin * (sky_amb * 0.75 + sun_lin * front_boost)
            dark_lin = base_lin * (sky_amb * (0.72 if rig.in_front else 0.52))
            rim_lin = base_lin * sky_amb * 0.6 + sun_lin * 0.14
        else:
            lit_lin = dark_lin = rim_lin = base_lin * (sky_amb * 0.85)

        c_lit, c_dark, c_rim = tid, tid + 1, tid + 2
        colors.extend([lit_lin, dark_lin, rim_lin])
        tid += 3

        w = h * trng.uniform(0.40, 0.52)
        tw = max(2.0, h * 0.05)
        lean = trng.uniform(-0.03, 0.03) * h
        id_draw.polygon([
            (x - tw / 2, y - y_min), (x + tw / 2, y - y_min),
            (x + lean + tw / 3, y - h * 0.16 - y_min), (x + lean - tw / 3, y - h * 0.16 - y_min)
        ], fill=1)

        tiers = trng.randint(3, 4)
        for i in range(tiers):
            t_frac = i / tiers
            top_y = y - h + t_frac * h * 0.68 - y_min
            bot_y = top_y + h * (0.48 - 0.04 * tiers)
            hw = w * (0.32 + 0.68 * ((i + 1) / tiers))
            skew = trng.uniform(-0.06, 0.06) * hw
            apex = (x + lean * (1.0 - t_frac), top_y)
            left = (x - hw + skew, bot_y)
            right = (x + hw + skew, bot_y)
            mid = (x + lean * (1.0 - t_frac) * 0.5, bot_y)

            if shadows:
                sun_left = (rig.x > x) if rig.in_front else (rig.x < x)
                l_id, r_id = (c_lit, c_dark) if sun_left else (c_dark, c_lit)
                id_draw.polygon([apex, left, mid], fill=l_id)
                id_draw.polygon([apex, mid, right], fill=r_id)
                if not rig.in_front and rig.el < 0.42:
                    s = -1.0 if sun_left else 1.0
                    id_draw.polygon([apex, (x + s * hw * 0.92, bot_y), (x + s * hw, bot_y)], fill=c_rim)
            else:
                id_draw.polygon([apex, left, right], fill=c_lit)

    idx_map = np.asarray(id_img, dtype=np.int32)
    mask = idx_map > 0
    if mask.any():
        lut = np.stack(colors, axis=0)
        hdr[y_min:y_max][mask] = lut[idx_map[mask]]
    return hdr

# ------------------------- сборка сцены -----------------------------------

def generate(args):
    rng_main = random.Random(args.seed)
    W, H = args.width * args.ss, args.height * args.ss
    shadows = not args.no_shadows
    pal = load_palette(args.palette, args.hue, args.sat)
    water_on = args.water > 0.01
    night = bool(pal["stars"])

    cols = max(8, int(30 * args.density))
    rows_m = max(3, int(7 * args.density))
    xs = list(np.linspace(-0.02 * W, 1.02 * W, cols + 1))
    grid = np.linspace(-0.02 * W, 1.02 * W, 257)
    L = args.layers if args.layers else {"range": 4, "massif": 3, "single": 2}[args.scale]

    # --- 1. Профили хребтов ---
    rng_mount = random.Random(args.seed + 202)
    layers = []
    for i in range(L):
        tt = i / (L - 1) if L > 1 else 1.0
        if args.scale == "single" and i == 0:
            base, amp = (0.74 if water_on else 0.84) * H, (0.60 if water_on else 0.64) * H
        elif args.scale == "single":
            base, amp = (0.78 if water_on else 0.90) * H, 0.05 * H
        else:
            b0, b1 = (0.46, 0.72) if water_on else (0.52, 0.82)
            a0, a1 = (0.33, 0.15) if water_on else (0.36, 0.17)
            if args.scale == "massif":
                a0 = 0.44 if water_on else 0.48
            base, amp = (b0 + (b1 - b0) * tt) * H, (a0 + (a1 - a0) * tt) * H
        amp = min(amp, base - 0.07 * H)
        raw = ridge_line(rng_mount, 257, 0.52 + 0.10 * tt, 1.35 + 0.4 * tt)
        if args.scale == "range":
            hgt = raw
        else:
            env = scale_envelope(args.scale, rng_mount, np.linspace(-0.02, 1.02, 257))
            hgt = env * (0.45 + 0.55 * raw)
        hgt = hgt / max(hgt.max(), 1e-6)
        layers.append(dict(tt=tt, ys=list(base - np.interp(xs, grid, hgt) * amp), base=base, amp=amp))

    x_full = np.arange(W, dtype=F32)
    sil_full = np.minimum.reduce([np.interp(x_full, xs, l["ys"]) for l in layers]).astype(F32)

    # --- 2. Позиционирование солнца ---
    sxf = args.sun_x if args.sun_x is not None else rng_main.uniform(0.2, 0.8)
    horizon_norm = 0.62
    vfov = 80.0

    if args.sun_elev is not None:
        elev = args.sun_elev
    elif args.light_pos == "behind":
        elev = {"sunset": 7.5, "day": 14.0, "winter": 10.0, "night": 14.0, "mono": 12.0}[args.palette]
    elif args.light_pos == "above":
        sil_at_sun = float(np.interp(sxf * W, x_full, sil_full))
        min_elev = (horizon_norm - (sil_at_sun - 0.06 * H) / H) * vfov
        default_elev = {"sunset": 22.0, "day": 45.0, "winter": 28.0, "night": 34.0, "mono": 35.0}[args.palette]
        elev = max(default_elev, min_elev)
    else:
        elev = {"sunset": 18.0, "day": 42.0, "winter": 26.0, "night": 32.0, "mono": 34.0}[args.palette]

    rig = SunRig(
        sxf, elev, W, H, horizon=horizon_norm, vfov=vfov,
        light_pos=args.light_pos, front=args.front_light,
        night=night, turbidity=args.turbidity
    )

    # --- 3. Небо, облака и небесные лучи ---
    aw = min(W, 1600)
    ah = max(4, int(round(aw * H / W)))
    sky = render_sky(pal, rig, aw, ah, args.sky_phys, args.turbidity, night)
    shader = Shader(rig, pal, W, H, args.ambient, args.backlight, shadows=shadows)

    if args.clouds > 0.01:
        cw, csteps, cls, vol = CLOUD_Q[args.cloud_quality]
        cf = CloudField(args.seed + 777, vol, args.clouds, args.cloud_type, args.cloud_warp)
        amb_c = 0.55 * srgb2lin(pal["sky"][0][1]) + 0.30 * srgb2lin(pal["haze"])
        crgb, ca, reg = cf.render(rig, cw, amb_c, srgb2lin(pal["haze"]), csteps, cls, seed=args.seed)
        composite_clouds(sky, crgb, ca, reg)
        shader.cloud_x, shader.cloud = cf.shadow_strip(rig)

    if args.rays > 0.01 and not rig.in_front:
        xp = np.arange(aw) * (W / aw)
        sil_aw = np.interp(xp, x_full, sil_full)
        god_rays(sky, sil_aw, rig, aw, ah, W, H, args.rays * 0.75)

    hdr = resize_rgb(sky, (H, W))

    if night:
        rng_stars = random.Random(args.seed + 105)
        star_img = Image.new("L", (W, int(H * 0.55)), 0)
        sd = ImageDraw.Draw(star_img)
        for _ in range(300):
            sx_, sy_ = rng_stars.uniform(0, W), rng_stars.uniform(0, H * 0.55)
            sr = rng_stars.uniform(0.6, 1.9) * (W / 3840)
            v = rng_stars.randint(90, 255)
            sd.ellipse([sx_ - sr, sy_ - sr, sx_ + sr, sy_ + sr], fill=v)
        smask = (np.asarray(star_img, dtype=F32) / 255.0) * 1.8
        hdr[:star_img.height] += smask[..., None] * rig.sun_col[None, None, :]

    # --- 4. Слои гор, туман с тенями и межгорные объёмные лучи ---
    np_rng_fog = np.random.default_rng(args.seed + 203)
    n_tri = 0
    front_rs = None
    for i, l in enumerate(layers):
        tt, ys = l["tt"], l["ys"]
        snow_y = l["base"] - l["amp"] * 0.53 if (pal["snow"] and i < max(1, math.ceil(L / 2))) else None
        rs = ridge_cast_shadow(xs, ys, rig, W) if shadows else None
        if i == L - 1:
            front_rs = rs

        shader.set_layer(pal["mount"], pal["mount_low"], 0.72 * (1 - tt) ** 1.2,
                         min(ys), y_base=l["base"], snow_y=snow_y, snow_col=pal["snow"], ridge_shadow=rs)
        n_tri += draw_mosaic_layer_hdr(hdr, rng_mount, W, H, xs, ys, shader,
                                       ValueNoise(rng_mount, 13, 11), 0.12 + 0.06 * tt, cols, rows_m)

        # Объёмные лучи между планами гор (проходят над этой грядой, но перекрываются следующей!)
        if args.rays > 0.01 and not rig.in_front and i < L - 1:
            sil_i = np.interp(x_full, xs, ys).astype(F32)
            shaft_strength = args.rays * 0.24 * (1.0 - tt * 0.5)
            hdr = crepuscular_shafts_hdr(hdr, sil_i, rig, W, H, strength=shaft_strength)

        if args.fog > 0 and i < L - 1:
            hdr = add_fog_band_hdr(hdr, np_rng_fog, rig, l["base"] - 0.05 * H, l["base"] + 0.13 * H,
                                   pal["haze"], args.fog * (0.45 + 0.4 * (1 - tt)), W, shadow_strip=rs)

    # --- 5. Земля / берега (принимает тень от ближайшего горного хребта) ---
    g_base = 0.755 if water_on else 0.93
    g_amp = 0.018 if water_on else 0.035
    graw = ridge_line(rng_mount, 257, 0.5, 1.0)
    gys = list(g_base * H - np.interp(xs, grid, graw) * g_amp * H)
    shader.set_layer(pal["ground"], scale_rgb(pal["ground"], 0.8), 0.0, min(gys),
                     y_base=g_base * H, ridge_shadow=front_rs)
    n_tri += draw_mosaic_layer_hdr(hdr, rng_mount, W, H, xs, gys, shader,
                                   ValueNoise(rng_mount), 0.05, cols, max(3, rows_m // 2))

    # --- 6. Птицы ---
    if not night:
        rng_birds = random.Random(args.seed + 303)
        b_img = Image.new("L", (W, int(H * 0.35)), 0)
        bd = ImageDraw.Draw(b_img)
        for _ in range(rng_birds.randint(3, 6)):
            bx, by = rng_birds.uniform(0.1, 0.9) * W, rng_birds.uniform(0.10, 0.30) * H
            s = rng_birds.uniform(0.006, 0.012) * W
            bd.line([bx - s, by, bx, by - s * 0.4, bx + s, by],
                    fill=255, width=max(2, int(W / 1600)))
        bmask = (np.asarray(b_img, dtype=F32) / 255.0)[..., None]
        bird_lin = srgb2lin(scale_rgb(pal["mount"], 0.45))[None, None, :]
        hdr[:b_img.height] = hdr[:b_img.height] * (1.0 - bmask) + bird_lin * bmask

    # --- 7. Деревья и озеро ---
    rng_trees = random.Random(args.seed + 505)
    far, near = [], []
    horizon_y = g_base * H

    if water_on:
        lake_mean = 0.775 * H
        lake_bot_mean = lake_mean + args.water * (0.975 * H - lake_mean)
        top = noisy_shore(rng_trees, W, lake_mean, 0.006 * H)
        bot = noisy_shore(rng_trees, W, lake_bot_mean, 0.008 * H)
        for _ in range(int(args.trees * 0.35)):
            x = rng_trees.uniform(0.01, 0.99) * W
            gy = float(np.interp(x, xs, gys))
            ty = float(np.interp(x, np.arange(W), top))
            y = rng_trees.uniform(gy + 0.004 * H, max(ty - 0.004 * H, gy + 0.008 * H))
            far.append((x, y, 0.022 * H * rng_trees.uniform(0.7, 1.3), rng_trees.randint(0, 10 ** 6)))
        for _ in range(args.trees - len(far)):
            x = rng_trees.uniform(0.01, 0.99) * W
            by = float(np.interp(x, np.arange(W), bot))
            y = rng_trees.uniform(by + 0.008 * H, 0.995 * H)
            depth = min(max((y - lake_bot_mean) / (H - lake_bot_mean), 0), 1)
            near.append((x, y, (0.045 + 0.085 * depth) * H * rng_trees.uniform(0.75, 1.3),
                         rng_trees.randint(0, 10 ** 6)))
    else:
        for _ in range(args.trees):
            x = rng_trees.uniform(0.01, 0.99) * W
            gy = float(np.interp(x, xs, gys))
            y = rng_trees.uniform(gy + 0.02 * H, 0.995 * H)
            depth = min(max((y / H - 0.90) / 0.10, 0.0), 1.0)
            near.append((x, y, (0.045 + 0.10 * depth) * H * rng_trees.uniform(0.75, 1.3),
                         rng_trees.randint(0, 10 ** 6)))

    far.sort(key=lambda t: t[1])
    near.sort(key=lambda t: t[1])

    hdr = draw_trees_hdr(hdr, far, pal, rig, horizon_y, W, H, shadows)
    if water_on:
        hdr = add_lake_hdr(hdr, args.seed, pal, rig, W, H, top, bot, sil_full, args.fog)
    hdr = draw_trees_hdr(hdr, near, pal, rig, horizon_y, W, H, shadows)

    if args.ss > 1:
        hdr = resize_rgb(hdr, (args.height, args.width))

    img = hdr_postprocess_and_tonemap(
        hdr, args.seed, bloom=args.bloom, exposure=args.exposure
    )
    img.save(args.out)
    return n_tri

# ------------------------- CLI --------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Процедурный горный пейзаж-мозаика v4.2 (HDR + межгорные лучи + барометрическая дымка).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--ss", type=int, default=2, help="суперсэмплинг 1..4 (антиалиасинг)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--palette", choices=list(PALETTES), default="sunset")
    p.add_argument("--hue", type=float, default=0.0, help="поворот оттенка, градусы")
    p.add_argument("--sat", type=float, default=1.0, help="множитель насыщенности")
    p.add_argument("--scale", choices=["range", "massif", "single"], default="range",
                   help="масштаб: хребет / массив / одна скала")
    p.add_argument("--trees", type=int, default=140)
    p.add_argument("--density", type=float, default=1.0, help="плотность мозаики")
    p.add_argument("--layers", type=int, default=None,
                   help="слоёв гор (по умолчанию зависит от --scale)")
    p.add_argument("--fog", type=float, default=0.45, help="туман в долинах, 0..1")
    p.add_argument("--clouds", type=float, default=0.5, help="облачность, 0..1")
    p.add_argument("--water", type=float, default=0.6,
                   help="размер озера, 0..1 (0 = без воды)")
    p.add_argument("--light-pos", choices=["behind", "above", "front"], default="behind",
                   help="позиция источника света: за горами / над горами / перед горами")
    p.add_argument("--sun-x", type=float, default=None, help="солнце по X, 0..1")
    p.add_argument("--sun-elev", type=float, default=None, help="высота светила, градусы")
    p.add_argument("--front-light", type=float, default=None,
                   help="ручной коэффициент фронтальности 0..1 (переопределяет --light-pos)")
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--out", default="landscape_hdr.png")
    p.add_argument("--turbidity", type=float, default=1.2, help="мутность атмосферы")
    p.add_argument("--sky-phys", type=float, default=0.75, help="доля физического неба 0..1")
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--ambient", type=float, default=0.5)
    p.add_argument("--backlight", type=float, default=0.55)
    p.add_argument("--cloud-type", choices=["cumulus", "stratus", "cirrus"], default="cumulus")
    p.add_argument("--cloud-quality", choices=list(CLOUD_Q), default="med")
    p.add_argument("--cloud-warp", type=float, default=0.4)
    p.add_argument("--rays", type=float, default=0.45)
    p.add_argument("--bloom", type=float, default=0.25)
    p.add_argument("--budget", type=int, default=120, help="лимит внутреннего буфера, Мпикс")
    args = p.parse_args()
    if args.seed is None:
        args.seed = random.randrange(2 ** 31)
    args.ss = max(1, min(4, args.ss))
    budget = max(1, args.budget) * 1_000_000
    if args.width * args.height * args.ss ** 2 > budget:
        args.ss = max(1, int((budget / (args.width * args.height)) ** 0.5))
        print(f"Предупреждение: суперсэмплинг снижен до {args.ss} (лимит --budget {args.budget} Мпикс).")
    for k in ("fog", "clouds", "water", "sky_phys", "ambient", "backlight",
              "rays", "bloom", "cloud_warp"):
        setattr(args, k, max(0.0, min(1.0, getattr(args, k))))
    if args.front_light is not None:
        args.front_light = max(0.0, min(1.0, args.front_light))

    t0 = time.time()
    n_tri = generate(args)
    print(f"Сохранено: {args.out}  ({args.width}x{args.height})")
    print(f"seed={args.seed}  палитра={args.palette}  свет={args.light_pos}  "
          f"треугольников={n_tri}  деревьев={args.trees}  время={time.time() - t0:.1f} c")

if __name__ == "__main__":
    main()