import colorsys
import numpy as np
from PIL import Image


def extract_palette(cover_path, count=5):
    img = Image.open(cover_path).convert("RGB")
    small = img.resize((60, 60), Image.Resampling.BOX)
    pixels = np.asarray(small, dtype=np.float32).reshape(-1, 3)

    rgb = pixels / 255.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / mx, 0.0)

    luma = np.dot(pixels, [0.299, 0.587, 0.114])

    if float(np.mean(sat)) < 0.06:
        levels = np.quantile(luma, np.linspace(0.15, 0.85, count)) / 255.0
        return [(float(x), float(x), float(x)) for x in levels]

    mask = (luma > 25) & (luma < 245) & (sat > 0.08)
    valid = rgb[mask]

    if len(valid) < count:
        return [
            (0.90, 0.90, 0.90),
            (0.70, 0.70, 0.70),
            (0.50, 0.50, 0.50),
            (0.30, 0.30, 0.30),
            (0.15, 0.15, 0.15),
        ]

    chosen = valid[np.random.choice(len(valid), count, replace=False)]

    final = []
    for r, g, b in chosen:
        h, l, s = colorsys.rgb_to_hls(float(r), float(g), float(b))
        s = max(s, 0.35)
        l = min(max(l, 0.35), 0.75)
        rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
        final.append((float(rr), float(gg), float(bb)))
    return final
