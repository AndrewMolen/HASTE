"""Generate the application icon: a multi-hole injector orifice plate.

The icon is what the tool actually designs -- a circular plate with a ring of
drilled orifices -- so it reads as this application rather than as a generic
Python script. Drawn at 1024 px and downsampled into a multi-resolution ``.ico``
so it stays legible from a 256 px tile down to a 16 px taskbar corner.

Run:
    python tools/make_icon.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets")
OUT_ICO = os.path.join(OUT_DIR, "n2o_injector.ico")
OUT_PNG = os.path.join(OUT_DIR, "n2o_injector.png")

SIZES = [256, 128, 64, 48, 32, 16]

PLATE = (0x2B, 0x3A, 0x55)      # deep steel blue
PLATE_EDGE = (0x1B, 0x27, 0x3B)
HOLE = (0x0D, 0x14, 0x20)
FLOW = (0x4F, 0xC3, 0xF7)       # nitrous-blue jets
CENTRE = (0x7E, 0xC8, 0xE3)


def draw_plate(px: int = 1024, detail: str = "full") -> Image.Image:
    """Render the plate at high resolution on a transparent background.

    ``detail`` trades hole count for legibility at small sizes: ``full`` is the
    19-hole showerhead, ``mid`` keeps one ring of larger bores, ``low`` is a
    plate with a single central bore for 16 px.
    """
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    c = px / 2
    r_outer = px * 0.46

    # Plate body, with a darker rim for a machined edge.
    d.ellipse([c - r_outer, c - r_outer, c + r_outer, c + r_outer],
              fill=PLATE, outline=PLATE_EDGE, width=int(px * 0.022))

    if detail == "low":
        hr = px * 0.15
        d.ellipse([c - hr * 1.35, c - hr * 1.35, c + hr * 1.35, c + hr * 1.35], fill=FLOW)
        d.ellipse([c - hr, c - hr, c + hr, c + hr], fill=HOLE)
        return img

    # Subtle inner relief ring (full detail only -- it muddies the mid sizes).
    if detail == "full":
        r_ring = px * 0.375
        d.ellipse([c - r_ring, c - r_ring, c + r_ring, c + r_ring],
                  outline=PLATE_EDGE, width=int(px * 0.012))

    # Orifices on bolt circles, plus one on the axis -- a real showerhead
    # pattern rather than decoration.
    def ring_of_holes(radius_frac: float, count: int, hole_frac: float, phase: float):
        rr = px * radius_frac
        hr = px * hole_frac
        for i in range(count):
            a = phase + 2 * math.pi * i / count
            hx, hy = c + rr * math.cos(a), c + rr * math.sin(a)
            # Flow glow, then the bore itself.
            d.ellipse([hx - hr * 1.55, hy - hr * 1.55,
                       hx + hr * 1.55, hy + hr * 1.55], fill=FLOW)
            d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=HOLE)

    if detail == "full":
        ring_of_holes(0.295, 12, 0.032, 0.0)
        ring_of_holes(0.165, 6, 0.030, math.pi / 6)
        hr = px * 0.036
    else:  # mid
        ring_of_holes(0.255, 6, 0.070, 0.0)
        hr = px * 0.070

    # Central orifice.
    d.ellipse([c - hr * 1.6, c - hr * 1.6, c + hr * 1.6, c + hr * 1.6], fill=CENTRE)
    d.ellipse([c - hr, c - hr, c + hr, c + hr], fill=HOLE)

    return img


def main() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Nineteen holes turn to mud below ~64 px, so each size gets the most
    # detailed variant that still resolves at that scale.
    variants = {d: draw_plate(1024, d) for d in ("full", "mid", "low")}

    def variant_for(s: int) -> str:
        if s >= 64:
            return "full"
        return "mid" if s >= 24 else "low"

    frames = [
        variants[variant_for(s)].resize((s, s), Image.LANCZOS) for s in SIZES
    ]
    base = variants["full"]

    frames[0].save(OUT_ICO, format="ICO",
                   sizes=[(f.width, f.height) for f in frames], append_images=frames[1:])
    base.resize((512, 512), Image.LANCZOS).save(OUT_PNG)

    print(f"wrote {OUT_ICO}  ({', '.join(str(s) for s in SIZES)} px)")
    print(f"wrote {OUT_PNG}")
    return OUT_ICO


if __name__ == "__main__":
    main()
