"""Cross-section geometry for the sector-port ("wagon wheel") fuel grain.

The layout is a round centre port surrounded by ``n`` annular-sector ports.
Each sector port is bounded by

* an inner arc at radius ``r1`` (concave, seen from inside the port),
* an outer arc at radius ``r2`` (convex),
* two straight sides, each parallel to a spoke centreline and offset from it
  by half the spoke thickness, so every spoke is a constant-thickness web.

Constant-thickness webs matter: a web burns from both faces at the same rate,
so a web of uniform thickness is consumed everywhere at once instead of
breaking through at its thinnest point and leaving the rest as a sliver.

**Regression.** Every port surface recedes normal to itself by the same
distance ``e``. For any closed port outline whose offset stays simple (no port
has met another, no concave arc has shrunk to a point), Steiner's formula
gives the area and perimeter exactly, sharp corners and concave arcs
included::

    A(e) = A0 + P0 e + pi e^2
    P(e) = P0 + 2 pi e

The ``pi e^2`` term is half the total turning of the outline, which is 2 pi
for any simple closed curve regardless of shape. It is exact until ports
merge, which for a balanced layout is the same instant the wall web burns
through, so no approximation is needed over the useful life of the grain.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq


def _F(r: float, h: float) -> float:
    """Antiderivative of ``r * asin(h / r)`` with respect to ``r``."""
    return 0.5 * r * r * math.asin(h / r) + 0.5 * h * math.sqrt(r * r - h * h)


def sector_port(r1: float, r2: float, h: float, n: int) -> tuple[float, float]:
    """Area and perimeter of one sector port, before any regression.

    ``r1``/``r2`` are the inner and outer arc radii, ``h`` is half the spoke
    thickness and ``n`` the number of sector ports (the sector spans
    ``2 pi / n``). Any consistent length unit.
    """
    alpha = 2.0 * math.pi / n
    area = 0.5 * alpha * (r2 * r2 - r1 * r1) - 2.0 * (_F(r2, h) - _F(r1, h))
    arcs = r2 * (alpha - 2.0 * math.asin(h / r2)) + r1 * (alpha - 2.0 * math.asin(h / r1))
    sides = 2.0 * (math.sqrt(r2 * r2 - h * h) - math.sqrt(r1 * r1 - h * h))
    return area, arcs + sides


def sector_errors(outer_d: float, centre_d: float, ring_web: float,
                  spoke_web: float, wall_web: float, n: int) -> list[str]:
    """Reasons this sector layout cannot exist, in the units given."""
    errs = []
    if n < 2:
        return ["a sector grain needs at least 2 sector ports"]
    if min(centre_d, ring_web, spoke_web, wall_web) <= 0:
        return ["centre port diameter and all web thicknesses must be positive"]
    r1 = 0.5 * centre_d + ring_web
    r2 = 0.5 * outer_d - wall_web
    h = 0.5 * spoke_web
    if r2 <= r1:
        errs.append(
            f"the sector ports have no radial depth: inner radius {r1 * 1e3:.2f} mm is "
            f"not inside outer radius {r2 * 1e3:.2f} mm. Thin the webs or shrink "
            "the centre port."
        )
        return errs
    alpha = 2.0 * math.pi / n
    if h >= r1 or 2.0 * math.asin(h / r1) >= alpha:
        errs.append(
            f"{n} spokes of {spoke_web * 1e3:.2f} mm do not fit around the "
            f"{r1 * 1e3:.2f} mm inner radius. Use fewer sectors or thinner spokes."
        )
    return errs


def balanced_sector(outer_d: float, port_area: float, n: int = 4) -> dict[str, float]:
    """A sector layout with evenly matched webs for a given total port area.

    Two conditions fix the two free dimensions:

    * **balanced webs** -- the spokes and the ring between the centre port and
      the sectors are twice the wall web, because they burn from both faces
      and the wall web burns from one. Every web is then consumed at the same
      regression depth, so no port breaks into another before the grain
      reaches the case, and no fuel is left behind as a sliver of web;
    * **equal hydraulic diameter** -- the centre port and the sector ports
      have the same ``4A/P``, so the oxidiser divides between them in
      proportion to area and each sees about the same flux.

    Returns ``centre_d``, ``ring_web``, ``spoke_web`` and ``wall_web`` in the
    units of the inputs.
    """
    Ro = 0.5 * outer_d

    def layout(rc: float, w: float) -> tuple[float, float, float]:
        r1, r2 = rc + 2.0 * w, Ro - w
        a, p = sector_port(r1, r2, w, n)
        return math.pi * rc * rc + n * a, a, p

    def wall_for(rc: float) -> float:
        # Port area falls monotonically as the webs thicken: from nearly the
        # whole cross-section at w -> 0 to just the centre port at the web
        # that closes the sectors to zero depth (r1 = r2 at w = (Ro - rc)/3).
        lo, hi = 1e-6 * outer_d, (Ro - rc) / 3.0 * 0.999
        return float(brentq(lambda w: layout(rc, w)[0] - port_area, lo, hi, xtol=1e-12 * outer_d))

    def dh_mismatch(rc: float) -> float:
        w = wall_for(rc)
        _, a, p = layout(rc, w)
        return 2.0 * rc - 4.0 * a / p

    rc_max = math.sqrt(port_area / math.pi) * 0.999
    rc = float(brentq(dh_mismatch, 0.02 * rc_max, rc_max * 0.98, xtol=1e-12 * outer_d))
    w = wall_for(rc)
    return {"centre_d": 2.0 * rc, "ring_web": 2.0 * w, "spoke_web": 2.0 * w, "wall_web": w}


def sector_outlines(outer_d: float, centre_d: float, ring_web: float,
                    spoke_web: float, n: int, wall_web: float):
    """Exact outline of each sector port, for drawing and DXF.

    Yields one dict per port with the inner and outer arcs as
    ``(radius, start_deg, end_deg)`` (counter-clockwise) and the two straight
    sides as ``((x0, y0), (x1, y1))``. Spoke ``k`` lies along
    ``k * 360/n`` degrees, so the ports sit between spokes.
    """
    r1 = 0.5 * centre_d + ring_web
    r2 = 0.5 * outer_d - wall_web
    h = 0.5 * spoke_web
    alpha = 2.0 * math.pi / n
    b1, b2 = math.asin(h / r1), math.asin(h / r2)
    for k in range(n):
        th0 = k * alpha
        a_in = (th0 + b1, th0 + alpha - b1)
        a_out = (th0 + b2, th0 + alpha - b2)
        side_a = ((r1 * math.cos(a_in[0]), r1 * math.sin(a_in[0])),
                  (r2 * math.cos(a_out[0]), r2 * math.sin(a_out[0])))
        side_b = ((r1 * math.cos(a_in[1]), r1 * math.sin(a_in[1])),
                  (r2 * math.cos(a_out[1]), r2 * math.sin(a_out[1])))
        yield {
            "inner": (r1, math.degrees(a_in[0]), math.degrees(a_in[1])),
            "outer": (r2, math.degrees(a_out[0]), math.degrees(a_out[1])),
            "sides": (side_a, side_b),
        }
