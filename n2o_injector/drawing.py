"""Manufacturing geometry for the orifice plate: hole layout, DXF and hole table.

This module is deliberately free of matplotlib so the layout can be generated
and exported headlessly (scripts, tests, CI). The drawing *sheet* itself is
rendered in :mod:`n2o_injector.plots`, from the same :class:`PlateLayout` that
is written to DXF -- so the printed drawing and the CAD file can never disagree
about where a hole is.

Two export formats, because a shop needs different things:

* **DXF (R12)** -- the geometry, to scale, in millimetres, on named layers so
  the annotation and section view can be switched off and the hole circles used
  directly as drill targets. R12 is the most widely readable DXF revision;
  every CAD and CAM package still imports it.
* **CSV hole table** -- X/Y coordinates from plate centre, for manual DRO work,
  a CMM, or a CAM package that prefers a point list to geometry.

Coordinates are in millimetres with the origin at the plate centre, X to the
right and Y up -- the same frame the drawing shows, so a coordinate read off
the table lands where the drawing puts it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime

#: Fraction of the plate radius kept clear between the outermost hole edge and
#: the plate rim. Holes too close to the edge leave no material for the seal
#: groove or bolt circle, and thin webs distort when drilled.
EDGE_MARGIN_FRAC = 0.12


@dataclass
class Ring:
    """One bolt circle of equally spaced holes."""

    radius: float  #: mm, bolt-circle *radius* (not diameter)
    count: int
    phase: float  #: rad, angular offset of the first hole from +X

    @property
    def bolt_circle_d(self) -> float:
        return 2.0 * self.radius

    @property
    def pitch_deg(self) -> float:
        return 360.0 / self.count if self.count else float("nan")

    @property
    def phase_deg(self) -> float:
        return math.degrees(self.phase)


@dataclass
class PlateLayout:
    """Everything needed to draw or machine the plate. All lengths in mm."""

    plate_d: float
    hole_d: float
    thickness: float
    holes: list[tuple[float, float]] = field(default_factory=list)
    rings: list[Ring] = field(default_factory=list)
    centre_hole: bool = False
    Cd: float = float("nan")

    @property
    def n_holes(self) -> int:
        return len(self.holes)

    @property
    def plate_r(self) -> float:
        return 0.5 * self.plate_d

    @property
    def hole_area(self) -> float:
        """mm^2 per hole."""
        return 0.25 * math.pi * self.hole_d**2

    @property
    def total_area(self) -> float:
        """mm^2 over all holes."""
        return self.n_holes * self.hole_area

    @property
    def L_over_D(self) -> float:
        return self.thickness / self.hole_d if self.hole_d > 0 else float("nan")

    @property
    def edge_margin(self) -> float:
        """Smallest distance from a hole *edge* to the plate rim, mm."""
        if not self.holes:
            return float("nan")
        far = max(math.hypot(x, y) for x, y in self.holes)
        return self.plate_r - far - 0.5 * self.hole_d

    @property
    def min_web(self) -> float:
        """Smallest edge-to-edge distance between any two holes, mm.

        Negative means the holes intersect -- the plate cannot be made as
        specified. This is the number that decides whether a hole count is
        realisable, so it is reported on the drawing and in the GUI title.
        """
        n = len(self.holes)
        if n < 2:
            return float("inf")
        gap = float("inf")
        for i in range(n):
            xi, yi = self.holes[i]
            for j in range(i + 1, n):
                xj, yj = self.holes[j]
                gap = min(gap, math.hypot(xi - xj, yi - yj) - self.hole_d)
        return gap

    @property
    def feasible(self) -> bool:
        m, e = self.min_web, self.edge_margin
        return (not math.isfinite(m) or m > 0) and (not math.isfinite(e) or e > 0)

    def area_sensitivity(self, tol: float = 0.02) -> float:
        """Percent change in total flow area for a +``tol`` mm hole-diameter error.

        Area goes as d^2, so a tolerance that looks negligible next to the hole
        diameter is amplified: this is the number to quote to the machinist.
        """
        if self.hole_d <= 0:
            return float("nan")
        return 100.0 * ((self.hole_d + tol) ** 2 / self.hole_d**2 - 1.0)


@dataclass
class DrawingMeta:
    """Title-block and notes content. Blank fields are left blank on the sheet."""

    title: str = "INJECTOR ORIFICE PLATE"
    project: str = ""
    part_no: str = ""
    material: str = ""
    drawn_by: str = ""
    flow_model: str = ""
    tolerance: str = ""
    date: str = ""
    notes: list[str] = field(default_factory=list)

    def stamped(self) -> "DrawingMeta":
        """A copy with the date filled in if it was left empty."""
        if self.date:
            return self
        d = DrawingMeta(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        d.date = f"{datetime.now():%Y-%m-%d}"
        return d


def ring_positions(n_holes, plate_r, hole_r, edge_margin_frac=EDGE_MARGIN_FRAC):
    """Distribute ``n_holes`` over concentric bolt circles.

    Real plates are drilled on bolt circles rather than scattered, so holes are
    spread over rings with the count per ring proportional to its
    circumference -- that keeps neighbour spacing roughly even, which is what
    governs whether the webs between holes survive drilling.

    Returns ``(rings, positions, centre_hole)``.
    """
    if n_holes <= 0 or plate_r <= 0:
        return [], [], False

    usable = plate_r * (1.0 - edge_margin_frac) - hole_r
    if usable <= 0 or n_holes == 1:
        # Nothing fits on a ring, or there is only one hole: put it on centre.
        return [], [(0.0, 0.0)], True

    n_rings = max(1, min(4, int(round((n_holes / 6.0) ** 0.5))))
    centre = n_holes % n_rings == 1 and n_holes > 6

    outer = n_holes - (1 if centre else 0)
    radii = [usable * (i + 1) / n_rings for i in range(n_rings)]
    total_r = sum(radii)
    counts = [max(1, int(round(outer * r / total_r))) for r in radii]

    # Rounding rarely lands exactly on the requested total; settle it on the
    # outer ring, which has the most room to absorb a hole either way.
    counts[-1] += outer - sum(counts)
    if counts[-1] < 1:
        counts[-1] = 1

    rings: list[Ring] = []
    pos: list[tuple[float, float]] = [(0.0, 0.0)] if centre else []
    for i, (r, c) in enumerate(zip(radii, counts)):
        c = max(int(c), 1)
        # Stagger alternate rings by half a pitch so holes on adjacent circles
        # interleave rather than lining up radially -- that maximises the web
        # between neighbours and is how plates are actually drilled.
        phase = (math.pi / c) if i % 2 else 0.0
        rings.append(Ring(radius=r, count=c, phase=phase))
        for k in range(c):
            a = phase + 2.0 * math.pi * k / c
            pos.append((r * math.cos(a), r * math.sin(a)))
    return rings, pos, centre


def plate_layout(n_holes, hole_d_mm, plate_d_mm, thickness_mm=0.0, Cd=float("nan"),
                 edge_margin_frac=EDGE_MARGIN_FRAC) -> PlateLayout:
    """Build a :class:`PlateLayout` from the sized plate's numbers."""
    rings, pos, centre = ring_positions(
        int(n_holes), 0.5 * plate_d_mm, 0.5 * hole_d_mm, edge_margin_frac
    )
    return PlateLayout(
        plate_d=float(plate_d_mm),
        hole_d=float(hole_d_mm),
        thickness=float(thickness_mm),
        holes=pos,
        rings=rings,
        centre_hole=centre,
        Cd=float(Cd),
    )


def layout_from_plate(plate, plate_d_mm=None, grain_od_mm=None) -> PlateLayout:
    """Convenience wrapper around a sized :class:`~n2o_injector.sizing.OrificePlate`.

    ``plate_d_mm`` of ``None`` or ``0`` falls back to the grain outer diameter,
    which is the usual envelope for the plate, and then to a sensible multiple
    of the hole diameter if that is unknown too.
    """
    hole_d = plate.hole_d * 1e3
    if not plate_d_mm or plate_d_mm <= 0:
        plate_d_mm = grain_od_mm if grain_od_mm else max(hole_d * 12.0, 40.0)
    return plate_layout(
        plate.n_holes, hole_d, plate_d_mm,
        thickness_mm=plate.plate_thickness * 1e3, Cd=plate.Cd,
    )


def hole_table(layout: PlateLayout) -> list[dict]:
    """Per-hole rows: index, ring, X, Y, polar radius and angle."""
    rows = []
    ring_of: list[int] = []
    if layout.centre_hole:
        ring_of.append(0)
    for i, r in enumerate(layout.rings, start=1):
        ring_of.extend([i] * r.count)
    for i, (x, y) in enumerate(layout.holes, start=1):
        rows.append({
            "hole": i,
            "ring": ring_of[i - 1] if i - 1 < len(ring_of) else 0,
            "x_mm": x,
            "y_mm": y,
            "radius_mm": math.hypot(x, y),
            "angle_deg": (math.degrees(math.atan2(y, x)) + 360.0) % 360.0,
            "diameter_mm": layout.hole_d,
        })
    return rows


def write_hole_table_csv(path: str, layout: PlateLayout,
                         meta: DrawingMeta | None = None) -> str:
    """Write the hole coordinate table for DRO, CMM or CAM use.

    The preamble lines start with ``#`` so a CAM importer can skip them while a
    human still sees the units and the plate the coordinates belong to -- a
    bare coordinate list with no datum statement is a machining error waiting
    to happen.
    """
    m = (meta or DrawingMeta()).stamped()
    rows = hole_table(layout)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {m.title}\n")
        if m.project:
            f.write(f"# project: {m.project}\n")
        if m.part_no:
            f.write(f"# part no: {m.part_no}\n")
        f.write("# units: mm. Origin at plate centre, X right, Y up, viewed from "
                "the injector inlet (upstream) face.\n")
        f.write(f"# plate dia {layout.plate_d:.3f}, thickness {layout.thickness:.3f}, "
                f"{layout.n_holes} holes of {layout.hole_d:.3f} dia, THRU\n")
        f.write(f"# total flow area {layout.total_area:.3f} mm^2, "
                f"L/D {layout.L_over_D:.2f}, Cd assumed {layout.Cd:.3f}\n")
        f.write(f"# min web between holes {layout.min_web:.3f}, "
                f"edge margin {layout.edge_margin:.3f}\n")
        f.write(f"# generated {m.date} by HASTE\n")
        cols = ["hole", "ring", "x_mm", "y_mm", "radius_mm", "angle_deg",
                "diameter_mm", "depth"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(
                f"{r['hole']},{r['ring']},{r['x_mm']:.4f},{r['y_mm']:.4f},"
                f"{r['radius_mm']:.4f},{r['angle_deg']:.3f},"
                f"{r['diameter_mm']:.4f},THRU\n"
            )
    return path


# --------------------------------------------------------------------- DXF
#
# Hand-written DXF R12 (AC1009). R12 is chosen over the newer revisions for
# one reason: it is the revision every CAD and CAM package still reads, and it
# needs no third-party dependency to write. The structure below is the minimum
# a conforming reader requires -- HEADER, a LTYPE table (which the LAYER table
# references), a LAYER table, then ENTITIES.

_LAYERS = [
    # name,             colour, linetype
    ("PLATE_OUTLINE",   7,  "CONTINUOUS"),
    ("HOLES",           1,  "CONTINUOUS"),
    ("HOLE_CENTRES",    1,  "CONTINUOUS"),
    ("CENTRELINES",     4,  "DASHED"),
    ("SECTION",         8,  "CONTINUOUS"),
    ("ANNOTATION",      3,  "CONTINUOUS"),
]


def _g(code: int, value) -> str:
    if isinstance(value, float):
        return f"{code}\n{value:.6f}\n"
    return f"{code}\n{value}\n"


def _circle(x, y, r, layer) -> str:
    return (_g(0, "CIRCLE") + _g(8, layer) + _g(10, float(x)) + _g(20, float(y))
            + _g(30, 0.0) + _g(40, float(r)))


def _line(x1, y1, x2, y2, layer) -> str:
    return (_g(0, "LINE") + _g(8, layer) + _g(10, float(x1)) + _g(20, float(y1))
            + _g(30, 0.0) + _g(11, float(x2)) + _g(21, float(y2)) + _g(31, 0.0))


def _point(x, y, layer) -> str:
    return (_g(0, "POINT") + _g(8, layer) + _g(10, float(x)) + _g(20, float(y))
            + _g(30, 0.0))


def _text(x, y, height, s, layer) -> str:
    # DXF R12 has no Unicode: keep annotation to plain ASCII so the file opens
    # identically everywhere. "%%c" is the AutoCAD control code for the
    # diameter symbol and is understood by every reader that matters.
    s = str(s).replace("ø", "%%c").replace("⌀", "%%c")
    s = s.replace("°", "%%d").replace("±", "%%p")
    s = s.encode("ascii", "replace").decode("ascii")
    return (_g(0, "TEXT") + _g(8, layer) + _g(10, float(x)) + _g(20, float(y))
            + _g(30, 0.0) + _g(40, float(height)) + _g(1, s) + _g(7, "STANDARD"))


def _centre_mark(x, y, size, layer) -> str:
    return (_line(x - size, y, x + size, y, layer)
            + _line(x, y - size, x, y + size, layer))


def write_dxf(path: str, layout: PlateLayout, meta: DrawingMeta | None = None,
              include_section: bool = True, include_text: bool = True) -> str:
    """Write the plate geometry as a DXF R12 file, full scale, in millimetres.

    Everything is on named layers so the drawing can be reduced to just what
    is wanted: switch off ``ANNOTATION`` and ``SECTION`` and what remains is
    the plate outline plus the hole circles, ready to be used as drill targets
    or extruded into a solid.
    """
    m = (meta or DrawingMeta()).stamped()
    R = layout.plate_r
    hr = 0.5 * layout.hole_d
    parts: list[str] = []

    # ---- plan view -----------------------------------------------------
    parts.append(_circle(0, 0, R, "PLATE_OUTLINE"))
    parts.append(_centre_mark(0, 0, R * 1.05, "CENTRELINES"))
    for ring in layout.rings:
        parts.append(_circle(0, 0, ring.radius, "CENTRELINES"))
    for (x, y) in layout.holes:
        parts.append(_circle(x, y, hr, "HOLES"))
        parts.append(_point(x, y, "HOLE_CENTRES"))
        parts.append(_centre_mark(x, y, hr * 1.6, "CENTRELINES"))

    # ---- side section, placed clear of the plan view --------------------
    if include_section and layout.thickness > 0:
        y0 = -(R + max(0.25 * R, 8.0) + layout.thickness)
        parts.append(_line(-R, y0, R, y0, "SECTION"))
        parts.append(_line(-R, y0 + layout.thickness, R, y0 + layout.thickness, "SECTION"))
        parts.append(_line(-R, y0, -R, y0 + layout.thickness, "SECTION"))
        parts.append(_line(R, y0, R, y0 + layout.thickness, "SECTION"))
        for (x, _y) in layout.holes:
            parts.append(_line(x - hr, y0, x - hr, y0 + layout.thickness, "SECTION"))
            parts.append(_line(x + hr, y0, x + hr, y0 + layout.thickness, "SECTION"))

    # ---- annotation ------------------------------------------------------
    if include_text:
        h = max(R * 0.045, 1.2)
        ty = R * 1.18
        lines = [
            f"{m.title}" + (f"  /  {m.part_no}" if m.part_no else ""),
            f"PLATE %%c{layout.plate_d:.2f} x {layout.thickness:.2f} THK",
            f"{layout.n_holes} x %%c{layout.hole_d:.3f} THRU  "
            f"(TOTAL AREA {layout.total_area:.2f} mm2, L/D {layout.L_over_D:.2f})",
        ]
        for ring in layout.rings:
            lines.append(
                f"{ring.count} x %%c{layout.hole_d:.3f} EQ SP ON "
                f"%%c{ring.bolt_circle_d:.2f} B.C."
                + (f" (STAGGERED {ring.phase_deg:.2f}%%d)" if ring.phase else "")
            )
        lines.append(f"MIN WEB {layout.min_web:.2f}   EDGE MARGIN {layout.edge_margin:.2f}")
        lines.append("UNITS mm. ORIGIN AT PLATE CENTRE. VIEWED FROM INLET FACE.")
        if m.flow_model:
            lines.append(f"SIZED WITH {m.flow_model} MODEL, Cd {layout.Cd:.3f}")
        if m.project:
            lines.append(f"PROJECT {m.project}")
        lines.append(f"GENERATED {m.date} BY HASTE")
        for i, s in enumerate(lines):
            parts.append(_text(-R, ty + (len(lines) - i - 1) * h * 1.8, h, s, "ANNOTATION"))

    body = "".join(parts)

    # Extents, so the file opens zoomed to the drawing rather than to origin.
    span_y_lo = -(R * 1.3 + (layout.thickness + max(0.25 * R, 8.0) if include_section else 0))
    span_y_hi = R * 1.2 + (len(layout.rings) + 8) * max(R * 0.045, 1.2) * 1.8
    header = (
        _g(0, "SECTION") + _g(2, "HEADER")
        + _g(9, "$ACADVER") + _g(1, "AC1009")
        + _g(9, "$INSUNITS") + _g(70, 4)          # 4 = millimetres
        + _g(9, "$MEASUREMENT") + _g(70, 1)       # 1 = metric
        + _g(9, "$EXTMIN") + _g(10, -R * 1.3) + _g(20, span_y_lo) + _g(30, 0.0)
        + _g(9, "$EXTMAX") + _g(10, R * 1.3) + _g(20, span_y_hi) + _g(30, 0.0)
        + _g(0, "ENDSEC")
    )

    ltypes = (
        _g(0, "TABLE") + _g(2, "LTYPE") + _g(70, 2)
        + _g(0, "LTYPE") + _g(2, "CONTINUOUS") + _g(70, 0) + _g(3, "Solid line")
        + _g(72, 65) + _g(73, 0) + _g(40, 0.0)
        + _g(0, "LTYPE") + _g(2, "DASHED") + _g(70, 0) + _g(3, "Dashed __ __ __")
        + _g(72, 65) + _g(73, 2) + _g(40, 15.0) + _g(49, 10.0) + _g(49, -5.0)
        + _g(0, "ENDTAB")
    )
    layers = _g(0, "TABLE") + _g(2, "LAYER") + _g(70, len(_LAYERS))
    for name, colour, lt in _LAYERS:
        layers += (_g(0, "LAYER") + _g(2, name) + _g(70, 0)
                   + _g(62, colour) + _g(6, lt))
    layers += _g(0, "ENDTAB")

    # Every TEXT entity names the STANDARD style, so the style has to exist --
    # some readers reject text that references an undefined style.
    styles = (
        _g(0, "TABLE") + _g(2, "STYLE") + _g(70, 1)
        + _g(0, "STYLE") + _g(2, "STANDARD") + _g(70, 0) + _g(40, 0.0)
        + _g(41, 1.0) + _g(50, 0.0) + _g(71, 0) + _g(42, 2.5)
        + _g(3, "txt") + _g(4, "")
        + _g(0, "ENDTAB")
    )

    tables = (_g(0, "SECTION") + _g(2, "TABLES") + ltypes + layers + styles
              + _g(0, "ENDSEC"))
    entities = _g(0, "SECTION") + _g(2, "ENTITIES") + body + _g(0, "ENDSEC")

    with open(path, "w", encoding="ascii", newline="\r\n") as f:
        f.write(header + tables + entities + _g(0, "EOF"))
    return path


def default_meta(cfg=None, model_name: str = "", project: str = "",
                 part_no: str = "") -> DrawingMeta:
    """Title-block content derived from the motor configuration."""
    proj = project
    if not proj and cfg is not None:
        proj = getattr(getattr(cfg, "propellant", None), "name", "") or ""
    return DrawingMeta(
        project=proj,
        part_no=part_no,
        flow_model=model_name or (
            cfg.injector.model.value if cfg is not None else ""
        ),
    ).stamped()


def suggested_filename(layout: PlateLayout, ext: str, part_no: str = "") -> str:
    """A filename that states the design, so files do not get mixed up."""
    stem = part_no.strip().replace(" ", "_") if part_no.strip() else "injector_plate"
    stem += f"_{layout.n_holes}x{layout.hole_d:.3f}mm_t{layout.thickness:.2f}"
    return stem.replace("..", ".") + (ext if ext.startswith(".") else "." + ext)


def export_all(directory: str, layout: PlateLayout, meta: DrawingMeta | None = None,
               part_no: str = "") -> list[str]:
    """Write the DXF and the hole table into ``directory``; return the paths."""
    os.makedirs(directory, exist_ok=True)
    out = []
    dxf = os.path.join(directory, suggested_filename(layout, ".dxf", part_no))
    out.append(write_dxf(dxf, layout, meta))
    csvp = os.path.join(directory, suggested_filename(layout, ".csv", part_no))
    out.append(write_hole_table_csv(csvp, layout, meta))
    return out
