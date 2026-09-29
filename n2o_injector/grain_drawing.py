"""Manufacturing geometry for the fuel grain: port outlines, DXF and port data.

The counterpart of :mod:`n2o_injector.drawing` for the grain. It is free of
matplotlib, and the drawing sheet in :mod:`n2o_injector.plots` is rendered
from the same :class:`GrainLayout` that is written to DXF, so the sheet and the
CAD file cannot disagree about where a port wall is.

What the drawing describes is the *cast* grain -- equivalently the casting
mandrels. All lengths are millimetres, origin on the grain axis, viewed from
the forward (injector) end, X right and Y up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .drawing import DrawingMeta, _arc, _centre_mark, _circle, _line, _text, write_dxf_document
from .grain_geometry import sector_outlines

_GRAIN_LAYERS = [
    # name,             colour, linetype
    ("GRAIN_OUTLINE",   7,  "CONTINUOUS"),
    ("PORTS",           1,  "CONTINUOUS"),
    ("CENTRELINES",     4,  "DASHED"),
    ("SECTION",         8,  "CONTINUOUS"),
    ("ANNOTATION",      3,  "CONTINUOUS"),
]


@dataclass
class GrainLayout:
    """Everything needed to draw or cast the grain. Lengths in mm."""

    outer_d: float
    length: float
    layout: str                  #: 'round' or 'sector'
    port_d: float                #: round port dia, or the centre port for 'sector'
    n_ports: int = 1             #: 'round' only
    port_circle_r: float = 0.0   #: 'round', n_ports > 1: radius the ports sit on
    n_sectors: int = 4
    ring_web: float = 0.0
    spoke_web: float = 0.0
    wall_web: float = 0.0
    #: Areas are exact for 'sector' and single-port grains; the multi-port
    #: round wall web depends on the assumed port circle and is reported so.
    port_area: float = float("nan")
    perimeter: float = float("nan")
    throat_area: float = float("nan")
    rho_fuel: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def R(self) -> float:
        return 0.5 * self.outer_d

    @property
    def r1(self) -> float:
        """Sector inner radius."""
        return 0.5 * self.port_d + self.ring_web

    @property
    def r2(self) -> float:
        """Sector outer radius."""
        return self.R - self.wall_web

    @property
    def round_ports(self) -> list[tuple[float, float]]:
        """Centres of the round ports ('round' layout)."""
        if self.layout != "round":
            return []
        if self.n_ports == 1:
            return [(0.0, 0.0)]
        return [(self.port_circle_r * math.cos(a), self.port_circle_r * math.sin(a))
                for a in (math.pi / 2 + 2 * math.pi * k / self.n_ports
                          for k in range(self.n_ports))]

    @property
    def section_angle(self) -> float:
        """Direction [rad] of the section line A-A through the axis.

        Sector grains are cut midway between two spokes, through the centre
        port and two sector ports. Round grains are cut through the port
        nearest 45 degrees, which keeps the section letters clear of the
        diameter dimension below the view.
        """
        if self.layout == "sector":
            return math.pi / self.n_sectors
        if self.n_ports == 1:
            return math.pi / 4
        angles = [math.atan2(y, x) for x, y in self.round_ports]
        return min(angles, key=lambda a: abs(((a - math.pi / 4) + math.pi / 2)
                                             % math.pi - math.pi / 2))

    @property
    def wall_web_actual(self) -> float:
        """Distance from the nearest port wall to the case, mm."""
        if self.layout == "sector":
            return self.wall_web
        if self.n_ports == 1:
            return self.R - 0.5 * self.port_d
        return self.R - self.port_circle_r - 0.5 * self.port_d

    @property
    def between_ports(self) -> float:
        """Thinnest web between neighbouring ports, mm (inf for one port)."""
        if self.layout == "sector":
            return min(self.ring_web, self.spoke_web)
        if self.n_ports == 1:
            return float("inf")
        chord = 2 * self.port_circle_r * math.sin(math.pi / self.n_ports)
        return chord - self.port_d

    @property
    def fuel_mass(self) -> float:
        """kg, from the exact port area."""
        vol_mm3 = (math.pi * self.R**2 - self.port_area) * self.length
        return self.rho_fuel * vol_mm3 * 1e-9

    @property
    def burn_surface(self) -> float:
        """Initial port-wall area, mm^2."""
        return self.perimeter * self.length

    @property
    def port_throat_ratio(self) -> float:
        return self.port_area / self.throat_area if self.throat_area > 0 else float("nan")


def grain_layout(grain, nozzle=None, propellant=None) -> GrainLayout:
    """Build a :class:`GrainLayout` from a :class:`~n2o_injector.motor.Grain`.

    A multi-port round grain carries no port positions, so the ports are placed
    on the circle that balances the web to the case against the web between
    ports -- the layout that burns through at the deepest regression. The
    drawing says it has assumed this.
    """
    mm = 1e3
    n = grain.n_ports
    layout = GrainLayout(
        outer_d=grain.outer_d * mm,
        length=grain.length * mm,
        layout=grain.layout,
        port_d=grain.port_id * mm,
        n_ports=n,
        n_sectors=grain.n_sectors,
        ring_web=grain.ring_web * mm,
        spoke_web=grain.spoke_web * mm,
        wall_web=grain.wall_web * mm,
        port_area=grain.port_area(grain.port_id) * mm**2,
        perimeter=grain.burn_perimeter(grain.port_id) * mm,
        throat_area=(nozzle.throat_area * mm**2 if nozzle is not None else float("nan")),
        rho_fuel=(propellant.rho_fuel if propellant is not None else float("nan")),
    )
    if grain.layout == "round" and n > 1:
        # Balanced: (chord - d)/2 == R - Rc - d/2  =>  Rc = R / (1 + sin(pi/n))
        layout.port_circle_r = layout.R / (1.0 + math.sin(math.pi / n))
        layout.notes.append(
            f"PORT POSITIONS ASSUMED: {n} PORTS ON R{layout.port_circle_r:.2f}, THE "
            "CIRCLE THAT BALANCES WALL WEB AGAINST WEB BETWEEN PORTS."
        )
    return layout


def write_grain_dxf(path: str, layout: GrainLayout, meta: DrawingMeta | None = None,
                    include_section: bool = True, include_text: bool = True) -> str:
    """Write the grain cross-section (and a side section) as DXF R12, in mm.

    Sector ports are written as true ARC and LINE entities rather than
    polylines, so a CAD package can dimension them and a CAM package can cut a
    mandrel from them without re-fitting curves.
    """
    m = (meta or DrawingMeta(title="FUEL GRAIN")).stamped()
    R = layout.R
    parts: list[str] = []

    # ---- cross-section, from the forward end ---------------------------
    parts.append(_circle(0, 0, R, "GRAIN_OUTLINE"))
    parts.append(_centre_mark(0, 0, R * 1.05, "CENTRELINES"))
    if layout.layout == "sector":
        parts.append(_circle(0, 0, 0.5 * layout.port_d, "PORTS"))
        for port in sector_outlines(layout.outer_d, layout.port_d, layout.ring_web,
                                    layout.spoke_web, layout.n_sectors, layout.wall_web):
            for r, a0, a1 in (port["inner"], port["outer"]):
                parts.append(_arc(0, 0, r, a0, a1, "PORTS"))
            for (x0, y0), (x1, y1) in port["sides"]:
                parts.append(_line(x0, y0, x1, y1, "PORTS"))
        for k in range(layout.n_sectors):
            a = 2 * math.pi * k / layout.n_sectors
            parts.append(_line(0, 0, R * 1.05 * math.cos(a), R * 1.05 * math.sin(a),
                               "CENTRELINES"))
    else:
        for (x, y) in layout.round_ports:
            parts.append(_circle(x, y, 0.5 * layout.port_d, "PORTS"))
            parts.append(_centre_mark(x, y, 0.5 * layout.port_d * 1.3, "CENTRELINES"))
        if layout.n_ports > 1:
            parts.append(_circle(0, 0, layout.port_circle_r, "CENTRELINES"))

    # ---- side section, placed clear of the cross-section ---------------
    L = layout.length
    y_top = -(R + max(0.3 * R, 10.0))
    if include_section:
        x0 = -0.5 * L
        parts.append(_line(x0, y_top, x0 + L, y_top, "SECTION"))
        parts.append(_line(x0, y_top - 2 * R, x0 + L, y_top - 2 * R, "SECTION"))
        parts.append(_line(x0, y_top, x0, y_top - 2 * R, "SECTION"))
        parts.append(_line(x0 + L, y_top, x0 + L, y_top - 2 * R, "SECTION"))
        for lo, hi in section_bands(layout):
            for yy in (lo, hi):
                parts.append(_line(x0, y_top - R + yy, x0 + L, y_top - R + yy, "SECTION"))

    # ---- annotation ------------------------------------------------------
    if include_text:
        h = max(R * 0.045, 1.2)
        ty = R * 1.18
        lines = [f"{m.title}" + (f"  /  {m.part_no}" if m.part_no else ""),
                 f"GRAIN %%c{layout.outer_d:.2f} OD x {L:.2f} LONG"]
        lines += [s.replace("⌀", "%%c") for s in port_callouts(layout)]
        lines.append(f"PORT AREA {layout.port_area:.1f} mm2, PERIMETER "
                     f"{layout.perimeter:.1f} mm, WALL WEB {layout.wall_web_actual:.2f}")
        lines.append("UNITS mm. ORIGIN ON GRAIN AXIS. VIEWED FROM FORWARD END.")
        if m.project:
            lines.append(f"PROJECT {m.project}")
        lines.append(f"GENERATED {m.date} BY HASTE")
        for i, s in enumerate(lines):
            parts.append(_text(-R, ty + (len(lines) - i - 1) * h * 1.8, h, s, "ANNOTATION"))

    span = max(R * 1.3, 0.5 * L + 5.0)
    y_lo = y_top - 2 * R - 5.0 if include_section else -R * 1.3
    y_hi = R * 1.2 + 12 * max(R * 0.045, 1.2) * 1.8
    return write_dxf_document(path, "".join(parts), (-span, y_lo), (span, y_hi),
                              _GRAIN_LAYERS)


def section_bands(layout: GrainLayout) -> list[tuple[float, float]]:
    """Port extents across the side section, as ``(lo, hi)`` offsets from the axis.

    The cut runs along :attr:`GrainLayout.section_angle`; see there.
    """
    if layout.layout == "sector":
        c = 0.5 * layout.port_d
        return [(-layout.r2, -layout.r1), (-c, c), (layout.r1, layout.r2)]
    a = layout.section_angle
    ux, uy = math.cos(a), math.sin(a)
    bands = []
    for (x, y) in layout.round_ports:
        if abs(x * uy - y * ux) < 1e-6:  # port centre lies on the section line
            t = x * ux + y * uy
            bands.append((t - 0.5 * layout.port_d, t + 0.5 * layout.port_d))
    return bands


def port_callouts(layout: GrainLayout) -> list[str]:
    """The port dimensions, as they would be called out on a drawing."""
    if layout.layout == "sector":
        return [
            f"CENTRE PORT ⌀{layout.port_d:.2f}",
            f"{layout.n_sectors} SECTOR PORTS R{layout.r1:.2f} TO R{layout.r2:.2f}, "
            f"EQ SP ({360 / layout.n_sectors:.0f}° PITCH)",
            f"SPOKES {layout.spoke_web:.2f} THK, RING WEB {layout.ring_web:.2f}, "
            f"WALL WEB {layout.wall_web:.2f}",
        ]
    if layout.n_ports == 1:
        return [f"PORT ⌀{layout.port_d:.2f} ON AXIS"]
    return [f"{layout.n_ports} PORTS ⌀{layout.port_d:.2f} EQ SP ON "
            f"⌀{2 * layout.port_circle_r:.2f} ({360 / layout.n_ports:.0f}° PITCH)"]


def suggested_grain_filename(layout: GrainLayout, ext: str, part_no: str = "") -> str:
    stem = part_no.strip().replace(" ", "_") if part_no.strip() else "fuel_grain"
    if layout.layout == "sector":
        stem += f"_sector{layout.n_sectors}_wall{layout.wall_web:.2f}"
    else:
        stem += f"_{layout.n_ports}x{layout.port_d:.2f}mm"
    stem += f"_OD{layout.outer_d:.1f}_L{layout.length:.0f}"
    return stem + (ext if ext.startswith(".") else "." + ext)
