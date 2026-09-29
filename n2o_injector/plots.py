"""Figure construction, shared between the GUI and scripted use.

Both entry points draw the same figures from the same code so that an exported
PNG is exactly what the GUI shows.
"""

from __future__ import annotations

import math

import numpy as np

BAR = 1e5


def add_startup_inset(ax, b, max_frac=0.25):
    """Overlay a zoomed view of the ignition ramp, if there is one to see.

    With the transient chamber model the chamber fills in a few milliseconds to
    a few tens of milliseconds, which against a multi-second burn is well under
    1% of the axis -- it renders as a vertical line at t=0 and looks like the
    plot simply starts at full pressure. The quasi-steady model has no ramp at
    all, so nothing is drawn there.
    """
    import numpy as np

    if len(b.t) < 20:
        return None
    pc = b.P_chamber
    peak = float(pc.max())
    if peak <= 0:
        return None

    # A ramp exists only if pressure starts well below its peak.
    if pc[0] > 0.5 * peak:
        return None

    i95 = int(np.argmax(pc >= 0.95 * peak))
    t95 = float(b.t[i95])
    if t95 <= 0:
        return None

    window = min(max(4.0 * t95, 0.05), max_frac * float(b.t[-1]))
    m = b.t <= window
    if m.sum() < 5:
        return None

    ins = ax.inset_axes([0.45, 0.12, 0.52, 0.44])
    ins.plot(b.t[m], b.P_tank[m] / BAR, lw=1.0)
    ins.plot(b.t[m], pc[m] / BAR, lw=1.4)
    ins.set_title(f"ignition, first {window * 1e3:.0f} ms", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.grid(alpha=0.3)
    ins.patch.set_alpha(0.92)
    return ins


def build_results_figure(fig, result, cfg):
    """Draw the six-panel burn summary onto ``fig``."""
    b = result.burn
    fig.clear()
    (a1, a2), (a3, a4), (a5, a6) = fig.subplots(3, 2)

    a1.plot(b.t, b.mdot_ox, label="oxidiser")
    a1.plot(b.t, b.mdot_fuel, label="fuel")
    a1.plot(b.t, b.mdot_ox + b.mdot_fuel, label="total", ls=":")
    a1.set_ylabel("mass flow [kg/s]")
    a1.set_xlabel("time [s]")
    a1.set_title("Mass flow rate")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    a2.plot(b.t, b.OF, color="tab:red", label="achieved")
    a2.plot(b.t, result.target.target_at(b.t), color="k", ls="--", lw=1, label="target")
    a2.set_ylabel("O/F")
    a2.set_xlabel("time [s]")
    a2.set_title(f"O/F ratio (mean {result.achieved_mean_OF:.2f})")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)

    a3.plot(b.t, b.P_tank / BAR, label="tank")
    a3.plot(b.t, b.P_chamber / BAR, label="chamber")
    a3.set_ylabel("pressure [bar]")
    a3.set_xlabel("time [s]")
    a3.set_title("Tank and chamber pressure")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3)
    add_startup_inset(a3, b)

    a4.plot(b.t, b.dP_frac * 100, color="tab:green")
    a4.axhline(result.target.min_dP_fraction * 100, color="r", ls="--", lw=1,
               label=f"{result.target.min_dP_fraction * 100:.0f}% minimum")
    a4.set_ylabel(r"$\Delta P_{inj}$ / $P_c$ [%]")
    a4.set_xlabel("time [s]")
    a4.set_title("Injector pressure-drop margin")
    a4.legend(fontsize=8)
    a4.grid(alpha=0.3)

    a5.plot(b.t, b.thrust, color="tab:orange")
    a5.set_ylabel("thrust [N]")
    a5.set_xlabel("time [s]")
    a5.set_title(f"Thrust (total impulse {b.total_impulse:.0f} N.s)")
    a5.grid(alpha=0.3)

    a6.plot(b.t, b.port_d * 1e3, color="tab:blue", label="port dia")
    a6.set_ylabel("port diameter [mm]")
    a6.set_xlabel("time [s]")
    a6.grid(alpha=0.3)
    a6t = a6.twinx()
    a6t.plot(b.t, b.G_ox, color="tab:brown", ls=":", label="$G_{ox}$")
    a6t.set_ylabel(r"$G_{ox}$ [kg/m$^2$/s]")
    a6.set_title("Port growth and oxidiser flux")
    lines = a6.get_lines() + a6t.get_lines()
    a6.legend(lines, [ln.get_label() for ln in lines], fontsize=8)

    fig.suptitle(
        f"{result.plate.n_holes} x {result.plate.hole_d * 1e3:.3f} mm  "
        f"(Cd {result.plate.Cd:.2f}, L/D {result.plate.L_over_D:.2f})  --  "
        f"{cfg.injector.model.value} model",
        fontsize=11,
    )
    return fig


def build_hrap_overlay_figure(fig, burn, hrap, model_name=""):
    """Overlay this tool's burn against an imported HRAP run.

    Solid lines are this tool, dashed are HRAP, so divergence is visible at a
    glance without needing the legend.
    """
    fig.clear()
    (a1, a2), (a3, a4), (a5, a6) = fig.subplots(3, 2)

    def pair(ax, ours, key, scale, ylabel, title):
        ax.plot(burn.t, ours * scale, lw=1.8, color="tab:blue", label="this tool")
        if hrap.has(key):
            ax.plot(hrap.t, hrap.channels[key] * scale, lw=1.6, ls="--",
                    color="tab:red", label="HRAP")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    pair(a1, burn.mdot_ox, "mdot_ox", 1.0, "kg/s", "Oxidiser mass flow")
    pair(a2, burn.OF, "OF", 1.0, "O/F", "O/F ratio")

    a3.plot(burn.t, burn.P_tank / BAR, lw=1.8, color="tab:blue", label="tank (this tool)")
    a3.plot(burn.t, burn.P_chamber / BAR, lw=1.8, color="tab:cyan", label="chamber (this tool)")
    if hrap.has("P_tank"):
        a3.plot(hrap.t, hrap.channels["P_tank"] / BAR, ls="--", lw=1.6,
                color="tab:red", label="tank (HRAP)")
    if hrap.has("P_chamber"):
        a3.plot(hrap.t, hrap.channels["P_chamber"] / BAR, ls="--", lw=1.6,
                color="tab:orange", label="chamber (HRAP)")
    a3.set_xlabel("time [s]")
    a3.set_ylabel("pressure [bar]")
    a3.set_title("Tank and chamber pressure")
    a3.legend(fontsize=7)
    a3.grid(alpha=0.3)

    pair(a4, burn.thrust, "thrust", 1.0, "thrust [N]", "Thrust")
    pair(a5, burn.port_d, "port_d", 1e3, "port diameter [mm]", "Port growth")
    pair(a6, burn.mdot_fuel, "mdot_fuel", 1.0, "kg/s", "Fuel mass flow")

    title = "This tool vs HRAP"
    if model_name:
        title += f"  ({model_name} injector model)"
    fig.suptitle(title, fontsize=11)
    return fig


def build_comparison_figure(fig, curve, up):
    """Draw the SPI / HEM / Dyer flux comparison and the kappa profile."""
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1)
    P = curve.P / BAR

    ax1.plot(P, curve.G_spi, label="SPI", lw=1.8)
    ax1.plot(P, curve.G_hem, label="HEM", lw=1.8)
    ax1.plot(P, curve.G, label="Dyer (NHNE)", lw=2.2, ls="--")
    ax1.set_xlabel("chamber (downstream) pressure [bar]")
    ax1.set_ylabel("mass flux $G$ [kg/m$^2$/s]")
    ax1.set_title(
        f"Injector mass flux vs chamber pressure\n"
        f"tank {up.T - 273.15:.1f} $^\\circ$C, feed {up.P / BAR:.2f} bar"
        f"{' (saturated)' if up.is_saturated else ' (subcooled)'}"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    if np.any(curve.choked):
        pc = curve.P_crit[0] / BAR
        ax1.axvline(pc, color="k", ls=":", lw=1)
        ax1.text(pc, ax1.get_ylim()[1] * 0.95, f"  HEM chokes at {pc:.1f} bar",
                 fontsize=8, va="top")

    ax2.plot(P, curve.kappa, color="tab:purple")
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax2.set_ylim(0, 5)
    ax2.set_xlabel("chamber (downstream) pressure [bar]")
    ax2.set_ylabel(r"Dyer $\kappa$")
    ax2.set_title(r"$\kappa=1$ (equal SPI/HEM weighting) for a saturated feed")
    ax2.grid(alpha=0.3)
    return fig


def ring_layout(n_holes, plate_r, hole_r, edge_margin_frac=0.12):
    """Place ``n_holes`` on concentric rings inside a plate of radius ``plate_r``.

    Thin wrapper over :func:`n2o_injector.drawing.ring_positions`, kept because
    it is the convenient form for plotting. Returns ``(positions, min_gap)``
    where ``min_gap`` is the smallest edge-to-edge distance between any two
    holes, in the same units as the inputs; negative means they overlap.
    """
    from .drawing import PlateLayout, ring_positions

    rings, pos, centre = ring_positions(n_holes, plate_r, hole_r, edge_margin_frac)
    if not pos:
        return [], float("nan")
    lay = PlateLayout(plate_d=2 * plate_r, hole_d=2 * hole_r, thickness=0.0,
                      holes=pos, rings=rings, centre_hole=centre)
    return pos, lay.min_web


# ------------------------------------------------------------------ drawing
#
# The plate drawing is laid out as a real A4 sheet: the sheet is the coordinate
# system (millimetres, origin bottom-left), so a 10 mm title-block row is 10 mm
# on paper. Line weights and text heights are given in millimetres too and
# converted to points against the current figure size, which keeps the sheet
# looking identical whether it is shown in a small GUI tab or exported at 300
# dpi -- only the overall size changes, never the proportions.

SHEET_W, SHEET_H = 297.0, 210.0  #: A4 landscape, mm

INK = "#12263a"
LIGHT = "#7d93a8"
ACCENT = "#8a2b2b"
HOLE_FILL = "#ffffff"

#: Preferred drawing scales, in the order a drawing office would try them.
STD_SCALES = (10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def _pick_scale(size_mm, room_mm):
    """Largest standard scale that fits ``size_mm`` into ``room_mm``."""
    if size_mm <= 0:
        return 1.0
    limit = room_mm / size_mm
    for s in STD_SCALES:
        if s <= limit:
            return s
    return STD_SCALES[-1]


def _scale_text(s):
    if s >= 1:
        return f"{s:g} : 1"
    return f"1 : {1 / s:g}"


class _Sheet:
    """Drawing primitives in sheet millimetres."""

    def __init__(self, ax, pt_per_mm):
        self.ax = ax
        self.pt = pt_per_mm

    def line(self, x1, y1, x2, y2, lw=0.25, color=INK, ls="-", zorder=2):
        self.ax.plot([x1, x2], [y1, y2], lw=lw * self.pt, color=color, ls=ls,
                     solid_capstyle="butt", zorder=zorder)

    def rect(self, x, y, w, h, lw=0.35, color=INK, face="none", ls="-", zorder=2, **kw):
        from matplotlib.patches import Rectangle

        self.ax.add_patch(Rectangle((x, y), w, h, fill=face != "none", facecolor=face,
                                    edgecolor=color, linewidth=lw * self.pt, ls=ls,
                                    zorder=zorder, **kw))

    def circle(self, x, y, r, lw=0.35, color=INK, face="none", ls="-", zorder=2):
        from matplotlib.patches import Circle

        self.ax.add_patch(Circle((x, y), r, fill=face != "none", facecolor=face,
                                 edgecolor=color, linewidth=lw * self.pt, ls=ls,
                                 zorder=zorder))

    def text(self, x, y, s, h=2.5, color=INK, ha="left", va="baseline",
             weight="normal", family="DejaVu Sans", zorder=5, **kw):
        self.ax.text(x, y, s, fontsize=h * self.pt, color=color, ha=ha, va=va,
                     fontweight=weight, family=family, zorder=zorder, **kw)

    def dim(self, x1, y1, x2, y2, label, h=2.4, off=1.2, color=ACCENT):
        """A dimension line with arrowheads at both ends and a centred label."""
        self.ax.annotate("", xy=(x1, y1), xytext=(x2, y2),
                         arrowprops=dict(arrowstyle="<|-|>", color=color,
                                         lw=0.25 * self.pt, shrinkA=0, shrinkB=0,
                                         mutation_scale=2.2 * self.pt))
        self.text((x1 + x2) / 2, (y1 + y2) / 2 + off, label, h=h, color=color,
                  ha="center", va="bottom")


def _sheet_axes(fig):
    """Clear ``fig`` and return ``(_Sheet, points-per-sheet-mm)``.

    The axes is centred and given the sheet's own aspect ratio, so a wide GUI
    tab shows the sheet with margins either side rather than a stretched one.
    """
    fig.clear()
    try:
        fig.set_layout_engine("none")
    except Exception:  # older matplotlib
        pass
    fig.patch.set_facecolor("white")

    w_in, h_in = fig.get_size_inches()
    fig_ar = (w_in / h_in) if h_in else 1.0
    sheet_ar = SHEET_W / SHEET_H
    if fig_ar > sheet_ar:
        aw, ah = sheet_ar / fig_ar, 1.0
    else:
        aw, ah = 1.0, fig_ar / sheet_ar

    ax = fig.add_axes([(1 - aw) / 2, (1 - ah) / 2, aw, ah])
    ax.set_xlim(0, SHEET_W)
    ax.set_ylim(0, SHEET_H)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor("white")
    return _Sheet(ax, w_in * aw * 72.0 / SHEET_W)


def build_plate_drawing(fig, layout, meta=None, version=""):
    """Render the orifice plate as a dimensioned A4 manufacturing drawing.

    ``layout`` is a :class:`n2o_injector.drawing.PlateLayout`; the same object
    is what :func:`n2o_injector.drawing.write_dxf` exports, so the sheet and
    the CAD file always describe the same plate.
    """
    from .drawing import DrawingMeta, hole_table

    m = (meta or DrawingMeta()).stamped()
    s = _sheet_axes(fig)

    # ---- sheet frame and zones ----------------------------------------
    M = 10.0                      # sheet margin
    s.rect(M, M, SHEET_W - 2 * M, SHEET_H - 2 * M, lw=0.7)
    blocks_top = 52.0             # title block / notes block share this strip
    tb_x = 180.0                  # title block left edge
    table_x = 178.0               # hole table left edge

    band_h = min(34.0, max(18.0, layout.thickness * 0.9 + 14.0))
    plan_y0 = blocks_top + band_h
    plan_cx = (M + table_x) / 2.0
    plan_cy = (plan_y0 + (SHEET_H - M)) / 2.0

    room = min((table_x - M) * 0.46, ((SHEET_H - M) - plan_y0) * 0.44) * 2.0
    scale = _pick_scale(layout.plate_d, room)
    R = 0.5 * layout.plate_d * scale
    hr = 0.5 * layout.hole_d * scale

    # ---- plan view ------------------------------------------------------
    s.circle(plan_cx, plan_cy, R, lw=0.6, face="#f4f7fa")
    # Long-dash-short-dash is the drawing convention for a centreline.
    cl = (0, (9, 2.5, 1.5, 2.5))
    s.line(plan_cx - R * 1.12, plan_cy, plan_cx + R * 1.12, plan_cy,
           lw=0.2, color=LIGHT, ls=cl)
    s.line(plan_cx, plan_cy - R * 1.12, plan_cx, plan_cy + R * 1.12,
           lw=0.2, color=LIGHT, ls=cl)
    for ring in layout.rings:
        s.circle(plan_cx, plan_cy, ring.radius * scale, lw=0.2, color=LIGHT, ls=cl)

    for (x, y) in layout.holes:
        px, py = plan_cx + x * scale, plan_cy + y * scale
        s.circle(px, py, hr, lw=0.35, face=HOLE_FILL, zorder=3)
        if hr > 0.9:  # centre marks only when they would be legible
            s.line(px - hr * 1.5, py, px + hr * 1.5, py, lw=0.15, color=LIGHT, zorder=4)
            s.line(px, py - hr * 1.5, px, py + hr * 1.5, lw=0.15, color=LIGHT, zorder=4)

    # Outside diameter, dimensioned below the view.
    dy = plan_cy - R - 8.0
    for sx in (-R, R):
        s.line(plan_cx + sx, plan_cy, plan_cx + sx, dy - 2.0, lw=0.15, color=ACCENT)
    s.dim(plan_cx - R, dy, plan_cx + R, dy, f"⌀{layout.plate_d:.2f}")

    # Hole callout, led out to the free corner above the view.
    if layout.holes:
        hx, hy = max(layout.holes, key=lambda p: (p[1], p[0]))
        px, py = plan_cx + hx * scale, plan_cy + hy * scale
        lx, ly = table_x - 6.0, SHEET_H - M - 8.0
        s.line(px, py, lx - 34.0, ly - 1.0, lw=0.2, color=ACCENT)
        s.line(lx - 34.0, ly - 1.0, lx, ly - 1.0, lw=0.2, color=ACCENT)
        s.text(lx, ly, f"{layout.n_holes}× ⌀{layout.hole_d:.3f} THRU",
               h=2.8, color=ACCENT, ha="right", weight="bold")
        s.text(lx, ly - 4.0, f"L/D {layout.L_over_D:.2f}   "
                             f"AREA {layout.total_area:.2f} mm²",
               h=2.2, color=ACCENT, ha="right")

    # Ring data table, top-left of the view area.
    if layout.rings:
        rx, ry = M + 3.0, SHEET_H - M - 5.0
        s.text(rx, ry, "RING DATA", h=2.4, weight="bold")
        cols = [(0, "RING"), (11, "B.C. ⌀"), (28, "HOLES"), (40, "PITCH"),
                (54, "STAGGER")]
        for dx, name in cols:
            s.text(rx + dx, ry - 4.0, name, h=2.0, color=LIGHT)
        s.line(rx, ry - 5.2, rx + 68.0, ry - 5.2, lw=0.15, color=LIGHT)
        row = ry - 8.4
        if layout.centre_hole:
            s.text(rx, row, "C", h=2.0)
            s.text(rx + 11, row, "on centre", h=2.0)
            s.text(rx + 28, row, "1", h=2.0)
            row -= 3.4
        for i, ring in enumerate(layout.rings, start=1):
            s.text(rx, row, str(i), h=2.0)
            s.text(rx + 11, row, f"{ring.bolt_circle_d:.2f}", h=2.0)
            s.text(rx + 28, row, str(ring.count), h=2.0)
            s.text(rx + 40, row, f"{ring.pitch_deg:.2f}°", h=2.0)
            s.text(rx + 54, row, f"{ring.phase_deg:.2f}°", h=2.0)
            row -= 3.4

    s.text(plan_cx, plan_y0 + 2.0, "VIEW ON INLET (UPSTREAM) FACE",
           h=2.2, color=LIGHT, ha="center")

    # ---- edge view ------------------------------------------------------
    sec_cy = blocks_top + band_h * 0.55
    t = max(layout.thickness * scale, 0.6)
    # Hatching takes its colour from the patch edge, so the hatch and the
    # outline are drawn as two patches -- light section fill, crisp outline.
    s.rect(plan_cx - R, sec_cy - t / 2, 2 * R, t, lw=0.0, color="#c3d0dc",
           face="#eef3f8", hatch="///", zorder=1)
    s.rect(plan_cx - R, sec_cy - t / 2, 2 * R, t, lw=0.5, zorder=3)
    # Projected hole walls stop being informative once they merge into a comb.
    crowded = len(layout.holes) > 48
    if not crowded:
        for (x, _y) in layout.holes:
            px = plan_cx + x * scale
            s.line(px - hr, sec_cy - t / 2, px - hr, sec_cy + t / 2, lw=0.25, zorder=4)
            s.line(px + hr, sec_cy - t / 2, px + hr, sec_cy + t / 2, lw=0.25, zorder=4)
    s.line(plan_cx + R + 3, sec_cy - t / 2, plan_cx + R + 3, sec_cy + t / 2,
           lw=0.15, color=ACCENT)
    s.text(plan_cx + R + 5, sec_cy - 1.0, f"{layout.thickness:.2f} THK",
           h=2.2, color=ACCENT)
    s.text(plan_cx - R, sec_cy + t / 2 + 2.5,
           "EDGE VIEW — HOLES OMITTED FOR CLARITY, ALL THRU" if crowded
           else "EDGE VIEW — HOLES PROJECTED, ALL THRU", h=2.2, color=LIGHT)

    # ---- hole coordinate table ------------------------------------------
    s.line(table_x, M, table_x, SHEET_H - M, lw=0.5)
    ty = SHEET_H - M - 5.0
    s.text(table_x + 3, ty, "HOLE TABLE", h=2.6, weight="bold")
    s.text(table_x + 3, ty - 4.0, "mm from plate centre, X right / Y up",
           h=1.9, color=LIGHT)
    cols = [(3, "NO", "left"), (13, "RING", "left"), (42, "X", "right"),
            (64, "Y", "right"), (84, "R", "right"), (104, "ANG", "right")]
    for dx, name, ha in cols:
        s.text(table_x + dx, ty - 8.5, name, h=2.0, color=LIGHT, ha=ha)
    s.line(table_x + 3, ty - 10.0, SHEET_W - M - 3, ty - 10.0, lw=0.15, color=LIGHT)

    rows = hole_table(layout)
    pitch = 3.3
    # The column stops at the title block, not at the sheet margin.
    room_rows = int((ty - 13.0 - (blocks_top + 4.0)) / pitch)
    shown = rows[: max(room_rows - (1 if len(rows) > room_rows else 0), 0)]
    y = ty - 13.5
    for r in shown:
        s.text(table_x + 3, y, f"{r['hole']}", h=2.0)
        s.text(table_x + 13, y, f"{r['ring'] or 'C'}", h=2.0)
        s.text(table_x + 42, y, f"{r['x_mm']:+.3f}", h=2.0, ha="right",
               family="DejaVu Sans Mono")
        s.text(table_x + 64, y, f"{r['y_mm']:+.3f}", h=2.0, ha="right",
               family="DejaVu Sans Mono")
        s.text(table_x + 84, y, f"{r['radius_mm']:.3f}", h=2.0, ha="right",
               family="DejaVu Sans Mono")
        s.text(table_x + 104, y, f"{r['angle_deg']:.2f}°", h=2.0, ha="right",
               family="DejaVu Sans Mono")
        y -= pitch
    if len(shown) < len(rows):
        s.text(table_x + 3, y, f"... {len(rows) - len(shown)} more — "
                               f"export the hole table CSV", h=2.0, color=ACCENT)
    elif y - (blocks_top + 4.0) > 34.0:
        # Spare room under a short table: spend it on the numbers a reviewer
        # asks for next, rather than leaving the column half empty.
        y -= 6.0
        s.line(table_x + 3, y + 3.0, SHEET_W - M - 3, y + 3.0, lw=0.15, color=LIGHT)
        s.text(table_x + 3, y - 2.0, "DESIGN DATA", h=2.4, weight="bold")
        data = [
            ("TOTAL FLOW AREA", f"{layout.total_area:.3f} mm²"),
            ("EFFECTIVE Cd·A", f"{layout.Cd * layout.total_area:.3f} mm²"),
            ("HOLE L/D", f"{layout.L_over_D:.2f}"),
            ("MIN WEB / EDGE", f"{layout.min_web:.2f} / {layout.edge_margin:.2f}"),
            ("AREA PER +0.02 ⌀", f"{layout.area_sensitivity(0.02):+.2f} %"),
        ]
        dy2 = y - 6.5
        for label, value in data:
            s.text(table_x + 3, dy2, label, h=2.0, color=LIGHT)
            s.text(table_x + 46, dy2, value, h=2.0)
            dy2 -= 3.6

    # ---- notes and title block ------------------------------------------
    _notes_block(s, _plate_notes(layout, m), M, blocks_top, tb_x)
    dash = "—"
    src = "HASTE" + (f" v{version}" if version else "")
    if m.flow_model:
        src += f" — {m.flow_model} model, Cd {layout.Cd:.3f}"
    _title_block(s, m.title, [
        ("PROJECT", m.project or dash),
        ("PART NO", m.part_no or dash),
        ("MATERIAL", m.material or dash),
        ("PLATE", f"⌀{layout.plate_d:.2f} × {layout.thickness:.2f} THK"),
        ("HOLES", f"{layout.n_holes} × ⌀{layout.hole_d:.3f} THRU"),
        ("SCALE", f"{_scale_text(scale)}   UNITS mm"),
        ("DRAWN", f"{m.drawn_by or dash}    {m.date}"),
    ], src, M, blocks_top, tb_x)
    return s.ax


def _notes_block(s, notes, M, blocks_top, tb_x):
    """Numbered notes left of the title block, plus the print-scale check bar.

    Lines beginning ``!`` are rendered as warnings.
    """
    import textwrap

    s.line(M, blocks_top, SHEET_W - M, blocks_top, lw=0.5)
    s.line(tb_x, M, tb_x, blocks_top, lw=0.5)
    s.text(M + 3, blocks_top - 5.0, "NOTES", h=2.4, weight="bold")

    # Wrap first, then choose the line pitch that fits what wrapping produced,
    # so a long note shrinks the block rather than running off the sheet.
    wrapped = []
    for i, note in enumerate(notes, start=1):
        bad = note.startswith("!")
        for k, part in enumerate(textwrap.wrap(note.lstrip("! "), 112)):
            wrapped.append((f"{i}." if k == 0 else "", part, bad))
    # The bottom strip of the block is reserved for the print-scale check bar.
    avail = blocks_top - 9.5 - (M + 8.5)
    pitch = min(3.4, avail / max(len(wrapped), 1))
    nh = min(2.0, pitch * 0.62)
    ny = blocks_top - 9.5
    for num, part, bad in wrapped:
        colour = ACCENT if bad else INK
        if num:
            s.text(M + 3, ny, num, h=nh, color=colour)
        s.text(M + 8, ny, part, h=nh, color=colour,
               weight="bold" if bad else "normal")
        ny -= pitch

    # Print-scale check bar: a printed sheet is only to scale if it was printed
    # at 100%, and a bar of known *paper* length is how the shop confirms that
    # before measuring anything off the drawing.
    bar, by = 50.0, M + 4.0
    bx = tb_x - 6.0 - bar
    s.line(bx, by, bx + bar, by, lw=0.5)
    for k in (0, 1):
        s.line(bx + k * bar, by - 1.3, bx + k * bar, by + 1.3, lw=0.5)
    s.text(bx - 3.0, by - 0.7, "CHECK BAR — 50 mm ON PAPER:", h=1.9,
           color=LIGHT, ha="right")


def _title_block(s, title, rows, src, M, blocks_top, tb_x):
    s.text(tb_x + 3, blocks_top - 6.0, title, h=3.4, weight="bold")
    ry = blocks_top - 11.0
    for label, value in rows:
        s.text(tb_x + 3, ry, label, h=1.9, color=LIGHT)
        s.text(tb_x + 24, ry, value, h=2.2)
        ry -= 4.3
    s.text(tb_x + 3, M + 2.5, src, h=1.8, color=LIGHT)


def _plate_notes(layout, meta):
    """The notes block. Lines beginning ``!`` are rendered as warnings."""
    tol = 0.02
    notes = [
        "ALL DIMENSIONS IN MILLIMETRES. ORIGIN AT PLATE CENTRE. "
        "PLAN VIEW IS FROM THE INLET (UPSTREAM) FACE.",
        f"{layout.n_holes} HOLES ⌀{layout.hole_d:.3f} THRU, EQUALLY SPACED ON THE "
        "BOLT CIRCLES LISTED IN RING DATA.",
        f"FLOW AREA GOES AS d²: A HOLE ⌀ ERROR OF +{tol:.2f} GIVES "
        f"{layout.area_sensitivity(tol):+.1f}% AREA, AND THE SAME SHIFT IN "
        "OXIDISER FLOW. SET THE TOLERANCE TO SUIT.",
        f"Cd {layout.Cd:.3f} ASSUMED FOR SHARP-EDGED DRILLED HOLES. DEBURR THE OUTLET "
        "FACE. DO NOT CHAMFER OR RADIUS THE INLET WITHOUT RE-SIZING — INLET "
        "GEOMETRY CHANGES Cd.",
        f"L/D {layout.L_over_D:.2f} FROM {layout.thickness:.2f} THICKNESS. THE FLOW "
        "MODEL ASSUMES THIS LENGTH; CHANGING THICKNESS CHANGES PREDICTED FLOW.",
    ]
    if meta.tolerance:
        notes.append(meta.tolerance)
    web, edge = layout.min_web, layout.edge_margin
    if web < 0:
        notes.append(f"! HOLES OVERLAP BY {abs(web):.2f} — NOT MANUFACTURABLE AS "
                     "DRAWN. REDUCE HOLE COUNT OR INCREASE PLATE DIAMETER.")
    elif web < 2 * layout.hole_d:
        notes.append(f"! MIN WEB BETWEEN HOLES {web:.2f} IS THIN NEXT TO "
                     f"⌀{layout.hole_d:.3f} — BACK THE PLATE AND PECK DRILL.")
    else:
        notes.append(f"MIN WEB BETWEEN HOLES {web:.2f}. EDGE MARGIN {edge:.2f}.")
    notes.append("PRINT AT 100% ON A4 LANDSCAPE FOR TRUE SCALE — CONFIRM WITH THE "
                 "CHECK BAR BEFORE MEASURING OFF THIS SHEET.")
    notes.extend(meta.notes)
    return notes


def build_injector_figure(fig, plate, plate_d_mm=None, grain_od_mm=None,
                          meta=None, version=""):
    """Draw the sized plate as a manufacturing sheet; returns the layout."""
    from .drawing import layout_from_plate

    layout = layout_from_plate(plate, plate_d_mm=plate_d_mm, grain_od_mm=grain_od_mm)
    build_plate_drawing(fig, layout, meta=meta, version=version)
    return layout


def save_plate_drawing(path, layout, meta=None, version="", dpi=300):
    """Write the drawing sheet to PNG / PDF / SVG at true A4-landscape size."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(SHEET_W / 25.4, SHEET_H / 25.4))
    FigureCanvasAgg(fig)
    build_plate_drawing(fig, layout, meta=meta, version=version)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path


def plt_circle(x, y, r, face="none", edge="k", lw=1.0, ls="-"):
    from matplotlib.patches import Circle

    return Circle((x, y), r, facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls)


# --------------------------------------------------------------------------
# Fuel grain drawing sheet
# --------------------------------------------------------------------------


def _sector_polygon(port, cx, cy, scale, n_arc=48):
    """Sheet-mm polygon of one sector port from its exact outline."""
    r1, a10, a11 = port["inner"]
    r2, a20, a21 = port["outer"]
    pts = []
    for k in range(n_arc + 1):  # outer arc, counter-clockwise
        a = math.radians(a20 + (a21 - a20) * k / n_arc)
        pts.append((cx + r2 * scale * math.cos(a), cy + r2 * scale * math.sin(a)))
    for k in range(n_arc + 1):  # inner arc, back the other way
        a = math.radians(a11 - (a11 - a10) * k / n_arc)
        pts.append((cx + r1 * scale * math.cos(a), cy + r1 * scale * math.sin(a)))
    return pts


def build_grain_drawing(fig, layout, meta=None, version=""):
    """Render the fuel grain as a dimensioned A4 drawing.

    ``layout`` is a :class:`n2o_injector.grain_drawing.GrainLayout`, the same
    object :func:`~n2o_injector.grain_drawing.write_grain_dxf` exports.
    """
    from matplotlib.patches import Polygon

    from .drawing import DrawingMeta
    from .grain_drawing import port_callouts, section_bands
    from .grain_geometry import sector_outlines

    m = (meta or DrawingMeta(title="FUEL GRAIN")).stamped()
    s = _sheet_axes(fig)

    M = 10.0
    s.rect(M, M, SHEET_W - 2 * M, SHEET_H - 2 * M, lw=0.7)
    blocks_top = 52.0
    tb_x = 180.0
    split_x = 150.0               # cross-section left, section + data right

    FUEL = "#f1ece0"
    cl = (0, (9, 2.5, 1.5, 2.5))

    # ---- cross-section --------------------------------------------------
    top = SHEET_H - M
    cx = (M + split_x) / 2.0
    cy = (blocks_top + top) / 2.0 - 4.0
    room = min(split_x - M - 16.0, top - blocks_top - 36.0)
    scale = _pick_scale(layout.outer_d, room)
    R = layout.R * scale
    s.circle(cx, cy, R, lw=0.6, face=FUEL)
    s.line(cx - R * 1.12, cy, cx + R * 1.12, cy, lw=0.2, color=LIGHT, ls=cl)
    s.line(cx, cy - R * 1.12, cx, cy + R * 1.12, lw=0.2, color=LIGHT, ls=cl)

    if layout.layout == "sector":
        s.circle(cx, cy, 0.5 * layout.port_d * scale, lw=0.4, face=HOLE_FILL, zorder=3)
        for port in sector_outlines(layout.outer_d, layout.port_d, layout.ring_web,
                                    layout.spoke_web, layout.n_sectors, layout.wall_web):
            s.ax.add_patch(Polygon(_sector_polygon(port, cx, cy, scale), closed=True,
                                   facecolor=HOLE_FILL, edgecolor=INK,
                                   linewidth=0.4 * s.pt, zorder=3))
        for k in range(layout.n_sectors):  # spoke centrelines
            a = 2 * math.pi * k / layout.n_sectors
            s.line(cx, cy, cx + R * 1.08 * math.cos(a), cy + R * 1.08 * math.sin(a),
                   lw=0.15, color=LIGHT, ls=cl, zorder=4)
        # Spoke thickness, across the spoke that points straight up if there is one.
        if any(abs(360.0 * k / layout.n_sectors - 90.0) < 1e-6
               for k in range(layout.n_sectors)):
            h = 0.5 * layout.spoke_web * scale
            ym = cy + 0.5 * (layout.r1 + layout.r2) * scale
            s.dim(cx - h, ym, cx + h, ym, f"{layout.spoke_web:.2f}", h=2.1, off=0.8)
        # Wall web, dimensioned radially through the middle of a lower sector.
        a_d = -math.pi / 4 if layout.n_sectors % 4 == 0 else -math.pi / layout.n_sectors
        ux, uy = math.cos(a_d), math.sin(a_d)
        s.ax.annotate("", xy=(cx + layout.r2 * scale * ux, cy + layout.r2 * scale * uy),
                      xytext=(cx + R * ux, cy + R * uy),
                      arrowprops=dict(arrowstyle="<|-|>", color=ACCENT,
                                      lw=0.25 * s.pt, shrinkA=0, shrinkB=0,
                                      mutation_scale=2.2 * s.pt), zorder=5)
        s.text(cx + (R + 2.0) * ux, cy + (R + 2.0) * uy,
               f"WALL WEB {layout.wall_web:.2f}", h=2.1, color=ACCENT, va="top")
    else:
        for (x, y) in layout.round_ports:
            s.circle(cx + x * scale, cy + y * scale, 0.5 * layout.port_d * scale,
                     lw=0.4, face=HOLE_FILL, zorder=3)
        if layout.n_ports > 1:
            s.circle(cx, cy, layout.port_circle_r * scale, lw=0.15, color=LIGHT, ls=cl)

    # Section line A-A through the axis.
    a_sec = layout.section_angle
    ex, ey = math.cos(a_sec) * R * 1.15, math.sin(a_sec) * R * 1.15
    s.line(cx - ex, cy - ey, cx + ex, cy + ey, lw=0.3, color=ACCENT,
           ls=(0, (6, 2)), zorder=4)
    for sgn in (-1, 1):
        s.text(cx + sgn * ex * 1.07, cy + sgn * ey * 1.07, "A", h=3.0, color=ACCENT,
               ha="center", va="center", weight="bold")

    # Outside diameter.
    dy = cy - R - 7.0
    for sx in (-R, R):
        s.line(cx + sx, cy, cx + sx, dy - 2.0, lw=0.15, color=ACCENT)
    s.dim(cx - R, dy, cx + R, dy, f"⌀{layout.outer_d:.2f}")

    # Port callouts, top-left.
    rx, ry = M + 3.0, top - 5.0
    s.text(rx, ry, "PORTS", h=2.4, weight="bold")
    for i, line in enumerate(port_callouts(layout)):
        s.text(rx, ry - 3.8 * (i + 1), line, h=2.0)
    s.text(cx, blocks_top + 3.0,
           f"CROSS-SECTION FROM FORWARD END   SCALE {_scale_text(scale)}",
           h=2.2, color=LIGHT, ha="center")

    # ---- side section A-A ------------------------------------------------
    s.line(split_x, blocks_top, split_x, top, lw=0.5)
    sx0, sx1 = split_x + 8.0, SHEET_W - M - 8.0
    sscale = min(_pick_scale(layout.length, sx1 - sx0), _pick_scale(layout.outer_d, 50.0))
    Ls, Rs = layout.length * sscale, layout.R * sscale
    scx = 0.5 * (sx0 + sx1)
    scy = top - 12.0 - Rs
    x_lo = scx - 0.5 * Ls
    s.rect(x_lo, scy - Rs, Ls, 2 * Rs, lw=0.0, color="#c9bfa8", face=FUEL,
           hatch="///", zorder=1)
    for lo, hi in section_bands(layout):
        s.rect(x_lo, scy + lo * sscale, Ls, (hi - lo) * sscale, lw=0.0,
               color=HOLE_FILL, face=HOLE_FILL, zorder=2)
        for yv in (lo, hi):
            s.line(x_lo, scy + yv * sscale, x_lo + Ls, scy + yv * sscale, lw=0.35, zorder=3)
    s.rect(x_lo, scy - Rs, Ls, 2 * Rs, lw=0.5, zorder=3)
    s.line(x_lo - 3, scy, x_lo + Ls + 3, scy, lw=0.15, color=LIGHT, ls=cl, zorder=4)
    s.dim(x_lo, scy - Rs - 4.5, x_lo + Ls, scy - Rs - 4.5, f"{layout.length:.2f}",
          h=2.1, off=0.8)
    s.text(scx, top - 5.0, f"SECTION A–A   SCALE {_scale_text(sscale)}",
           h=2.2, color=LIGHT, ha="center")

    # ---- design data -----------------------------------------------------
    dx0, dyy = split_x + 5.0, scy - Rs - 15.0
    s.text(dx0, dyy, "DESIGN DATA", h=2.4, weight="bold")
    between = layout.between_ports
    data = [
        ("PORT AREA", f"{layout.port_area:.1f} mm²"),
        ("BURNING PERIMETER", f"{layout.perimeter:.1f} mm"),
        ("BURNING SURFACE", f"{layout.burn_surface / 100:.0f} cm²"),
        ("WALL WEB (TO CASE)", f"{layout.wall_web_actual:.2f} mm"),
        ("WEB BETWEEN PORTS", f"{between:.2f} mm" if math.isfinite(between) else "—"),
    ]
    if math.isfinite(layout.port_throat_ratio):
        data.append(("PORT / THROAT AREA", f"{layout.port_throat_ratio:.2f}"))
    if math.isfinite(layout.fuel_mass):
        data.append(("FUEL MASS", f"{layout.fuel_mass:.3f} kg"))
    yy = dyy - 5.0
    for label, value in data:
        s.text(dx0, yy, label, h=2.0, color=LIGHT)
        s.text(dx0 + 48, yy, value, h=2.0)
        yy -= 3.6

    # ---- notes and title block ------------------------------------------
    _notes_block(s, _grain_notes(layout, m), M, blocks_top, tb_x)
    dash = "—"
    ports = (f"CENTRE + {layout.n_sectors} SECTOR" if layout.layout == "sector"
             else f"{layout.n_ports} × ⌀{layout.port_d:.2f}")
    _title_block(s, m.title, [
        ("PROJECT", m.project or dash),
        ("PART NO", m.part_no or dash),
        ("MATERIAL", m.material or dash),
        ("GRAIN", f"⌀{layout.outer_d:.2f} × {layout.length:.2f}"),
        ("PORTS", ports),
        ("SCALE", f"{_scale_text(scale)} / {_scale_text(sscale)}   UNITS mm"),
        ("DRAWN", f"{m.drawn_by or dash}    {m.date}"),
    ], "HASTE" + (f" v{version}" if version else ""), M, blocks_top, tb_x)
    return s.ax


def _grain_notes(layout, meta):
    """The grain notes block. Lines beginning ``!`` are rendered as warnings."""
    notes = [
        "ALL DIMENSIONS IN MILLIMETRES. ORIGIN ON THE GRAIN AXIS. CROSS-SECTION IS "
        "VIEWED FROM THE FORWARD (INJECTOR) END. PORTS RUN THE FULL LENGTH.",
    ]
    if layout.layout == "sector":
        fillet = 2.0
        loss = layout.n_sectors * 4 * (1 - math.pi / 4) * fillet**2
        notes.append(
            "SPOKES ARE CONSTANT THICKNESS: EACH SECTOR SIDE IS PARALLEL TO ITS SPOKE "
            "CENTRELINE, NOT RADIAL. INNER AND OUTER PORT WALLS ARE ARCS ON THE GRAIN AXIS."
        )
        notes.append(
            f"PORT CORNERS ARE DRAWN SHARP. CAST WITH R{fillet:g} FILLETS TO AVOID "
            f"CRACKING; THIS REMOVES ABOUT {loss:.0f} mm² "
            f"({100 * loss / layout.port_area:.1f}%) OF PORT AREA."
        )
        if (abs(layout.ring_web - 2 * layout.wall_web) > 0.05
                or abs(layout.spoke_web - 2 * layout.wall_web) > 0.05):
            notes.append("! WEBS ARE NOT BALANCED: SPOKE AND RING WEBS SHOULD BE TWICE THE "
                         "WALL WEB, OR THE PORTS MERGE BEFORE THE WALL BURNS THROUGH (OR "
                         "THE REVERSE) AND LEAVE SLIVERS.")
    notes += layout.notes
    if math.isfinite(layout.port_throat_ratio) and layout.port_throat_ratio < 1.0:
        notes.append(f"! PORT AREA IS {layout.port_throat_ratio:.2f}× THE THROAT AREA — "
                     "THE GRAIN, NOT THE NOZZLE, WOULD CHOKE THE FLOW.")
    notes.append("PRINT AT 100% ON A4 LANDSCAPE FOR TRUE SCALE — CONFIRM WITH THE "
                 "CHECK BAR BEFORE MEASURING OFF THIS SHEET.")
    notes.extend(meta.notes)
    return notes


def save_grain_drawing(path, layout, meta=None, version="", dpi=300):
    """Write the grain sheet to PNG / PDF / SVG at true A4-landscape size."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(SHEET_W / 25.4, SHEET_H / 25.4))
    FigureCanvasAgg(fig)
    build_grain_drawing(fig, layout, meta=meta, version=version)
    fig.savefig(path, dpi=dpi, facecolor="white")
    return path
