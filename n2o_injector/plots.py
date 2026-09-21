"""Figure construction, shared between the GUI and scripted use.

Both entry points draw the same figures from the same code so that an exported
PNG is exactly what the GUI shows.
"""

from __future__ import annotations

import numpy as np

BAR = 1e5


def add_startup_inset(ax, b, max_frac=0.25):
    """Overlay a zoomed view of the ignition ramp, if there is one to see.

    With the transient chamber model the chamber fills in ~100-250 ms, which
    against a multi-second burn is 1-2% of the axis -- it renders as a vertical
    line at t=0 and looks like the plot simply starts at full pressure. The
    quasi-steady model has no ramp at all, so nothing is drawn there.
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

    Real plates are drilled on bolt circles rather than scattered, so holes are
    distributed over rings with the count per ring proportional to its
    circumference -- that keeps the spacing between neighbours roughly even,
    which is what governs whether the webs between holes survive.

    Returns ``(positions, min_gap)`` where ``min_gap`` is the smallest
    edge-to-edge distance between any two holes, in the same units as the
    inputs. A negative value means the holes overlap.
    """
    import numpy as np

    if n_holes <= 0 or plate_r <= 0:
        return [], float("nan")

    usable = plate_r * (1.0 - edge_margin_frac) - hole_r
    if usable <= 0:
        return [(0.0, 0.0)], float("nan")

    if n_holes == 1:
        return [(0.0, 0.0)], float("inf")

    # Pick a ring count that keeps per-ring crowding reasonable.
    n_rings = max(1, min(4, int(round((n_holes / 6.0) ** 0.5))))
    centre = n_holes % n_rings == 1 and n_holes > 6

    outer = n_holes - (1 if centre else 0)
    radii = [usable * (i + 1) / n_rings for i in range(n_rings)]
    weights = np.array(radii, dtype=float)
    counts = np.maximum(1, np.round(outer * weights / weights.sum()).astype(int))

    # Rounding rarely lands exactly on the requested total; fix on the outer ring.
    counts[-1] += outer - int(counts.sum())
    if counts[-1] < 1:
        counts[-1] = 1

    pos = [(0.0, 0.0)] if centre else []
    for ring_i, (r, c) in enumerate(zip(radii, counts)):
        c = max(int(c), 1)
        # Stagger alternate rings by half a step so holes on adjacent circles
        # interleave rather than lining up radially -- that maximises the web
        # between neighbours and is how plates are actually drilled.
        phase = (np.pi / c) if ring_i % 2 else 0.0
        for k in range(c):
            a = phase + 2 * np.pi * k / c
            pos.append((r * np.cos(a), r * np.sin(a)))

    gap = float("inf")
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
            gap = min(gap, d - 2 * hole_r)
    return pos, gap


def build_injector_figure(fig, plate, plate_d_mm=None, grain_od_mm=None):
    """Top-down view of the orifice plate: the plate outline and every hole."""
    import numpy as np

    fig.clear()
    ax = fig.subplots()

    hole_d = plate.hole_d * 1e3
    if plate_d_mm is None or plate_d_mm <= 0:
        plate_d_mm = grain_od_mm if grain_od_mm else max(hole_d * 12, 40.0)
    R = plate_d_mm / 2.0

    pos, gap = ring_layout(plate.n_holes, R, hole_d / 2.0)

    ax.add_patch(plt_circle(0, 0, R, face="#e9edf2", edge="#33465e", lw=2.0))
    ax.add_patch(plt_circle(0, 0, R * 0.88, face="none", edge="#9bb0c6", lw=0.8, ls=":"))
    for (x, y) in pos:
        ax.add_patch(plt_circle(x, y, hole_d / 2.0, face="#123048", edge="#0b1f2f", lw=0.6))

    ax.set_aspect("equal")
    lim = R * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.grid(alpha=0.25)

    gap_txt = "n/a" if not np.isfinite(gap) else f"{gap:.2f} mm"
    warn = "  <-- HOLES OVERLAP" if (np.isfinite(gap) and gap < 0) else ""
    ax.set_title(
        f"Injector plate: {plate.n_holes} x {hole_d:.3f} mm  "
        f"(plate {plate_d_mm:.1f} mm, Cd {plate.Cd:.2f}, L/D {plate.L_over_D:.2f})\n"
        f"total area {plate.total_area * 1e6:.2f} mm²   "
        f"min web between holes {gap_txt}{warn}",
        fontsize=10,
    )
    return ax


def plt_circle(x, y, r, face="none", edge="k", lw=1.0, ls="-"):
    from matplotlib.patches import Circle

    return Circle((x, y), r, facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls)
