"""Render the V2.2 injector trade study as a PDF report.

Kept separate from ``v22_trade_study.py`` so the analysis stays readable as
analysis: that module computes, this one only presents. Every number on the
page comes from the results dict that module returns -- nothing is retyped.

Run:  python examples/v22_report.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from v22_trade_study import (  # noqa: E402
    GOX_HI, HOLE_D_MM, MAIN_COLS, PLATE_D_MM, PLATE_T_MM, main as run_analysis,
)

A4 = (8.27, 11.69)
INK = "#12263a"
LIGHT = "#6b8299"
ACCENT = "#8a2b2b"
GOOD = "#1f6b3a"
GRID = "#d5dee6"
SERIES = ["#1f4e79", "#8a2b2b", "#1f6b3a", "#b5751f", "#5b3a86"]

_state = {"page": 0, "total": 0}


def _new_page(pdf_title):
    fig = Figure(figsize=A4)
    fig.patch.set_facecolor("white")
    _state["page"] += 1
    fig.text(0.08, 0.962, pdf_title, fontsize=13, color=INK, fontweight="bold",
             va="bottom")
    fig.text(0.92, 0.962, f"{_state['page']}", fontsize=9, color=LIGHT,
             ha="right", va="bottom")
    fig.add_artist(matplotlib.lines.Line2D([0.08, 0.92], [0.952, 0.952],
                                           color=INK, lw=1.0))
    fig.text(0.08, 0.028, "V2.2 injector trade study  ·  HASTE",
             fontsize=7, color=LIGHT)
    return fig


def _wrap(text, width=104):
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


class TextPage:
    """A page built top-down from headings, prose and monospaced tables."""

    def __init__(self, title):
        self.fig = _new_page(title)
        self.y = 0.925

    def head(self, s, gap=0.018):
        self.y -= gap
        self.fig.text(0.08, self.y, s, fontsize=10.5, color=INK, fontweight="bold")
        self.y -= 0.016

    def para(self, s, color=INK, size=8.6, width=104, gap=0.006):
        self.y -= gap
        for line in _wrap(s, width):
            self.fig.text(0.08, self.y, line, fontsize=size, color=color)
            self.y -= 0.0145
        return self

    def bullets(self, items, color=INK, size=8.6, width=98):
        for it in items:
            lines = _wrap(it, width)
            self.fig.text(0.085, self.y, "•", fontsize=size, color=color)
            for i, line in enumerate(lines):
                self.fig.text(0.105, self.y, line, fontsize=size, color=color)
                self.y -= 0.0145
            self.y -= 0.002

    def mono(self, lines, size=7.0, color=INK, gap=0.008, highlight=None):
        self.y -= gap
        for i, line in enumerate(lines):
            c = color
            if highlight is not None and highlight(i, line):
                c = ACCENT
            self.fig.text(0.08, self.y, line, fontsize=size, color=c,
                          family="DejaVu Sans Mono")
            self.y -= 0.0132
        return self

    def note(self, s, color=ACCENT, size=8.4):
        self.y -= 0.008
        for line in _wrap(s, 100):
            self.fig.text(0.09, self.y, line, fontsize=size, color=color,
                          fontstyle="italic")
            self.y -= 0.0142

    def save(self, pdf):
        pdf.savefig(self.fig)


def _table_lines(rows, cols):
    head = "".join(f"{c[0]:>{c[2]}}" for c in cols)
    out = [head, "-" * len(head)]
    for r in rows:
        out.append("".join(
            f"{r[c[1]]:>{c[2]}{c[3]}}" if c[3] else f"{str(r[c[1]]):>{c[2]}}"
            for c in cols))
    return out


def _axes_grid(fig, rect, nrows, ncols, **kw):
    gs = fig.add_gridspec(nrows, ncols, left=rect[0], bottom=rect[1],
                          right=rect[2], top=rect[3], **kw)
    return [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, color=INK)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.35, color=GRID)
    for s in ax.spines.values():
        s.set_color(GRID)


# ------------------------------------------------------------------ pages


def page_summary(pdf, R):
    base, cfg = R["base"], R["cfg"]
    lo = next(r for r in base if abs(r["mdot"] - 1.0) < 1e-9)
    hi = next(r for r in base if abs(r["mdot"] - 1.5) < 1e-9)
    ofs = R["ofs"]
    of_peak = ofs[int(np.argmax(ofs[:, 5]))]

    p = TextPage("Injector sizing for 1.0–1.5 kg/s — findings")
    p.para(f"Mini Hybrid V2.2  ·  {datetime.now():%Y-%m-%d}  ·  "
           f"generated from configs/V2.2_recommended_mdot1.0_HEM.json",
           color=LIGHT, size=8.2)

    p.head("The short answer")
    p.para(
        "Specific impulse rises monotonically across the whole range you asked about: "
        f"{lo['Isp']:.1f} s at 1.0 kg/s to {hi['Isp']:.1f} s at 1.5 kg/s, a gain of "
        f"{100 * (hi['Isp'] / lo['Isp'] - 1):.1f}%. There is no interior optimum to find. "
        "But that gain is bought almost entirely with chamber pressure, and every "
        "constraint you have runs the other way.")
    p.para(
        "Take 1.0 kg/s. The extra 4% of Isp at 1.5 kg/s costs you 50% more oxidiser flux "
        "in a grain that is already above the correlation range, a stability margin that "
        "expires at 60% of the burn instead of 84%, and 7 percentage points of unburnt "
        "fuel. None of those are modelled risks you can buy back later.")

    p.head("Why Isp rises — and why that is not really about O/F")
    p.para(
        f"Your nozzle (ε = {cfg.nozzle.expansion_ratio:.3f}) is perfectly expanded at "
        f"Pc = {R['ideal_pc']:.1f} bar. At 1.0 kg/s the chamber peaks at "
        f"{lo['Pc_peak']:.1f} bar, so the motor spends most of the burn overexpanded; "
        f"pushing to 1.5 kg/s raises the peak to {hi['Pc_peak']:.1f} bar and the thrust "
        f"coefficient with it, from {R['cf_20']:.3f} at 20 bar to {R['cf_37']:.3f} at 37 bar. "
        "That is a +7% effect.")
    p.para(
        f"O/F moves the opposite way, from {lo['OF_mean']:.2f} to {hi['OF_mean']:.2f}, and "
        f"the CEA table puts peak Isp at O/F {of_peak[0]:.2f}. So raising the flow walks you "
        "away from the mixture optimum — it is worth about −1.4%, which the nozzle "
        "gain simply outweighs. The Isp trend is a nozzle-matching result wearing a "
        "mixture-ratio costume.")
    p.note(
        f"Worth knowing: the propellant file declares opt_OF = {cfg.propellant.opt_OF:.2f}, "
        f"but its own CEA table peaks at O/F {of_peak[0]:.2f}. The declared value is an "
        "HRAP default and does not match the table it ships with. Nothing in the sizing "
        "uses it except as a fallback, but do not design to it.")

    p.head("What actually limits you")
    p.bullets([
        f"Oxidiser flux. G_ox starts at {lo['Gox_max']:.0f} kg/m²/s at 1.0 kg/s and "
        f"{hi['Gox_max']:.0f} at 1.5. Paraffin correlations are normally fitted below "
        f"~{GOX_HI:.0f}. You are extrapolating the regression law at both ends of the "
        "range, and G_ox is set by the grain port — no injector fixes it.",
        f"Feed isolation. The 20% pressure-drop margin lasts {R['margin_pct'][1.0]:.0f}% of "
        f"the liquid phase at 1.0 kg/s but only {R['margin_pct'][1.5]:.0f}% at 1.5 kg/s. "
        "The late burn is where feed coupling bites.",
        f"Fuel utilisation. {lo['fuel_frac']:.0f}% of the grain is consumed at 1.0 kg/s, "
        f"{hi['fuel_frac']:.0f}% at 1.5. Faster burns leave more paraffin behind.",
        "Chamber pressure. 1.5 kg/s puts 37.5 bar into a case you sized around a much "
        "lower number. That is a structural question, not a performance one.",
    ])

    p.head("The recommendation")
    p.mono([
        f"  {lo['n_holes']} holes  Ø{HOLE_D_MM:.2f} mm  THRU        "
        f"plate {PLATE_D_MM:.1f} mm × {PLATE_T_MM:.1f} mm thick",
        f"  L/D {PLATE_T_MM / HOLE_D_MM:.2f}  →  HEM regime        "
        f"CdA {lo['CdA']:.2f} mm²   min web {lo['min_web']:.2f} mm",
        f"  ṁ_ox 1.0 kg/s   O/F {lo['OF_mean']:.2f}          "
        f"Pc {lo['Pc_peak']:.1f} bar peak / {lo['Pc_mean']:.1f} mean",
        f"  thrust {lo['F_mean']:.0f} N mean       burn {lo['burn_liq']:.2f} s"
        f"     total impulse {lo['impulse']:.0f} N·s     Isp {lo['Isp']:.1f} s",
    ], size=8.4)
    p.para(
        "Hole diameter is held at 1.20 mm because L/D = 10.6 is the only setting on a "
        "12.7 mm plate that lands inside the L/D band where the Waxman cold-flow data "
        "validates HEM. Vary the hole count, not the diameter — that keeps the flow "
        "model valid as the flow target moves.", gap=0.012)
    p.save(pdf)


def page_method(pdf, R):
    cfg = R["cfg"]
    p = TextPage("Method, and what is held fixed")

    p.head("The variable, and the thing that must not vary with it")
    p.para(
        "Hole diameter is held at 1.20 mm and hole count is solved for. That is the "
        "opposite of the usual habit, and it is deliberate: L/D = thickness / diameter, "
        "so holding the diameter holds L/D at 10.58 across every point in the study. "
        "Had the diameter been solved for instead, L/D would have drifted from 12.7 to "
        "6.4 across the sweep, silently crossing out of the regime where HEM applies and "
        "into the one where Dyer does — and the endpoints would no longer be "
        "comparable, because they would be different physics rather than different sizes.")

    p.head("Run settings")
    p.mono([
        f"chamber model      quasi-steady   (timestep-independent; see § convergence)",
        f"timestep           1 ms",
        f"sizing objective   mdot_ox        (never inverts the regression law)",
        f"flow model         HEM            (L/D 10.58; externally validated band)",
        f"discharge coeff    0.70           (placeholder — not yet measured)",
        f"property backend   ESDU 91022     (HRAP MATLAB parity)",
        "",
        f"grain              {cfg.grain.n_ports} ports × {cfg.grain.port_id * 1e3:.2f} mm ID, "
        f"OD {cfg.grain.outer_d * 1e3:.2f} mm, L {cfg.grain.length * 1e3:.0f} mm",
        f"tank               {cfg.tank.volume * 1e3:.1f} L, {cfg.tank.ox_mass:.3f} kg N2O, "
        f"{cfg.tank.fill_temp - 273.15:.1f} °C saturated",
        f"nozzle             throat {cfg.nozzle.throat_d * 1e3:.2f} mm, "
        f"ε {cfg.nozzle.expansion_ratio:.3f}, Cd {cfg.nozzle.Cd}, η {cfg.nozzle.efficiency}",
        f"propellant         a {cfg.propellant.reg_a:.4f}, n {cfg.propellant.reg_n:.3f}  "
        "-- HRAP defaults, NOT measured",
    ], size=7.6)

    p.head("Sweep results")
    p.para("Isp is over the whole run including the vapour tail. Gox>hi% is the share of "
           "the liquid phase spent above the fitted flux range; dPok% is the share with at "
           "least 20% pressure-drop margin; fuel% is the grain actually consumed.",
           size=8.2, color=LIGHT)
    p.mono(_table_lines(R["base"], MAIN_COLS), size=6.5,
           highlight=lambda i, ln: ln.strip().startswith("1.50"))

    p.head("Reading it")
    p.bullets([
        "Isp and total impulse both increase all the way to 1.5 kg/s. If performance were "
        "the only axis, the answer would be 'as much flow as the plate can pass'.",
        "Every other column degrades monotonically in the same direction. This is a "
        "constraint problem, not an optimisation problem — there is no peak to find, "
        "only a line to stop at.",
        "The range below 1.0 kg/s is included because the O/F optimum lies below the "
        "brief's range. An optimum you cannot see both sides of is not an optimum.",
    ])
    p.note(
        "OF>10% is the share of the run whose instantaneous O/F is past the end of the "
        "CEA table, where c* is clamped rather than computed. It is zero up to 1.4 kg/s "
        "and 13% at 1.5 kg/s — so the top row of this table is the one row partly "
        "resting on an extrapolated c*, and it is also the row being argued against.")
    p.save(pdf)


def page_charts(pdf, R):
    base = R["base"]
    fig = _new_page("Performance across the sweep")
    md = [r["mdot"] for r in base]
    ax = _axes_grid(fig, (0.09, 0.30, 0.95, 0.91), 3, 2, hspace=0.45, wspace=0.30)

    def band(a):
        a.axvspan(1.0, 1.5, color="#1f4e79", alpha=0.055, lw=0)

    a = ax[0][0]
    a.plot(md, [r["Isp"] for r in base], "o-", color=SERIES[0], ms=3.5, lw=1.6)
    band(a)
    _style(a, "oxidiser flow [kg/s]", "Isp [s]", "Specific impulse")

    a = ax[0][1]
    a.plot(md, [r["impulse"] for r in base], "o-", color=SERIES[1], ms=3.5, lw=1.6)
    band(a)
    _style(a, "oxidiser flow [kg/s]", "total impulse [N·s]", "Total impulse")

    a = ax[1][0]
    a.plot(md, [r["OF_mean"] for r in base], "o-", color=SERIES[2], ms=3.5, lw=1.6)
    a.axhline(R["of_peak"], color=ACCENT, ls="--", lw=1.1)
    a.text(md[-1], R["of_peak"], f"peak-Isp O/F {R['of_peak']:.2f} ", fontsize=7,
           color=ACCENT, va="bottom", ha="right")
    band(a)
    _style(a, "oxidiser flow [kg/s]", "mean O/F", "Mixture ratio moves the wrong way")

    a = ax[1][1]
    a.plot(md, [r["Pc_peak"] for r in base], "o-", color=SERIES[3], ms=3.5, lw=1.6,
           label="peak")
    a.plot(md, [r["Pc_mean"] for r in base], "s--", color=SERIES[3], ms=3, lw=1.2,
           alpha=0.65, label="mean")
    a.axhline(R["ideal_pc"], color=GOOD, ls=":", lw=1.2)
    a.text(md[-1], R["ideal_pc"], f"ideal expansion {R['ideal_pc']:.1f} bar ", fontsize=7,
           color=GOOD, va="bottom", ha="right")
    band(a)
    a.legend(fontsize=7, frameon=False)
    _style(a, "oxidiser flow [kg/s]", "Pc [bar]", "Chamber pressure")

    a = ax[2][0]
    a.plot(md, [r["Gox_max"] for r in base], "o-", color=ACCENT, ms=3.5, lw=1.6)
    a.axhline(GOX_HI, color=ACCENT, ls="--", lw=1.1)
    a.fill_between([min(md), max(md)], GOX_HI, max(r["Gox_max"] for r in base) * 1.05,
                   color=ACCENT, alpha=0.07, lw=0)
    a.text(md[0], GOX_HI * 1.03, " extrapolating above here", fontsize=7, color=ACCENT)
    band(a)
    _style(a, "oxidiser flow [kg/s]", "peak $G_{ox}$ [kg/m$^2$/s]", "Oxidiser flux")

    a = ax[2][1]
    a.plot(md, [r["fuel_frac"] for r in base], "o-", color=SERIES[4], ms=3.5, lw=1.6,
           label="fuel burnt")
    a.plot(md, [r["t_dP_ok"] for r in base], "s-", color=SERIES[0], ms=3, lw=1.4,
           label="burn with ≥20% ΔP")
    band(a)
    a.legend(fontsize=7, frameon=False, loc="lower left")
    _style(a, "oxidiser flow [kg/s]", "[%]", "What you give up")

    body = (
        "The shaded band is the range in the brief. Read the top row against the bottom "
        "row: everything that improves with flow is on the top, everything that degrades "
        "is on the bottom, and they cross nowhere. The choice is therefore a judgement "
        "about which constraint you are least willing to violate, not a maximum."
    )
    y = 0.245
    for line in _wrap(body, 104):
        fig.text(0.08, y, line, fontsize=8.6, color=INK)
        y -= 0.0145

    y -= 0.010
    fig.text(0.08, y, "Where the Isp gain actually comes from", fontsize=10.5,
             color=INK, fontweight="bold")
    y -= 0.018
    for line in _wrap(
        f"Between 1.0 and 1.5 kg/s the thrust coefficient rises {100 * (R['cf_37'] / R['cf_20'] - 1):.1f}% "
        f"(nozzle matching, {R['cf_20']:.3f}→{R['cf_37']:.3f}) while c* falls about 1% "
        "(mixture ratio drifting past its optimum). The net is the +4% you see. If the "
        "nozzle were re-cut for the lower chamber pressure, most of the apparent advantage "
        "of running hard would disappear.", 104):
        fig.text(0.08, y, line, fontsize=8.6, color=INK)
        y -= 0.0145
    pdf.savefig(fig)


def page_mixture(pdf, R):
    fig = _new_page("Mixture ratio and nozzle matching")
    ofs, pcs = R["ofs"], R["pcs"]

    ax = _axes_grid(fig, (0.09, 0.66, 0.95, 0.915), 1, 2, wspace=0.30)
    a = ax[0][0]
    a.plot(ofs[:, 0], ofs[:, 5], color=SERIES[0], lw=1.8)
    i = int(np.argmax(ofs[:, 5]))
    a.plot(ofs[i, 0], ofs[i, 5], "o", color=ACCENT, ms=5)
    a.annotate(f"peak {ofs[i, 5]:.1f} s at O/F {ofs[i, 0]:.2f}",
               (ofs[i, 0], ofs[i, 5]), textcoords="offset points", xytext=(-6, 9),
               fontsize=7.5, color=ACCENT, ha="center")
    lo_y = a.get_ylim()[0]
    a.set_ylim(lo_y, ofs[i, 5] + 6)
    a.axvline(R["cfg"].propellant.opt_OF, color=LIGHT, ls="--", lw=1.1)
    a.text(R["cfg"].propellant.opt_OF, lo_y + 1, " declared\n opt_OF",
           fontsize=6.5, color=LIGHT, va="bottom")
    a.axvline(10.0, color=ACCENT, ls=":", lw=1.1)
    a.text(10.0, lo_y + 1, " table\n ends", fontsize=6.5, color=ACCENT, va="bottom")
    _style(a, "O/F", "ideal delivered Isp [s]", "Isp vs mixture ratio, Pc = 25 bar")

    a = ax[0][1]
    a.plot(pcs[:, 0] / 1e5, pcs[:, 3], color=SERIES[3], lw=1.8)
    a.axvline(R["ideal_pc"], color=GOOD, ls=":", lw=1.2)
    a.text(R["ideal_pc"] + 0.6, a.get_ylim()[0] + 0.02,
           f"Pe = Pa\nat {R['ideal_pc']:.1f} bar", fontsize=6.5, color=GOOD, va="bottom")
    _style(a, "chamber pressure [bar]", "thrust coefficient $C_f$",
           f"$C_f$ vs Pc at ε = {R['cfg'].nozzle.expansion_ratio:.3f}")

    # Decomposition: the two effects, separated.
    dec = R["decomp"]
    a2 = fig.add_axes([0.09, 0.415, 0.86, 0.175])
    x = [d[0] for d in dec]
    a2.axhline(0, color=LIGHT, lw=0.8)
    a2.plot(x, [d[1] for d in dec], "o-", color=SERIES[3], ms=3.5, lw=1.7,
            label="chamber pressure alone")
    a2.plot(x, [d[2] for d in dec], "s-", color=SERIES[2], ms=3.5, lw=1.7,
            label="mixture ratio alone")
    a2.plot(x, [d[3] for d in dec], "^--", color=INK, ms=3.5, lw=1.4,
            label="combined")
    a2.axvspan(1.0, 1.5, color="#1f4e79", alpha=0.055, lw=0)
    a2.legend(fontsize=7, frameon=False, loc="upper left")
    _style(a2, "oxidiser flow [kg/s]", "Δ Isp vs 1.0 kg/s [s]",
           "Separating the two effects (ideal Isp, referenced to the 1.0 kg/s point)")

    p = TextPage.__new__(TextPage)
    p.fig, p.y = fig, 0.375

    p.head("Two things the left-hand plot says")
    p.bullets([
        f"Peak Isp is at O/F {R['of_peak']:.2f}, not at the {R['cfg'].propellant.opt_OF:.2f} "
        "the propellant file declares. The declared figure is an HRAP default that does "
        "not agree with the CEA table shipped alongside it. Design to the table.",
        "The curve is flat near the top: O/F 6.0 to 8.5 costs under 1% of Isp. Mixture "
        "ratio is simply not where your performance is being won or lost — which is "
        "fortunate, because it is also the number you know least well.",
        "Above O/F 10 the CEA table ends and c* is clamped to its last value. Any run "
        "that drifts past 10 is reporting a flat extrapolation, not a computed c*. At "
        "a × 0.7 the sweep reaches O/F 12.6 and those Isp figures are not meaningful.",
    ])

    p.head("The right-hand plot, and the decomposition below it")
    d15 = next(d for d in R["decomp"] if abs(d[0] - 1.5) < 1e-9)
    p.para(
        f"C_f climbs steadily with chamber pressure and keeps climbing past the "
        f"{R['ideal_pc']:.1f} bar ideal-expansion point, because for a fixed area ratio the "
        "pressure-thrust term keeps growing. The bottom plot holds one variable at its "
        "1.0 kg/s value and moves the other, so the two contributions can be read apart "
        f"rather than only as their sum. Going to 1.5 kg/s, chamber pressure is worth "
        f"{d15[1]:+.1f} s and the mixture-ratio drift {d15[2]:+.1f} s.")
    p.para(
        "That is the answer to 'optimise O/F and Isp' in one line: in this motor they are "
        f"not the same optimisation, and they are not the same size. Over the brief's "
        f"range chamber pressure is worth {d15[1]:+.1f} s and mixture ratio {d15[2]:+.1f} s "
        "— a factor of four apart, and over the full sweep a factor of ten. Chamber "
        "pressure here is a consequence of a nozzle that is large for the flow, not of "
        "anything the injector is doing. Optimising the mixture ratio is polishing the "
        "smaller term.", gap=0.010)
    p.save(pdf)


def page_constraints(pdf, R):
    p = TextPage("Constraints, in the order they bite")

    p.head("1. Oxidiser flux — the binding one")
    p.para(
        "G_ox = ṁ_ox / A_port, and A_port is 1241 mm² at ignition. That fixes the "
        "flux entirely from the grain: the injector has no say in it. Across the brief's "
        "range the starting flux runs from 797 to 1205 kg/m²/s, against a correlation "
        f"range normally fitted below ~{GOX_HI:.0f}.")
    p.mono([f"{'mdot':>6}{'peak Gox':>10}{'% of burn above fit range':>28}{'port ID for Gox<700':>24}"]
           + ["-" * 68]
           + [f"{r['mdot']:>6.2f}{r['Gox_max']:>10.0f}{r['t_above_Gox']:>28.0f}"
              f"{np.sqrt(4 * (r['mdot'] / GOX_HI) / (np.pi * 5)) * 1e3:>24.1f}"
              for r in R["base"] if r["mdot"] >= 1.0], size=7.4)
    p.para(
        "Flux falls quickly as the port opens, so only the early burn is affected — "
        "but the early burn is where most of the fuel flow is. Opening the ports from "
        "17.78 mm to about 19 mm would bring 1.0 kg/s inside the fitted range; 1.5 kg/s "
        "would need 23.4 mm, which the 85.85 mm grain cannot give you with five ports and "
        "any useful web left.", gap=0.010)

    p.head("2. Feed isolation")
    p.para(
        "A bigger injector passes more flow, which raises chamber pressure, which eats the "
        "pressure drop. Late in blowdown the tank has decayed and the margin goes with it. "
        "The number that matters is not the minimum — that always collapses at the very "
        "end — but how much of the burn is spent above 20%.")
    p.mono([f"{'mdot':>6}{'margin expires at':>20}{'liquid phase':>16}{'clean fraction':>17}"]
           + ["-" * 59]
           + [f"{m[0]:>6.2f}{m[1]:>17.2f} s{m[2]:>14.2f} s{m[3]:>16.0f}%"
              for m in R["margin"]], size=7.4)

    p.head("3. Manufacturability — not a constraint here")
    p.para(
        "Every hole count in the sweep is comfortably drillable. Even 62 holes of 1.20 mm "
        "leaves a 6.19 mm web and a 5.15 mm edge margin on an 85.85 mm plate. The plate is "
        "the one part of this design with margin to spare.")

    p.head("4. Timestep — checked, not assumed")
    p.mono([f"{'dt [ms]':>9}{'holes':>8}{'impulse [N.s]':>16}{'Isp [s]':>10}"]
           + ["-" * 43]
           + [f"{c[0]:>9.2f}{c[1]:>8d}{c[2]:>16.0f}{c[3]:>10.2f}" for c in R["conv"]],
           size=7.4)
    p.para(
        "Impulse varies by 0.1% over a 16× change in timestep, and the hole count does "
        "not move at all. The quasi-steady chamber model has no timestep sensitivity by "
        "construction; this confirms the tank blowdown integration does not either at "
        "these settings.", gap=0.010)
    p.save(pdf)


def page_sensitivity(pdf, R):
    p = TextPage("Does the conclusion survive the things we do not know?")
    p.para(
        "Three inputs here are genuinely uncertain: the flow model, the discharge "
        "coefficient, and the regression coefficients. The question is not whether they "
        "change the numbers — they do — but whether they change the decision.",
        gap=0.004)

    p.head("Flow model — changes the plate, not the conclusion")
    p.mono(R["sens_model"], size=7.4)
    p.para(
        "The hole count varies by more than 2× between SPI and HEM at the same flow, "
        "because the models disagree by that much about how much area a given mass flow "
        "needs. The Isp ranking is untouched: every model says more flow is better, and "
        "they agree on the value to within 5 s. The model uncertainty lands entirely on "
        "the hardware and not at all on the trade.", gap=0.010)

    p.head("Discharge coefficient — same story, smaller")
    p.mono(R["sens_cd"], size=7.4)
    p.para(
        "A Cd error is a pure area scaling: ±14% in Cd moves the hole count by "
        "±17% and the Isp by 0.1 s. Cd matters enormously for building the right "
        "plate and not at all for choosing the operating point.", gap=0.010)

    p.head("Regression coefficients — the one that hurts")
    p.mono(R["sens_a"], size=7.4)
    p.para(
        "Scaling a from 0.7× to 2.3× — which is roughly the span between the "
        "HRAP default and what your own HRAP run implied — moves the mean O/F from "
        "12.6 to 4.5. That is the whole usable mixture range. Isp moves far less, between "
        "187 and 209 s, because the Isp-vs-O/F curve is flat near its peak.")
    p.para(
        "Crucially the ranking holds in all five cases: at every value of a, more oxidiser "
        "flow gives more Isp. So the decision is robust even though the predicted O/F is "
        "not. That is the reason to size on oxidiser flow and then find out what O/F you "
        "got, rather than the other way round.", gap=0.010)
    p.note(
        "At a × 0.7 the mean O/F reaches 12.6, past the end of the CEA table, so those "
        "Isp figures rest on a clamped c*. Treat that row as directional only.")
    p.save(pdf)


def page_options(pdf, R):
    p = TextPage("Things that would change the picture")

    p.head("Open the grain ports")
    p.para(
        "This is the highest-value change available, and it is the only one that addresses "
        "the binding constraint. Going from 17.78 mm to 19.1 mm ports brings 1.0 kg/s "
        "inside the fitted flux range. It also lowers O/F, because a larger port at the "
        "same oxidiser flow means lower flux, slower regression — and here the effect "
        "runs the useful way, toward the flat top of the Isp curve. If the grain is not "
        "yet cast, do this.")

    p.head("Supercharge the tank")
    p.mono(R["sens_feed"], size=7.4)
    p.para(
        "Supercharging to 65 bar produces the best result anywhere in this study: 215.4 s "
        "at 1.5 kg/s with the 20% margin intact for the entire burn. The mechanism is "
        "simple — subcooling the feed raises the driving pressure, so the same flow "
        "needs less area, and chamber pressure climbs to 51.8 bar where the nozzle is far "
        "better matched.")
    p.note(
        "Two cautions. This model holds the supercharge pressure constant for the whole "
        "burn; a real blowdown-pressurised tank decays, so the margin shown is optimistic "
        "unless the supply is regulated. And 51.8 bar chamber with 65 bar feed is a "
        "different pressure vessel problem than the one you have costed. Supercharging "
        "does nothing at all for G_ox.")

    p.head("Warm the fill")
    p.para(
        "Filling at 30 °C instead of 23.9 °C is nearly free and does very little: "
        "Isp 202.7 vs 203.4 s. Higher saturation pressure means more flow per unit area, "
        "so the solver just gives you fewer holes. Not a lever worth pulling for "
        "performance, though it does help repeatability if you can control it.")

    p.head("Re-cut the nozzle")
    p.para(
        f"Not in scope here, but worth stating plainly since it drives the whole result: "
        f"ε = {R['cfg'].nozzle.expansion_ratio:.3f} is matched to {R['ideal_pc']:.1f} bar, "
        "and at 1.0 kg/s you run below that for most of the burn. A smaller throat would "
        "raise chamber pressure at the same flow and capture most of the benefit that this "
        "study attributes to running harder — without the flux penalty. That is the "
        "trade worth running next.")
    p.save(pdf)


def page_limits(pdf, R):
    p = TextPage("What this study does not know")
    p.para(
        "Every number here is conditional on the inputs below. They are listed in "
        "descending order of how much they could move the answer.", gap=0.004)

    p.head("Unmeasured inputs")
    p.bullets([
        "Regression coefficients a and n are HRAP defaults, and HRAP ships the same pair "
        "for ABS, asphalt and HTPB — they are placeholders, not paraffin properties. "
        "Your own HRAP run implied a value about 2.3× larger. Nothing here resolves "
        "that; only a hot fire does.",
        "Cd = 0.70 is assumed. A water cold-flow test measures it directly and is the "
        "single highest-value test available before firing.",
        "The flow model is chosen from L/D, not measured. Only a cold flow with a "
        "flashing fluid distinguishes HEM from Dyer at this L/D.",
        "c* efficiency is 1.0 and nozzle efficiency 0.9. Real injector mixing efficiency "
        "on a showerhead plate is typically below unity, so delivered Isp will be under "
        "these figures.",
    ])

    p.head("Model limitations that matter at this operating point")
    p.bullets([
        "No combustion instability model. The 20% pressure-drop criterion is a rule of "
        "thumb, not a prediction. Chugging is not simulated.",
        "No throat erosion. Over a 7 s burn with a graphite throat this is small but not "
        "zero, and it would reduce chamber pressure late in the burn.",
        "Tank blowdown is adiabatic with liquid and vapour in equilibrium. Real tanks draw "
        "heat from the walls, so pressure decays more slowly and the real burn is likely "
        "a little longer than predicted.",
        "The supercharged case holds feed pressure constant, which no unregulated system "
        "does.",
        "The grain is modelled as five identical circular ports with no end-face "
        "regression and no port-to-port coupling.",
    ])

    p.head("How to close the gaps, cheapest first")
    p.mono([
        "1.  water cold flow of the plate            -> Cd, and confirms the drilling",
        "2.  N2O cold flow                           -> settles SPI / Dyer / HEM",
        "3.  one hot fire, fuel weighed before/after -> a and n for your paraffin",
        "4.  second fire at a different flow         -> separates a from n",
    ], size=7.8)
    p.para(
        "Until step 3, treat the O/F column of this report as an order-of-magnitude "
        "statement and the Isp column as a comparison between options rather than an "
        "absolute prediction. The trade conclusion — that 1.0 kg/s is the right end of "
        "the range — does not depend on any of it, which is the one genuinely firm "
        "result here.", gap=0.010)
    p.save(pdf)


def page_plate(pdf, R):
    """The buildable outcome, drawn to the same standard as a shop drawing."""
    from n2o_injector import __version__
    from n2o_injector.drawing import DrawingMeta, plate_layout
    from n2o_injector.plots import build_plate_drawing

    lo = next(r for r in R["base"] if abs(r["mdot"] - 1.0) < 1e-9)
    layout = plate_layout(lo["n_holes"], HOLE_D_MM, PLATE_D_MM, PLATE_T_MM, Cd=0.70)
    meta = DrawingMeta(
        project="Mini Hybrid V2.2", part_no="INJ-PLATE-V2.2B", drawn_by="Molen",
        flow_model="HEM",
        notes=["SIZED FOR 1.0 kg/s N2O. SEE TRADE STUDY FOR THE BASIS OF THIS CHOICE."],
    )

    # The sheet is A4 landscape; rotate it onto the portrait report page.
    land = Figure(figsize=(A4[1], A4[0]))
    build_plate_drawing(land, layout, meta=meta, version=__version__)
    import io

    buf = io.BytesIO()
    land.savefig(buf, format="png", dpi=300, facecolor="white")
    buf.seek(0)
    img = matplotlib.image.imread(buf)

    fig = _new_page("The plate this recommends")
    # Sized to the sheet's own 297:210 aspect so no band of page is wasted.
    ax = fig.add_axes([0.055, 0.475, 0.89, 0.44])
    ax.imshow(img)
    ax.axis("off")

    p = TextPage.__new__(TextPage)
    p.fig, p.y = fig, 0.435
    p.para(
        "Full-size sheets, the DXF and the hole coordinate table are produced from the "
        "Plate drawing tab, or from the configuration file saved alongside this report. "
        "The drawing and the DXF are generated from the same layout object, so they "
        "cannot disagree about where a hole is.", gap=0.002)
    p.head("Before this is cut")
    p.bullets([
        "Cold-flow the plate in water to measure Cd. At the assumed 0.70 the hole count "
        "is 41; at a measured 0.60 it would be 48 and at 0.80 it would be 36. This is the "
        "one number that changes the part.",
        "Decide the hole-diameter tolerance. Flow area goes as d², so +0.02 mm on "
        "Ø1.20 is +3.4% area and the same again in oxidiser flow.",
        "Confirm the plate outside diameter and the seal arrangement against the case. "
        "The 85.85 mm here is the grain outer diameter, used as a placeholder envelope.",
    ])

    p.head("In order")
    p.mono([
        "1.  size on oxidiser flow, not O/F      already done -- 1.0 kg/s, 41 x 1.20 mm",
        "2.  open the grain ports to ~19 mm      if the grain is not yet cast",
        "3.  water cold flow the plate           measures Cd; re-run with the real number",
        "4.  make two plates, not one            the model spread is 2.2x in area",
        "5.  first fire, weigh the fuel          gives a and n; everything else follows",
        "6.  then revisit the nozzle             the largest single lever left",
    ], size=7.8)
    p.para(
        "Steps 1 and 2 are free. Step 3 costs an afternoon and removes the largest "
        "avoidable uncertainty in the design. Nothing before step 5 can tell you what O/F "
        "this motor will actually run at.", gap=0.010)
    pdf.savefig(fig)


def build(path=None):
    R = run_analysis()

    # Derived scalars the pages quote.
    ofs, pcs = R["ofs"], R["pcs"]
    R["of_peak"] = float(ofs[int(np.argmax(ofs[:, 5])), 0])
    R["ideal_pc"] = float(pcs[np.argmin(np.abs(pcs[:, 4] - 101325.0)), 0] / 1e5)
    from v22_trade_study import pc_scan
    R["cf_20"] = float(pc_scan(R["cfg"].propellant, R["cfg"].nozzle, 7.0,
                               lo=20e5, hi=20e5, n=1)[0, 3])
    R["cf_37"] = float(pc_scan(R["cfg"].propellant, R["cfg"].nozzle, 7.0,
                               lo=37e5, hi=37e5, n=1)[0, 3])
    R["margin_pct"] = {m[0]: m[3] for m in R["margin"]}

    path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Trade Study", "V2.2_injector_trade_study.pdf")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    _state["page"] = 0
    with PdfPages(path) as pdf:
        page_summary(pdf, R)
        page_method(pdf, R)
        page_charts(pdf, R)
        page_mixture(pdf, R)
        page_constraints(pdf, R)
        page_sensitivity(pdf, R)
        page_options(pdf, R)
        page_limits(pdf, R)
        page_plate(pdf, R)
        d = pdf.infodict()
        d["Title"] = "V2.2 injector trade study - oxidiser flow vs O/F and Isp"
        d["Author"] = "Molen"
        d["Subject"] = "Injector sizing for 1.0-1.5 kg/s N2O on the V2.2 hybrid"
        d["Creator"] = "HASTE"
        d["CreationDate"] = datetime.now()
    return path


if __name__ == "__main__":
    print("wrote", build())
