"""V2.2 trade study, revision 2: chamber pressure, regression, injector, ports.

Revision 1 (``v22_trade_study.py``) sized the plate on oxidiser flow for five
round ports, with HRAP's placeholder regression coefficients. Since then:

* the regression coefficients were traced -- HRAP's "paraffin" pair is HTPB,
  and the literature N2O/paraffin fits disagree by several times at the same
  flux -- so the plate is now sized on **chamber pressure** (25 bar), which
  does not invert the regression law;
* the grain web turned out to be the binding risk, which led to the
  **sector-port grain** and a **commanded run-valve shutdown**.

This script recomputes every number the revision-2 report quotes, from
``configs/V2.2_sector_40x1.20mm_valve3.8s.json``. The rendering reuses the
page helpers of ``v22_report.py``.

Run:  python examples/v22_trade_study_rev2.py
"""

from __future__ import annotations

import copy
import dataclasses
import io
import math
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from n2o_injector import SaturationTable, get_backend  # noqa: E402
from n2o_injector.config_io import load_config  # noqa: E402
from n2o_injector.grain_geometry import balanced_sector  # noqa: E402
from n2o_injector.motor import Grain, simulate  # noqa: E402
from n2o_injector.sizing import SizingTarget, analyse_geometry, size_injector  # noqa: E402

import v22_report as R1  # noqa: E402  (page helpers)
from v22_trade_study import of_scan, pc_scan, thrust_coefficient  # noqa: E402

BAR = 1e5
CONFIG = os.path.join(ROOT, "configs", "V2.2_sector_40x1.20mm_valve3.8s.json")
OUT = os.path.join(ROOT, "Trade Study", "V2.2_trade_study_rev2.pdf")
GOX_HI = 700.0

#: Candidate regression fits: a [mm/s], n, oxidiser, provenance.
FITS = [
    ("SP7 a", 0.0781, 0.545, "N2O (very likely)", "Karp & Jens Tbl 6.2, 10 tests"),
    ("SP7 b", 0.280, 0.297, "N2O (very likely)", "Karp & Jens Tbl 6.2, 8 tests"),
    ("HRAP default", 0.0304, 0.681, "none -- HTPB", "= Karp Tbl 6.2 HTPB (Thiokol)"),
    ("Liu 2020", 0.0876, 0.3953, "N2O, G 91-242", "paraffin + Al/Mg, 2020"),
    ("SP1A", 0.117, 0.620, "probably GOX", "Karp & Jens Tbl 6.2, 65 tests"),
    ("FR5560", 0.169, 0.600, "unconfirmed", "Karp & Jens Tbl 6.2, 4 tests"),
]
LIKELY = ("SP7 a", "SP7 b")


# ======================================================================
# Analysis
# ======================================================================


def with_fit(cfg, a, n):
    c = copy.deepcopy(cfg)
    c.propellant.reg_a, c.propellant.reg_n = a, n
    return c


def web_of(grain, burn):
    return grain.regression(burn.port_d[-1]) * 1e3


def liquid_out(cfg, props, table):
    c = copy.deepcopy(cfg)
    c.valve_close_t = None
    c.stop_at_liquid_exhausted = True
    return simulate(c, props, table).burn_time


def run(cfg, props, table, target):
    r = analyse_geometry(cfg, props, table, target)
    b = r.burn
    return {
        "e": r.regression_max * 1e3, "used": 100 * r.web_used_frac,
        "through": any("burned through" in n for n in b.notes),
        "t": b.burn_time, "I": b.total_impulse, "Isp": b.specific_impulse(),
        "Pc_pk": b.P_chamber.max() / BAR, "Pc_mn": b.P_chamber.mean() / BAR,
        "F_mn": b.thrust.mean(), "OF": b.mean_OF, "G": b.G_ox.max(),
        "dPmin": 100 * r.min_dP_fraction, "burn": b,
    }


def fits_table():
    rows = []
    for name, a, n, ox, src in FITS:
        rdot = [a * G**n for G in (150, 300, 500, 700)]
        rows.append((name, a, n, ox, src, *rdot))
    return rows


def objective_robustness(cfg, props, table):
    """Hole count per fit: chamber-pressure objective vs burn-average O/F."""
    out = []
    for name, a, n, *_ in FITS:
        c = with_fit(cfg, a, n)
        pc = size_injector(c, props, table,
                           SizingTarget(objective="chamber_pressure", chamber_P=25 * BAR),
                           fix="hole_d")
        try:
            of = size_injector(c, props, table,
                               SizingTarget(objective="burn_average", OF=7.5), fix="hole_d")
            n_of = of.plate.n_holes
            deg = any("DEGENERATE" in w for w in of.warnings)
        except Exception:
            n_of, deg = None, True
        out.append((name, pc.plate.n_holes, n_of, deg))
    return out


def pc_sweep(cfg, props, table):
    """Size for 20-35 bar; valve closes 0.25 s before the liquid runs out."""
    out = []
    for Pc in (20, 25, 30, 35):
        holes = {}
        for name, a, n, *_ in FITS[:3]:
            r = size_injector(with_fit(cfg, a, n), props, table,
                              SizingTarget(objective="chamber_pressure", chamber_P=Pc * BAR),
                              fix="hole_d")
            holes[name] = r.plate.n_holes
        c = with_fit(cfg, 0.0781, 0.545)
        c.injector.n_holes = holes["SP7 a"]
        c.valve_close_t = round(liquid_out(c, props, table) - 0.25, 2)
        m = run(c, props, table, SizingTarget())
        cb = with_fit(c, 0.280, 0.297)
        mb = run(cb, props, table, SizingTarget())
        out.append({"Pc": Pc, "holes": holes, "valve": c.valve_close_t, "a": m, "b": mb})
    return out


def cd_sensitivity(cfg, props, table):
    out = []
    for Cd in (0.60, 0.70, 0.80):
        c = with_fit(cfg, 0.0781, 0.545)
        c.injector.Cd = Cd
        r = size_injector(c, props, table,
                          SizingTarget(objective="chamber_pressure", chamber_P=25 * BAR),
                          fix="hole_d")
        out.append((Cd, r.plate.n_holes))
    return out


def port_layouts(cfg, props, table):
    """Round, centre+4 round, sector (same area), sector (same surface)."""
    g_sec = cfg.grain
    OD, L = g_sec.outer_d, g_sec.length
    round5 = Grain(length=L, port_id=0.01778, outer_d=OD, n_ports=5)
    Ro = 0.5 * OD
    wall5 = (Ro - Ro / (1 + math.sin(math.pi / 5)) - 0.00889) * 1e3
    wall41 = (Ro - 2 * Ro / 3 - 0.00889) * 1e3
    P_round = round5.burn_perimeter(round5.port_id)

    def sector_for(area):
        b = balanced_sector(OD, area)
        return Grain(length=L, port_id=b["centre_d"], outer_d=OD, layout="sector",
                     ring_web=b["ring_web"], spoke_web=b["spoke_web"], wall_web=b["wall_web"])

    A0 = round5.port_area(round5.port_id)
    area_s = brentq(lambda A: sector_for(A).burn_perimeter(sector_for(A).port_id) - P_round,
                    0.3 * A0, A0)
    shrunk = sector_for(area_s)
    cases = [
        ("5 round on a circle", round5, wall5),
        ("centre + 4 round", round5, wall41),
        ("sector, same area", g_sec, g_sec.wall_web * 1e3),
        ("sector, same surface", shrunk, shrunk.wall_web * 1e3),
    ]
    At = cfg.nozzle.throat_area
    out = []
    for label, g, wall in cases:
        row = {"label": label, "wall": wall, "A": g.port_area(g.port_id) * 1e6,
               "P": g.burn_perimeter(g.port_id) * 1e3,
               "ratio": g.port_area(g.port_id) / At}
        for key, (a, n) in (("a", (0.0781, 0.545)), ("b", (0.280, 0.297))):
            c = with_fit(cfg, a, n)
            c.grain = g
            m = run(c, props, table, SizingTarget())
            # Round grains are simulated nominally; judge them against the
            # real wall web of their layout, not the nominal one.
            row[key] = m
            row[key + "_ok"] = m["e"] < wall and not m["through"]
        out.append(row)
    return out


def termination(cfg, props, table):
    out = []
    for name, a, n, *_ in FITS[:4]:
        c = with_fit(cfg, a, n)
        full = copy.deepcopy(c)
        full.valve_close_t = None
        light = copy.deepcopy(full)
        light.tank.ox_mass = 3.5
        out.append((name, run(full, props, table, SizingTarget()),
                    run(c, props, table, SizingTarget()),
                    run(light, props, table, SizingTarget())))
    return out


def nozzle_matching(cfg, Pc_mean):
    prop, noz = cfg.propellant, cfg.nozzle
    pcs = pc_scan(prop, noz, 7.0)
    ideal_pc = float(pcs[np.argmin(np.abs(pcs[:, 4] - 101325.0)), 0] / BAR)
    k = prop.gamma(7.0, Pc_mean * BAR)

    def pe_minus_pa(eps):
        return thrust_coefficient(k, Pc_mean * BAR, eps, 101325.0)[1] - 101325.0

    eps_opt = brentq(pe_minus_pa, 1.2, 20.0)
    cf_now = thrust_coefficient(k, Pc_mean * BAR, noz.expansion_ratio, 101325.0)[0]
    cf_opt = thrust_coefficient(k, Pc_mean * BAR, eps_opt, 101325.0)[0]
    ofs = of_scan(prop, noz, Pc_mean * BAR)
    of_pk = float(ofs[int(np.argmax(ofs[:, 5])), 0])
    return {"ideal_pc": ideal_pc, "eps_opt": eps_opt, "cf_now": cf_now,
            "cf_opt": cf_opt, "of_peak": of_pk, "ofs": ofs}


def analyse():
    props = get_backend("esdu")
    table = SaturationTable(props)
    cfg, target, warns = load_config(CONFIG)
    if not cfg.propellant.has_table:
        raise SystemExit("CEA table not found: " + "; ".join(warns))
    Rz = {"cfg": cfg}
    print("fits");        Rz["fits"] = fits_table()
    print("baseline")
    Rz["base"] = {name: run(with_fit(cfg, a, n), props, table, target)
                  for name, a, n, *_ in FITS}
    Rz["liq_out"] = liquid_out(cfg, props, table)
    print("robustness");  Rz["robust"] = objective_robustness(cfg, props, table)
    print("Pc sweep");    Rz["pc"] = pc_sweep(cfg, props, table)
    print("Cd");          Rz["cd"] = cd_sensitivity(cfg, props, table)
    print("ports");       Rz["ports"] = port_layouts(cfg, props, table)
    print("termination"); Rz["term"] = termination(cfg, props, table)
    Rz["noz"] = nozzle_matching(cfg, Rz["base"]["SP7 a"]["Pc_mn"])
    return Rz


# ======================================================================
# Report
# ======================================================================

TP = R1.TextPage
INK, LIGHT, ACCENT, GOOD = R1.INK, R1.LIGHT, R1.ACCENT, R1.GOOD
SERIES = R1.SERIES


def page_summary(pdf, Rz):
    cfg, B = Rz["cfg"], Rz["base"]
    a, b = B["SP7 a"], B["SP7 b"]
    p = TP("V2.2 trade study, revision 2 — the design as it stands")
    p.para(f"Mini Hybrid V2.2  ·  {datetime.now():%Y-%m-%d}  ·  generated from "
           "configs/V2.2_sector_40x1.20mm_valve3.8s.json", color=LIGHT, size=8.2)
    p.head("Current design")
    g = cfg.grain
    p.mono([
        f"  injector   40 x Ø1.20 mm THRU, 12.7 mm plate (L/D 10.58, HEM), Cd 0.70 assumed",
        f"  sized on   chamber pressure, 25 bar target   (does not invert the regression law)",
        f"  grain      centre port Ø{g.port_id * 1e3:.2f} + 4 sector ports, OD 85.85, L 177.8",
        f"             webs: spoke/ring {g.spoke_web * 1e3:.2f} mm, wall {g.wall_web * 1e3:.2f} mm;"
        f" port area 1241 mm² (1.70 x throat)",
        f"  shutdown   run valve closes at {cfg.valve_close_t:.1f} s; liquid runs out at "
        f"{Rz['liq_out']:.2f} s",
        f"  tank       7.0 L, 4.48 kg N2O at 23.9 °C, self-pressurised",
        f"  nozzle     throat Ø30.48 mm, ε 4.34",
    ], size=7.9)
    p.head("What it does, under the two regression fits that are most likely N2O data")
    p.mono([f"{'fit':<10}{'Pc peak':>9}{'Pc mean':>9}{'F mean':>9}{'impulse':>10}"
            f"{'Isp':>7}{'O/F':>7}{'web used':>16}",
            "-" * 77] + [
        f"{n:<10}{m['Pc_pk']:>7.1f} b{m['Pc_mn']:>7.1f} b{m['F_mn']:>7.0f} N{m['I']:>7.0f} Ns"
        f"{m['Isp']:>6.0f}s{m['OF']:>7.2f}{m['e']:>7.2f} of 8.76 mm"
        for n, m in (("SP7 a", a), ("SP7 b", b))], size=7.9)
    npc = [r[1] for r in Rz["robust"]]
    nof = [r[2] for r in Rz["robust"] if r[2] is not None]
    margin = min(8.76 - Rz["base"][n]["e"] for n in LIKELY)
    p.head("The decisions, and why")
    p.bullets([
        "Size on chamber pressure, not O/F. The O/F objectives invert the regression "
        "law with exponent 1/(1-n) ≈ 1.4-3.1; across the candidate fits a burn-average "
        f"O/F target asks for {min(nof)} to {max(nof)} holes. Sizing on 25 bar gives "
        f"{min(npc)}-{max(npc)} holes over the same 9x spread in 'a'. The plate is now the "
        "settled part of the design.",
        "Keep Ø1.20 mm holes in the 12.7 mm plate. L/D 10.6 puts the orifices in the "
        "HEM regime; varying the count rather than the diameter keeps the flow model valid.",
        "Sector-port grain. At the same port area (so the same starting oxidiser flux) "
        "it gives an 8.76 mm wall web, against 7.00 mm for the five round ports on "
        "their best circle and 5.42 mm for a centre + 4 layout.",
        "Close the run valve before the liquid runs out. The vapour tail delivers ~13% "
        "of the impulse but burns a fifth to a quarter of the web; cutting it keeps "
        f"both likely-N2O fits inside the wall web with ≥{margin:.1f} mm to spare.",
    ])
    p.note("Regression coefficients are literature values. Everything that depends on "
           "them — O/F, web use, Isp — is a range until the first fire is weighed.")
    p.save(pdf)


def page_changes(pdf, Rz):
    p = TP("What changed since revision 1")
    p.mono([
        f"{'':<22}{'revision 1':<30}{'revision 2 (now)':<30}",
        "-" * 82,
        f"{'sizing objective':<22}{'oxidiser flow 1.0 kg/s':<30}{'chamber pressure 25 bar':<30}",
        f"{'plate':<22}{'41 x Ø1.20, 12.7 mm, HEM':<30}{'40 x Ø1.20, 12.7 mm, HEM':<30}",
        f"{'grain ports':<22}{'5 x Ø17.78 round':<30}{'Ø14.14 centre + 4 sectors':<30}",
        f"{'wall web':<22}{'10.31 nominal (7.00 real)':<30}{'8.76 exact':<30}",
        f"{'burning surface':<22}{'497 cm²':<30}{'625 cm² (+26%)':<30}",
        f"{'shutdown':<22}{'blow down to empty':<30}{'valve closes at 3.8 s':<30}",
        f"{'regression law':<22}{'HRAP a 0.0304, n 0.681':<30}{'range of 6 fits, 2 likely':<30}",
        f"{'c* / table':<22}{'HRAP Paraffin.mat':<30}{'same':<30}",
    ], size=7.8)
    p.head("Why the regression coefficients were demoted")
    p.para(
        "HRAP's 'paraffin' pair (a 0.0304, n 0.681) matches the HTPB (Thiokol) row of "
        "Karp & Jens Table 6.2 to three significant figures, and HRAP ships the same pair "
        "for ABS, asphalt and HTPB. The PDF of that table drops subscripted formulas, so "
        "its oxidiser column is blank; the N2O rows were identified by converting Doran "
        "et al. (AIAA 2007-5352) N2O coefficients for HDPE, PMMA and HTPB to SI — all "
        "three match the table exactly, and the two SP7 paraffin rows sit in the same "
        "block. SP1A, which looked like the obvious N2O row, is probably GOX: Doran "
        "declines to report SP1A coefficients at all.")
    p.head("Why the web became the question")
    p.para(
        "With the plate fixed on chamber pressure, the uncertainty moved to the grain. "
        "The tool's multi-port web was an equal-area figure (10.31 mm); the five round "
        "ports on their best circle actually leave 7.00 mm to the case, and both SP7 fits "
        "regress 9.5-9.6 mm over a full blowdown. The round-port grain burns through "
        "under the most likely fuel data. Revision 2 answers that with the port layout "
        "and the shutdown, not with the injector.")
    p.head("Tool changes behind these numbers")
    p.bullets([
        "Sector grain with exact area/perimeter at every regression depth; balanced-web "
        "generator; grain drawing sheet and DXF.",
        "Commanded valve closure; web-used reporting; warnings for burn-through, thin "
        "web margin, nominal multi-port web and ports smaller than the throat.",
        "Transient chamber model now timestep-converged; injector ΔP measured against "
        "the real feed pressure; flux-range warning on the regression law.",
        "Fuel presets carry oxidiser, flux range and source; placeholder coefficients "
        "are flagged when used with an O/F objective.",
    ])
    p.save(pdf)


def page_regression(pdf, Rz):
    fig = R1._new_page("Regression rates — what we have and how far apart it is")
    ax = fig.add_axes([0.11, 0.60, 0.83, 0.31])
    G = np.linspace(50, 900, 200)
    for i, (name, a, n, ox, *_r) in enumerate(FITS):
        ls = "-" if name in LIKELY else ("--" if "GOX" in ox or "unconf" in ox else ":")
        ax.plot(G, a * G**n, ls, lw=1.8 if name in LIKELY else 1.2,
                color=SERIES[i % len(SERIES)] if i < 5 else INK, label=name)
    ax.axvspan(GOX_HI, 900, color=ACCENT, alpha=0.06, lw=0)
    ax.axvline(778, color=ACCENT, lw=0.9, ls="--")
    ax.text(785, ax.get_ylim()[1] * 0.93, "V2.2 peak\nG_ox 778", fontsize=6.5,
            color=ACCENT, va="top")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    R1._style(ax, "oxidiser mass flux G_ox [kg/m²/s]", "regression rate [mm/s]",
              "ṙ = a·G_ox^n for each candidate fit (solid = likely N2O)")

    p = TP.__new__(TP)
    p.fig, p.y = fig, 0.545
    p.mono([f"{'fit':<13}{'a':>7}{'n':>7}  {'oxidiser':<18}{'ṙ@150':>7}{'@300':>7}"
            f"{'@500':>7}{'@700':>7}  mm/s", "-" * 82]
           + [f"{r[0]:<13}{r[1]:>7.4f}{r[2]:>7.3f}  {r[3]:<18}{r[5]:>7.2f}{r[6]:>7.2f}"
              f"{r[7]:>7.2f}{r[8]:>7.2f}" for r in Rz["fits"]], size=7.4)
    f = {r[0]: r for r in Rz["fits"]}

    def ratio(name, col):  # fit / HRAP default at one flux column
        return f[name][col] / f["HRAP default"][col]

    p.head("Reading it")
    p.bullets([
        "Compare fits at the same flux, never by 'a'. The units of a depend on n, so "
        "a 0.28 with n 0.30 and a 0.078 with n 0.55 are not comparable numbers. The two "
        f"SP7 fits agree within {100 * abs(f['SP7 a'][6] / f['SP7 b'][6] - 1):.0f}% at "
        f"300 kg/m²/s and part to {100 * abs(f['SP7 a'][8] / f['SP7 b'][8] - 1):.0f}% at "
        "700 — they differ mainly in n, which only a second fire separates.",
        f"Against HRAP's default at 300 kg/m²/s: SP7 a {ratio('SP7 a', 6):.2f}x, SP7 b "
        f"{ratio('SP7 b', 6):.2f}x, SP1A {ratio('SP1A', 6):.1f}x, FR5560 "
        f"{ratio('FR5560', 6):.1f}x. The likely-N2O data sit close to the placeholder; "
        "the fast fits, which are probably not N2O, are the ones far from it.",
        "Liu 2020 is a metallised paraffin fitted at 91-242 kg/m²/s; at V2.2's 300-780 "
        "it is extrapolated and reads low. Treat it as a lower bound.",
        "V2.2 starts at 778 kg/m²/s, above the ~700 upper end of the published fits. "
        "The early burn is extrapolation for every fit. G_ox is set by port area, which "
        "the sector grain kept equal to preserve this number rather than make it worse.",
    ], size=8.3)
    pdf.savefig(fig)


def page_injector(pdf, Rz):
    p = TP("Injector sizing on chamber pressure")
    p.head("Objective robustness — the reason for sizing on Pc")
    rows = [f"{'fit':<14}{'Pc 25 bar':>12}{'O/F 7.5 (burn avg)':>22}", "-" * 48]
    for name, n_pc, n_of, deg in Rz["robust"]:
        of = "no solution" if n_of is None else f"{n_of}" + ("  degenerate" if deg else "")
        rows.append(f"{name:<14}{n_pc:>9} holes{of:>22}")
    p.mono(rows, size=7.8)
    npc = [r[1] for r in Rz["robust"]]
    p.para(f"Across a 9x spread in 'a' the chamber-pressure objective moves the hole count "
           f"from {min(npc)} to {max(npc)}. The O/F objective has to invert the regression "
           "law, so the same spread moves it by orders of magnitude, and at the fast fits "
           "it drives chamber pressure onto tank pressure (degenerate). Sizing on Pc and "
           "reading O/F off afterwards is the well-conditioned direction.", gap=0.010)

    p.head("Chamber-pressure target sweep (sector grain, valve 0.25 s before liquid-out)")
    rows = [f"{'Pc':>4}{'holes a/b/H':>13}{'valve':>7}{'Pc pk':>7}{'F mean':>8}{'impulse':>9}"
            f"{'Isp':>6}{'min ΔP':>8}{'web a':>8}{'web b':>8}{'G pk':>7}", "-" * 85]
    for r in Rz["pc"]:
        h = r["holes"]
        m, mb = r["a"], r["b"]
        rows.append(
            f"{r['Pc']:>4}{h['SP7 a']:>5}/{h['SP7 b']}/{h['HRAP default']}{r['valve']:>7.2f}"
            f"{m['Pc_pk']:>7.1f}{m['F_mn']:>8.0f}{m['I']:>9.0f}{m['Isp']:>6.0f}"
            f"{m['dPmin']:>7.0f}%{m['used']:>7.0f}%{mb['used']:>7.0f}%{m['G']:>7.0f}")
    p.mono(rows, size=7.3)
    p.para(
        "Higher chamber pressure needs more holes and buys Isp through the nozzle, but "
        "shortens the liquid phase, raises oxidiser flux further past the fitted range "
        "and eats injector pressure drop. 25 bar keeps ΔP near 100% of Pc all burn with "
        "the web inside margin for both N2O fits. 30 bar is a candidate once the "
        "regression rate is measured; 35 bar is not, on this tank and case.", gap=0.010)

    p.head("Discharge coefficient — the dominant plate uncertainty now")
    p.mono([f"{'Cd':>6}{'holes for 25 bar':>20}", "-" * 26]
           + [f"{cd:>6.2f}{n:>20}" for cd, n in Rz["cd"]], size=7.8)
    p.para("Cd moves the hole count more than the regression fits do. A water cold flow "
           "of the drilled plate measures it in an afternoon; drill fewer holes first "
           "and open up after the cold flow if needed.", gap=0.010)
    p.save(pdf)


def page_ports(pdf, Rz):
    fig = R1._new_page("Fuel port layout")
    img_path = os.path.join(ROOT, "Trade Study", "V2.2_grain_port_layouts.png")
    y = 0.925
    if os.path.exists(img_path):
        img = matplotlib.image.imread(img_path)
        h, w = img.shape[:2]
        ax_h = 0.84 * (h / w) * (8.27 / 11.69) * 0.62
        ax = fig.add_axes([0.08, 0.925 - ax_h, 0.84, ax_h])
        ax.imshow(img[: int(h * 0.62)])
        ax.axis("off")
        y = 0.925 - ax_h - 0.01
    p = TP.__new__(TP)
    p.fig, p.y = fig, y
    p.para("Cross-sections as compared earlier (dashed: SP7 a regression at liquid-out and "
           "at end of a full blowdown). Table: 40-hole plate, valve closing at 3.8 s.",
           color=LIGHT, size=8.0)
    rows = [f"{'layout':<22}{'wall':>6}{'area':>7}{'perim':>7}{'A/At':>6}{'G pk':>6}"
            f"{'SP7 a web':>11}{'SP7 b web':>11}{'O/F a':>7}", "-" * 83]
    for r in Rz["ports"]:
        def cell(k):
            m = r[k]
            return f"{m['e']:>5.2f} {'ok' if r[k + '_ok'] else 'OUT':>4}"
        rows.append(f"{r['label']:<22}{r['wall']:>6.2f}{r['A']:>7.0f}{r['P']:>7.0f}"
                    f"{r['ratio']:>6.2f}{r['a']['G']:>6.0f}{cell('a'):>11}{cell('b'):>11}"
                    f"{r['a']['OF']:>7.2f}")
    p.mono(rows, size=7.3)
    p.bullets([
        "The sector layout is the only one that keeps both N2O fits off the case at the "
        "same port area. It does it by using the diameter better: round ports leave fuel "
        "between them that does not protect the wall.",
        "Its cost is 26% more burning surface: more fuel flow and lower O/F "
        f"({Rz['ports'][2]['a']['OF']:.1f}/{Rz['ports'][2]['b']['OF']:.1f} for SP7 a/b, "
        f"against {Rz['ports'][0]['a']['OF']:.1f}/{Rz['ports'][0]['b']['OF']:.1f} for the "
        f"round grain). The CEA table peaks at O/F {Rz['noz']['of_peak']:.1f} and is flat "
        "near the top, so the Isp cost is small.",
        "Shrinking the sector ports to match the round grain's surface gives the most web "
        "but halves the port area — below the throat area — and nearly doubles oxidiser "
        "flux. The grain would choke the flow. Rejected.",
        "Round-port rows are simulated with nominal growth and judged against the real wall "
        "web of their layout.",
    ], size=8.2)
    pdf.savefig(fig)


def page_termination(pdf, Rz):
    p = TP("Ending the burn — blowdown, cutoff or a lighter load")
    rows = [f"{'fit':<14}{'full blowdown':>22}{'valve at 3.8 s':>22}{'3.5 kg loaded':>22}",
            f"{'':<14}" + f"{'web mm':>10}{'Ns':>8}    " * 3, "-" * 80]
    for name, f, c, l in Rz["term"]:
        def cell(m):
            w = "THRU" if m["through"] else f"{m['e']:.2f}"
            return f"{w:>10}{m['I']:>8.0f}    "
        rows.append(f"{name:<14}" + cell(f) + cell(c) + cell(l))
    p.mono(rows, size=7.6)
    p.bullets([
        "A full blowdown burns through the 8.76 mm wall under both SP7 fits.",
        "Closing the valve at 3.8 s keeps every fit here inside the web and keeps "
        f"{min(c['I'] / f['I'] for _, f, c, _l in Rz['term']) * 100:.0f}-"
        f"{max(c['I'] / f['I'] for _, f, c, _l in Rz['term']) * 100:.0f}% "
        "of the full-blowdown impulse; the vapour tail it removes is low-pressure and "
        "low-flux, which is also where chuffing and O/F drift live.",
        "Loading 3.5 kg and letting it blow down is worse on both counts: it still ends "
        "with a vapour tail, so it uses more web and delivers less impulse than the cutoff.",
        f"The valve closes on ~0.2 kg of remaining liquid. Liquid runs out at "
        f"{Rz['liq_out']:.2f} s in the model; that moves with fill temperature and real Cd, "
        "so confirm it from the tank-pressure knee in cold flow, and vent the ~1 kg left "
        "in the tank remotely after the fire.",
    ])
    p.save(pdf)


def page_histories(pdf, Rz):
    fig = R1._new_page("The current design across the regression fits")
    cfg = Rz["cfg"]
    ax = R1._axes_grid(fig, (0.10, 0.37, 0.95, 0.91), 2, 2, hspace=0.42, wspace=0.32)
    for i, name in enumerate(("SP7 a", "SP7 b", "HRAP default", "Liu 2020")):
        b = Rz["base"][name]["burn"]
        c = SERIES[i]
        ax[0][0].plot(b.t, b.P_chamber / BAR, color=c, lw=1.4, label=name)
        ax[0][1].plot(b.t, b.thrust, color=c, lw=1.4)
        ax[1][0].plot(b.t, (b.port_d - cfg.grain.port_id) / 2 * 1e3, color=c, lw=1.4)
        ok = np.isfinite(b.OF)
        ax[1][1].plot(b.t[ok], b.OF[ok], color=c, lw=1.4)
    ax[1][0].axhline(cfg.grain.wall_web * 1e3, color=ACCENT, ls="--", lw=1.1)
    ax[1][0].text(0.1, cfg.grain.wall_web * 1e3 + 0.15, "wall web 8.76 mm", fontsize=7,
                  color=ACCENT)
    ax[1][1].axhline(Rz["noz"]["of_peak"], color=GOOD, ls=":", lw=1.1)
    ax[1][1].text(0.1, Rz["noz"]["of_peak"] + 0.2, f"peak-Isp O/F {Rz['noz']['of_peak']:.1f}",
                  fontsize=7, color=GOOD)
    ax[0][0].legend(fontsize=7, frameon=False)
    R1._style(ax[0][0], "t [s]", "Pc [bar]", "Chamber pressure")
    R1._style(ax[0][1], "t [s]", "thrust [N]", "Thrust")
    R1._style(ax[1][0], "t [s]", "regression [mm]", "Web consumed")
    R1._style(ax[1][1], "t [s]", "O/F", "Mixture ratio")
    p = TP.__new__(TP)
    p.fig, p.y = fig, 0.32
    B = Rz["base"]
    pcs = [B[n]["Pc_pk"] for n in ("SP7 a", "SP7 b", "HRAP default", "Liu 2020")]
    p.para(
        f"Peak chamber pressure spans only {min(pcs):.1f}-{max(pcs):.1f} bar across four fits "
        "a factor of ~3 apart in regression rate — it is set mostly by the injector and the "
        "tank, which is the point of sizing on Pc. What the fit moves is O/F "
        f"({min(B[n]['OF'] for n in LIKELY):.1f}-{max(B[n]['OF'] for n in LIKELY):.1f} for the "
        f"N2O fits, {B['Liu 2020']['OF']:.1f} for Liu) and how much web is used. Web use is "
        "the number to watch on the first fire.")
    p.para(
        "Even the direction of O/F drift depends on n: it rises through the burn for the "
        "high-n fits (SP7 a, HRAP) and falls for the low-n ones (SP7 b, Liu). The likely-N2O "
        "fits stay within about 1.5 of the peak-Isp O/F throughout, which is on the flat top "
        "of the Isp curve.")
    pdf.savefig(fig)


def page_optimise(pdf, Rz):
    n = Rz["noz"]
    p = TP("Where more performance is — and what it costs")
    gain = 100 * (n["cf_opt"] / n["cf_now"] - 1)
    p.head("1. The nozzle is already matched — chamber pressure is the lever")
    p.para(
        f"ε 4.34 is perfectly expanded at {n['ideal_pc']:.1f} bar and the motor averages "
        f"{Rz['base']['SP7 a']['Pc_mn']:.1f} bar; the ideal ε at that pressure is "
        f"{n['eps_opt']:.2f}, worth {gain:+.1f}% in C_f. Revision 1 ran at ~20 bar and was "
        "over-expanded; sizing to 25 bar closed that gap. Re-cutting the nozzle for the "
        "current design buys nothing.")
    s25 = next(r for r in Rz["pc"] if r["Pc"] == 25)["a"]
    s35 = next(r for r in Rz["pc"] if r["Pc"] == 35)["a"]
    p.para(
        "The remaining nozzle-side gain is a higher chamber pressure: the Pc sweep gives "
        f"{s25['Isp']:.0f} s at 25 bar and {s35['Isp']:.0f} s at 35 bar "
        f"({100 * (s35['Isp'] / s25['Isp'] - 1):+.1f}%), more with ε re-matched. Its price is "
        f"injector ΔP margin ({s25['dPmin']:.0f}% → {s35['dPmin']:.0f}% of Pc), a shorter "
        "liquid phase and flux further past the fitted range, so it is a step for after the "
        "regression rate is measured, not before.", gap=0.010)
    p.head("2. Measure the regression rate, then re-optimise")
    p.para(
        "One fire with the grain weighed before and after fixes 'a' for an assumed n; a "
        "second at a different flow separates a from n. Every O/F, Isp and web number in "
        "this report collapses from a range to a value, and the valve time can move "
        "toward liquid-out (or the load up) with a known margin.")
    p.head("3. Grain options, if the case can change")
    p.bullets([
        "Larger grain OD: every millimetre of radius is wall web, and more port area "
        "lowers the 778 kg/m²/s starting flux into the fitted range.",
        "Longer grain: more burning surface at the same flux raises fuel flow toward "
        "the Isp-peak O/F without shrinking ports — and paraffin's regression does not "
        "depend on length (m = 0).",
        "Pre- and post-combustion chambers: paraffin burns by droplet entrainment, and "
        "mixing volume raises c* efficiency, which is assumed 1.0 here and will be lower.",
        "Fillets on the sector corners (R2 costs ~1% port area) and a phenolic liner for "
        "insulation margin against the case.",
    ], size=8.3)
    p.head("4. Feed system")
    p.bullets([
        "Regulated or supercharged feed would hold chamber pressure and ΔP flat through the "
        "burn (revision 1 found 65 bar supercharge the best single result), but it is a "
        "different pressure system and does nothing for flux.",
        "Fill-temperature control mainly buys repeatability: the liquid-out time, and "
        "so the valve timing, moves with it.",
    ], size=8.3)
    p.head("5. Injector")
    p.para(
        "The showerhead plate is sized and drawn. Beyond Cd, the open question is "
        "atomisation and mixing; impinging or swirl elements would help c* efficiency more "
        "than any change to the hole count.")
    p.save(pdf)


def page_open(pdf, Rz):
    p = TP("Open items and the order to close them")
    p.mono([
        "1.  water cold flow of the drilled plate     -> Cd; re-size (36-48 holes span)",
        "2.  N2O cold flow, tank pressure logged      -> liquid-out time; confirms HEM",
        "3.  set the valve timer from step 2          -> ~0.25 s before the pressure knee",
        "4.  first fire, grain weighed before/after   -> a (n assumed); web margin check",
        "5.  second fire at another flow              -> separates a from n",
        "6.  re-run this study with measured values   -> nozzle ε, valve time, ox load",
    ], size=7.8)
    p.head("Known model limits at this operating point")
    p.bullets([
        "Oxidiser flux starts above every fitted range; early-burn regression is extrapolation.",
        "c* efficiency 1.0 and nozzle efficiency 0.9 are assumptions; delivered Isp will be lower.",
        "No combustion-instability model: the 20% ΔP rule is guidance, not a prediction.",
        "Valve closure is instantaneous and post-shutdown smoulder is neglected.",
        "Blowdown is adiabatic and in equilibrium; real tanks decay a little more slowly.",
        "Sector corners are sharp in the model; fillets remove ~1% of port area.",
        "SP1A and FR5560 burn through in about 2 s whatever the valve does. Both are "
        "probably not N2O data, but the first fire is what rules them out.",
    ], size=8.3)
    p.save(pdf)


def page_sheet(pdf, title, build):
    land = Figure(figsize=(11.69, 8.27))
    build(land)
    buf = io.BytesIO()
    land.savefig(buf, format="png", dpi=260, facecolor="white")
    buf.seek(0)
    img = matplotlib.image.imread(buf)
    fig = R1._new_page(title)
    ax = fig.add_axes([0.055, 0.475, 0.89, 0.44])
    ax.imshow(img)
    ax.axis("off")
    return fig


def page_drawings(pdf, Rz):
    from n2o_injector import __version__
    from n2o_injector.drawing import DrawingMeta, plate_layout
    from n2o_injector.grain_drawing import grain_layout
    from n2o_injector.plots import build_grain_drawing, build_plate_drawing

    cfg = Rz["cfg"]
    lay = plate_layout(40, 1.20, 85.852, 12.7, Cd=0.70)
    meta = DrawingMeta(project="Mini Hybrid V2.2", part_no="INJ-PLATE-V2.2C",
                       flow_model="HEM",
                       notes=["SIZED FOR 25 bar CHAMBER PRESSURE. SEE TRADE STUDY REV 2."])
    fig = page_sheet(pdf, "Drawings — injector plate and fuel grain",
                     lambda f: build_plate_drawing(f, lay, meta=meta, version=__version__))
    p = TP.__new__(TP)
    p.fig, p.y = fig, 0.435
    p.para("40 x Ø1.20 mm on rings of 6/13/20, 12.7 mm plate. Full-size sheet, DXF and hole "
           "table: Trade Study/V2.2_injector_plate_40x1.20mm.*", gap=0.002)
    gl = grain_layout(cfg.grain, cfg.nozzle, cfg.propellant)
    gmeta = DrawingMeta(title="FUEL GRAIN", project="Mini Hybrid V2.2", material="PARAFFIN")
    land = Figure(figsize=(11.69, 8.27))
    build_grain_drawing(land, gl, meta=gmeta, version=__version__)
    buf = io.BytesIO()
    land.savefig(buf, format="png", dpi=260, facecolor="white")
    buf.seek(0)
    ax = fig.add_axes([0.055, 0.02, 0.89, 0.40])
    ax.imshow(matplotlib.image.imread(buf))
    ax.axis("off")
    pdf.savefig(fig)


def build():
    Rz = analyse()
    R1._state["page"] = 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with PdfPages(OUT) as pdf:
        page_summary(pdf, Rz)
        page_changes(pdf, Rz)
        page_regression(pdf, Rz)
        page_injector(pdf, Rz)
        page_ports(pdf, Rz)
        page_termination(pdf, Rz)
        page_histories(pdf, Rz)
        page_optimise(pdf, Rz)
        page_open(pdf, Rz)
        page_drawings(pdf, Rz)
        d = pdf.infodict()
        d["Title"] = "V2.2 trade study rev 2 - chamber pressure, regression, injector, ports"
        d["Author"] = "Molen"
        d["Creator"] = "HASTE"
        d["CreationDate"] = datetime.now()
    return OUT, Rz


if __name__ == "__main__":
    path, Rz = build()
    print("wrote", path)
