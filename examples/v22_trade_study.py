"""Injector trade study for the V2.2 hybrid: oxidiser flow vs O/F and Isp.

The question this answers: with the plate thickness fixed at 12.7 mm and the
oxidiser flow free to sit anywhere in 1.0-1.5 kg/s, what hole pattern gives the
best specific impulse, and what actually limits the choice?

Method. Hole *diameter* is held at 1.20 mm and the hole *count* is solved for.
That keeps L/D fixed at 10.6 across the entire sweep, which matters because L/D
is what decides whether HEM, Dyer or SPI applies -- varying the diameter would
silently change the flow model partway through the study and make the endpoints
incomparable. Chamber pressure is evaluated quasi-steady (timestep-independent)
and the sizing objective is ``mdot_ox`` (never inverts the regression law).

Run:  python examples/v22_trade_study.py
Writes the report PDF and prints every table it contains.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
from scipy.optimize import brentq  # noqa: E402

from n2o_injector import SaturationTable, FlowModel, SizingTarget, get_backend  # noqa: E402
from n2o_injector.config_io import load_config  # noqa: E402
from n2o_injector.drawing import plate_layout  # noqa: E402
from n2o_injector.sizing import size_injector  # noqa: E402

G0 = 9.80665
BAR = 1e5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(ROOT, "configs", "V2.2_recommended_mdot1.0_HEM.json")

#: Held fixed across the whole study.
HOLE_D_MM = 1.20
PLATE_T_MM = 12.7
PLATE_D_MM = 85.852

#: Oxidiser flows swept. The brief asks about 1.0-1.5; the range is extended
#: downwards because the O/F optimum turns out to lie below it, and an optimum
#: you cannot see the far side of is not an optimum.
MDOTS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

#: Oxidiser flux bounds for a paraffin regression correlation. These are the
#: rough bounds of the published fits, not hard physical limits: below the low
#: bound radiation and non-uniform burning start to dominate, above the high
#: bound the entrainment mechanism is being extrapolated. See USER_GUIDE §1.7.
GOX_LO, GOX_HI = 50.0, 700.0


def cfg_for(mdot, model=FlowModel.HEM, Cd=0.70, a_scale=1.0, hole_d_mm=HOLE_D_MM):
    cfg, _ = load_config(BASE_CONFIG)[:2]
    cfg.chamber_mode = "quasi-steady"
    cfg.dt = 1e-3
    cfg.injector.model = model
    cfg.injector.Cd = Cd
    cfg.injector.hole_d = hole_d_mm / 1e3
    cfg.injector.plate_thickness = PLATE_T_MM / 1e3
    cfg.propellant.reg_a *= a_scale
    target = SizingTarget(objective="mdot_ox", mdot_ox=mdot, min_dP_fraction=0.20)
    return cfg, target


def thrust_coefficient(k, Pc, eps, Pa):
    """``(Cf, Pe)`` for an ideal nozzle -- the same relation ``motor._thrust`` uses."""

    def area_ratio(M):
        return (
            ((k + 1) / 2) ** (-(k + 1) / (2 * (k - 1)))
            * (1 + (k - 1) / 2 * M**2) ** ((k + 1) / (2 * (k - 1)))
            / M
            - eps
        )

    M = brentq(area_ratio, 1.0000001, 20.0)
    Pe = Pc * (1 + 0.5 * (k - 1) * M**2) ** (-k / (k - 1))
    Cf = np.sqrt(
        ((2 * k**2) / (k - 1))
        * (2 / (k + 1)) ** ((k + 1) / (k - 1))
        * (1 - (Pe / Pc) ** ((k - 1) / k))
    ) + ((Pe - Pa) * eps) / Pc
    return Cf, Pe


def metrics(result, cfg):
    """Everything the study reports for one sized design."""
    b = result.burn
    liq = b.m_liq > 1e-3
    if not liq.any():
        liq = np.ones_like(b.t, dtype=bool)

    ox, fuel = b.ox_consumed, b.fuel_consumed
    prop_mass = ox + fuel
    Isp = b.total_impulse / (prop_mass * G0) if prop_mass > 0 else float("nan")

    # Liquid-phase-only figures: the vapour tail contributes little impulse but
    # drags the averages, and most motors are shut down or burnt out by then.
    t_l = b.t[liq]
    I_liq = float(np.trapezoid(b.thrust[liq], t_l)) if len(t_l) > 1 else 0.0
    ox_l = float(np.trapezoid(b.mdot_ox[liq], t_l)) if len(t_l) > 1 else 0.0
    fu_l = float(np.trapezoid(b.mdot_fuel[liq], t_l)) if len(t_l) > 1 else 0.0
    Isp_liq = I_liq / ((ox_l + fu_l) * G0) if (ox_l + fu_l) > 0 else float("nan")

    OF_inst = b.OF[np.isfinite(b.OF)]
    fuel_loaded = cfg.grain.fuel_mass(cfg.grain.port_id, cfg.propellant.rho_fuel)

    return {
        "n_holes": result.plate.n_holes,
        "hole_d": result.plate.hole_d * 1e3,
        "CdA": result.plate.CdA * 1e6,
        "L_over_D": result.plate.L_over_D,
        "Pc_peak": b.P_chamber.max() / BAR,
        "Pc_mean": b.P_chamber[liq].mean() / BAR,
        "F_mean": b.thrust[liq].mean(),
        "F_peak": b.thrust.max(),
        "burn": b.t[-1],
        "burn_liq": t_l[-1] if len(t_l) else 0.0,
        "impulse": b.total_impulse,
        "impulse_liq": I_liq,
        "OF_mean": result.achieved_mean_OF,
        "OF_min": float(np.nanmin(b.OF[liq])) if liq.any() else float("nan"),
        "OF_max": float(np.nanmax(b.OF[liq])) if liq.any() else float("nan"),
        "OF_over10": 100.0 * float(np.mean(OF_inst > 10.0)) if len(OF_inst) else 0.0,
        "Isp": Isp,
        "Isp_liq": Isp_liq,
        "Gox_max": b.G_ox.max(),
        "Gox_mean": float(b.G_ox[liq].mean()),
        "t_above_Gox": 100.0 * float(np.mean(b.G_ox[liq] > GOX_HI)),
        "dP_min": 100.0 * float(np.nanmin(b.dP_frac[liq])),
        "dP_mean": 100.0 * float(np.nanmean(b.dP_frac[liq])),
        "t_dP_ok": 100.0 * float(np.mean(b.dP_frac[liq] >= 0.20)),
        "choked": 100.0 * float(b.choked.mean()),
        "fuel_used": fuel,
        "fuel_frac": 100.0 * fuel / fuel_loaded if fuel_loaded > 0 else float("nan"),
        "port_end": b.port_d[-1] * 1e3,
        "web_left": (cfg.grain.outer_d * 1e3 - np.sqrt(5) * b.port_d[-1] * 1e3) / 2,
    }


def run_sweep(mdots=None, **kw):
    """Size at each oxidiser flow and collect the metrics."""
    props, table = get_backend("esdu"), None
    table = SaturationTable(props)
    out = []
    for md in mdots or MDOTS:
        cfg, target = cfg_for(md, **kw)
        res = size_injector(cfg, props, table, target, fix="hole_d")
        m = metrics(res, cfg)
        m["mdot"] = md
        lay = plate_layout(m["n_holes"], m["hole_d"], PLATE_D_MM, PLATE_T_MM, Cd=cfg.injector.Cd)
        m["min_web"] = lay.min_web
        m["rings"] = [r.count for r in lay.rings]
        m["warnings"] = res.warnings
        out.append(m)
    return out


def of_scan(prop, nozzle, Pc, Pa=101325.0, lo=3.0, hi=12.0, n=91):
    """Ideal delivered Isp against O/F at one chamber pressure."""
    rows = []
    for OF in np.linspace(lo, hi, n):
        cs = prop.cstar(OF, Pc)
        k = prop.gamma(OF, Pc)
        Cf, Pe = thrust_coefficient(k, Pc, nozzle.expansion_ratio, Pa)
        rows.append((OF, cs, k, Cf, Pe, cs * Cf / G0 * nozzle.efficiency))
    return np.array(rows)


def pc_scan(prop, nozzle, OF, Pa=101325.0, lo=8e5, hi=45e5, n=60):
    """Ideal delivered Isp against chamber pressure at one O/F."""
    rows = []
    for Pc in np.linspace(lo, hi, n):
        cs = prop.cstar(OF, Pc)
        k = prop.gamma(OF, Pc)
        Cf, Pe = thrust_coefficient(k, Pc, nozzle.expansion_ratio, Pa)
        rows.append((Pc, cs, k, Cf, Pe, cs * Cf / G0 * nozzle.efficiency))
    return np.array(rows)


def print_table(rows, cols, title):
    print()
    print(title)
    print("-" * len(title))
    head = "".join(f"{c[0]:>{c[2]}}" for c in cols)
    print(head)
    for r in rows:
        print("".join(f"{r[c[1]]:>{c[2]}{c[3]}}" if c[3] else f"{str(r[c[1]]):>{c[2]}}"
                      for c in cols))


MAIN_COLS = [
    ("mdot", "mdot", 6, ".2f"), ("N", "n_holes", 4, "d"), ("CdA", "CdA", 7, ".2f"),
    ("Pc pk", "Pc_peak", 7, ".1f"), ("Pc mn", "Pc_mean", 7, ".1f"),
    ("F mean", "F_mean", 8, ".0f"), ("burn", "burn_liq", 7, ".2f"),
    ("Itot", "impulse", 8, ".0f"), ("O/F", "OF_mean", 6, ".2f"),
    ("Isp", "Isp", 7, ".1f"), ("Gox", "Gox_max", 6, ".0f"),
    ("Gox>hi%", "t_above_Gox", 8, ".0f"), ("fuel%", "fuel_frac", 7, ".1f"),
    ("dPok%", "t_dP_ok", 7, ".0f"), ("OF>10%", "OF_over10", 7, ".0f"),
    ("web", "min_web", 6, ".2f"),
]


def main():
    props = get_backend("esdu")
    table = SaturationTable(props)
    cfg0, _ = load_config(BASE_CONFIG)[:2]
    prop, nozzle = cfg0.propellant, cfg0.nozzle

    print("=" * 110)
    print("V2.2 INJECTOR TRADE STUDY -- oxidiser flow vs O/F and specific impulse")
    print("=" * 110)
    print(f"Fixed: {PLATE_T_MM} mm plate, {HOLE_D_MM} mm holes (L/D "
          f"{PLATE_T_MM / HOLE_D_MM:.2f}), HEM, Cd 0.70, quasi-steady, dt 1 ms")
    print(f"Grain: {cfg0.grain.n_ports} ports x {cfg0.grain.port_id * 1e3:.2f} mm ID, "
          f"OD {cfg0.grain.outer_d * 1e3:.2f} mm, L {cfg0.grain.length * 1e3:.0f} mm")
    print(f"Tank:  {cfg0.tank.volume * 1e3:.1f} L, {cfg0.tank.ox_mass:.3f} kg N2O at "
          f"{cfg0.tank.fill_temp - 273.15:.1f} C")
    print(f"Nozzle: throat {cfg0.nozzle.throat_d * 1e3:.2f} mm, eps "
          f"{cfg0.nozzle.expansion_ratio:.3f}, Cd {cfg0.nozzle.Cd}, eta {cfg0.nozzle.efficiency}")

    base = run_sweep()
    print_table(base, MAIN_COLS, "1. BASELINE SWEEP (HEM, Cd 0.70)")

    ofs = of_scan(prop, nozzle, 25 * BAR)
    i = int(np.argmax(ofs[:, 5]))
    print(f"\n2. IDEAL Isp vs O/F at Pc = 25 bar: peak {ofs[i, 5]:.1f} s at "
          f"O/F {ofs[i, 0]:.2f}  (c* peaks at O/F "
          f"{ofs[int(np.argmax(ofs[:, 1])), 0]:.2f})")
    print(f"   declared opt_OF in the propellant file: {prop.opt_OF:.2f}")
    print(f"   CEA table O/F range: {prop.OF_grid.min():.1f} to {prop.OF_grid.max():.1f} "
          "-- c* is clamped outside it")

    pcs = pc_scan(prop, nozzle, 7.0)
    ideal = pcs[np.argmin(np.abs(pcs[:, 4] - 101325.0))]
    print(f"\n3. NOZZLE MATCHING: eps {nozzle.expansion_ratio:.3f} is perfectly "
          f"expanded at Pc = {ideal[0] / BAR:.1f} bar (Pe = Pa)")
    print(f"   Cf at 20 bar {pc_scan(prop, nozzle, 7.0, lo=20 * BAR, hi=20 * BAR, n=1)[0, 3]:.3f}"
          f"  -> at 37 bar {pc_scan(prop, nozzle, 7.0, lo=37 * BAR, hi=37 * BAR, n=1)[0, 3]:.3f}")

    flows = [1.0, 1.25, 1.5]
    hdr = f"{'':<12}" + "".join(f"{f'{f:.2f} kg/s':>22}" for f in flows)

    print("\n4. MODEL SENSITIVITY (does the ranking survive the flow model?)")
    sens_model = [hdr, "-" * len(hdr)]
    for m in (FlowModel.SPI, FlowModel.DYER, FlowModel.HEM):
        rows = run_sweep(flows, model=m)
        sens_model.append(f"{m.value:<12}" + "".join(
            f"{r['n_holes']:>13d} holes" + f"{r['Isp']:>7.1f}s" for r in rows))
        print("   " + sens_model[-1])

    print("\n5. Cd SENSITIVITY")
    sens_cd = [hdr, "-" * len(hdr)]
    for cd in (0.60, 0.70, 0.80):
        rows = run_sweep(flows, Cd=cd)
        sens_cd.append(f"{'Cd ' + f'{cd:.2f}':<12}" + "".join(
            f"{r['n_holes']:>13d} holes" + f"{r['Isp']:>7.1f}s" for r in rows))
        print("   " + sens_cd[-1])

    print("\n6. REGRESSION SENSITIVITY (a scaled; O/F and Isp for each flow)")
    sens_a = [hdr, "-" * len(hdr)]
    for sc in (0.7, 1.0, 1.3, 1.6, 2.3):
        rows = run_sweep(flows, a_scale=sc)
        sens_a.append(f"{'a x ' + f'{sc:.2f}':<12}" + "".join(
            f"{'O/F ':>9}{r['OF_mean']:>6.2f}" + f"{r['Isp']:>7.1f}s" for r in rows))
        print("   " + sens_a[-1])

    print("\n7. WHAT PORT SIZE WOULD KEEP Gox IN RANGE?")
    port_fix = []
    for md in (1.0, 1.25, 1.5):
        need_A = md / GOX_HI
        d = np.sqrt(4 * need_A / (np.pi * cfg0.grain.n_ports)) * 1e3
        port_fix.append((md, need_A * 1e6, d))
        print(f"   {md:.2f} kg/s needs {need_A * 1e6:.0f} mm2 of port to stay under "
              f"{GOX_HI:.0f} kg/m2/s -> {cfg0.grain.n_ports} ports of {d:.1f} mm "
              f"(have {cfg0.grain.port_id * 1e3:.2f} mm)")

    print("\n8. WHEN DOES THE PRESSURE-DROP MARGIN EXPIRE?")
    margin = []
    for md in (0.8, 1.0, 1.2, 1.5):
        cfg, target = cfg_for(md)
        res = size_injector(cfg, props, table, target, fix="hole_d")
        b = res.burn
        liq = b.m_liq > 1e-3
        t, f = b.t[liq], b.dP_frac[liq]
        below = t[f < 0.20]
        t20 = below[0] if len(below) else float("nan")
        margin.append((md, t20, t[-1], 100 * t20 / t[-1] if t[-1] else float("nan")))
        print(f"   {md:.2f} kg/s: drops under 20% at t = {t20:5.2f} s of a "
              f"{t[-1]:5.2f} s liquid phase ({100 * t20 / t[-1]:4.0f}% of the burn is clean)")

    print("\n9. DOES A WARMER FILL OR A SUPERCHARGE RESCUE THE HIGH-FLOW CASES?")
    feed = []
    sens_feed = [f"{'feed at 1.50 kg/s':<22}{'holes':>7}{'Isp':>8}{'Pc peak':>10}"
                 f"{'clean margin':>14}{'peak Gox':>10}",
                 "-" * 71]
    for label, T_C, sup_bar in (("saturated 23.9 C", 23.85, None),
                                ("saturated 30 C", 30.0, None),
                                ("supercharged 65 bar", 23.85, 65.0),
                                ("supercharged 75 bar", 23.85, 75.0)):
        cfg, target = cfg_for(1.5)
        cfg.tank.fill_temp = T_C + 273.15
        cfg.tank.supercharge_P = None if sup_bar is None else sup_bar * BAR
        try:
            res = size_injector(cfg, props, table, target, fix="hole_d")
            m = metrics(res, cfg)
            feed.append((label, m))
            sens_feed.append(f"{label:<22}{m['n_holes']:>7d}{m['Isp']:>7.1f}s"
                             f"{m['Pc_peak']:>8.1f} bar{m['t_dP_ok']:>13.0f}%"
                             f"{m['Gox_max']:>10.0f}")
            print("   " + sens_feed[-1])
        except Exception as exc:
            sens_feed.append(f"{label:<22}   not reachable")
            print(f"   {label:22s}: not reachable -- {exc}")

    print("\n10. HOLE DIAMETER AT FIXED 12.7 mm PLATE (L/D = 12.7/d)")
    dia = []
    for hd in (1.00, 1.20, 1.40, 1.60, 2.00):
        cfg, target = cfg_for(1.0, hole_d_mm=hd)
        res = size_injector(cfg, props, table, target, fix="hole_d")
        lay = plate_layout(res.plate.n_holes, hd, PLATE_D_MM, PLATE_T_MM, Cd=0.70)
        rec = res.model_recommendation or "-"
        dia.append((hd, res.plate.n_holes, PLATE_T_MM / hd, lay.min_web, rec))
        print(f"   {hd:.2f} mm: {res.plate.n_holes:3d} holes, L/D {PLATE_T_MM / hd:5.2f}, "
              f"web {lay.min_web:5.2f} mm, model recommendation: {rec}")

    print("\n11. Isp DECOMPOSITION: how much of the gain is O/F, how much is Pc?")
    # Hold one variable at its 1.0 kg/s value and move the other, so the two
    # effects can be read separately instead of only as their sum.
    ref = next(r for r in base if abs(r["mdot"] - 1.0) < 1e-9)
    OF_ref, Pc_ref = ref["OF_mean"], ref["Pc_mean"] * BAR

    def ideal_isp(OF, Pc):
        cs = prop.cstar(OF, Pc)
        k = prop.gamma(OF, Pc)
        Cf, _ = thrust_coefficient(k, Pc, nozzle.expansion_ratio, Pa=101325.0)
        return cs * Cf / G0 * nozzle.efficiency

    base_isp = ideal_isp(OF_ref, Pc_ref)
    decomp = []
    for r in base:
        Pc = r["Pc_mean"] * BAR
        d_pc = ideal_isp(OF_ref, Pc) - base_isp          # pressure alone
        d_of = ideal_isp(r["OF_mean"], Pc_ref) - base_isp  # mixture alone
        both = ideal_isp(r["OF_mean"], Pc) - base_isp
        decomp.append((r["mdot"], d_pc, d_of, both))
        print(f"   {r['mdot']:.2f} kg/s: from Pc {d_pc:+6.2f} s, from O/F {d_of:+6.2f} s, "
              f"combined {both:+6.2f} s")

    print("\n12. TIMESTEP CONVERGENCE OF THE RECOMMENDATION")
    conv = []
    for dt_ms in (4.0, 2.0, 1.0, 0.5, 0.25):
        cfg, target = cfg_for(1.0)
        cfg.dt = dt_ms / 1e3
        res = size_injector(cfg, props, table, target, fix="hole_d")
        m = metrics(res, cfg)
        conv.append((dt_ms, m["n_holes"], m["impulse"], m["Isp"]))
        print(f"   dt {dt_ms:5.2f} ms: {m['n_holes']} holes, impulse {m['impulse']:.0f} Ns, "
              f"Isp {m['Isp']:.2f} s")

    return dict(base=base, ofs=ofs, pcs=pcs, cfg=cfg0, margin=margin,
                port_fix=port_fix, feed=feed, dia=dia, conv=conv,
                sens_model=sens_model, sens_cd=sens_cd, sens_a=sens_a,
                sens_feed=sens_feed, decomp=decomp)


if __name__ == "__main__":
    main()
