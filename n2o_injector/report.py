"""Design report generation and HRAP export.

The HRAP motor-configuration writer targets the MATLAB build's ``.mat``
format.  Field names, the ``cfg`` struct layout, and the permitted unit
strings were all taken from HRAP itself -- the shipped
``motor_configs/*.mat`` files and the unit drop-down definitions inside
``HRAP.mlapp`` -- rather than assumed, so the emitted file uses only values
HRAP is known to accept.

The injector fields HRAP consumes are exactly the three this tool sizes:
``inj_D`` (orifice diameter), ``inj_N`` (orifice count) and ``inj_Cd``
(discharge coefficient).
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np

from .injector import FlowModel
from .motor import MotorConfig
from .sizing import SizingResult

BAR = 1e5


def _fmt(x: float, digits: int = 3, unit: str = "") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}{unit}"


def build_report(result: SizingResult, cfg: MotorConfig, props_name: str) -> str:
    """Render the full on-screen / exportable design report."""
    b = result.burn
    p = result.plate
    L = []
    add = L.append

    add("=" * 78)
    add("  N2O / PARAFFIN HYBRID -- INJECTOR SIZING REPORT")
    add(f"  generated {datetime.now():%Y-%m-%d %H:%M:%S}")
    add("=" * 78)
    add("")

    # ---- headline result ----
    add("RECOMMENDED INJECTOR")
    add("-" * 78)
    add(f"  Orifice count            : {p.n_holes}")
    add(f"  Orifice diameter         : {p.hole_d * 1e3:.3f} mm")
    add(f"  Total geometric area     : {p.total_area * 1e6:.2f} mm^2")
    add(f"  Effective area (Cd*A)    : {p.CdA * 1e6:.2f} mm^2")
    add(f"  Discharge coefficient    : {p.Cd:.3f}")
    add(f"  Plate thickness          : {p.plate_thickness * 1e3:.2f} mm")
    add(f"  Orifice L/D              : {p.L_over_D:.2f}")
    add("")
    add(f"  Ideal Cd*A from solver   : {result.CdA_required * 1e6:.3f} mm^2")
    add(f"  Realised Cd*A            : {result.CdA_achieved * 1e6:.3f} mm^2")
    area_err = (result.CdA_achieved - result.CdA_required) / result.CdA_required * 100
    add(f"  Area rounding error      : {area_err:+.2f} %")
    add("")

    # ---- model / assumptions ----
    add("MODEL AND ASSUMPTIONS")
    add("-" * 78)
    add(f"  Injector flow model      : {cfg.injector.model.value}")
    add(f"  N2O property source      : {props_name}")
    add(f"  Chamber pressure model   : {cfg.chamber_mode}")
    add(f"  Propellant               : {cfg.propellant.name}")
    add(
        f"  Regression law           : rdot[mm/s] = {cfg.propellant.reg_a:g}"
        f" * G_ox^{cfg.propellant.reg_n:g}"
        + (f" * L^{cfg.propellant.reg_m:g}" if cfg.propellant.reg_m else "")
    )
    add(f"  Fuel density             : {cfg.propellant.rho_fuel:.0f} kg/m^3")
    if cfg.propellant.has_table:
        add("  c* source                : CEA/RPA table from HRAP propellant config")
    else:
        add(
            f"  c* source                : constant {cfg.propellant.cstar_const:.0f} m/s"
            f" (eff {cfg.propellant.cstar_eff:.2f})"
        )
    add(f"  Timestep                 : {cfg.dt * 1e3:.1f} ms")
    if result.model_recommendation:
        add("")
        add(f"  Regime note: {result.model_recommendation}")
    add("")

    # ---- design point ----
    add("DESIGN POINT (t = 0)")
    add("-" * 78)
    if len(b.t):
        add(f"  Tank temperature         : {b.T_tank[0] - 273.15:.2f} C")
        add(f"  Tank pressure            : {b.P_tank[0] / BAR:.2f} bar")
        add(f"  Chamber pressure         : {b.P_chamber[0] / BAR:.2f} bar")
        add(f"  Injector dP              : {b.dP_inj[0] / BAR:.2f} bar "
            f"({b.dP_frac[0] * 100:.1f} % of Pc)")
        add(f"  Oxidiser mass flow       : {b.mdot_ox[0]:.4f} kg/s")
        add(f"  Fuel mass flow           : {b.mdot_fuel[0]:.4f} kg/s")
        add(f"  O/F                      : {b.OF[0]:.3f}")
        add(f"  Port mass flux G_ox      : {b.G_ox[0]:.1f} kg/m^2/s")
        add(f"  Thrust                   : {b.thrust[0]:.0f} N")
    add("")

    # ---- burn summary ----
    add("BURN SUMMARY")
    add("-" * 78)
    tgt_desc = (
        "profile" if result.target.has_profile else f"{result.target.OF:.3f}"
    )
    add(f"  Objective                : {result.target.objective}")
    if result.target.objective == "mdot_ox":
        add(f"  Target oxidiser flow     : {result.target.mdot_ox:.4f} kg/s")
    elif result.target.objective == "chamber_pressure":
        add(f"  Target chamber pressure  : {result.target.chamber_P / BAR:.2f} bar")
    add(f"  Target O/F               : {tgt_desc}")
    if math.isfinite(result.conditioning):
        add(
            f"  Regression conditioning  : 1/(1-n) = {result.conditioning:.2f}  "
            f"(a 10% error in 'a' -> ~{(1.10 ** result.conditioning - 1) * 100:.0f}% "
            "in required oxidiser flow)"
        )
    add(f"  Achieved mean O/F        : {_fmt(result.achieved_mean_OF)}")
    add(f"  O/F error vs target      : {_fmt(result.OF_error_pct, 2, ' %')}")
    if len(b.t):
        ok = np.isfinite(b.OF)
        if np.any(ok):
            add(f"  O/F range over burn      : {b.OF[ok].min():.2f} -> {b.OF[ok].max():.2f}")
    add(f"  Burn time                : {b.burn_time:.2f} s")
    if b.liquid_exhausted_t is not None:
        add(f"  Liquid exhausted at      : {b.liquid_exhausted_t:.2f} s")
    add(f"  Oxidiser consumed        : {b.ox_consumed:.3f} kg")
    add(f"  Fuel consumed            : {b.fuel_consumed:.3f} kg")
    add(f"  Total impulse            : {b.total_impulse:.0f} N.s")
    add(f"  Isp (delivered)          : {b.specific_impulse():.1f} s")
    if len(b.t):
        add(f"  Peak thrust              : {b.thrust.max():.0f} N")
        add(f"  Port diameter            : {b.port_d[0] * 1e3:.1f} -> {b.port_d[-1] * 1e3:.1f} mm")
        add(f"  Tank pressure            : {b.P_tank[0] / BAR:.1f} -> {b.P_tank[-1] / BAR:.1f} bar")
        add(f"  Chamber pressure         : {b.P_chamber[0] / BAR:.1f} -> {b.P_chamber[-1] / BAR:.1f} bar")
    add("")

    # ---- margins ----
    add("MARGINS AND FLAGS")
    add("-" * 78)
    add(f"  Required dP margin       : {result.target.min_dP_fraction * 100:.0f} % of Pc")
    add(f"  Minimum dP over burn     : {_fmt(result.min_dP_fraction * 100, 1, ' %')}")
    add(f"  Mean dP over burn        : {_fmt(result.mean_dP_fraction * 100, 1, ' %')}")
    add(f"  dP margin satisfied      : {'YES' if result.dP_margin_ok else 'NO'}")
    add(f"  Choked fraction of burn  : {_fmt(result.choked_fraction * 100, 0, ' %')}")
    add(f"  Solver converged         : {'YES' if result.converged else 'NO'}")
    add("")

    if result.warnings:
        add("WARNINGS")
        add("-" * 78)
        for w in result.warnings:
            add(f"  [!] {w}")
        add("")

    if result.notes:
        add("NOTES")
        add("-" * 78)
        for n in result.notes:
            add(f"  -  {n}")
        add("")

    add("HRAP CROSS-CHECK")
    add("-" * 78)
    add(f"  HRAP inj_N               : {p.n_holes}")
    add(f"  HRAP inj_D               : {p.hole_d * 1e3:.4f} mm")
    add(f"  HRAP inj_Cd              : {p.Cd:.4f}")
    add(f"  HRAP inj_CdA (per hole)  : {p.CdA_per_hole:.6e} m^2")
    add("")
    add("  HRAP models the liquid injector as single-phase incompressible")
    add("  (mdot = Cd*N*A*sqrt(2*rho_l*dP)). If this design was sized with HEM or")
    add("  Dyer, HRAP will predict a HIGHER oxidiser flow than this tool for the")
    add("  same geometry. Re-run here with model = SPI to reproduce HRAP exactly.")
    if cfg.injector.model is not FlowModel.SPI:
        add("")
        add("  -> This design was sized with "
            f"{cfg.injector.model.value}, so expect that discrepancy.")
    add("")
    add("=" * 78)
    return "\n".join(L)


# --------------------------------------------------------------------------
# HRAP export
# --------------------------------------------------------------------------


def export_hrap_mat(
    path: str,
    result: SizingResult,
    cfg: MotorConfig,
    motor_name: str = "sized_motor",
    propellant_file: str = "",
) -> str:
    """Write an HRAP (MATLAB) motor configuration ``.mat``.

    Units are emitted using only strings present in HRAP's own drop-downs.
    HRAP's efficiencies are percentages, and its regression coefficients are
    the ``[a, n, m]`` set with ``a`` in mm/s -- both are converted here.
    """
    from scipy.io import savemat

    p = result.plate
    tank, grain, noz, prop = cfg.tank, cfg.grain, cfg.nozzle, cfg.propellant

    cfg_struct = {
        "mtr_nm": motor_name,
        # -- tank --
        "tnk_V": tank.volume * 1e3,          # m^3 -> L
        "tnk_V_unit": "L",
        "tnk_V_state": 0,
        "tnk_L": 0.0,
        "tnk_L_unit": "mm",
        "tnk_D": 0.0,
        "tnk_D_unit": "mm",
        "tnk_dd": "Starting Tank Temperature",
        "tnk_cond": tank.fill_temp,
        "T_tnk_unit": "K",
        "fill_dd": "Starting Oxidizer Mass",
        "fill": tank.ox_mass,
        "fill_unit": "kg",
        # -- chamber --
        "cmbr_V_state": 1,
        "cmbr_V": cfg.chamber_volume * 1e3,
        "cmbr_V_unit": "L",
        "P_cmbr": 1.0,
        "P_cmbr_unit": "atm",
        "Pa": cfg.ambient_P,
        "Pa_unit": "Pa",
        # -- nozzle --
        "noz_thrt": noz.throat_d * 1e3,
        "noz_thrt_unit": "mm",
        "noz_def": "Nozzle Expansion Ratio",
        "noz_ex": noz.expansion_ratio,
        "noz_ex_unit": "mm",
        "noz_eff": noz.efficiency * 100.0,   # HRAP stores efficiencies as %
        "noz_Cd": noz.Cd,
        # -- grain / propellant --
        "grn_ID": grain.port_id * 1e3,
        "grn_ID_unit": "mm",
        "grn_OD": grain.outer_d * 1e3,
        "grn_OD_unit": "mm",
        "grn_L": grain.length * 1e3,
        "grn_L_unit": "mm",
        "prop_file": propellant_file,
        "prop_nm": prop.name,
        "prop_rho": prop.rho_fuel,
        "prop_rho_unit": "kg/m^3",
        "prop_a": prop.reg_a,
        "prop_n": prop.reg_n,
        "prop_m": prop.reg_m,
        "const_OF": prop.opt_OF,
        "cstar_eff": prop.cstar_eff * 100.0,
        "reg_model": "Shifting OF",
        # -- injector (the sized result) --
        "inj_D": p.hole_d * 1e3,
        "inj_D_unit": "mm",
        "inj_N": p.n_holes,
        "inj_Cd": p.Cd,
        # -- vent --
        "vnt_state": "None",
        "vnt_D": 0.0,
        "vnt_D_unit": "mm",
        "vnt_Cd": 0.75,
        # -- mass properties (not modelled here) --
        "mp_state": 0,
        "tnk_X": 0.0, "tnk_X_unit": "mm",
        "cmbr_X": 0.0, "cmbr_X_unit": "mm",
        "mtr_cg": 0.0, "mtr_cg_unit": "mm",
        "mtr_m": 0.0, "mtr_m_unit": "kg",
        # -- run control --
        "t_max": cfg.max_time,
        "t_burn": 0.0,
        "dt": cfg.dt,
    }

    savemat(path, {"cfg": cfg_struct}, oned_as="row")
    return path


def export_injector_json(path: str, result: SizingResult, cfg: MotorConfig) -> str:
    """Write a compact machine-readable injector definition."""
    p = result.plate
    b = result.burn
    data = {
        "injector": {
            "n_holes": p.n_holes,
            "hole_diameter_m": p.hole_d,
            "hole_diameter_mm": p.hole_d * 1e3,
            "total_area_m2": p.total_area,
            "CdA_total_m2": p.CdA,
            "CdA_per_hole_m2": p.CdA_per_hole,
            "Cd": p.Cd,
            "plate_thickness_m": p.plate_thickness,
            "L_over_D": p.L_over_D,
        },
        "hrap_fields": {
            "inj_N": p.n_holes,
            "inj_D_mm": p.hole_d * 1e3,
            "inj_Cd": p.Cd,
            "inj_CdA_m2": p.CdA_per_hole,
        },
        "model": {
            "flow_model": cfg.injector.model.value,
            "chamber_mode": cfg.chamber_mode,
            "hrap_liquid_model": "SPI (incompressible) -- HRAP's only liquid model",
        },
        "target": {
            "objective": result.target.objective,
            "OF": result.target.OF,
            "min_dP_fraction": result.target.min_dP_fraction,
        },
        "performance": {
            "mean_OF": result.achieved_mean_OF,
            "OF_error_pct": result.OF_error_pct,
            "burn_time_s": b.burn_time,
            "total_impulse_Ns": b.total_impulse,
            "isp_s": b.specific_impulse(),
            "ox_consumed_kg": b.ox_consumed,
            "fuel_consumed_kg": b.fuel_consumed,
            "min_dP_fraction": result.min_dP_fraction,
            "dP_margin_ok": result.dP_margin_ok,
            "choked_fraction": result.choked_fraction,
        },
        "warnings": result.warnings,
        "notes": result.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(data), f, indent=2)
    return path


def export_timeseries_csv(path: str, result: SizingResult) -> str:
    """Write the burn time histories as CSV."""
    b = result.burn
    cols = [
        ("time_s", b.t),
        ("P_tank_Pa", b.P_tank),
        ("P_chamber_Pa", b.P_chamber),
        ("T_tank_K", b.T_tank),
        ("mdot_ox_kg_s", b.mdot_ox),
        ("mdot_fuel_kg_s", b.mdot_fuel),
        ("OF", b.OF),
        ("port_d_m", b.port_d),
        ("m_ox_kg", b.m_ox),
        ("m_liq_kg", b.m_liq),
        ("m_fuel_kg", b.m_fuel),
        ("thrust_N", b.thrust),
        ("dP_inj_Pa", b.dP_inj),
        ("dP_fraction", b.dP_frac),
        ("G_ox_kg_m2_s", b.G_ox),
        ("choked", b.choked.astype(int)),
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(name for name, _ in cols) + "\n")
        for i in range(len(b.t)):
            f.write(",".join(f"{arr[i]:.8g}" for _, arr in cols) + "\n")
    return path


def _clean(obj):
    """Make numpy scalars and non-finite floats JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj
