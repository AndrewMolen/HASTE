"""Import HRAP motor configurations and simulation output, for cross-checking.

Everything here was derived from HRAP itself rather than inferred: the unit
factors, the state-flag semantics and the CSV column set were read out of
``HRAP.mlapp`` (its ``run_sim`` initialiser and ``save_csv`` handler), and the
field layout from the shipped ``motor_configs/*.mat``.

Two importers are provided:

:func:`load_hrap_motor`
    Reads an HRAP ``.mat`` motor configuration into a :class:`MotorConfig`,
    resolving every unit and mode flag exactly as HRAP's own initialiser does.

:func:`load_hrap_output`
    Reads an HRAP simulation output CSV so its time histories can be overlaid
    against this tool's.

.. note::
   **HRAP's CSV column "Injector Pressure Drop (kPa)" is not the injector
   pressure drop.** In ``tank.m`` the stored ``x.dP`` is ``Pv(T_new) -
   P_tank_old`` -- the change in *tank* pressure across one timestep, which the
   blowdown model reuses as a running average. It is negative and of order
   0.01 kPa, not the tens of bar an injector drop would be. :func:`load_hrap_output`
   therefore recomputes the true injector drop as ``P_tank - P_chamber`` from
   the two pressure columns and reports the original column separately.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .injector import FlowModel
from .motor import Grain, InjectorSpec, MotorConfig, Nozzle, Tank
from .propellant import Propellant
from .properties import ESDUProperties

# --------------------------------------------------------------------------
# Unit tables (verbatim from HRAP.mlapp's run_sim initialiser)
# --------------------------------------------------------------------------

_PSI = 101325.0 / 14.696  # HRAP's own psi definition, not the 6894.76 SI value

LENGTH = {"in": 0.0254, "ft": 0.3048, "cm": 0.01, "mm": 0.001, "m": 1.0}
VOLUME = {
    "in^3": 0.0254**3, "ft^3": 0.3048**3, "cm^3": 0.01**3,
    "L": 0.001, "Gal": 0.00378541, "m^3": 1.0,
}
PRESSURE = {
    "psi": _PSI, "psf": _PSI * 144.0, "atm": 101325.0,
    "MPa": 1e6, "kPa": 1e3, "Bar": 1e5, "Pa": 1.0,
}
MASS = {"lbm": 0.453592, "oz": 0.0283495, "g": 0.001, "kg": 1.0}
DENSITY = {
    "lb/in^3": 1.0 / (2.205 * 0.0254**3),
    "lb/ft^3": 1.0 / (2.205 * 0.3048**3),
    "g/cm^3": 1000.0,
    "kg/m^3": 1.0,
}


def _to_kelvin(value: float, unit: str) -> float:
    """HRAP's temperature handling, including its Rankine convention."""
    u = (unit or "K").strip()
    if u == "C":
        return value + 273.15
    if u == "R":
        return value / 1.8
    if u == "F":
        return (value - 32.0) / 1.8 + 273.15
    return value


@dataclass
class ImportedMotor:
    """Result of importing an HRAP motor configuration."""

    config: MotorConfig
    motor_name: str = ""
    propellant_file: str = ""
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Motor configuration import
# --------------------------------------------------------------------------


def _struct_reader(path: str):
    """Return a ``(get, names)`` pair for HRAP's ``cfg`` struct or flat layout."""
    from scipy.io import loadmat

    raw = loadmat(path)
    if "cfg" in raw:
        s = raw["cfg"][0, 0]
        return (lambda k: s[k]), set(s.dtype.names or ())
    if "s" in raw and raw["s"].dtype.names:
        s = raw["s"][0, 0]
        return (lambda k: s[k]), set(s.dtype.names or ())
    return (lambda k: raw[k]), {k for k in raw if not k.startswith("__")}


def load_hrap_motor(
    path: str,
    propellant: Propellant | None = None,
    match_hrap_models: bool = True,
) -> ImportedMotor:
    """Load an HRAP ``.mat`` motor configuration.

    ``propellant`` optionally supplies a already-loaded propellant (with its CEA
    table); otherwise the config's referenced ``prop_file`` is loaded if it can
    be found, falling back to the regression coefficients stored in the config
    itself with a constant ``c*``.

    ``match_hrap_models`` sets this tool to reproduce HRAP as closely as it can
    -- SPI injector flow and the transient chamber ODE -- which is what you want
    when cross-referencing. Set it False to keep your own model choices.
    """
    get, names = _struct_reader(path)
    warnings: list[str] = []
    info: dict = {}

    def num(key, default=None):
        if key not in names:
            return default
        arr = np.asarray(get(key)).ravel()
        if arr.size == 0:
            return default
        try:
            return float(arr[0])
        except (TypeError, ValueError):
            return default

    def text(key, default=""):
        if key not in names:
            return default
        arr = np.asarray(get(key)).ravel()
        if arr.size == 0:
            return default
        return str(arr[0]).strip()

    def unit(key, table, default_key, what):
        """Look up a unit string, warning (not failing) on anything unexpected."""
        u = text(key, default_key)
        if u in table:
            return table[u]
        warnings.append(
            f"unrecognised {what} unit {u!r} in {os.path.basename(path)}; "
            f"assuming {default_key}"
        )
        return table[default_key]

    required = {"grn_ID", "grn_OD", "grn_L", "noz_thrt", "inj_D", "inj_N"}
    missing = required - names
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is not an HRAP motor configuration "
            f"(missing {', '.join(sorted(missing))}). If this is a *propellant* "
            "config, load it with the propellant loader instead."
        )

    motor_name = text("mtr_nm", os.path.splitext(os.path.basename(path))[0])

    # ---- grain ---------------------------------------------------------
    grn_ID = num("grn_ID", 0.0) * unit("grn_ID_unit", LENGTH, "in", "grain ID")
    grn_OD = num("grn_OD", 0.0) * unit("grn_OD_unit", LENGTH, "in", "grain OD")
    grn_L = num("grn_L", 0.0) * unit("grn_L_unit", LENGTH, "in", "grain length")
    grain = Grain(length=grn_L, port_id=grn_ID, outer_d=grn_OD, n_ports=1)

    # ---- tank ----------------------------------------------------------
    # HRAP: tnk_V_state == 1 means "volume from length x diameter".
    if num("tnk_V_state", 0.0) == 1:
        tnk_L = num("tnk_L", 0.0) * unit("tnk_L_unit", LENGTH, "in", "tank length")
        tnk_D = num("tnk_D", 0.0) * unit("tnk_D_unit", LENGTH, "in", "tank diameter")
        tnk_V = tnk_L * 0.25 * math.pi * tnk_D**2
        info["tank_volume_from"] = "length x diameter"
    else:
        tnk_V = num("tnk_V", 0.0) * unit("tnk_V_unit", VOLUME, "cm^3", "tank volume")
        info["tank_volume_from"] = "explicit volume"

    # Tank initial condition: temperature directly, or inverted from pressure.
    tnk_dd = text("tnk_dd", "Starting Tank Temperature")
    tnk_cond = num("tnk_cond", 293.15)
    t_unit = text("T_tnk_unit", "K")
    if tnk_dd == "Starting Tank Pressure":
        P = tnk_cond * PRESSURE.get(t_unit, 1.0)
        try:
            fill_temp = ESDUProperties().T_sat(P)
        except ValueError as exc:
            raise ValueError(
                f"tank pressure {P / 1e5:.2f} bar in {os.path.basename(path)} "
                f"cannot be inverted to a saturation temperature: {exc}"
            ) from exc
        info["tank_condition"] = f"{tnk_cond:g} {t_unit} -> {fill_temp - 273.15:.2f} C"
    else:
        fill_temp = _to_kelvin(tnk_cond, t_unit)
        info["tank_condition"] = f"{tnk_cond:g} {t_unit}"

    # Oxidiser load: explicit mass, or a fill percentage HRAP resolves against
    # the saturated densities at the fill temperature.
    fill_dd = text("fill_dd", "Starting Oxidizer Mass")
    fill = num("fill", 0.0)
    if fill_dd == "Tank Fill Percentage":
        sat = ESDUProperties().sat(fill_temp)
        f = fill / 100.0
        ox_mass = f * tnk_V * sat.rho_l + (1.0 - f) * tnk_V * sat.rho_v
        info["oxidiser_load"] = f"{fill:g} % fill -> {ox_mass:.3f} kg"
    else:
        ox_mass = fill * unit("fill_unit", MASS, "kg", "oxidiser mass")
        info["oxidiser_load"] = f"{ox_mass:.3f} kg"

    tank = Tank(volume=tnk_V, fill_temp=fill_temp, ox_mass=ox_mass)

    # ---- nozzle --------------------------------------------------------
    noz_thrt = num("noz_thrt", 0.0) * unit("noz_thrt_unit", LENGTH, "in", "throat")
    noz_def = text("noz_def", "Nozzle Expansion Ratio")
    noz_ex = num("noz_ex", 1.0)
    if noz_def == "Nozzle Exit Diameter":
        d_exit = noz_ex * unit("noz_ex_unit", LENGTH, "in", "nozzle exit")
        expansion = (d_exit**2) / (noz_thrt**2) if noz_thrt > 0 else 1.0
        info["nozzle"] = f"exit dia {d_exit * 1e3:.2f} mm -> ER {expansion:.3f}"
    else:
        expansion = noz_ex
        info["nozzle"] = f"ER {expansion:g}"
    nozzle = Nozzle(
        throat_d=noz_thrt,
        expansion_ratio=expansion,
        Cd=num("noz_Cd", 0.95),
        efficiency=num("noz_eff", 95.0) / 100.0,  # HRAP stores percent
    )

    # ---- injector ------------------------------------------------------
    inj_D = num("inj_D", 0.0) * unit("inj_D_unit", LENGTH, "in", "injector dia")
    inj_N = int(round(num("inj_N", 1.0)))
    inj_Cd = num("inj_Cd", 0.7)
    injector = InjectorSpec(
        n_holes=max(inj_N, 1),
        hole_d=inj_D,
        Cd=inj_Cd,
        # HRAP has no orifice-length concept; assume a thin plate so L/D stays
        # in the Dyer-appropriate regime if the user switches models later.
        plate_thickness=max(inj_D, 1e-4),
        model=FlowModel.SPI if match_hrap_models else FlowModel.DYER,
    )
    if match_hrap_models:
        info["injector_model"] = "SPI (matching HRAP)"
    else:
        warnings.append(
            "imported with the Dyer model rather than SPI: predicted oxidiser flow "
            "will be lower than HRAP's for this geometry. Use SPI to cross-check."
        )

    # ---- propellant / regression ---------------------------------------
    prop_file = text("prop_file", "")
    prop_nm = text("prop_nm", "imported")
    cstar_eff = num("cstar_eff", 100.0) / 100.0  # HRAP stores percent

    if propellant is not None:
        prop = propellant
        prop.cstar_eff = cstar_eff
    else:
        prop, prop_warn = _resolve_propellant(path, prop_file, prop_nm, cstar_eff)
        warnings += prop_warn

    prop.rho_fuel = num("prop_rho", prop.rho_fuel) * unit(
        "prop_rho_unit", DENSITY, "kg/m^3", "fuel density"
    )

    reg_model = text("reg_model", "Shifting OF")
    a = num("prop_a", 0.0)
    n = num("prop_n", 0.0)
    m = num("prop_m", 0.0)
    const_OF = num("const_OF", prop.opt_OF)

    if reg_model == "Constant OF":
        regression_mode = "constant_OF"
        info["regression"] = f"constant O/F = {const_OF:g}"
    else:
        regression_mode = "shifting"
        if a > 0:
            prop.reg_a, prop.reg_n, prop.reg_m = a, n, m
        else:
            warnings.append(
                "config uses the shifting-O/F model but stores a = 0; keeping the "
                f"existing coefficients (a={prop.reg_a:g}, n={prop.reg_n:g})"
            )
        info["regression"] = (
            f"shifting O/F, a={prop.reg_a:g}, n={prop.reg_n:g}, m={prop.reg_m:g}"
        )
    prop.opt_OF = const_OF if const_OF > 0 else prop.opt_OF

    # ---- chamber volume ------------------------------------------------
    if num("cmbr_V_state", 1.0) == 1:
        chamber_volume = grn_L * 0.25 * math.pi * grn_OD**2
        info["chamber_volume_from"] = "grain envelope"
    else:
        chamber_volume = num("cmbr_V", 0.0) * unit(
            "cmbr_V_unit", VOLUME, "cm^3", "chamber volume"
        )
        info["chamber_volume_from"] = "explicit volume"

    # ---- vent ----------------------------------------------------------
    vnt_state = text("vnt_state", "None")
    if vnt_state != "None":
        warnings.append(
            f"config specifies a '{vnt_state}' vent, which this tool does not model. "
            "Oxidiser flow and blowdown will differ from HRAP by the vent flow."
        )

    if num("mp_state", 0.0) == 1:
        info["mass_properties"] = "present in config, not modelled here"

    dt = num("dt", 0.001)
    t_max = num("t_max", 30.0)
    if num("t_burn", 0.0) > 0:
        warnings.append(
            f"config sets a forced burn time of {num('t_burn', 0.0):g} s; this tool "
            "runs until propellant depletion instead."
        )

    cfg = MotorConfig(
        tank=tank,
        grain=grain,
        nozzle=nozzle,
        injector=injector,
        propellant=prop,
        ambient_P=num("Pa", 1.0) * unit("Pa_unit", PRESSURE, "atm", "ambient pressure"),
        dt=dt,
        max_time=t_max,
        # 'transient-hrap' rather than 'transient': matching HRAP means
        # matching its forward-Euler discretisation too, not just the ODE.
        chamber_mode="transient-hrap" if match_hrap_models else "quasi-steady",
        chamber_volume=chamber_volume,
        regression_mode=regression_mode,
        const_OF=const_OF,
        initial_Pc=num("P_cmbr", 1.0)
        * unit("P_cmbr_unit", PRESSURE, "atm", "initial chamber pressure"),
    )

    if not prop.has_table:
        warnings.append(
            "no CEA table available for this propellant, so a constant c* of "
            f"{prop.cstar_const:.0f} m/s is being used. HRAP always interpolates a "
            "table, so chamber pressure and thrust will not match closely until you "
            "load the matching propellant .mat."
        )

    return ImportedMotor(
        config=cfg,
        motor_name=motor_name,
        propellant_file=prop_file,
        warnings=warnings,
        info=info,
    )


def _resolve_propellant(
    motor_path: str, prop_file: str, prop_nm: str, cstar_eff: float
) -> tuple[Propellant, list[str]]:
    """Find the propellant config a motor file refers to.

    HRAP stores an absolute path from whatever machine wrote the file, which is
    almost never valid elsewhere, so the propellant name is also searched for
    near the motor file.
    """
    warnings: list[str] = []
    candidates = []

    if prop_file and not prop_file.startswith("("):
        candidates.append(prop_file)

    base = os.path.dirname(os.path.abspath(motor_path))
    name = os.path.basename(prop_file) if prop_file else f"{prop_nm}.mat"
    if not name.lower().endswith(".mat"):
        name = f"{prop_nm}.mat"
    for rel in (".", "..", "../propellant_configs", "propellant_configs"):
        candidates.append(os.path.normpath(os.path.join(base, rel, name)))

    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                return Propellant.from_hrap_mat(cand, cstar_eff=cstar_eff), warnings
            except Exception as exc:
                warnings.append(f"found {cand} but could not load it: {exc}")

    warnings.append(
        f"could not locate the propellant config for '{prop_nm}'"
        + (f" (config points at {prop_file})" if prop_file else "")
        + ". Load it manually to get HRAP-matching c*."
    )
    p = Propellant(name=prop_nm, cstar_eff=cstar_eff)
    return p, warnings


# --------------------------------------------------------------------------
# Simulation output import
# --------------------------------------------------------------------------

#: Maps HRAP CSV headers onto internal names and an SI conversion factor.
_CSV_COLUMNS = {
    "time": (r"^time", 1.0),
    "thrust": (r"^thrust", 1.0),
    "P_tank": (r"oxidizer tank pressure", 1e3),          # kPa -> Pa
    "P_chamber": (r"combustion chamber pressure", 1e3),  # kPa -> Pa
    "dP_reported": (r"injector pressure drop", 1e3),     # see module docstring
    "m_ox": (r"^oxidizer mass \(", 1.0),
    "m_fuel": (r"^fuel mass", 1.0),
    "m_total": (r"total motor mass", 1.0),
    "mdot_ox": (r"oxidizer mass flow rate", 1.0),
    "mdot_fuel": (r"fuel mass flow rate", 1.0),
    "mdot_nozzle": (r"exhaust mass flow rate", 1.0),
    "OF": (r"oxidizer to fuel ratio", 1.0),
    "port_d": (r"grain id", 0.01),                       # cm -> m
    "rdot": (r"regression rate", 1e-3),                  # mm/s -> m/s
    "cg": (r"center of mass", 0.01),
}


@dataclass
class HrapRun:
    """Time histories read from an HRAP output CSV, converted to SI."""

    channels: dict[str, np.ndarray]
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, name):
        ch = self.__dict__.get("channels", {})
        if name in ch:
            return ch[name]
        raise AttributeError(name)

    def has(self, name: str) -> bool:
        return name in self.channels and np.any(np.isfinite(self.channels[name]))

    @property
    def t(self) -> np.ndarray:
        return self.channels["time"]

    @property
    def burn_time(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def total_impulse(self) -> float:
        if not self.has("thrust"):
            return float("nan")
        return float(np.trapezoid(self.channels["thrust"], self.t))

    @property
    def mean_OF(self) -> float:
        if not (self.has("mdot_ox") and self.has("mdot_fuel")):
            return float("nan")
        ox = float(np.trapezoid(self.channels["mdot_ox"], self.t))
        fu = float(np.trapezoid(self.channels["mdot_fuel"], self.t))
        return ox / fu if fu > 0 else float("nan")


def load_hrap_output(path: str) -> HrapRun:
    """Read an HRAP simulation output CSV.

    Handles both the 13-column form and the 15-column form HRAP writes when
    mass properties are enabled, and is tolerant of column order.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{os.path.basename(path)} is empty")

    header = [h.strip().lower() for h in rows[0]]
    if not any("time" in h for h in header):
        raise ValueError(
            f"{os.path.basename(path)} does not look like an HRAP output CSV "
            "(no 'Time' column found in the header row). Export it from HRAP with "
            "the 'Export CSV' button."
        )

    index: dict[str, tuple[int, float]] = {}
    for key, (pattern, scale) in _CSV_COLUMNS.items():
        for i, h in enumerate(header):
            if re.search(pattern, h):
                index[key] = (i, scale)
                break

    warnings: list[str] = []
    data: list[list[float]] = []
    for row in rows[1:]:
        if not row or all(not c.strip() for c in row):
            continue
        try:
            data.append([float(c) if c.strip() else math.nan for c in row])
        except ValueError:
            warnings.append(f"skipped an unparseable row: {row[:3]}...")
    if not data:
        raise ValueError(f"{os.path.basename(path)} has a header but no data rows")

    width = max(len(r) for r in data)
    arr = np.full((len(data), width), np.nan)
    for i, r in enumerate(data):
        arr[i, : len(r)] = r

    channels = {
        key: arr[:, i] * scale for key, (i, scale) in index.items() if i < width
    }

    if "time" not in channels:
        raise ValueError("could not locate the time column")

    # HRAP's "Injector Pressure Drop" column is the per-timestep *tank* pressure
    # change, not the injector drop -- see the module docstring. Recompute the
    # real quantity from the two pressure columns.
    if "P_tank" in channels and "P_chamber" in channels:
        channels["dP_inj"] = channels["P_tank"] - channels["P_chamber"]
        if "dP_reported" in channels:
            reported = channels["dP_reported"]
            true_dP = channels["dP_inj"]
            finite = np.isfinite(reported) & np.isfinite(true_dP)
            if np.any(finite) and np.nanmax(np.abs(reported[finite])) < 0.05 * np.nanmax(
                np.abs(true_dP[finite])
            ):
                warnings.append(
                    "HRAP's 'Injector Pressure Drop' column holds the per-timestep "
                    "tank pressure change, not the injector drop. The comparison "
                    "uses P_tank - P_chamber instead."
                )

    return HrapRun(channels=channels, source=os.path.basename(path), warnings=warnings)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

#: Channels compared, mapped to this tool's :class:`BurnResult` attributes.
COMPARE_CHANNELS = [
    ("mdot_ox", "mdot_ox", "oxidiser mass flow", "kg/s"),
    ("mdot_fuel", "mdot_fuel", "fuel mass flow", "kg/s"),
    ("P_tank", "P_tank", "tank pressure", "Pa"),
    ("P_chamber", "P_chamber", "chamber pressure", "Pa"),
    ("OF", "OF", "O/F ratio", "-"),
    ("thrust", "thrust", "thrust", "N"),
    ("port_d", "port_d", "port diameter", "m"),
    ("m_ox", "m_ox", "oxidiser mass", "kg"),
    ("dP_inj", "dP_inj", "injector pressure drop", "Pa"),
]


@dataclass
class ChannelComparison:
    key: str
    label: str
    unit: str
    rms_pct: float
    max_pct: float
    mean_ours: float
    mean_hrap: float
    bias_pct: float


@dataclass
class RunComparison:
    """Quantitative agreement between this tool and an HRAP run."""

    channels: list[ChannelComparison] = field(default_factory=list)
    burn_time_ours: float = float("nan")
    burn_time_hrap: float = float("nan")
    impulse_ours: float = float("nan")
    impulse_hrap: float = float("nan")
    mean_OF_ours: float = float("nan")
    mean_OF_hrap: float = float("nan")
    overlap: float = 0.0
    warnings: list[str] = field(default_factory=list)


def compare_to_hrap(burn, hrap: HrapRun) -> RunComparison:
    """Compare a :class:`BurnResult` against an imported HRAP run.

    Both runs are resampled onto the overlapping portion of their time bases
    before differencing, since the two simulations generally stop at different
    times.
    """
    cmp = RunComparison(
        burn_time_ours=burn.burn_time,
        burn_time_hrap=hrap.burn_time,
        impulse_ours=burn.total_impulse,
        impulse_hrap=hrap.total_impulse,
        mean_OF_ours=burn.mean_OF,
        mean_OF_hrap=hrap.mean_OF,
        warnings=list(hrap.warnings),
    )

    if len(burn.t) < 2 or len(hrap.t) < 2:
        cmp.warnings.append("not enough samples to compare")
        return cmp

    t_end = min(float(burn.t[-1]), float(hrap.t[-1]))
    if t_end <= 0:
        cmp.warnings.append("the two runs do not overlap in time")
        return cmp
    cmp.overlap = t_end
    grid = np.linspace(0.0, t_end, 400)

    if abs(burn.burn_time - hrap.burn_time) > 0.1 * max(burn.burn_time, 1e-9):
        cmp.warnings.append(
            f"burn times differ by "
            f"{abs(burn.burn_time - hrap.burn_time) / max(hrap.burn_time, 1e-9) * 100:.1f}% "
            f"({burn.burn_time:.2f} s here vs {hrap.burn_time:.2f} s in HRAP); only the "
            f"overlapping first {t_end:.2f} s is compared."
        )

    for ours_key, hrap_key, label, unit in COMPARE_CHANNELS:
        if not hasattr(burn, ours_key) or not hrap.has(hrap_key):
            continue
        a = np.asarray(getattr(burn, ours_key), dtype=float)
        b = hrap.channels[hrap_key]
        if len(a) != len(burn.t):
            continue

        ok_a, ok_b = np.isfinite(a), np.isfinite(b)
        if ok_a.sum() < 2 or ok_b.sum() < 2:
            continue
        ai = np.interp(grid, burn.t[ok_a], a[ok_a])
        bi = np.interp(grid, hrap.t[ok_b], b[ok_b])

        scale = np.nanmax(np.abs(bi))
        if not np.isfinite(scale) or scale <= 0:
            continue
        err = (ai - bi) / scale * 100.0
        cmp.channels.append(
            ChannelComparison(
                key=ours_key,
                label=label,
                unit=unit,
                rms_pct=float(np.sqrt(np.mean(err**2))),
                max_pct=float(np.max(np.abs(err))),
                mean_ours=float(np.mean(ai)),
                mean_hrap=float(np.mean(bi)),
                bias_pct=float(np.mean(err)),
            )
        )

    return cmp


def format_comparison(cmp: RunComparison, model_name: str = "") -> str:
    """Render the comparison as a text table."""
    L = []
    add = L.append
    add("=" * 78)
    add("  CROSS-REFERENCE vs HRAP")
    if model_name:
        add(f"  this tool's injector model: {model_name}")
    add("=" * 78)
    add("")
    add(f"  {'':24} {'this tool':>14} {'HRAP':>14} {'diff':>10}")
    add("  " + "-" * 66)

    def line(label, a, b, fmt="{:.3f}"):
        if not (math.isfinite(a) and math.isfinite(b)):
            return
        d = (a - b) / b * 100.0 if b else float("nan")
        ds = f"{d:+.2f} %" if math.isfinite(d) else "n/a"
        add(f"  {label:24} {fmt.format(a):>14} {fmt.format(b):>14} {ds:>10}")

    line("burn time [s]", cmp.burn_time_ours, cmp.burn_time_hrap)
    line("total impulse [N.s]", cmp.impulse_ours, cmp.impulse_hrap, "{:.1f}")
    line("mean O/F", cmp.mean_OF_ours, cmp.mean_OF_hrap)
    add("")
    add(f"  Compared over the overlapping first {cmp.overlap:.2f} s.")
    add("")
    add("  Per-channel error (% of that channel's HRAP peak)")
    add("  " + "-" * 66)
    add(f"  {'channel':24} {'RMS':>9} {'max':>9} {'bias':>9}")
    for c in sorted(cmp.channels, key=lambda c: -c.rms_pct):
        add(f"  {c.label:24} {c.rms_pct:8.2f}% {c.max_pct:8.2f}% {c.bias_pct:+8.2f}%")

    if cmp.warnings:
        add("")
        add("  NOTES")
        add("  " + "-" * 66)
        for w in cmp.warnings:
            add(f"  [!] {w}")
    add("")
    add("=" * 78)
    return "\n".join(L)
