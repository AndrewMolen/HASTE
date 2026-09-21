"""Motor geometry, tank blowdown, and the coupled burn simulation.

The tank model reproduces HRAP's saturated-equilibrium blowdown.  The liquid
phase uses the evaporative-cooling derivative form from HRAP's Python
implementation (``HRAP - Python/hrap/tank.py``)::

    A = (m_liq drho_l/dT / rho_l^2 + m_vap drho_v/dT / rho_v^2) / (1/rho_v - 1/rho_l)
    B = -mdot_ox / (rho_l/rho_v - 1)
    C = -h_fg / ((m_liq + m_vap) cp)
    Tdot = B C / (1 - A C)

which is the differential equivalent of the finite-difference bookkeeping in
the MATLAB ``tank.m``.  Once the liquid is exhausted the vapour is assumed to
stay on the saturation line, following HRAP's ``Z``-iteration adiabatic
expansion.

Chamber pressure can be evaluated two ways:

``quasi-steady`` (default for sizing)
    ``Pc = mdot_total * c* / (Cd_noz A_t)``, solved as a fixed point because
    ``c*`` depends on O/F and ``Pc``.  This is the steady solution of HRAP's
    chamber ODE and is what an injector sizing study actually cares about; it
    has no ignition transient and no dependence on an assumed initial ``Pc``.

``transient``
    The same chamber ODE, ``dP/P = dm_g/m_g - dV/V``, but integrated in the
    form it actually has.  The equation says ``P`` is proportional to
    ``m_g/V``, so a step is taken as the exact ratio
    ``P_new = P_old (m_g_new/m_g_old)(V_old/V_new)`` rather than as an
    additive Euler increment, with sub-stepping when the gas inventory changes
    quickly.  This matters at ignition: the chamber is seeded with air, so
    ``m_g`` is tiny and ``dm_g/m_g`` is enormous, and an additive step there is
    limited by the step size rather than by the physics.

``transient-hrap``
    HRAP's own forward-Euler discretisation of the same ODE, kept verbatim so
    a comparison against an HRAP run reproduces its numerics as well as its
    physics.  Use it only for that comparison: its ignition ramp is a function
    of the timestep, not of the chamber, so the ramp takes a roughly fixed
    number of steps and its duration scales with ``dt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .injector import FlowModel, OrificeCurve, SaturationTable
from .propellant import Propellant
from .properties import GAMMA_N2O_VAP, R_N2O, PropertyBackend

#: Accepted values for :attr:`MotorConfig.chamber_mode`.
CHAMBER_MODES = ("quasi-steady", "transient", "transient-hrap")

#: Largest fractional change in chamber gas inventory allowed in one
#: sub-step of the transient model. The ignition ramp starts from an almost
#: empty chamber, where the inventory can multiply many times over in a single
#: outer timestep; splitting the step keeps the nozzle outflow (which depends
#: on the pressure being solved for) evaluated close to where it applies.
CHAMBER_SUBSTEP_TOL = 0.05
#: Cap on sub-steps per outer step, so a pathological config cannot stall.
CHAMBER_MAX_SUBSTEPS = 256


# --------------------------------------------------------------------------
# Geometry / configuration
# --------------------------------------------------------------------------


@dataclass
class Grain:
    """Cylindrical fuel grain with ``n_ports`` identical circular ports."""

    length: float = 0.40  #: m
    port_id: float = 0.035  #: initial port diameter [m]
    outer_d: float = 0.090  #: grain outer diameter [m]
    n_ports: int = 1

    def port_area(self, d: float) -> float:
        """Total flow area of all ports [m^2]."""
        return self.n_ports * 0.25 * np.pi * d**2

    def burn_perimeter(self, d: float) -> float:
        """Total burning perimeter of all ports [m]."""
        return self.n_ports * np.pi * d

    def fuel_volume(self, d: float) -> float:
        """Remaining solid fuel volume [m^3]."""
        return max(
            0.25 * np.pi * (self.outer_d**2 - self.n_ports * d**2) * self.length, 0.0
        )

    def fuel_mass(self, d: float, rho: float) -> float:
        return rho * self.fuel_volume(d)

    @property
    def web(self) -> float:
        """Available web thickness [m] (single-port exact, multi-port nominal)."""
        if self.n_ports == 1:
            return 0.5 * (self.outer_d - self.port_id)
        # Outer web to the case for a ring of ports on a bolt circle is
        # geometry-dependent; report the equivalent-area web as a nominal value.
        d_final = np.sqrt(self.outer_d**2 / self.n_ports)
        return 0.5 * (d_final - self.port_id)

    def validate(self) -> list[str]:
        errs = []
        if self.length <= 0:
            errs.append("grain length must be positive")
        if self.port_id <= 0:
            errs.append("initial port diameter must be positive")
        if self.n_ports < 1:
            errs.append("number of ports must be at least 1")
        if self.n_ports * self.port_id**2 >= self.outer_d**2:
            errs.append(
                f"{self.n_ports} port(s) of {self.port_id * 1e3:.1f} mm do not fit "
                f"inside a {self.outer_d * 1e3:.1f} mm grain"
            )
        return errs


@dataclass
class Tank:
    """Self-pressurising N2O tank."""

    volume: float = 0.010  #: m^3
    fill_temp: float = 293.15  #: K
    ox_mass: float = 6.0  #: initial oxidiser mass [kg]
    supercharge_P: float | None = None  #: absolute feed pressure if supercharged [Pa]

    def validate(self, props: PropertyBackend) -> list[str]:
        errs = []
        if self.volume <= 0:
            errs.append("tank volume must be positive")
        if self.ox_mass <= 0:
            errs.append("oxidiser mass must be positive")
        if not props.in_range(self.fill_temp):
            errs.append(
                f"fill temperature {self.fill_temp - 273.15:.1f} C is outside the "
                "N2O correlation range (-90 to +36 C)"
            )
            return errs
        sat = props.sat(self.fill_temp)
        rho_bulk = self.ox_mass / self.volume
        if rho_bulk > sat.rho_l:
            errs.append(
                f"tank is liquid-full: {self.ox_mass:.2f} kg in {self.volume * 1e3:.2f} L "
                f"is {rho_bulk:.0f} kg/m^3 but saturated liquid is only "
                f"{sat.rho_l:.0f} kg/m^3 at {self.fill_temp - 273.15:.1f} C"
            )
        if rho_bulk < sat.rho_v:
            errs.append(
                f"tank contains no liquid: bulk density {rho_bulk:.1f} kg/m^3 is below "
                f"saturated vapour density {sat.rho_v:.1f} kg/m^3"
            )
        if self.supercharge_P is not None and self.supercharge_P < sat.P:
            errs.append(
                f"supercharge pressure {self.supercharge_P / 1e5:.1f} bar is below the "
                f"{sat.P / 1e5:.1f} bar saturation pressure at the fill temperature"
            )
        return errs


@dataclass
class Nozzle:
    throat_d: float = 0.030  #: m
    expansion_ratio: float = 4.0
    Cd: float = 0.95
    efficiency: float = 0.95

    @property
    def throat_area(self) -> float:
        return 0.25 * np.pi * self.throat_d**2


@dataclass
class InjectorSpec:
    """Injector plate definition."""

    n_holes: int = 20
    hole_d: float = 0.0015  #: m
    Cd: float = 0.7
    plate_thickness: float = 0.003  #: m, sets orifice length
    min_hole_d: float = 0.0008  #: manufacturing minimum [m]
    model: FlowModel = FlowModel.DYER
    ld_ref: float = 5.0  #: reference L/D for the L/D-weighted blend

    @property
    def hole_area(self) -> float:
        return 0.25 * np.pi * self.hole_d**2

    @property
    def total_area(self) -> float:
        return self.n_holes * self.hole_area

    @property
    def CdA(self) -> float:
        """Total effective area ``Cd * N * A_hole`` [m^2]."""
        return self.Cd * self.total_area

    @property
    def CdA_per_hole(self) -> float:
        """Per-hole ``CdA`` -- this is HRAP's ``inj_CdA`` convention."""
        return self.Cd * self.hole_area

    @property
    def L_over_D(self) -> float:
        return self.plate_thickness / self.hole_d if self.hole_d > 0 else 0.0

    def validate(self) -> list[str]:
        errs = []
        if self.n_holes < 1:
            errs.append("hole count must be at least 1")
        if self.hole_d <= 0:
            errs.append("hole diameter must be positive")
        elif self.hole_d < self.min_hole_d:
            errs.append(
                f"hole diameter {self.hole_d * 1e3:.3f} mm is below the "
                f"{self.min_hole_d * 1e3:.3f} mm manufacturing minimum"
            )
        if not (0.0 < self.Cd <= 1.0):
            errs.append("discharge coefficient must be in (0, 1]")
        return errs


@dataclass
class MotorConfig:
    """Everything needed to run a burn."""

    tank: Tank = field(default_factory=Tank)
    grain: Grain = field(default_factory=Grain)
    nozzle: Nozzle = field(default_factory=Nozzle)
    injector: InjectorSpec = field(default_factory=InjectorSpec)
    propellant: Propellant = field(default_factory=Propellant)

    ambient_P: float = 101325.0  #: Pa
    dt: float = 0.005  #: s
    max_time: float = 60.0  #: s
    #: ``'quasi-steady'``, ``'transient'`` or ``'transient-hrap'``; see the
    #: module docstring for what each one integrates.
    chamber_mode: str = "quasi-steady"
    chamber_volume: float = 0.0  #: m^3; 0 => port volume only (HRAP convention)
    stop_at_liquid_exhausted: bool = True

    #: Tail-off cutoff. Once the motor has lit, the run ends when chamber
    #: pressure falls back to this multiple of ambient. Without it a
    #: vapour-phase blowdown dribbles on until ``max_time``, producing a burn
    #: time and impulse that depend on the time limit rather than the physics.
    #: HRAP's own runs terminate at roughly 1.09x ambient.
    tail_cutoff_frac: float = 1.05

    #: ``'shifting'`` uses the a/n/m regression law; ``'constant_OF'`` forces
    #: fuel flow to hold ``const_OF``, matching HRAP's ``const_OF.m``. Every
    #: motor config HRAP ships uses the latter, so imports need it.
    regression_mode: str = "shifting"
    const_OF: float = 8.0

    #: Chamber pressure at ignition, used only by the transient model.
    initial_Pc: float | None = None

    def validate(self, props: PropertyBackend) -> list[str]:
        errs = []
        errs += self.tank.validate(props)
        errs += self.grain.validate()
        errs += self.injector.validate()
        if self.nozzle.throat_d <= 0:
            errs.append("nozzle throat diameter must be positive")
        if self.dt <= 0:
            errs.append("timestep must be positive")
        if self.regression_mode == "shifting" and self.propellant.reg_n >= 1.0:
            errs.append(
                f"regression exponent n = {self.propellant.reg_n:.3f} must be < 1 "
                "(the required-oxidiser back-solve is singular at n = 1)"
            )
        if self.regression_mode == "constant_OF" and self.const_OF <= 0:
            errs.append("constant O/F must be positive")
        if self.regression_mode not in ("shifting", "constant_OF"):
            errs.append(
                f"unknown regression mode {self.regression_mode!r} "
                "(expected 'shifting' or 'constant_OF')"
            )
        if self.chamber_mode not in CHAMBER_MODES:
            errs.append(
                f"unknown chamber mode {self.chamber_mode!r} (expected one of "
                + ", ".join(repr(m) for m in CHAMBER_MODES) + ")"
            )
        return errs


# --------------------------------------------------------------------------
# Required oxidiser flow for a target O/F
# --------------------------------------------------------------------------


def required_mdot_ox(OF_target: float, grain: Grain, prop: Propellant, port_d: float) -> float:
    """Oxidiser flow [kg/s] that yields ``OF_target`` at the current port size.

    With :math:`\\dot r = 10^{-3} a G_{ox}^n L^m` and
    :math:`\\dot m_f = \\rho \\dot r P L` (``P`` = burn perimeter), the O/F
    definition inverts in closed form:

    .. math::
        \\dot m_{ox} = \\left[ \\frac{OF \\cdot \\rho\\, 10^{-3} a\\, L^{m}\\, P\\, L}
                                    {A_{port}^{\\,n}} \\right]^{1/(1-n)}

    which is exact provided ``n < 1``.
    """
    if OF_target <= 0:
        raise ValueError("target O/F must be positive")
    if prop.reg_n >= 1.0:
        raise ValueError("regression exponent n must be < 1 to invert for mdot_ox")

    A = grain.port_area(port_d)
    P = grain.burn_perimeter(port_d)
    L = grain.length
    coeff = OF_target * prop.rho_fuel * 1e-3 * prop.reg_a * L**prop.reg_m * P * L
    return float((coeff / A**prop.reg_n) ** (1.0 / (1.0 - prop.reg_n)))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def fuel_flow(cfg: MotorConfig, d: float, mdot_ox: float) -> tuple[float, float, float]:
    """Fuel regression response to an oxidiser flow.

    Returns ``(rdot, mdot_fuel, OF)`` for the current port diameter ``d``.

    ``'shifting'`` applies the a/n/m mass-flux law (HRAP's ``shift_OF``);
    ``'constant_OF'`` instead pins the O/F and back-solves the regression rate
    that delivers it (HRAP's ``const_OF``), which is what every motor config
    HRAP ships actually uses.
    """
    prop, grain = cfg.propellant, cfg.grain
    perim = grain.burn_perimeter(d)
    L = grain.length

    if cfg.regression_mode == "constant_OF":
        mdot_f = mdot_ox / cfg.const_OF if cfg.const_OF > 0 else 0.0
        denom = prop.rho_fuel * perim * L
        rdot = mdot_f / denom if denom > 0 else 0.0
        return rdot, mdot_f, (cfg.const_OF if mdot_f > 0 else float("nan"))

    A_port = grain.port_area(d)
    G_ox = mdot_ox / A_port if A_port > 0 else 0.0
    rdot = prop.regression_rate(G_ox, L)
    mdot_f = prop.rho_fuel * rdot * perim * L
    OF = mdot_ox / mdot_f if mdot_f > 1e-12 else float("nan")
    return rdot, mdot_f, OF


@dataclass
class BurnResult:
    """Time histories from a burn simulation."""

    t: np.ndarray
    P_tank: np.ndarray
    P_chamber: np.ndarray
    T_tank: np.ndarray
    mdot_ox: np.ndarray
    mdot_fuel: np.ndarray
    OF: np.ndarray
    port_d: np.ndarray
    m_ox: np.ndarray
    m_liq: np.ndarray
    m_fuel: np.ndarray
    thrust: np.ndarray
    dP_inj: np.ndarray  #: injector pressure drop [Pa]
    dP_frac: np.ndarray  #: dP / Pc [-]
    G_ox: np.ndarray  #: oxidiser port mass flux [kg/m^2/s]
    choked: np.ndarray  #: HEM branch choked flag
    kappa: np.ndarray
    liquid_exhausted_t: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def burn_time(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.thrust, self.t)) if len(self.t) > 1 else 0.0

    @property
    def mean_OF(self) -> float:
        """Mass-averaged O/F over the burn."""
        ox = float(np.trapezoid(self.mdot_ox, self.t))
        fu = float(np.trapezoid(self.mdot_fuel, self.t))
        return ox / fu if fu > 0 else float("nan")

    @property
    def ox_consumed(self) -> float:
        return float(np.trapezoid(self.mdot_ox, self.t))

    @property
    def fuel_consumed(self) -> float:
        return float(np.trapezoid(self.mdot_fuel, self.t))

    def specific_impulse(self, g0: float = 9.80665) -> float:
        m = self.ox_consumed + self.fuel_consumed
        return self.total_impulse / (m * g0) if m > 0 else float("nan")


def _thrust(cfg: MotorConfig, Pc: float, k: float) -> float:
    """Nozzle thrust [N] using HRAP's ``nozzle.m`` formulation."""
    if Pc <= cfg.ambient_P:
        return 0.0
    At = cfg.nozzle.throat_area
    er = cfg.nozzle.expansion_ratio

    def area_ratio(M):
        return (
            ((k + 1) / 2) ** (-(k + 1) / (2 * (k - 1)))
            * (1 + (k - 1) / 2 * M**2) ** ((k + 1) / (2 * (k - 1)))
            / M
            - er
        )

    try:
        M = brentq(area_ratio, 1.0000001, 20.0)
    except ValueError:
        return 0.0
    Pe = Pc * (1 + 0.5 * (k - 1) * M**2) ** (-k / (k - 1))
    Cf = np.sqrt(
        ((2 * k**2) / (k - 1))
        * (2 / (k + 1)) ** ((k + 1) / (k - 1))
        * (1 - (Pe / Pc) ** ((k - 1) / k))
    ) + ((Pe - cfg.ambient_P) * At * er) / (Pc * At)
    return max(cfg.nozzle.efficiency * Cf * At * Pc * cfg.nozzle.Cd, 0.0)


def _vapour_mass_flow(CdA: float, P_t: float, T: float, Z: float, P_down: float) -> float:
    """HRAP's compressible real-gas vapour injector model."""
    if P_t <= 0 or P_down >= P_t:
        return 0.0
    g = GAMMA_N2O_VAP
    M = np.sqrt(Z * g * R_N2O * T * (P_down / P_t) ** ((g - 1) / g))
    M = min(M, 1.0)
    return float(
        (CdA * P_t / np.sqrt(T))
        * np.sqrt(g / (Z * R_N2O))
        * M
        * (1 + (g - 1) / 2 * M**2) ** (-(g + 1) / (2 * (g - 1)))
    )


def simulate(
    cfg: MotorConfig,
    props: PropertyBackend,
    table: SaturationTable,
    CdA_override: float | None = None,
) -> BurnResult:
    """Integrate a full burn including tank blowdown.

    ``CdA_override`` replaces the injector's total ``Cd*A``; the sizing solver
    uses it to sweep area without rebuilding the configuration.
    """
    CdA = cfg.injector.CdA if CdA_override is None else CdA_override
    prop = cfg.propellant
    grain = cfg.grain
    dt = cfg.dt

    T = cfg.tank.fill_temp
    m_ox = cfg.tank.ox_mass
    d = grain.port_id
    m_fuel = grain.fuel_mass(d, prop.rho_fuel)
    m_fuel_0 = m_fuel

    Pc = cfg.ambient_P if cfg.initial_Pc is None else cfg.initial_Pc

    # Initial chamber gas inventory. HRAP seeds the free chamber volume with
    # air at 1.225 kg/m^3 (`x.m_g = 1.225*(cmbr_V - solid grain volume)`), and
    # this value sets the whole ignition transient: starting from ~0 instead
    # makes the chamber take orders of magnitude longer to come up to pressure.
    # Matching it is required for any meaningful comparison against HRAP.
    if cfg.chamber_volume > 0.0:
        solid_V = 0.25 * np.pi * (grain.outer_d**2 - grain.port_id**2) * grain.length
        m_gas = max(1.225 * (cfg.chamber_volume - solid_V), 1e-9)
    else:
        m_gas = max(1.225 * grain.port_area(d) * grain.length, 1e-9)
    notes: list[str] = []
    lit = False  # set once chamber pressure rises clear of ambient
    liquid_exhausted_t: float | None = None

    hist: dict[str, list[float]] = {
        k: []
        for k in (
            "t", "P_tank", "P_chamber", "T_tank", "mdot_ox", "mdot_fuel", "OF",
            "port_d", "m_ox", "m_liq", "m_fuel", "thrust", "dP_inj", "dP_frac",
            "G_ox", "choked", "kappa",
        )
    }

    t = 0.0
    n_steps = int(cfg.max_time / dt)

    for _ in range(n_steps):
        if m_ox <= 1e-6 or m_fuel <= 1e-9:
            break
        if not props.in_range(T):
            notes.append(
                f"tank temperature reached {T - 273.15:.1f} C, outside the N2O "
                "correlation range; simulation stopped"
            )
            break

        sat = props.sat(T)

        # Phase split from tank volume and total oxidiser mass.
        m_liq = max(
            (cfg.tank.volume - m_ox / sat.rho_v) / (1.0 / sat.rho_l - 1.0 / sat.rho_v), 0.0
        )
        m_liq = min(m_liq, m_ox)
        m_vap = m_ox - m_liq

        if m_liq <= 1e-3 and liquid_exhausted_t is None:
            liquid_exhausted_t = t
            if cfg.stop_at_liquid_exhausted:
                notes.append(f"liquid oxidiser exhausted at t = {t:.3f} s")
                break

        # ---- injector ----
        up = props.upstream_state(T, cfg.tank.supercharge_P)
        # The pressure the injector actually sees. For a self-pressurising tank
        # this is the saturation pressure; for a supercharged one it is the
        # supercharge pressure, with the liquid subcooled beneath it. Reporting
        # the drop against the saturation pressure instead would understate the
        # margin on a supercharged feed by exactly the supercharge.
        P_tank = up.P
        liquid_phase = m_liq > 1e-3
        curve = (
            OrificeCurve(
                table, up, model=cfg.injector.model,
                L_over_D=cfg.injector.L_over_D, ld_ref=cfg.injector.ld_ref,
            )
            if liquid_phase
            else None
        )
        if liquid_phase:
            det = curve.detail(Pc)
            mdot_ox = CdA * det.G
            choked, kappa = det.choked, det.kappa
        else:
            mdot_ox = _vapour_mass_flow(CdA, P_tank, T, sat.Z, Pc)
            choked, kappa = True, float("nan")

        # ---- grain regression ----
        A_port = grain.port_area(d)
        G_ox = mdot_ox / A_port if A_port > 0 else 0.0
        rdot, mdot_f, OF = fuel_flow(cfg, d, mdot_ox)
        if m_fuel <= 0:
            mdot_f, rdot, OF = 0.0, 0.0, float("nan")

        # ---- chamber pressure ----
        if cfg.chamber_mode == "quasi-steady":
            Pc = _quasi_steady_Pc(cfg, curve, d, T, sat, CdA, up)
            # Recompute the operating point at the converged chamber pressure so
            # the recorded flow, flux and O/F are mutually consistent.
            if liquid_phase:
                det = curve.detail(Pc)
                mdot_ox = CdA * det.G
                choked, kappa = det.choked, det.kappa
            else:
                mdot_ox = _vapour_mass_flow(CdA, P_tank, T, sat.Z, Pc)
            G_ox = mdot_ox / A_port if A_port > 0 else 0.0
            rdot, mdot_f, OF = fuel_flow(cfg, d, mdot_ox)
            if m_fuel <= 0:
                mdot_f, rdot, OF = 0.0, 0.0, float("nan")
            cstar = prop.cstar(OF if np.isfinite(OF) else prop.opt_OF, Pc)
            mdot_n = Pc * cfg.nozzle.Cd * cfg.nozzle.throat_area / cstar
        else:
            OF_eff = OF if np.isfinite(OF) else prop.opt_OF
            cstar = prop.cstar(OF_eff, Pc)
            if cfg.chamber_volume == 0.0:
                V = A_port * grain.length
            else:
                V = cfg.chamber_volume - (
                    0.25 * np.pi * grain.outer_d**2 - A_port
                ) * grain.length
            V = max(V, 1e-9)
            dV = grain.burn_perimeter(d) * rdot * grain.length
            Pc, m_gas, mdot_n = _chamber_step(
                cfg, Pc, m_gas, V, dV, mdot_ox, mdot_f, cstar, dt, P_feed=up.P
            )

        k_gas = prop.gamma(OF if np.isfinite(OF) else prop.opt_OF, Pc)
        F = _thrust(cfg, Pc, k_gas)

        dP_inj = P_tank - Pc
        dP_frac = dP_inj / Pc if Pc > 0 else float("nan")

        for key, val in (
            ("t", t), ("P_tank", P_tank), ("P_chamber", Pc), ("T_tank", T),
            ("mdot_ox", mdot_ox), ("mdot_fuel", mdot_f), ("OF", OF),
            ("port_d", d), ("m_ox", m_ox), ("m_liq", m_liq), ("m_fuel", m_fuel),
            ("thrust", F), ("dP_inj", dP_inj), ("dP_frac", dP_frac),
            ("G_ox", G_ox), ("choked", float(bool(choked))), ("kappa", kappa),
        ):
            hist[key].append(val)

        # ---- tail-off termination ----
        # Only after the motor has actually lit, so the ignition ramp (which
        # starts at ambient) is never mistaken for burnout.
        if Pc > 1.5 * cfg.ambient_P:
            lit = True
        if lit and Pc <= cfg.tail_cutoff_frac * cfg.ambient_P:
            notes.append(
                f"tail-off: chamber pressure decayed to {cfg.tail_cutoff_frac:.2f}x "
                f"ambient at t = {t:.3f} s"
            )
            break

        # ---- integrate ----
        if m_liq > 1e-3:
            Tdot = _liquid_blowdown_Tdot(props, T, sat, m_liq, m_vap, mdot_ox)
        else:
            Tdot = _vapour_blowdown_Tdot(props, T, m_ox, mdot_ox, sat)

        T += Tdot * dt
        m_ox = max(m_ox - mdot_ox * dt, 0.0)
        d += 2.0 * rdot * dt
        m_fuel = max(m_fuel - mdot_f * dt, 0.0)
        t += dt

        if grain.n_ports * d**2 >= grain.outer_d**2:
            notes.append(f"fuel web burned through at t = {t:.3f} s")
            break
    else:
        notes.append(
            f"simulation hit the {cfg.max_time:.0f} s time limit before the "
            "oxidiser or fuel was exhausted"
        )

    if m_fuel <= 1e-9 < m_fuel_0:
        notes.append("fuel grain fully consumed before oxidiser ran out")

    arrays = {k: np.asarray(v, dtype=float) for k, v in hist.items()}
    choked_arr = arrays.pop("choked").astype(bool)
    return BurnResult(
        **arrays, choked=choked_arr, liquid_exhausted_t=liquid_exhausted_t, notes=notes
    )


def _chamber_step(
    cfg: MotorConfig,
    Pc: float,
    m_gas: float,
    V: float,
    dV: float,
    mdot_ox: float,
    mdot_f: float,
    cstar: float,
    dt: float,
    P_feed: float = 0.0,
) -> tuple[float, float, float]:
    """Advance the transient chamber over one timestep.

    Returns ``(Pc, m_gas, mdot_n)``.

    The governing equation is HRAP's

    .. math:: \\frac{dP}{P} = \\frac{dm_g}{m_g} - \\frac{dV}{V}

    whose solution is simply :math:`P \\propto m_g / V`.  Taking the step in
    that ratio form rather than as :math:`P + P(\\ldots)\\,dt` is what makes
    the ignition ramp a property of the chamber instead of a property of the
    timestep: at ignition the chamber holds only its seeded air, so
    :math:`dm_g/m_g` is of order :math:`10^4\\,\\mathrm{s^{-1}}` and an
    additive step of any usable size simply multiplies the pressure by a
    bounded factor each step, which makes the ramp take a fixed *number of
    steps* and therefore a duration proportional to ``dt``.

    ``'transient-hrap'`` keeps the additive form deliberately, because
    reproducing an HRAP run means reproducing its discretisation too.
    """
    At = cfg.nozzle.Cd * cfg.nozzle.throat_area
    mdot_in = mdot_ox + mdot_f

    if cfg.chamber_mode == "transient-hrap":
        mdot_n = Pc * At / cstar
        dm_g = mdot_in - mdot_n
        m_gas = max(m_gas + dm_g * dt, 1e-9)
        Pc = max(Pc + Pc * (dm_g / m_gas - dV / V) * dt, cfg.ambient_P)
        return Pc, m_gas, mdot_n

    # Split the step so no sub-step changes the inventory by more than
    # CHAMBER_SUBSTEP_TOL. Away from ignition this is always one sub-step.
    mdot_n = Pc * At / cstar
    swing = abs(mdot_in - mdot_n) * dt / max(m_gas, 1e-12)
    n_sub = int(min(max(1, np.ceil(swing / CHAMBER_SUBSTEP_TOL)), CHAMBER_MAX_SUBSTEPS))
    h = dt / n_sub

    for _ in range(n_sub):
        mdot_n = Pc * At / cstar
        # Oxidiser flow is held at its start-of-step value, which is fine while
        # the injector is choked but breaks down if the chamber approaches feed
        # pressure inside one step -- the frozen inflow then keeps filling a
        # chamber the tank could no longer push into, and Pc runs away above
        # tank pressure. Tapering the inflow over the last 5% of the available
        # drop is a numerical guard, not a model of the flow: in a converged
        # run the chamber never gets there and the taper never engages.
        taper = 1.0
        if P_feed > 0.0:
            taper = min(max((P_feed - Pc) / (0.05 * P_feed), 0.0), 1.0)
        m_new = max(m_gas + (mdot_f + mdot_ox * taper - mdot_n) * h, 1e-12)
        V_new = max(V + dV * h, 1e-12)
        # Exact integral of dP/P = dm_g/m_g - dV/V over the sub-step.
        Pc = max(Pc * (m_new / m_gas) * (V / V_new), cfg.ambient_P)
        m_gas, V = m_new, V_new

    return Pc, m_gas, mdot_n


def _quasi_steady_Pc(
    cfg: MotorConfig, curve, d: float, T: float, sat, CdA: float, up
) -> float:
    """Solve ``Pc = mdot_total c* / (Cd_noz A_t)`` as a fixed point in ``Pc``."""
    prop = cfg.propellant
    At = cfg.nozzle.throat_area * cfg.nozzle.Cd

    def residual(Pc: float) -> float:
        if curve is not None:
            mdot_ox = CdA * curve.flux(Pc)
        else:
            # up.P, not sat.P: the feed pressure is what drives the orifice,
            # and the two differ on a supercharged tank.
            mdot_ox = _vapour_mass_flow(CdA, up.P, T, sat.Z, Pc)
        _, mdot_f, OF = fuel_flow(cfg, d, mdot_ox)
        if not np.isfinite(OF):
            OF = prop.opt_OF
        cstar = prop.cstar(OF, Pc)
        return (mdot_ox + mdot_f) * cstar / At - Pc

    lo, hi = cfg.ambient_P, max(up.P * 0.999, cfg.ambient_P * 1.01)
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo <= 0.0:
        return lo
    if f_hi >= 0.0:
        # Nozzle cannot pass the flow below tank pressure -- the motor would
        # stagnate. Clamp just under tank pressure and let the caller's dP
        # margin check flag it.
        return hi
    return float(brentq(residual, lo, hi, xtol=1.0, rtol=1e-6, maxiter=100))


def _liquid_blowdown_Tdot(props, T: float, sat, m_liq: float, m_vap: float, mdot_out: float) -> float:
    """Evaporative-cooling temperature rate, per HRAP's ``liq_blowdow``."""
    if m_liq + m_vap <= 0 or mdot_out <= 0:
        return 0.0
    eps = 1e-3
    s_hi, s_lo = props.sat(T + eps), props.sat(T - eps)
    drho_l = (s_hi.rho_l - s_lo.rho_l) / (2 * eps)
    drho_v = (s_hi.rho_v - s_lo.rho_v) / (2 * eps)

    denom = 1.0 / sat.rho_v - 1.0 / sat.rho_l
    if abs(denom) < 1e-12:
        return 0.0
    A = (m_liq * drho_l / sat.rho_l**2 + m_vap * drho_v / sat.rho_v**2) / denom
    B = -(-mdot_out) / (sat.rho_l / sat.rho_v - 1.0)
    C = -sat.h_fg / ((m_liq + m_vap) * sat.cp_l)
    if abs(1.0 - A * C) < 1e-12:
        return 0.0
    return float(B * C / (1.0 - A * C))


def _vapour_blowdown_Tdot(props, T: float, m_ox: float, mdot_out: float, sat) -> float:
    """Saturation-line vapour blowdown, per HRAP's ``sat_vap_blowdown``."""
    if m_ox <= 0 or mdot_out <= 0:
        return 0.0
    delta = 1e-5
    m_ratio = max((m_ox - mdot_out * delta) / m_ox, 1e-9)
    Z1 = sat.Z
    Z_i, T_new = Z1, T
    for _ in range(50):
        T_try = T * (Z_i / Z1 * m_ratio) ** 0.3
        if not props.in_range(T_try):
            return 0.0
        Z_new = props.sat(T_try).Z
        if abs(Z_i - Z_new) < 1e-9:
            T_new = T_try
            break
        Z_i = 0.5 * (Z_i + Z_new)
        T_new = T_try
    return float((T_new - T) / delta)
