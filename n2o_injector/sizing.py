"""Injector orifice sizing against an O/F target across the whole burn.

The sizing problem is not a single-point calculation.  A drilled orifice plate
has *one* fixed area, but the operating point moves continuously: the tank
blows down, the port opens up, and the fuel mass flow responds to oxidiser
flux with a fractional-power law.  So the tool solves for the single ``Cd*A``
whose *burn-long* behaviour best matches the target, then reports how far the
instantaneous O/F wanders from that target along the way.

Three objectives are available:

``design_point``
    Match the target exactly at one instant (default ``t = 0``).  Closed-form:
    the required oxidiser flow follows from inverting the regression law, and
    the required area follows from the injector model.  Fast, and the right
    choice when the user wants the classic "size it at ignition" answer.

``burn_average``
    Match the mass-averaged O/F over the burn.  This is usually the better
    objective, because it balances the early-burn and late-burn excursions
    rather than pinning one end.

``least_squares``
    Minimise the integrated squared O/F error against a target *profile*.
    Use when the target varies over the burn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .injector import FlowModel, OrificeCurve, check_validity, recommend_model
from .motor import BurnResult, MotorConfig, fuel_flow, required_mdot_ox, simulate
from .properties import PropertyBackend

#: Approximate bounds of the oxidiser mass flux range over which published
#: paraffin regression correlations are fitted, kg/m^2/s. These are soft
#: bounds on the *data*, not physical limits: outside them ``a G^n`` is an
#: extrapolation rather than a fit, which is worth saying out loud because the
#: arithmetic gives an answer either way. See USER_GUIDE.md §1.7.
GOX_FIT_LO, GOX_FIT_HI = 50.0, 700.0


@dataclass
class SizingTarget:
    """What the injector is being sized to achieve.

    **On choosing an objective.** The O/F-based objectives must invert the
    regression law to find the oxidiser flow that yields the target ratio:

    .. math:: \\dot m_{ox} \\propto \\left[\\cdots\\right]^{1/(1-n)}

    For paraffin (``n ~ 0.68``) that exponent is ``~3.1``, so a 10% error in
    the regression coefficient ``a`` becomes a 35% error in required flow --
    and then a second amplification follows, because a larger injector raises
    chamber pressure, which cuts the pressure drop, which demands more area
    again. **This ill-conditioning belongs to the inverse problem, not to the
    regression law itself**: run forwards (fixed geometry, predict O/F) the
    same 10% error produces only a ~9% shift.

    If your ``a`` and ``n`` are literature values rather than measured on your
    own motor, prefer ``'mdot_ox'`` or ``'chamber_pressure'``. Those size the
    orifice from the feed state alone, never touch the regression law, and are
    numerically well-conditioned -- you then run forwards to see what O/F you
    actually get, which is the benign direction. See ``VALIDATION.md`` §3.1.
    """

    OF: float = 8.0
    #: Optional time-varying target: ``(times [s], O/F values)``.
    profile_t: np.ndarray | None = None
    profile_OF: np.ndarray | None = None

    #: Target oxidiser mass flow [kg/s], for ``objective='mdot_ox'``.
    mdot_ox: float = 1.0
    #: Target chamber pressure [Pa], for ``objective='chamber_pressure'``.
    chamber_P: float = 30e5

    #: design_point | burn_average | least_squares | mdot_ox | chamber_pressure
    objective: str = "burn_average"
    design_time: float = 0.0  #: s, used by design_point

    #: Minimum acceptable injector pressure drop as a fraction of chamber
    #: pressure. 15-20% is the usual guidance for feed-coupled stability margin.
    min_dP_fraction: float = 0.20

    @property
    def has_profile(self) -> bool:
        return self.profile_t is not None and self.profile_OF is not None

    def target_at(self, t) -> np.ndarray:
        """Target O/F at time(s) ``t``."""
        if self.has_profile:
            return np.interp(t, self.profile_t, self.profile_OF)
        return np.full_like(np.asarray(t, dtype=float), self.OF)


@dataclass
class OrificePlate:
    """A realisable multi-hole orifice plate."""

    n_holes: int
    hole_d: float  #: m
    Cd: float
    plate_thickness: float

    @property
    def hole_area(self) -> float:
        return 0.25 * math.pi * self.hole_d**2

    @property
    def total_area(self) -> float:
        return self.n_holes * self.hole_area

    @property
    def CdA(self) -> float:
        return self.Cd * self.total_area

    @property
    def CdA_per_hole(self) -> float:
        """HRAP's ``inj_CdA`` convention (per hole)."""
        return self.Cd * self.hole_area

    @property
    def L_over_D(self) -> float:
        return self.plate_thickness / self.hole_d if self.hole_d > 0 else 0.0


@dataclass
class SizingResult:
    """Outcome of a sizing run."""

    plate: OrificePlate
    CdA_required: float  #: ideal effective area before rounding to a hole count
    CdA_achieved: float
    burn: BurnResult
    target: SizingTarget

    achieved_mean_OF: float = float("nan")
    OF_error_pct: float = float("nan")
    #: 1/(1-n): how strongly an error in the regression coefficients is
    #: amplified into the sized area. Only meaningful for the O/F objectives.
    conditioning: float = float("nan")
    min_dP_fraction: float = float("nan")
    mean_dP_fraction: float = float("nan")
    dP_margin_ok: bool = True
    choked_fraction: float = float("nan")

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    model_recommendation: str = ""
    converged: bool = True


# --------------------------------------------------------------------------
# Plate realisation
# --------------------------------------------------------------------------


def plate_from_CdA(
    CdA: float,
    Cd: float,
    plate_thickness: float,
    n_holes: int | None = None,
    hole_d: float | None = None,
    min_hole_d: float = 0.0008,
) -> tuple[OrificePlate, list[str]]:
    """Turn a required ``Cd*A`` into a drillable plate.

    Exactly one of ``n_holes`` or ``hole_d`` is held fixed and the other is
    solved for.  Fixing the hole count gives an exact area match with a
    non-standard drill size; fixing the diameter forces an integer hole count,
    so the achieved area is rounded and the residual error is reported.
    """
    warnings: list[str] = []
    A_req = CdA / Cd

    if n_holes is not None and hole_d is not None:
        raise ValueError("fix either the hole count or the hole diameter, not both")

    if n_holes is not None:
        if n_holes < 1:
            raise ValueError("hole count must be at least 1")
        d = math.sqrt(4.0 * A_req / (n_holes * math.pi))
        if d < min_hole_d:
            n_max = max(int(A_req / (0.25 * math.pi * min_hole_d**2)), 1)
            warnings.append(
                f"{n_holes} holes would need a {d * 1e3:.3f} mm diameter, below the "
                f"{min_hole_d * 1e3:.3f} mm manufacturing minimum. Use at most "
                f"{n_max} hole(s) at that minimum diameter, or relax the constraint."
            )
        return OrificePlate(n_holes, d, Cd, plate_thickness), warnings

    if hole_d is None:
        raise ValueError("specify either a hole count or a hole diameter")

    if hole_d < min_hole_d:
        warnings.append(
            f"specified hole diameter {hole_d * 1e3:.3f} mm is below the "
            f"{min_hole_d * 1e3:.3f} mm manufacturing minimum"
        )
    a_hole = 0.25 * math.pi * hole_d**2
    n_exact = A_req / a_hole
    n = max(int(round(n_exact)), 1)
    err = (n * a_hole - A_req) / A_req * 100.0
    if abs(err) > 2.0:
        warnings.append(
            f"rounding {n_exact:.2f} holes to {n} changes the injector area by "
            f"{err:+.1f}%. Adjust the hole diameter for a closer match."
        )
    return OrificePlate(n, hole_d, Cd, plate_thickness), warnings


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


def design_point_CdA(
    cfg: MotorConfig,
    props: PropertyBackend,
    table,
    target: SizingTarget,
) -> tuple[float, float, float]:
    """Closed-form ``Cd*A`` to hit the target O/F at the design instant.

    Returns ``(CdA, mdot_ox_required, Pc)``.  Chamber pressure is not known in
    advance -- it depends on the very flow being sized -- so it is found by
    iterating the nozzle relation to convergence alongside the required flow.
    """
    prop, grain, noz = cfg.propellant, cfg.grain, cfg.nozzle
    OF_t = float(target.target_at(target.design_time))
    d = grain.port_id

    mdot_ox = required_mdot_ox(OF_t, grain, prop, d)
    mdot_f = mdot_ox / OF_t
    At = noz.throat_area * noz.Cd

    Pc = cfg.ambient_P
    for _ in range(100):
        cstar = prop.cstar(OF_t, Pc)
        Pc_new = (mdot_ox + mdot_f) * cstar / At
        if abs(Pc_new - Pc) < 1.0:
            Pc = Pc_new
            break
        Pc = 0.5 * (Pc + Pc_new)

    up = props.upstream_state(cfg.tank.fill_temp, cfg.tank.supercharge_P)
    if Pc >= up.P:
        raise ValueError(
            f"the target O/F needs {mdot_ox:.3f} kg/s, which drives chamber pressure "
            f"to {Pc / 1e5:.1f} bar -- at or above the {up.P / 1e5:.1f} bar tank "
            "pressure. Increase the nozzle throat, lower the target O/F, or raise "
            "the tank temperature."
        )

    curve = OrificeCurve(
        table, up, model=cfg.injector.model,
        L_over_D=cfg.injector.L_over_D, ld_ref=cfg.injector.ld_ref,
    )
    G = curve.flux(Pc)
    if G <= 0:
        raise ValueError("injector model returns zero mass flux at the design point")
    return mdot_ox / G, mdot_ox, Pc


def flow_target_CdA(
    cfg: MotorConfig,
    props: PropertyBackend,
    table,
    target: SizingTarget,
) -> tuple[float, float, float]:
    """Size from a mass-flow or chamber-pressure target.

    Returns ``(CdA, mdot_ox, Pc)``.

    Unlike the O/F objectives, this never inverts the regression law. For a
    mass-flow target the fuel flow follows *forwards* from the given oxidiser
    flow, chamber pressure follows from the nozzle relation, and the area is
    simply ``mdot / G``. For a chamber-pressure target the area is found by a
    single monotone root-find. Both are well-conditioned: a 10% error in the
    regression coefficients moves the answer by roughly 10%, not 150%.
    """
    prop, grain, noz = cfg.propellant, cfg.grain, cfg.nozzle
    d = grain.port_id
    At = noz.throat_area * noz.Cd
    up = props.upstream_state(cfg.tank.fill_temp, cfg.tank.supercharge_P)
    curve = OrificeCurve(
        table, up, model=cfg.injector.model,
        L_over_D=cfg.injector.L_over_D, ld_ref=cfg.injector.ld_ref,
    )

    if target.objective == "mdot_ox":
        mdot_ox = target.mdot_ox
        if mdot_ox <= 0:
            raise ValueError("target oxidiser mass flow must be positive")
        # Chamber pressure is a fixed point in Pc alone; fuel flow is forward.
        Pc = cfg.ambient_P
        for _ in range(200):
            _, mdot_f, OF = fuel_flow(cfg, d, mdot_ox)
            if not np.isfinite(OF):
                OF = prop.opt_OF
            Pc_new = (mdot_ox + mdot_f) * prop.cstar(OF, Pc) / At
            if abs(Pc_new - Pc) < 1.0:
                Pc = Pc_new
                break
            Pc = 0.5 * (Pc + Pc_new)
        if Pc >= up.P:
            raise ValueError(
                f"an oxidiser flow of {mdot_ox:.3f} kg/s drives chamber pressure to "
                f"{Pc / 1e5:.1f} bar, at or above the {up.P / 1e5:.1f} bar tank "
                "pressure. Enlarge the nozzle throat or reduce the target flow."
            )
        G = curve.flux(Pc)
        if G <= 0:
            raise ValueError("injector model returns zero mass flux at this point")
        return mdot_ox / G, mdot_ox, Pc

    # chamber_pressure objective
    Pc = target.chamber_P
    if Pc <= cfg.ambient_P:
        raise ValueError("target chamber pressure must exceed ambient")
    if Pc >= up.P:
        raise ValueError(
            f"target chamber pressure {Pc / 1e5:.1f} bar is at or above the "
            f"{up.P / 1e5:.1f} bar tank pressure; no flow is possible"
        )
    G = curve.flux(Pc)
    if G <= 0:
        raise ValueError("injector model returns zero mass flux at the target pressure")

    def residual(CdA: float) -> float:
        mdot_ox = CdA * G
        _, mdot_f, OF = fuel_flow(cfg, d, mdot_ox)
        if not np.isfinite(OF):
            OF = prop.opt_OF
        return (mdot_ox + mdot_f) * prop.cstar(OF, Pc) / At - Pc

    lo, hi = 1e-9, 1e-2
    if residual(lo) > 0 or residual(hi) < 0:
        raise ValueError(
            f"could not reach {Pc / 1e5:.1f} bar chamber pressure by varying "
            "injector area alone; check the throat diameter and regression law"
        )
    CdA = float(brentq(residual, lo, hi, xtol=1e-12, rtol=1e-9, maxiter=100))
    return CdA, CdA * G, Pc


def _burn_metric(
    cfg: MotorConfig, props, table, CdA: float, target: SizingTarget
) -> tuple[float, BurnResult]:
    """Scalar objective value for a candidate ``Cd*A``."""
    burn = simulate(cfg, props, table, CdA_override=CdA)
    if len(burn.t) < 2:
        return float("nan"), burn
    if target.objective == "least_squares":
        tgt = target.target_at(burn.t)
        ok = np.isfinite(burn.OF)
        if not np.any(ok):
            return float("nan"), burn
        return float(np.mean((burn.OF[ok] - tgt[ok]) ** 2)), burn
    return burn.mean_OF, burn


def solve_CdA(
    cfg: MotorConfig,
    props: PropertyBackend,
    table,
    target: SizingTarget,
    CdA_guess: float,
) -> tuple[float, BurnResult, bool, list[str]]:
    """Find the ``Cd*A`` that best meets the target over the full burn."""
    notes: list[str] = []

    if target.objective in ("design_point", "mdot_ox", "chamber_pressure"):
        burn = simulate(cfg, props, table, CdA_override=CdA_guess)
        return CdA_guess, burn, True, notes

    if target.objective == "least_squares":
        res = minimize_scalar(
            lambda lg: _burn_metric(cfg, props, table, math.exp(lg), target)[0],
            bracket=(math.log(CdA_guess * 0.6), math.log(CdA_guess * 1.4)),
            method="brent",
            options={"xtol": 1e-4, "maxiter": 60},
        )
        CdA = math.exp(float(res.x))
        burn = simulate(cfg, props, table, CdA_override=CdA)
        if not res.success:
            notes.append("least-squares optimiser did not fully converge")
        return CdA, burn, bool(res.success), notes

    # burn_average: mean O/F rises monotonically with injector area, so bracket
    # and bisect on log(CdA).
    OF_goal = target.OF

    def residual(lg: float) -> float:
        val, _ = _burn_metric(cfg, props, table, math.exp(lg), target)
        if not np.isfinite(val):
            # A failed burn (no flow, or instant stall) reads as "far too small".
            return -OF_goal
        return val - OF_goal

    lg0 = math.log(CdA_guess)
    lo = hi = lg0
    f0 = residual(lg0)
    if abs(f0) < 1e-9:
        burn = simulate(cfg, props, table, CdA_override=CdA_guess)
        return CdA_guess, burn, True, notes

    step = 0.25
    f_lo = f_hi = f0
    for _ in range(24):
        if f_lo > 0:
            lo -= step
            f_lo = residual(lo)
        elif f_hi < 0:
            hi += step
            f_hi = residual(hi)
        if f_lo <= 0 <= f_hi:
            break
    else:
        notes.append(
            "could not bracket the target O/F by varying injector area alone; "
            "returning the closest area found. The target may be unreachable for "
            "this grain and nozzle combination."
        )
        CdA = math.exp(lo if abs(f_lo) < abs(f_hi) else hi)
        return CdA, simulate(cfg, props, table, CdA_override=CdA), False, notes

    lg = brentq(residual, lo, hi, xtol=1e-5, maxiter=80)
    CdA = math.exp(float(lg))
    return CdA, simulate(cfg, props, table, CdA_override=CdA), True, notes


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------


def size_injector(
    cfg: MotorConfig,
    props: PropertyBackend,
    table,
    target: SizingTarget,
    fix: str = "n_holes",
) -> SizingResult:
    """Size the injector plate.

    ``fix`` selects which plate parameter is held constant: ``'n_holes'``
    solves for the hole diameter, ``'hole_d'`` solves for the hole count.
    """
    errs = cfg.validate(props)
    if errs:
        raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errs))

    if cfg.regression_mode == "constant_OF" and target.objective in (
        "design_point", "burn_average", "least_squares"
    ):
        # In this mode fuel flow is defined as mdot_ox / const_OF, so O/F is
        # pinned regardless of injector area. Sizing to an O/F target is
        # degenerate and the solver would silently fail to bracket.
        raise ValueError(
            "cannot size to an O/F target with the constant-O/F regression model: "
            f"O/F is pinned at {cfg.const_OF:g} for any injector area. Switch to "
            "the shifting-O/F model (regression coefficients a, n, m) to size the "
            "injector, or use constant-O/F only for reproducing an HRAP run."
        )

    warnings: list[str] = []
    notes_pre: list[str] = []

    # The closed-form design-point area is only an initial guess for the
    # burn-long objectives. It can legitimately be infeasible (a target that
    # needs more flow than the nozzle can pass below tank pressure at t=0)
    # while the burn-average solve still has a solution, so a failure here must
    # not abort those objectives -- fall back to a geometric guess instead.
    if target.objective in ("mdot_ox", "chamber_pressure"):
        # Well-conditioned path: never inverts the regression law.
        CdA_guess, _, _ = flow_target_CdA(cfg, props, table, target)
    else:
        try:
            CdA_guess, _, _ = design_point_CdA(cfg, props, table, target)
        except ValueError as exc:
            if target.objective == "design_point":
                raise
            CdA_guess = cfg.injector.CdA
            notes_pre.append(
                f"the design-point guess was infeasible ({exc}); the solver started "
                "from the current injector geometry instead."
            )

    CdA, burn, converged, notes = solve_CdA(cfg, props, table, target, CdA_guess)
    notes = notes_pre + notes

    inj = cfg.injector
    if fix == "n_holes":
        plate, plate_warn = plate_from_CdA(
            CdA, inj.Cd, inj.plate_thickness, n_holes=inj.n_holes,
            min_hole_d=inj.min_hole_d,
        )
    elif fix == "hole_d":
        plate, plate_warn = plate_from_CdA(
            CdA, inj.Cd, inj.plate_thickness, hole_d=inj.hole_d,
            min_hole_d=inj.min_hole_d,
        )
    else:
        raise ValueError("fix must be 'n_holes' or 'hole_d'")
    warnings += plate_warn

    # Re-run with the realised plate so every reported number reflects hardware
    # that can actually be drilled, not the idealised area.
    cfg_final = _with_plate(cfg, plate)
    burn = simulate(cfg_final, props, table, CdA_override=plate.CdA)

    result = SizingResult(
        plate=plate,
        CdA_required=CdA,
        CdA_achieved=plate.CdA,
        burn=burn,
        target=target,
        converged=converged,
        warnings=warnings,
        notes=notes + list(burn.notes),
    )

    _populate_metrics(result, cfg_final, props, table, target)
    return result


def analyse_geometry(
    cfg: MotorConfig,
    props: PropertyBackend,
    table,
    target: SizingTarget | None = None,
) -> SizingResult:
    """Run a *fixed* injector forwards, without sizing anything.

    This is the question HRAP answers: given this orifice plate, what does the
    motor do? Nothing is solved for -- the O/F that comes out is whatever the
    geometry produces, and may be nowhere near ``target.OF``.

    Use this to cross-reference against HRAP, or whenever you already have
    hardware and want its behaviour rather than a recommendation. It returns
    the same :class:`SizingResult` as :func:`size_injector`, so the report,
    plots and margin checks all work identically.
    """
    errs = cfg.validate(props)
    if errs:
        raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errs))

    inj = cfg.injector
    plate = OrificePlate(
        n_holes=inj.n_holes,
        hole_d=inj.hole_d,
        Cd=inj.Cd,
        plate_thickness=inj.plate_thickness,
    )
    burn = simulate(cfg, props, table, CdA_override=plate.CdA)

    if target is None:
        target = SizingTarget(OF=cfg.propellant.opt_OF, objective="as-built")

    result = SizingResult(
        plate=plate,
        CdA_required=plate.CdA,   # nothing was solved for
        CdA_achieved=plate.CdA,
        burn=burn,
        target=target,
        converged=True,
        warnings=list(inj.validate()),
        notes=["forward run of the entered geometry -- no sizing was performed"]
        + list(burn.notes),
    )
    _populate_metrics(result, cfg, props, table, target)
    return result


def _with_plate(cfg: MotorConfig, plate: OrificePlate) -> MotorConfig:
    """Copy ``cfg`` with the injector replaced by the realised plate."""
    import copy

    new = copy.deepcopy(cfg)
    new.injector.n_holes = plate.n_holes
    new.injector.hole_d = plate.hole_d
    new.injector.Cd = plate.Cd
    new.injector.plate_thickness = plate.plate_thickness
    return new


def _populate_metrics(
    result: SizingResult, cfg: MotorConfig, props, table, target: SizingTarget
) -> None:
    burn = result.burn
    if len(burn.t) == 0:
        result.warnings.append("simulation produced no timesteps")
        return

    result.achieved_mean_OF = burn.mean_OF
    goal = float(np.mean(target.target_at(burn.t)))
    if goal > 0 and np.isfinite(burn.mean_OF):
        result.OF_error_pct = (burn.mean_OF - goal) / goal * 100.0

    # Conditioning of the O/F objectives against regression-coefficient error.
    n = cfg.propellant.reg_n
    if target.objective in ("design_point", "burn_average", "least_squares"):
        if cfg.regression_mode == "shifting" and n < 1.0:
            k = 1.0 / (1.0 - n)
            result.conditioning = k
            if k >= 2.5:
                amp = (1.10**k - 1.0) * 100.0
                result.warnings.append(
                    f"ILL-CONDITIONED OBJECTIVE: sizing to an O/F target inverts the "
                    f"regression law with exponent 1/(1-n) = {k:.2f}, so a 10% error "
                    f"in the coefficient 'a' becomes roughly {amp:.0f}% in required "
                    f"oxidiser flow (and more in area, since a bigger injector raises "
                    f"chamber pressure and cuts the pressure drop). If a and n are "
                    f"literature values rather than measured on your motor, size with "
                    f"the 'mdot_ox' or 'chamber_pressure' objective instead and check "
                    f"the resulting O/F afterwards -- that direction is well-conditioned."
                )

    ok = np.isfinite(burn.dP_frac)
    if np.any(ok):
        result.min_dP_fraction = float(np.min(burn.dP_frac[ok]))
        result.mean_dP_fraction = float(np.mean(burn.dP_frac[ok]))
        result.dP_margin_ok = result.min_dP_fraction >= target.min_dP_fraction
        if result.min_dP_fraction < 0.02:
            # Below a couple of percent the injector is no longer metering: the
            # solver has driven chamber pressure onto the tank pressure to pass
            # the demanded flow. It "converges", but the result is not a design.
            result.warnings.append(
                f"DEGENERATE RESULT: injector pressure drop has collapsed to "
                f"{result.min_dP_fraction * 100:.2f}% of chamber pressure. Chamber "
                f"pressure has risen to meet tank pressure, so the injector is no "
                f"longer metering the flow and the returned area is not a usable "
                f"design. The target O/F is unreachable with this grain and nozzle "
                f"-- enlarge the throat, reduce the target O/F, or re-check the "
                f"regression coefficients (a, n)."
            )
        if not result.dP_margin_ok:
            result.warnings.append(
                f"minimum injector pressure drop is {result.min_dP_fraction * 100:.1f}% "
                f"of chamber pressure, below the {target.min_dP_fraction * 100:.0f}% "
                "target. The feed system is weakly isolated from the chamber and "
                "may couple into combustion instability. Reduce injector area "
                "(fewer or smaller holes), or raise tank pressure."
            )
        elif result.min_dP_fraction > 1.5:
            result.notes.append(
                f"injector pressure drop is very high ({result.min_dP_fraction * 100:.0f}% "
                "of chamber pressure). Stability margin is ample but the feed system "
                "is doing a lot of throttling; a larger injector would raise chamber "
                "pressure and performance."
            )

    if len(burn.choked):
        result.choked_fraction = float(np.mean(burn.choked))

    # Validity of the chosen model at the design point.
    up = props.upstream_state(cfg.tank.fill_temp, cfg.tank.supercharge_P)
    curve = OrificeCurve(
        table, up, model=cfg.injector.model,
        L_over_D=cfg.injector.L_over_D, ld_ref=cfg.injector.ld_ref,
    )
    Pc0 = float(burn.P_chamber[0])
    det = curve.detail(Pc0)
    result.warnings += check_validity(det, up, Pc0, result.plate.L_over_D)

    rec_model, rec_text = recommend_model(result.plate.L_over_D)
    result.model_recommendation = rec_text
    if rec_model is not cfg.injector.model and cfg.injector.model is not FlowModel.DYER_LD:
        result.notes.append(
            f"selected model is {cfg.injector.model.value}; based on the resulting "
            f"L/D the recommendation is {rec_model.value}."
        )

    # Oxidiser flux against the range the regression law was fitted over. The
    # tool will happily evaluate a*G^n at any flux, and the answer looks no
    # different -- so an extrapolation has to be called out explicitly or it
    # passes unnoticed. The design consequence is real: G_ox is set by the
    # grain port, not by the injector, so the fix is never a different plate.
    if len(burn.G_ox):
        liq = burn.m_liq > 1e-3
        G = burn.G_ox[liq] if np.any(liq) else burn.G_ox
        G = G[np.isfinite(G) & (G > 0)]
        if len(G):
            G_hi = float(np.max(G))
            if G_hi > GOX_FIT_HI:
                over = 100.0 * float(np.mean(G > GOX_FIT_HI))
                result.warnings.append(
                    f"oxidiser flux peaks at {G_hi:.0f} kg/m^2/s, above the "
                    f"~{GOX_FIT_HI:.0f} kg/m^2/s upper end of the range paraffin "
                    f"regression correlations are normally fitted over, for {over:.0f}% "
                    "of the liquid phase. The regression law is being extrapolated, so "
                    "the fuel flow and O/F here carry more uncertainty than the "
                    "coefficients alone imply. G_ox is set by the grain port area, not "
                    "by the injector -- open the ports or accept the extrapolation."
                )
            elif G_hi < GOX_FIT_LO:
                result.warnings.append(
                    f"oxidiser flux peaks at only {G_hi:.0f} kg/m^2/s, below the "
                    f"~{GOX_FIT_LO:.0f} kg/m^2/s lower end of the usual correlation "
                    "range. At low flux, radiation and non-uniform burning matter and "
                    "the a*G^n law under-predicts."
                )

    # O/F excursion across the burn.
    ok_of = np.isfinite(burn.OF)
    if np.any(ok_of):
        tgt = target.target_at(burn.t)[ok_of]
        dev = (burn.OF[ok_of] - tgt) / tgt * 100.0
        worst = float(np.max(np.abs(dev)))
        if worst > 15.0:
            result.warnings.append(
                f"instantaneous O/F deviates from target by up to {worst:.0f}% during "
                "the burn. A single fixed orifice area cannot track the target across "
                "blowdown; consider whether the excursion is acceptable at the ends "
                "of the burn."
            )
