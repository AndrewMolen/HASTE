"""Injector orifice flow models for self-pressurising nitrous oxide.

Three mass-flux models are implemented, plus the Dyer blend:

SPI -- Single Phase Incompressible
    The classical incompressible orifice relation
    ``G = sqrt(2 rho_l dP)``.  Valid when the liquid stays subcooled all the
    way through the orifice.  This is the *only* liquid-phase injector model
    HRAP implements (``tnk_inj_liq_model = 'Incompressible'`` in
    ``HRAP - Python/hrap/tank.py``; the same expression appears in
    ``HRAP - Matlab/util/tank.m``), so an SPI run of this tool reproduces
    HRAP's oxidiser flow exactly for a given ``CdA``.

HEM -- Homogeneous Equilibrium Model
    Liquid and vapour are assumed to reach thermodynamic equilibrium
    instantaneously.  The flow expands isentropically from the upstream state;
    at each candidate exit pressure the equilibrium quality follows from the
    entropy balance and the mass flux is ``G = sqrt(2 (h1 - h2)) / v2``.  The
    delivered flux is the *maximum* of ``G`` over exit pressure, which is the
    definition of choking for a two-phase flow.

Dyer / NHNE -- Non-Homogeneous Non-Equilibrium
    A weighted blend of SPI and HEM (Dyer et al., "Modeling Feed System Flow
    Physics for Self-Pressurizing Propellants", AIAA 2007-5702, with the
    Solomon/Whitmore form of the weighting parameter):

    .. math::

        \\kappa = \\sqrt{\\frac{P_1 - P_2}{P_{sat}(T_1) - P_2}}, \\qquad
        G = \\frac{\\kappa}{1+\\kappa} G_{SPI} + \\frac{1}{1+\\kappa} G_{HEM}

    For a saturated (self-pressurising) feed ``P_1 = P_sat``, so ``kappa = 1``
    and the blend is the familiar 50/50 average.

All mass fluxes are per unit *geometric* area; the discharge coefficient is
applied by :func:`mass_flow` when converting flux to mass flow rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .properties import (
    T_MAX_VALID,
    T_MIN_VALID,
    FluidState,
    PropertyBackend,
)


class FlowModel(str, Enum):
    """Selectable injector flow model."""

    SPI = "SPI"
    HEM = "HEM"
    DYER = "Dyer"
    DYER_LD = "Dyer (L/D weighted)"

    @classmethod
    def from_str(cls, s: str) -> "FlowModel":
        key = s.strip().lower().replace("_", " ")
        for m in cls:
            if key == m.value.lower() or key == m.name.lower():
                return m
        if key.startswith("dyer") and "l/d" in key:
            return cls.DYER_LD
        if key.startswith("dyer"):
            return cls.DYER
        raise ValueError(f"unknown flow model {s!r}")


@dataclass
class FluxResult:
    """Outcome of a mass-flux evaluation at one operating point."""

    G: float  #: delivered mass flux [kg/m^2/s], excluding Cd
    G_spi: float  #: SPI branch [kg/m^2/s]
    G_hem: float  #: HEM branch [kg/m^2/s]
    kappa: float  #: Dyer non-equilibrium weighting parameter [-]
    w_hem: float  #: weight applied to the HEM branch [-]
    choked: bool  #: True when the HEM branch is choked
    P_crit: float  #: HEM critical (choking) back pressure [Pa]
    model: FlowModel
    notes: tuple[str, ...] = ()

    @property
    def w_spi(self) -> float:
        return 1.0 - self.w_hem


class SaturationTable:
    """Dense, vectorised saturation table built once from a property backend.

    The HEM solver needs saturated properties at every trial exit pressure of
    every timestep of every sizing iteration.  Sampling the correlations onto a
    monotone grid once, then interpolating with ``numpy``, turns that inner
    loop into vector arithmetic and is what makes full-burn sizing interactive.
    Grid density is chosen so interpolation error is far below the ~1-2%
    spread between the ESDU fits and the CoolProp reference EOS.
    """

    def __init__(
        self,
        props: PropertyBackend,
        n: int = 3000,
        T_min: float = T_MIN_VALID,
        T_max: float = T_MAX_VALID,
    ) -> None:
        self.props = props
        T = np.linspace(T_min, T_max, n)
        states = [props.sat(float(t)) for t in T]
        self.T = T
        self.P = np.array([s.P for s in states])
        self.rho_l = np.array([s.rho_l for s in states])
        self.rho_v = np.array([s.rho_v for s in states])
        self.h_l = np.array([s.h_l for s in states])
        self.h_v = np.array([s.h_v for s in states])
        self.s_l = np.array([s.s_l for s in states])
        self.s_v = np.array([s.s_v for s in states])
        self.P_min = float(self.P[0])
        self.P_max = float(self.P[-1])

    def at_P(self, P):
        """Interpolate saturated properties at pressure(s) ``P``."""
        P = np.clip(np.asarray(P, dtype=float), self.P_min, self.P_max)
        return {
            "T": np.interp(P, self.P, self.T),
            "rho_l": np.interp(P, self.P, self.rho_l),
            "rho_v": np.interp(P, self.P, self.rho_v),
            "h_l": np.interp(P, self.P, self.h_l),
            "h_v": np.interp(P, self.P, self.h_v),
            "s_l": np.interp(P, self.P, self.s_l),
            "s_v": np.interp(P, self.P, self.s_v),
        }


# --------------------------------------------------------------------------
# Individual models
# --------------------------------------------------------------------------


def spi_mass_flux(up: FluidState, P2: float) -> float:
    """Single-phase incompressible mass flux [kg/m^2/s] (Cd not applied)."""
    dP = up.P - P2
    if dP <= 0.0:
        return 0.0
    return float(np.sqrt(2.0 * up.rho * dP))


def _isentropic_flux_curve(table: SaturationTable, up: FluidState, P: np.ndarray) -> np.ndarray:
    """Mass flux for isentropic expansion of ``up`` to each pressure in ``P``.

    Above the upstream saturation pressure the fluid is still subcooled liquid
    and the incompressible relation applies; below it the flow flashes and the
    equilibrium quality sets the two-phase specific volume.
    """
    P = np.asarray(P, dtype=float)
    G = np.zeros_like(P)

    liquid = P >= up.P_sat
    if np.any(liquid):
        dP = np.maximum(up.P - P[liquid], 0.0)
        G[liquid] = np.sqrt(2.0 * up.rho * dP)

    two_phase = ~liquid
    if np.any(two_phase):
        sat = table.at_P(P[two_phase])
        s_fg = sat["s_v"] - sat["s_l"]
        h_fg = sat["h_v"] - sat["h_l"]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(s_fg > 0.0, (up.s - sat["s_l"]) / s_fg, 0.0)
        x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
        h2 = sat["h_l"] + x * h_fg
        v2 = (1.0 - x) / sat["rho_l"] + x / sat["rho_v"]
        dh = np.maximum(up.h - h2, 0.0)
        G[two_phase] = np.sqrt(2.0 * dh) / v2

    return G


def hem_mass_flux(
    table: SaturationTable,
    up: FluidState,
    P2: float,
    n_coarse: int = 240,
    n_fine: int = 120,
) -> tuple[float, bool, float]:
    """Homogeneous-equilibrium mass flux with choking.

    Returns ``(G, choked, P_crit)``.  ``G`` is the maximum isentropic flux over
    back pressures between ``P2`` and the upstream pressure -- the flow chokes
    when that maximum occurs at a pressure *above* ``P2``, in which case the
    orifice cannot sense the lower downstream pressure and ``P_crit`` is the
    pressure that is actually reached at the throat.
    """
    if P2 >= up.P:
        return 0.0, False, P2

    # The grid is anchored to the *table*, not to P2, so that the choked
    # plateau evaluates to bit-identical values regardless of the back
    # pressure asked for. Anchoring at P2 would shift every sample point and
    # make a physically constant plateau wobble at the 1e-5 level.
    grid = np.linspace(max(table.P_min, 1.0), up.P, n_coarse)
    G = _isentropic_flux_curve(table, up, grid)

    above = grid >= P2
    if not np.any(above):
        return 0.0, False, P2
    idx = np.flatnonzero(above)
    i_rel = int(np.argmax(G[idx]))
    i = int(idx[i_rel])

    # Refine around the coarse maximum so the reported critical pressure is not
    # limited by grid spacing.
    lo = grid[max(i - 1, idx[0])]
    hi = grid[min(i + 1, len(grid) - 1)]
    G_max, P_crit = float(G[i]), float(grid[i])
    if hi > lo:
        fine = np.linspace(lo, hi, n_fine)
        Gf = _isentropic_flux_curve(table, up, fine)
        j = int(np.argmax(Gf))
        if Gf[j] >= G_max:
            G_max, P_crit = float(Gf[j]), float(fine[j])

    # Choked when the optimum sits above the requested back pressure: the
    # throat cannot sense P2, so lowering it further changes nothing.
    choked = bool(grid[i] > grid[idx[0]])
    if not choked:
        G_max = float(np.interp(P2, grid, G))
        P_crit = P2
    return G_max, choked, P_crit


def dyer_kappa(up: FluidState, P2: float) -> float:
    """Dyer non-equilibrium parameter ``sqrt((P1-P2)/(Psat-P2))``.

    Returns ``inf`` when the downstream pressure is at or above the upstream
    saturation pressure -- the flow cannot flash, so the blend collapses to
    pure SPI.
    """
    denom = up.P_sat - P2
    if denom <= 0.0:
        return float("inf")
    num = up.P - P2
    if num <= 0.0:
        return 0.0
    return float(np.sqrt(num / denom))


def ld_hem_weight(L_over_D: float, ld_ref: float = 5.0) -> float:
    """Heuristic HEM weight as a function of orifice ``L/D``.

    .. warning::
       This is an *engineering interpolation*, not Dyer's published parameter.
       It exists to expose the regime trend the literature describes -- a
       vanishingly short orifice gives the flow no residence time to flash
       (SPI-like), while a long one lets it approach equilibrium (HEM-like) --
       and to make that limiting behaviour testable.  Use :data:`FlowModel.DYER`
       for design work; use this only for regime exploration.
    """
    if ld_ref <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-max(L_over_D, 0.0) / ld_ref))


# --------------------------------------------------------------------------
# Precomputed operating curve (fast path for burn simulation)
# --------------------------------------------------------------------------


class OrificeCurve:
    """Injector mass flux vs. chamber pressure for one fixed upstream state.

    The burn simulation solves for chamber pressure at every timestep, which
    means evaluating the injector many times against the *same* upstream state.
    Recomputing the HEM curve for each of those evaluations is pure waste,
    because of a useful structural fact:

        ``G_HEM(P_2) = max{ G_isentropic(P) : P_2 <= P <= P_1 }``

    is a *suffix maximum* of a single curve that does not depend on ``P_2``.
    So the isentropic flux curve is built once here, its suffix maximum gives
    the HEM branch at every back pressure simultaneously, and each subsequent
    query is an interpolation.  This is what makes full-burn sizing (hundreds
    of simulations, each with hundreds of timesteps) run interactively.
    """

    def __init__(
        self,
        table: SaturationTable,
        up: FluidState,
        model: FlowModel = FlowModel.DYER,
        L_over_D: float = 2.0,
        ld_ref: float = 5.0,
        n: int = 600,
    ) -> None:
        self.up = up
        self.model = model
        self.L_over_D = L_over_D

        P = np.linspace(max(table.P_min, 1.0), up.P, n)
        self.P = P

        G_iso = _isentropic_flux_curve(table, up, P)

        # HEM branch: suffix maximum of the isentropic curve.
        G_hem = np.maximum.accumulate(G_iso[::-1])[::-1]

        # Critical pressure: for each back pressure, the lowest-index location
        # at or above it that attains the suffix maximum.
        idx = np.arange(n)
        is_peak = G_iso >= G_hem - 1e-9 * np.maximum(G_hem, 1.0)
        cand = np.where(is_peak, idx, n - 1)
        crit_idx = np.minimum.accumulate(cand[::-1])[::-1]
        self.P_crit = P[crit_idx]
        self.choked = crit_idx > idx

        self.G_hem = G_hem
        self.G_spi = np.sqrt(2.0 * up.rho * np.maximum(up.P - P, 0.0))

        denom = up.P_sat - P
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.sqrt(np.maximum(up.P - P, 0.0) / denom)
        # Downstream at or above the upstream saturation pressure => no
        # flashing => the blend must collapse onto SPI.
        self.kappa = np.where(denom > 0.0, kappa, np.inf)

        if model is FlowModel.SPI:
            self.w_hem = np.zeros(n)
        elif model is FlowModel.HEM:
            self.w_hem = np.ones(n)
        elif model is FlowModel.DYER_LD:
            self.w_hem = np.full(n, ld_hem_weight(L_over_D, ld_ref))
        else:
            self.w_hem = np.where(np.isfinite(self.kappa), 1.0 / (1.0 + self.kappa), 0.0)

        self.G = (1.0 - self.w_hem) * self.G_spi + self.w_hem * self.G_hem

    def flux(self, P2: float) -> float:
        """Delivered mass flux [kg/m^2/s] at chamber pressure ``P2``."""
        if P2 >= self.up.P:
            return 0.0
        return float(np.interp(P2, self.P, self.G))

    def detail(self, P2: float) -> FluxResult:
        """Full :class:`FluxResult` at chamber pressure ``P2``."""
        if P2 >= self.up.P:
            return FluxResult(
                G=0.0, G_spi=0.0, G_hem=0.0, kappa=float("nan"), w_hem=0.0,
                choked=False, P_crit=P2, model=self.model,
                notes=("downstream pressure is at or above upstream pressure: no flow",),
            )
        i = int(np.clip(np.searchsorted(self.P, P2), 0, len(self.P) - 1))
        return FluxResult(
            G=self.flux(P2),
            G_spi=float(np.interp(P2, self.P, self.G_spi)),
            G_hem=float(np.interp(P2, self.P, self.G_hem)),
            kappa=float(self.kappa[i]),
            w_hem=float(self.w_hem[i]),
            choked=bool(self.choked[i]),
            P_crit=float(self.P_crit[i]),
            model=self.model,
        )


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------


def mass_flux(
    table: SaturationTable,
    up: FluidState,
    P2: float,
    model: FlowModel = FlowModel.DYER,
    L_over_D: float = 2.0,
    ld_ref: float = 5.0,
) -> FluxResult:
    """Evaluate injector mass flux for the selected model.

    ``table`` supplies saturated properties, ``up`` is the upstream liquid
    state, and ``P2`` is the chamber (downstream) pressure.
    """
    notes: list[str] = []

    if P2 >= up.P:
        return FluxResult(
            G=0.0, G_spi=0.0, G_hem=0.0, kappa=float("nan"), w_hem=0.0,
            choked=False, P_crit=P2, model=model,
            notes=("downstream pressure is at or above upstream pressure: no flow",),
        )

    G_spi = spi_mass_flux(up, P2)

    if model is FlowModel.SPI:
        # Still report the HEM branch so the user can see how far apart the
        # models are at this operating point.
        G_hem, choked, P_crit = hem_mass_flux(table, up, P2)
        if G_spi > 1.25 * G_hem > 0.0:
            notes.append(
                "SPI exceeds HEM by >25% at this point: flashing is likely and "
                "SPI will over-predict flow. Consider Dyer or HEM."
            )
        return FluxResult(
            G=G_spi, G_spi=G_spi, G_hem=G_hem, kappa=float("inf"), w_hem=0.0,
            choked=False, P_crit=P2, model=model, notes=tuple(notes),
        )

    G_hem, choked, P_crit = hem_mass_flux(table, up, P2)

    if model is FlowModel.HEM:
        return FluxResult(
            G=G_hem, G_spi=G_spi, G_hem=G_hem, kappa=0.0, w_hem=1.0,
            choked=choked, P_crit=P_crit, model=model, notes=tuple(notes),
        )

    if model is FlowModel.DYER_LD:
        w_hem = ld_hem_weight(L_over_D, ld_ref)
        kappa = (1.0 - w_hem) / w_hem if w_hem > 0.0 else float("inf")
    else:  # FlowModel.DYER
        kappa = dyer_kappa(up, P2)
        w_hem = 0.0 if np.isinf(kappa) else 1.0 / (1.0 + kappa)

    G = (1.0 - w_hem) * G_spi + w_hem * G_hem

    if np.isinf(kappa):
        notes.append(
            "chamber pressure exceeds upstream saturation pressure: flow cannot "
            "flash, Dyer collapses to SPI"
        )
    if choked:
        notes.append(
            f"HEM branch is choked: throat reaches {P_crit / 1e5:.2f} bar and the "
            "orifice is insensitive to further reductions in chamber pressure"
        )
    return FluxResult(
        G=G, G_spi=G_spi, G_hem=G_hem, kappa=kappa, w_hem=w_hem,
        choked=choked, P_crit=P_crit, model=model, notes=tuple(notes),
    )


def mass_flow(
    table: SaturationTable,
    up: FluidState,
    P2: float,
    CdA_total: float,
    model: FlowModel = FlowModel.DYER,
    L_over_D: float = 2.0,
    ld_ref: float = 5.0,
) -> tuple[float, FluxResult]:
    """Oxidiser mass flow [kg/s] for a total effective area ``CdA_total``.

    ``CdA_total`` is ``Cd * N_holes * A_hole`` -- i.e. HRAP's ``inj_CdA *
    inj_N`` with the discharge coefficient folded in.
    """
    res = mass_flux(table, up, P2, model=model, L_over_D=L_over_D, ld_ref=ld_ref)
    return CdA_total * res.G, res


def required_CdA(
    table: SaturationTable,
    up: FluidState,
    P2: float,
    mdot_target: float,
    model: FlowModel = FlowModel.DYER,
    L_over_D: float = 2.0,
    ld_ref: float = 5.0,
) -> tuple[float, FluxResult]:
    """Effective area ``Cd*A`` needed to pass ``mdot_target`` at this point."""
    res = mass_flux(table, up, P2, model=model, L_over_D=L_over_D, ld_ref=ld_ref)
    if res.G <= 0.0:
        raise ValueError(
            "no flow at this operating point (chamber pressure at or above tank "
            "pressure); cannot size an orifice"
        )
    return mdot_target / res.G, res


# --------------------------------------------------------------------------
# Regime guidance
# --------------------------------------------------------------------------


def recommend_model(L_over_D: float) -> tuple[FlowModel, str]:
    """Suggest a flow model from the orifice length-to-diameter ratio.

    The thresholds follow the common hybrid-rocket guidance: a thin-plate
    orifice gives the flow too little residence time to reach equilibrium, so
    the Dyer blend is the practical standard; a long tube-like passage lets
    flashing develop and trends towards HEM.
    """
    if L_over_D < 1.0:
        return FlowModel.DYER, (
            f"L/D = {L_over_D:.2f}: very short orifice. Flow has little residence "
            "time to flash, so behaviour is SPI-leaning. Dyer is recommended; "
            "expect kappa-weighting to favour the SPI branch."
        )
    if L_over_D <= 5.0:
        return FlowModel.DYER, (
            f"L/D = {L_over_D:.2f}: thin-plate regime. Dyer (NHNE) is the "
            "standard choice here and is recommended."
        )
    if L_over_D <= 10.0:
        return FlowModel.DYER, (
            f"L/D = {L_over_D:.2f}: transitional. Flashing has meaningful time to "
            "develop. Dyer is still reasonable, but cross-check against HEM."
        )
    return FlowModel.HEM, (
        f"L/D = {L_over_D:.2f}: long orifice. The flow has time to approach "
        "equilibrium, so HEM is the more appropriate model."
    )


def check_validity(res: FluxResult, up: FluidState, P2: float, L_over_D: float) -> list[str]:
    """Flag operating points that fall outside a model's validity range."""
    warnings: list[str] = []

    if res.model is FlowModel.SPI and up.is_saturated:
        warnings.append(
            "SPI with a saturated feed: the liquid is on the saturation line, so "
            "any pressure drop flashes it. SPI will over-predict mass flow."
        )
    if res.model is FlowModel.HEM and L_over_D < 1.0:
        warnings.append(
            f"HEM at L/D = {L_over_D:.2f}: the orifice is too short for the flow "
            "to reach equilibrium. HEM will under-predict mass flow."
        )
    if res.model in (FlowModel.DYER, FlowModel.DYER_LD) and L_over_D > 10.0:
        warnings.append(
            f"Dyer at L/D = {L_over_D:.2f}: long orifices approach equilibrium; "
            "HEM is likely more accurate here."
        )
    if res.G_hem > 0.0 and res.G_spi > 3.0 * res.G_hem:
        warnings.append(
            "SPI and HEM differ by more than 3x at this point; the choice of "
            "model dominates the result, so treat the sizing as provisional "
            "until validated against a cold-flow test."
        )
    return warnings
