"""Saturated nitrous oxide thermophysical properties.

Two interchangeable backends are provided:

``ESDUProperties``
    Curve fits from ESDU 91022 (Beaton & Walton, *Thermophysical Properties
    of Nitrous Oxide*, ESDU International, London, 1991).  These are the
    *exact* correlations used by HRAP's MATLAB implementation
    (``HRAP - Matlab/util/NOX.m``), reproduced coefficient-for-coefficient so
    that this tool and an HRAP MATLAB run see identical fluid properties.

``CoolPropProperties``
    Reference-equation-of-state properties via CoolProp, which is what the
    HRAP *Python* implementation uses (``HRAP - Python/hrap/fluid.py`` calls
    ``CP.PropsSI(..., 'NitrousOxide')``).  Requires the optional ``CoolProp``
    package.

Entropy is required by the Homogeneous Equilibrium Model but is *not* part of
the ESDU correlation set that HRAP carries.  For the ESDU backend it is
reconstructed thermodynamically (see :meth:`ESDUProperties._build_entropy`),
which is exact to within the saturated-liquid-``cp`` approximation and, more
importantly, is self-consistent with the ESDU enthalpies that the HEM
solver uses alongside it.

Units are SI throughout: K, Pa, kg/m^3, J/kg, J/kg/K.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Critical-point constants (as used by HRAP / ESDU 91022)
# --------------------------------------------------------------------------

T_CRIT = 309.57  # K
P_CRIT = 7.251e6  # Pa
RHO_CRIT = 452.0  # kg/m^3
R_N2O = 188.91  # J/kg/K, specific gas constant
GAMMA_N2O_VAP = 1.31  # ratio of specific heats used by HRAP's vapour model

T_TRIPLE = 182.33  # K
#: Validity window of the ESDU correlations (-90 C to +36 C).
T_MIN_VALID = 183.15
T_MAX_VALID = 309.15


@dataclass(frozen=True)
class SatState:
    """Saturated-fluid state at a single temperature."""

    T: float  #: temperature [K]
    P: float  #: saturation (vapour) pressure [Pa]
    rho_l: float  #: saturated liquid density [kg/m^3]
    rho_v: float  #: saturated vapour density [kg/m^3]
    h_l: float  #: saturated liquid specific enthalpy [J/kg]
    h_v: float  #: saturated vapour specific enthalpy [J/kg]
    s_l: float  #: saturated liquid specific entropy [J/kg/K]
    s_v: float  #: saturated vapour specific entropy [J/kg/K]
    cp_l: float  #: saturated liquid specific heat [J/kg/K]
    Z: float  #: saturated vapour compressibility factor [-]

    @property
    def h_fg(self) -> float:
        """Latent heat of vaporisation [J/kg]."""
        return self.h_v - self.h_l

    @property
    def s_fg(self) -> float:
        """Entropy of vaporisation [J/kg/K]."""
        return self.s_v - self.s_l


@dataclass(frozen=True)
class FluidState:
    """A general (not necessarily saturated) upstream fluid state."""

    T: float  #: temperature [K]
    P: float  #: pressure [Pa]
    rho: float  #: density [kg/m^3]
    h: float  #: specific enthalpy [J/kg]
    s: float  #: specific entropy [J/kg/K]
    P_sat: float  #: saturation pressure at T [Pa]

    @property
    def subcooling(self) -> float:
        """Pressure margin above saturation [Pa]. Positive => subcooled."""
        return self.P - self.P_sat

    @property
    def is_saturated(self) -> bool:
        """True when the feed is on (or effectively on) the saturation line."""
        return self.subcooling <= 1e-3 * max(self.P_sat, 1.0)


class PropertyBackend:
    """Interface shared by the property backends."""

    name = "abstract"

    def sat(self, T: float) -> SatState:  # pragma: no cover - interface
        raise NotImplementedError

    def P_sat(self, T: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def T_sat(self, P: float) -> float:
        """Invert the vapour-pressure curve.

        Raises ``ValueError`` if ``P`` lies outside the correlation range.
        """
        p_lo = self.P_sat(T_MIN_VALID)
        p_hi = self.P_sat(T_MAX_VALID)
        if not (p_lo <= P <= p_hi):
            raise ValueError(
                f"saturation pressure {P / 1e5:.3f} bar outside supported range "
                f"[{p_lo / 1e5:.3f}, {p_hi / 1e5:.3f}] bar"
            )
        return brentq(lambda T: self.P_sat(T) - P, T_MIN_VALID, T_MAX_VALID, xtol=1e-10)

    def upstream_state(self, T: float, P: float | None = None) -> FluidState:
        """Build the injector upstream state.

        With ``P is None`` (or ``P`` at/below saturation) the feed is taken as
        saturated liquid at ``T`` -- the self-pressurising tank case.  With
        ``P`` above saturation the liquid is subcooled/supercharged and is
        treated as an incompressible liquid, so that

        .. math:: h = h_l(T) + (P - P_{sat}) / \\rho_l,\\qquad s = s_l(T).
        """
        sat = self.sat(T)
        if P is None or P <= sat.P:
            return FluidState(T=T, P=sat.P, rho=sat.rho_l, h=sat.h_l, s=sat.s_l, P_sat=sat.P)
        h = sat.h_l + (P - sat.P) / sat.rho_l
        return FluidState(T=T, P=P, rho=sat.rho_l, h=h, s=sat.s_l, P_sat=sat.P)

    def in_range(self, T: float) -> bool:
        return T_MIN_VALID <= T <= T_MAX_VALID


class ESDUProperties(PropertyBackend):
    """ESDU 91022 curve fits -- identical to HRAP MATLAB's ``NOX.m``."""

    name = "ESDU 91022 (HRAP MATLAB)"

    # Vapour pressure
    _A = (-6.71893, 1.35966, -1.37790, -4.05100)
    # Saturated liquid density
    _B = (1.72328, -0.83950, 0.51060, -0.10412)
    # Saturated vapour density
    _C = (-1.00900, -6.28792, 7.50332, -7.90463, 0.629427)
    # Saturated liquid enthalpy [kJ/kg]
    _D = (-200.0, 116.043, -917.225, 794.779, -589.587)
    # Saturated vapour enthalpy [kJ/kg]
    _E = (-200.0, 440.055, -459.701, 434.081, -485.338)
    # Saturated liquid isobaric specific heat [kJ/kg/K]
    _F = (2.49973, 0.023454, -3.80136, 13.0945, -14.5180)

    def __init__(self) -> None:
        self._s_grid_T, self._s_grid_val = self._build_entropy()

    # -- individual correlations -------------------------------------------

    def P_sat(self, T: float) -> float:
        Tr = T / T_CRIT
        t = 1.0 - Tr
        a1, a2, a3, a4 = self._A
        return P_CRIT * np.exp(
            (1.0 / Tr) * (a1 * t + a2 * t**1.5 + a3 * t**2.5 + a4 * t**5)
        )

    def rho_liquid(self, T: float) -> float:
        t = 1.0 - T / T_CRIT
        b1, b2, b3, b4 = self._B
        return RHO_CRIT * np.exp(
            b1 * t ** (1 / 3) + b2 * t ** (2 / 3) + b3 * t + b4 * t ** (4 / 3)
        )

    def rho_vapour(self, T: float) -> float:
        u = T_CRIT / T - 1.0
        c1, c2, c3, c4, c5 = self._C
        return RHO_CRIT * np.exp(
            c1 * u ** (1 / 3)
            + c2 * u ** (2 / 3)
            + c3 * u
            + c4 * u ** (4 / 3)
            + c5 * u ** (5 / 3)
        )

    def h_liquid(self, T: float) -> float:
        """Saturated liquid enthalpy [J/kg] (ESDU datum, offset is arbitrary)."""
        t = 1.0 - T / T_CRIT
        d1, d2, d3, d4, d5 = self._D
        return 1e3 * (d1 + d2 * t ** (1 / 3) + d3 * t ** (2 / 3) + d4 * t + d5 * t ** (4 / 3))

    def h_vapour(self, T: float) -> float:
        """Saturated vapour enthalpy [J/kg] (same datum as :meth:`h_liquid`)."""
        t = 1.0 - T / T_CRIT
        e1, e2, e3, e4, e5 = self._E
        return 1e3 * (e1 + e2 * t ** (1 / 3) + e3 * t ** (2 / 3) + e4 * t + e5 * t ** (4 / 3))

    def cp_liquid(self, T: float) -> float:
        t = 1.0 - T / T_CRIT
        f1, f2, f3, f4, f5 = self._F
        return 1e3 * f1 * (1.0 + f2 / t + f3 * t + f4 * t**2 + f5 * t**3)

    def Z_vapour(self, T: float) -> float:
        return self.P_sat(T) / (self.rho_vapour(T) * R_N2O * T)

    # -- entropy reconstruction --------------------------------------------

    def _build_entropy(self, n: int = 20001):
        """Reconstruct saturated-liquid entropy *from the ESDU enthalpy fit*.

        The HEM solver evaluates ``h_1 - h_2`` for a near-isentropic expansion
        of saturated liquid.  That difference is a small residual between two
        much larger quantities -- typically a few kJ/kg out of ~50 kJ/kg -- so
        it is only meaningful if ``h`` and ``s`` come from *mutually
        consistent* correlations.  Integrating ``cp_l`` independently of the
        ESDU enthalpy fit introduces a discrepancy of the same order as the
        residual itself, which drives the computed flux to zero.

        Instead, entropy is derived from the enthalpy fit via the exact
        relation ``T ds = dh - v dP`` applied along the saturation line:

        .. math::
            \\frac{ds_l}{dT} = \\frac{1}{T}
                \\left( \\frac{dh_l}{dT} - \\frac{1}{\\rho_l}\\frac{dP_{sat}}{dT} \\right)

        so ``s_l`` inherits exactly the temperature dependence that ``h_l``
        already encodes.  The additive datum is arbitrary and cancels in HEM,
        which uses only ``s_1 - s_l(T_2)`` and ``s_fg = h_fg / T``.

        Saturated vapour entropy then follows exactly from ``s_v = s_l +
        h_fg / T``, inheriting the ESDU latent heat directly.
        """
        T = np.linspace(T_MIN_VALID, T_MAX_VALID, n)
        h_l = np.array([self.h_liquid(t) for t in T])
        P = np.array([self.P_sat(t) for t in T])
        rho_l = np.array([self.rho_liquid(t) for t in T])

        dh_dT = np.gradient(h_l, T)
        dP_dT = np.gradient(P, T)
        ds_dT = (dh_dT - dP_dT / rho_l) / T

        ds = 0.5 * (ds_dT[1:] + ds_dT[:-1]) * np.diff(T)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        return T, s

    def s_liquid(self, T: float) -> float:
        return float(np.interp(T, self._s_grid_T, self._s_grid_val))

    def s_vapour(self, T: float) -> float:
        return self.s_liquid(T) + (self.h_vapour(T) - self.h_liquid(T)) / T

    # -- assembled state ----------------------------------------------------

    def sat(self, T: float) -> SatState:
        h_l = self.h_liquid(T)
        h_v = self.h_vapour(T)
        s_l = self.s_liquid(T)
        return SatState(
            T=T,
            P=self.P_sat(T),
            rho_l=self.rho_liquid(T),
            rho_v=self.rho_vapour(T),
            h_l=h_l,
            h_v=h_v,
            s_l=s_l,
            s_v=s_l + (h_v - h_l) / T,
            cp_l=self.cp_liquid(T),
            Z=self.Z_vapour(T),
        )


class CoolPropProperties(PropertyBackend):
    """CoolProp reference EOS -- matches HRAP's Python implementation."""

    name = "CoolProp (HRAP Python)"

    def __init__(self, fluid: str = "NitrousOxide") -> None:
        try:
            import CoolProp.CoolProp as CP
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CoolProp is required for the CoolProp property backend. "
                "Install it with 'pip install CoolProp', or select the ESDU backend."
            ) from exc
        self._CP = CP
        self._fluid = fluid

    def P_sat(self, T: float) -> float:
        return float(self._CP.PropsSI("P", "T", T, "Q", 0, self._fluid))

    def sat(self, T: float) -> SatState:
        CP, f = self._CP, self._fluid
        h_l = float(CP.PropsSI("H", "T", T, "Q", 0, f))
        h_v = float(CP.PropsSI("H", "T", T, "Q", 1, f))
        return SatState(
            T=T,
            P=float(CP.PropsSI("P", "T", T, "Q", 0, f)),
            rho_l=float(CP.PropsSI("D", "T", T, "Q", 0, f)),
            rho_v=float(CP.PropsSI("D", "T", T, "Q", 1, f)),
            h_l=h_l,
            h_v=h_v,
            s_l=float(CP.PropsSI("S", "T", T, "Q", 0, f)),
            s_v=float(CP.PropsSI("S", "T", T, "Q", 1, f)),
            cp_l=float(CP.PropsSI("CPMASS", "T", T, "Q", 0, f)),
            Z=float(CP.PropsSI("Z", "T", T, "Q", 1, f)),
        )


class CachedProperties(PropertyBackend):
    """Temperature-quantising cache in front of any backend.

    Saturated-property evaluation dominates the cost of a blowdown sweep (the
    HEM solver asks for a saturation state at every trial exit pressure).  Tank
    temperature moves slowly and smoothly, so rounding ``T`` to ``resolution``
    before the lookup costs far less accuracy than the underlying correlation
    fit while removing almost all repeat evaluation.
    """

    def __init__(self, backend: PropertyBackend, resolution: float = 1e-3, maxsize: int = 200_000):
        self._backend = backend
        self._res = resolution
        self.name = backend.name
        self._sat_cached = lru_cache(maxsize=maxsize)(self._sat_quantised)
        self._psat_cached = lru_cache(maxsize=maxsize)(self._psat_quantised)
        self._tsat_grid_P: list[float] | None = None
        self._tsat_grid_T: list[float] | None = None

    def _sat_quantised(self, key: int) -> SatState:
        return self._backend.sat(key * self._res)

    def _psat_quantised(self, key: int) -> float:
        return self._backend.P_sat(key * self._res)

    def sat(self, T: float) -> SatState:
        return self._sat_cached(int(round(T / self._res)))

    def P_sat(self, T: float) -> float:
        return self._psat_cached(int(round(T / self._res)))

    def T_sat(self, P: float) -> float:
        """Invert P_sat by interpolation on a precomputed monotone table.

        Root-finding per call is the single hottest path in the HEM solver;
        the vapour-pressure curve is smooth and monotone, so a dense table with
        linear interpolation is both far faster and accurate to well under the
        correlation's own uncertainty.
        """
        if self._tsat_grid_P is None:
            Ts = np.linspace(T_MIN_VALID, T_MAX_VALID, 6001)
            Ps = np.array([self._backend.P_sat(t) for t in Ts])
            self._tsat_grid_T = Ts
            self._tsat_grid_P = Ps
        Ps, Ts = self._tsat_grid_P, self._tsat_grid_T
        if P <= Ps[0]:
            return float(Ts[0])
        if P >= Ps[-1]:
            return float(Ts[-1])
        return float(np.interp(P, Ps, Ts))

    def in_range(self, T: float) -> bool:
        return self._backend.in_range(T)


def get_backend(name: str = "esdu", cached: bool = True) -> PropertyBackend:
    """Factory: ``'esdu'`` (HRAP MATLAB parity) or ``'coolprop'`` (HRAP Python)."""
    key = name.strip().lower()
    if key in ("esdu", "esdu91022", "hrap", "hrap-matlab"):
        backend: PropertyBackend = ESDUProperties()
    elif key in ("coolprop", "cp", "hrap-python", "nist"):
        backend = CoolPropProperties()
    else:
        raise ValueError(f"unknown property backend {name!r} (expected 'esdu' or 'coolprop')")
    return CachedProperties(backend) if cached else backend


def coolprop_available() -> bool:
    try:
        import CoolProp  # noqa: F401
    except ImportError:
        return False
    return True
