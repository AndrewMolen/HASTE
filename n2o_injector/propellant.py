"""Propellant configuration, including HRAP ``.mat`` interoperability.

HRAP stores each propellant as a MATLAB struct with the fields defined in
``propellant_configs/propellant_template.m``:

===============  =========================================================
``prop_Pc``      1xm chamber pressures [Pa] spanning the CEA table
``prop_OF``      nx1 O/F ratios spanning the CEA table
``prop_k``       ratio of specific heats on the (OF, Pc) grid
``prop_M``       exhaust molar mass [g/mol] on the same grid
``prop_T``       adiabatic flame temperature [K] on the same grid
``prop_Reg``     ``[a, n, m]`` regression coefficients
``prop_Rho``     solid fuel density [kg/m^3]
``opt_OF``       optimum O/F ratio
``prop_nm``      display name
===============  =========================================================

Regression follows HRAP's ``shift_OF.m`` / ``d_grain_shiftOF``:

.. math:: \\dot r \\,[\\mathrm{m/s}] = 0.001\\, a\\, G_{ox}^{\\,n}\\, L^{m}

with the oxidiser mass flux ``G_ox`` in kg/m^2/s and grain length ``L`` in m
(the 0.001 converts the mm/s in which ``a`` is expressed).  Note that HRAP
uses the *oxidiser* flux, not the total flux.

Characteristic velocity is reconstructed exactly as HRAP's ``comb.m`` does:

.. math:: c^* = \\eta \\sqrt{\\frac{R T_c}{k\\,(2/(k+1))^{(k+1)/(k-1)}}},
          \\quad R = 8314.5 / M
"""

from __future__ import annotations

import bisect
import math
import os
from dataclasses import dataclass, field

import numpy as np

#: HRAP's paraffin configuration (``propellant_configs/Paraffin.mat``).
#: Regression coefficients are ``[a, n, m]`` with ``a`` in mm/s.
PARAFFIN_DEFAULTS = {
    "name": "Paraffin (HRAP default)",
    "reg_a": 0.0304,
    "reg_n": 0.681,
    "reg_m": 0.0,
    "rho_fuel": 900.0,
    "opt_OF": 8.27,
}

#: A few other HRAP-shipped fuels, for reference in the GUI.
FUEL_PRESETS = {
    "Paraffin": dict(PARAFFIN_DEFAULTS),
    "HTPB/Paraffin 50/50": {
        "name": "50P (HRAP)", "reg_a": 0.1146, "reg_n": 0.5036, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 7.893,
    },
    "HTPB": {
        "name": "HTPB (HRAP)", "reg_a": 0.198, "reg_n": 0.325, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 7.95,
    },
}


@dataclass
class Propellant:
    """Fuel regression law plus an optional CEA table for ``c*``."""

    name: str = PARAFFIN_DEFAULTS["name"]
    reg_a: float = PARAFFIN_DEFAULTS["reg_a"]  #: mm/s
    reg_n: float = PARAFFIN_DEFAULTS["reg_n"]
    reg_m: float = PARAFFIN_DEFAULTS["reg_m"]
    rho_fuel: float = PARAFFIN_DEFAULTS["rho_fuel"]  #: kg/m^3
    opt_OF: float = PARAFFIN_DEFAULTS["opt_OF"]

    #: Constant c* fallback [m/s], used when no CEA table is loaded.
    cstar_const: float = 1550.0
    cstar_eff: float = 0.95

    # CEA table (optional)
    OF_grid: np.ndarray | None = field(default=None, repr=False)
    Pc_grid: np.ndarray | None = field(default=None, repr=False)
    k_grid: np.ndarray | None = field(default=None, repr=False)
    M_grid: np.ndarray | None = field(default=None, repr=False)
    T_grid: np.ndarray | None = field(default=None, repr=False)

    @property
    def has_table(self) -> bool:
        return self.OF_grid is not None and self.k_grid is not None

    # -- regression ---------------------------------------------------------

    def regression_rate(self, G_ox: float, L: float) -> float:
        """Radial regression rate [m/s] from oxidiser mass flux [kg/m^2/s]."""
        if G_ox <= 0.0:
            return 0.0
        return 1e-3 * self.reg_a * G_ox**self.reg_n * L**self.reg_m

    # -- combustion ---------------------------------------------------------

    def _axes(self):
        """Cache the table axes and grids as plain Python lists.

        This interpolation runs several times per timestep inside the chamber
        pressure root-find, always on *scalars*.  NumPy's scalar path
        (``np.clip``/``np.searchsorted`` on floats) dominated the simulation
        runtime, so the axes are held as Python lists and the arithmetic is
        done with builtins, which is roughly an order of magnitude faster here.
        """
        cache = getattr(self, "_axis_cache", None)
        if cache is None:
            cache = {
                "OF": [float(v) for v in self.OF_grid],
                "Pc": [float(v) for v in self.Pc_grid],
                "k": self.k_grid.tolist(),
                "M": self.M_grid.tolist(),
                "T": self.T_grid.tolist(),
            }
            object.__setattr__(self, "_axis_cache", cache)
        return cache

    def _interp2(self, which: str, OF: float, Pc: float) -> float:
        """Bilinear interpolation on the (OF, Pc) CEA grid, with clamping.

        HRAP's ``interp2x`` clamps to the table edges rather than extrapolating;
        the same is done here so that off-nominal transients (ignition, tail-off)
        stay bounded instead of producing nonsense ``c*``.
        """
        cache = self._axes()
        of_ax, pc_ax, grid = cache["OF"], cache["Pc"], cache[which]

        OF = of_ax[0] if OF < of_ax[0] else (of_ax[-1] if OF > of_ax[-1] else OF)
        Pc = pc_ax[0] if Pc < pc_ax[0] else (pc_ax[-1] if Pc > pc_ax[-1] else Pc)

        i = bisect.bisect_right(of_ax, OF) - 1
        j = bisect.bisect_right(pc_ax, Pc) - 1
        i = 0 if i < 0 else (len(of_ax) - 2 if i > len(of_ax) - 2 else i)
        j = 0 if j < 0 else (len(pc_ax) - 2 if j > len(pc_ax) - 2 else j)

        of0, of1 = of_ax[i], of_ax[i + 1]
        pc0, pc1 = pc_ax[j], pc_ax[j + 1]
        tu = 0.0 if of1 == of0 else (OF - of0) / (of1 - of0)
        tv = 0.0 if pc1 == pc0 else (Pc - pc0) / (pc1 - pc0)
        r0, r1 = grid[i], grid[i + 1]
        return (
            r0[j] * (1 - tu) * (1 - tv)
            + r1[j] * tu * (1 - tv)
            + r0[j + 1] * (1 - tu) * tv
            + r1[j + 1] * tu * tv
        )

    def combustion(self, OF: float, Pc: float) -> tuple[float, float, float]:
        """Return ``(k, M, T_flame)`` at the given operating point."""
        if not self.has_table:
            return 1.25, 30.0, 3000.0
        return (
            self._interp2("k", OF, Pc),
            self._interp2("M", OF, Pc),
            self._interp2("T", OF, Pc),
        )

    def cstar(self, OF: float, Pc: float) -> float:
        """Characteristic velocity [m/s], including ``cstar_eff``."""
        if not self.has_table:
            return self.cstar_eff * self.cstar_const
        k, M, T = self.combustion(OF, Pc)
        R = 8314.5 / M
        return self.cstar_eff * math.sqrt(
            (R * T) / (k * (2.0 / (k + 1.0)) ** ((k + 1.0) / (k - 1.0)))
        )

    def gamma(self, OF: float, Pc: float) -> float:
        if not self.has_table:
            return 1.25
        return self._interp2("k", OF, Pc)

    # -- HRAP interop -------------------------------------------------------

    @classmethod
    def from_hrap_mat(cls, path: str, cstar_eff: float = 0.95) -> "Propellant":
        """Load an HRAP propellant configuration (``.mat``).

        Accepts both the struct layout HRAP saves (``s.prop_*``) and a flat
        layout where the ``prop_*`` variables sit at the top level.
        """
        from scipy.io import loadmat

        raw = loadmat(path)
        if "s" in raw:
            s = raw["s"][0, 0]
            get = lambda key: s[key]  # noqa: E731
            names = set(s.dtype.names or ())
        else:
            get = lambda key: raw[key]  # noqa: E731
            names = set(raw.keys())

        required = {"prop_OF", "prop_Pc", "prop_k", "prop_M", "prop_T"}
        missing = required - names
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} is not an HRAP propellant config "
                f"(missing {', '.join(sorted(missing))})"
            )

        OF = np.asarray(get("prop_OF"), dtype=float).ravel()
        Pc = np.asarray(get("prop_Pc"), dtype=float).ravel()

        def orient(g):
            """Return the grid as (len(OF), len(Pc)).

            HRAP's template documents the CEA arrays as (OF, Pc) but the shipped
            ``.mat`` files store them transposed, so the orientation is resolved
            from the array shape rather than trusted.
            """
            g = np.asarray(g, dtype=float)
            if g.shape == (len(OF), len(Pc)):
                return g
            if g.shape == (len(Pc), len(OF)):
                return g.T
            raise ValueError(
                f"CEA grid shape {g.shape} matches neither (OF, Pc) = "
                f"{(len(OF), len(Pc))} nor its transpose"
            )

        reg = np.asarray(get("prop_Reg"), dtype=float).ravel() if "prop_Reg" in names else None
        rho = float(np.asarray(get("prop_Rho")).ravel()[0]) if "prop_Rho" in names else 900.0
        opt = float(np.asarray(get("opt_OF")).ravel()[0]) if "opt_OF" in names else 8.0
        if "prop_nm" in names:
            nm = np.asarray(get("prop_nm")).ravel()
            name = str(nm[0]) if nm.size else os.path.basename(path)
        else:
            name = os.path.splitext(os.path.basename(path))[0]

        # Sort ascending; HRAP's interpolators assume monotone axes.
        io, ip = np.argsort(OF), np.argsort(Pc)
        k_g, M_g, T_g = (orient(get(f"prop_{v}"))[np.ix_(io, ip)] for v in ("k", "M", "T"))

        return cls(
            name=name,
            reg_a=float(reg[0]) if reg is not None and reg.size > 0 else PARAFFIN_DEFAULTS["reg_a"],
            reg_n=float(reg[1]) if reg is not None and reg.size > 1 else PARAFFIN_DEFAULTS["reg_n"],
            reg_m=float(reg[2]) if reg is not None and reg.size > 2 else 0.0,
            rho_fuel=rho,
            opt_OF=opt,
            cstar_eff=cstar_eff,
            OF_grid=OF[io],
            Pc_grid=Pc[ip],
            k_grid=k_g,
            M_grid=M_g,
            T_grid=T_g,
        )
