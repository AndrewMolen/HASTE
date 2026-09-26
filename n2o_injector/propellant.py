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
#:
#: .. warning::
#:    These are almost certainly **not paraffin coefficients**. The pair
#:    ``a = 0.0304, n = 0.681`` matches, to three significant figures on both
#:    parameters, the *HTPB (Thiokol)* entry of Table 6.2 in Karp & Jens,
#:    *Hybrid Rocket Propulsion Design Handbook* (Elsevier, 2024) -- whose SI
#:    coefficient is 3.04e-5, i.e. 0.0304 in the mm/s convention used here.
#:    HRAP also ships identical coefficients for ABS, asphalt and HTPB, which
#:    is the signature of a placeholder rather than a measured fuel property.
#:
#:    The values are kept **unchanged on purpose**: they are what reproduces an
#:    HRAP run, and HRAP cross-referencing is a validated capability of this
#:    tool. Do not "fix" them here -- pick a different preset instead, and see
#:    ``FUEL_PRESETS`` for published alternatives with their provenance.
PARAFFIN_DEFAULTS = {
    "name": "Paraffin (HRAP default)",
    "reg_a": 0.0304,
    "reg_n": 0.681,
    "reg_m": 0.0,
    "rho_fuel": 900.0,
    "opt_OF": 8.27,
}

#: The regression pair that is really an HTPB dataset. Used by the sizing
#: warning to detect that a design rests on the placeholder.
PLACEHOLDER_REG = (0.0304, 0.681)


def is_placeholder_regression(reg_a: float, reg_n: float, tol: float = 1e-4) -> bool:
    """True if these coefficients are HRAP's mislabelled HTPB pair."""
    return (abs(reg_a - PLACEHOLDER_REG[0]) < tol
            and abs(reg_n - PLACEHOLDER_REG[1]) < tol)


#: Selectable fuels, each carrying its provenance.
#:
#: Every entry states the oxidiser it was fitted against, the oxidiser flux
#: range it was fitted over, and the fuel it actually describes. A bare
#: ``(a, n)`` pair with no provenance is exactly how HRAP's HTPB coefficients
#: came to be used as paraffin for years, so the metadata is not decoration.
#:
#: Units follow this tool's convention throughout:
#: ``rdot [m/s] = 0.001 * a * G_ox^n``, with ``G_ox`` in kg/m^2/s. Sources that
#: publish ``a`` in SI (rdot in m/s) are converted by x1000. Note that the
#: units of ``a`` depend on ``n``, so two fits with different ``n`` cannot be
#: compared by their ``a`` values alone -- compare regression rate at a matched
#: flux instead (Karp & Jens section 6.5.2).
FUEL_PRESETS = {
    "Paraffin (HRAP - HTPB coeffs)": {
        **PARAFFIN_DEFAULTS,
        "oxidiser": "unspecified",
        "flux_range": None,
        "composition": "labelled paraffin; coefficients match HTPB (Thiokol)",
        "source": "HRAP Paraffin.mat; = Karp & Jens Table 6.2 HTPB (Thiokol)",
        "note": "PLACEHOLDER. Keep for HRAP cross-referencing; do not design on it.",
    },
    "Paraffin SP1A (Karp 6.2)": {
        "name": "Paraffin SP1A", "reg_a": 0.117, "reg_n": 0.620, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 8.27,
        # Probably GOX/O2, NOT N2O. Doran et al. (AIAA 2007-5352) Table 1
        # reports N2O coefficients for HDPE, PMMA and HTPB in g/cm^2-s; convert
        # by a_SI = a * 10^-n / 1000 and all three match Table 6.2 exactly on
        # both parameters (1.16e-4/0.331, 1.31e-4/0.335, 1.88e-4/0.347). That
        # locates the N2O block in the table. SP1A is its *first* row, far from
        # that block, and figure 6.6 -- "HTPB combusting with O2", sourced from
        # Table 6.2 -- points at the early rows. Doran also explicitly does not
        # report SP1A coefficients (only 3 tests), so the 65-test row is from
        # elsewhere. Peregrine burning SP1A on N2O is not enough to overcome
        # the block ordering.
        "oxidiser": "probably GOX/O2, NOT N2O",
        "flux_range": None,
        "composition": "neat paraffin, SP1A (Stanford; the Peregrine fuel)",
        "source": "Karp & Jens Table 6.2 (65 tests)",
        "note": "Sits in what appears to be the O2 block of Table 6.2. Note "
                "n differs strongly by oxidiser, so a GOX fit is the wrong "
                "thing to borrow for an N2O motor. Prefer the SP7 entries.",
    },
    "Paraffin FR5560 (Karp 6.2)": {
        "name": "Paraffin FR5560", "reg_a": 0.169, "reg_n": 0.600, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 8.27,
        "oxidiser": "UNCONFIRMED",
        "flux_range": None,
        "composition": "neat paraffin, FR5560",
        "source": "Karp & Jens Table 6.2 (4 tests)",
        "note": "Oxidiser unconfirmed, as above. Only 4 tests.",
    },
    # The two SP7 rows sit immediately after the three rows proven to be
    # Doran's N2O data, i.e. inside the same merged oxidiser cell. SP7 is the
    # Stanford/Karabeyoglu paraffin developed for the Mars Ascent Vehicle. The
    # two fits disagree sharply and the table gives no basis to choose, so both
    # are offered rather than one being picked arbitrarily.
    "Paraffin SP7 a (Karp 6.2)": {
        "name": "Paraffin SP7 (fit a)", "reg_a": 0.0781, "reg_n": 0.545,
        "reg_m": 0.0, "rho_fuel": 900.0, "opt_OF": 8.27,
        "oxidiser": "N2O (very likely)",
        "flux_range": None,
        "composition": "neat paraffin, SP7 (Mars Ascent Vehicle fuel)",
        "source": "Karp & Jens Table 6.2 (10 tests)",
        "note": "Adjacent to the rows proven to be N2O. Two SP7 fits exist and "
                "they disagree; this is the lower one. Compare against 'SP7 b'.",
    },
    "Paraffin SP7 b (Karp 6.2)": {
        "name": "Paraffin SP7 (fit b)", "reg_a": 0.280, "reg_n": 0.297,
        "reg_m": 0.0, "rho_fuel": 900.0, "opt_OF": 8.27,
        "oxidiser": "N2O (very likely)",
        "flux_range": None,
        "composition": "neat paraffin, SP7 (Mars Ascent Vehicle fuel)",
        "source": "Karp & Jens Table 6.2 (8 tests)",
        "note": "The other SP7 fit. Much flatter (n = 0.297) and much higher a. "
                "The spread between the two SP7 rows is itself the uncertainty.",
    },
    # The only row in Table 6.2 whose oxidiser is *proven* rather than
    # inferred: it matches Doran et al. AIAA 2007-5352 Table 1 exactly. Kept as
    # a sanity anchor -- not a fuel you are likely to burn.
    "HTPB (N2O, Doran 2007)": {
        "name": "HTPB (N2O, Stanford)", "reg_a": 0.1876, "reg_n": 0.347,
        "reg_m": 0.0, "rho_fuel": 920.0, "opt_OF": 7.95,
        "oxidiser": "N2O (CONFIRMED)",
        "flux_range": (30.0, 350.0),
        "composition": "HTPB",
        "source": "Doran et al., AIAA 2007-5352 Table 1 (15 tests)",
        "note": "Oxidiser confirmed from the source paper, not inferred. "
                "Published as a = 0.417 in g/cm2-s; converted here by "
                "a x 10^-n. Useful as a known-good N2O anchor.",
    },
    "Paraffin+Al/Mg, N2O (Liu 2020)": {
        "name": "Paraffin-HTPB-Al/Mg (N2O)",
        "reg_a": 0.0876, "reg_n": 0.3953, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 5.3,
        "oxidiser": "N2O",
        "flux_range": (91.0, 242.0),
        "composition": "15% HTPB matrix, 65% paraffin, 4% PE, 5% Mg, 10% Al, "
                       "copper chromite catalyst",
        "source": "Liu et al., Aerosp. Sci. Technol. 107 (2020) 106269",
        "note": "The only N2O-confirmed fit here, but NOT neat paraffin: metal "
                "loaded and HTPB bound, so it regresses slower than neat "
                "paraffin and its rate depends on chamber pressure. rho_fuel "
                "is NOT the measured value for this composition.",
    },
    "Paraffin+Al/Mg, GOX (Liu 2020)": {
        "name": "Paraffin-HTPB-Al/Mg (GOX)",
        "reg_a": 0.0431, "reg_n": 0.7232, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 1.9,
        "oxidiser": "GOX",
        "flux_range": (24.9, 58.5),
        "composition": "as above",
        "source": "Liu et al., Aerosp. Sci. Technol. 107 (2020) 106269",
        "note": "Same fuel as the N2O entry. Included to show how strongly the "
                "exponent depends on oxidiser: n = 0.723 here vs 0.395 on N2O.",
    },
    "HTPB/Paraffin 50/50 (HRAP)": {
        "name": "50P (HRAP)", "reg_a": 0.1146, "reg_n": 0.5036, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 7.893,
        "oxidiser": "unspecified", "flux_range": None,
        "composition": "50/50 HTPB/paraffin", "source": "HRAP 50P.mat",
        "note": "",
    },
    "HTPB (HRAP)": {
        "name": "HTPB (HRAP)", "reg_a": 0.198, "reg_n": 0.325, "reg_m": 0.0,
        "rho_fuel": 900.0, "opt_OF": 7.95,
        "oxidiser": "unspecified", "flux_range": None,
        "composition": "HTPB", "source": "HRAP HTPB.mat",
        "note": "",
    },
}


#: Preset selected on startup. Deliberately still the HRAP pair: it is what
#: reproduces an HRAP run, and changing the out-of-box numbers would silently
#: alter every existing configuration. The GUI shows its provenance warning
#: next to the selector instead.
DEFAULT_FUEL_PRESET = "Paraffin (HRAP - HTPB coeffs)"


def preset_provenance(key: str) -> str:
    """One-line provenance for a preset, for display next to the selector."""
    p = FUEL_PRESETS.get(key)
    if not p:
        return ""
    bits = [f"oxidiser: {p.get('oxidiser', '?')}"]
    fr = p.get("flux_range")
    if fr:
        bits.append(f"fitted over G_ox {fr[0]:.0f}-{fr[1]:.0f} kg/m2/s")
    if p.get("source"):
        bits.append(p["source"])
    line = "  |  ".join(bits)
    return f"{line}\n{p['note']}" if p.get("note") else line


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
