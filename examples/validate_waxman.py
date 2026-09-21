"""External validation against NASA/Stanford cold-flow N2O injector data.

Source
------
B. S. Waxman, J. E. Zimmerman, B. J. Cantwell (Stanford University) and
G. G. Zilliac (NASA Ames Research Center), *Mass Flow Rate and Isolation
Characteristics of Injectors for Use with Self-Pressurizing Oxidizers in
Hybrid Rockets*, NTRS 20190001326.

This is the first check in this project against **real measured data** rather
than internal consistency or agreement with HRAP's equations. The paper's
models are the same ones implemented here, and its equations were used to
confirm the implementations line-for-line:

* SPI      -- paper Eqn. (2):  mdot = Cd A sqrt(2 rho dP)
* HEM      -- paper Eqns. (3)-(5): mdot = Cd A rho2 sqrt(2(h1-h2)), s1 = s2,
              with the critical flow found by maximising over back pressure
* Dyer     -- paper Eqns. (8)-(9): kappa = sqrt((P1-P2)/(Pv-P2)), and
              mdot = kappa/(1+kappa) mdot_SPI + 1/(1+kappa) mdot_HEM

Test case
---------
Injector 3: straight hole, D = 1.50 mm, L = 18.4 mm (L/D = 12.3), rounded
inlet. Nitrous oxide, upstream P1 = 704 psia, T1 = 280 K, supercharged
(subcooled) by 169 psi. Measured SPI discharge coefficient Cd = 0.77.

.. warning::
   The reference values below are quoted **in the paper's prose and figure
   captions**, not read from a data table -- the full dataset lives in an
   appendix of a previous work by the same authors which is not to hand. The
   effective Cd of "approximately 0.6" is explicitly approximate, so treat the
   comparison as accurate to a few percent at best, not to the last digit.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2o_injector import (  # noqa: E402
    FlowModel,
    SaturationTable,
    get_backend,
    mass_flux,
    recommend_model,
)

PSI = 101325.0 / 14.696  # the paper's psi, consistent with HRAP's definition

# ---- Test conditions (Waxman et al., injector 3, nitrous oxide) ----------
T1 = 280.0            # K
P1 = 704.0 * PSI      # Pa, absolute
D = 1.50e-3           # m
L = 18.4e-3           # m
CD_SPI = 0.77         # measured single-phase discharge coefficient

# ---- Reference values quoted in the paper --------------------------------
REF_SUPERCHARGE_PSI = 169.0   # stated upstream supercharge
REF_SPI_UPPER_PSI = 200.0     # "SPI valid for 30 psi < dP < 200 psi"
REF_CD_EFF = 0.60             # "approximately 0.6" at dP = 330 psi
REF_CD_EFF_DP_PSI = 330.0

A = 0.25 * math.pi * D**2
L_OVER_D = L / D


def main():
    props = get_backend("esdu")
    table = SaturationTable(props)
    up = props.upstream_state(T1, P1)

    print("=" * 74)
    print("  VALIDATION vs Waxman, Zimmerman, Cantwell & Zilliac (NASA/Stanford)")
    print("  Injector 3: D = 1.50 mm, L = 18.4 mm, L/D = 12.3, rounded inlet")
    print("=" * 74)

    # -- 1. Thermodynamic state -------------------------------------------
    Psat = props.P_sat(T1)
    super_psi = (P1 - Psat) / PSI
    print("\n1. UPSTREAM STATE  (paper used REFPROP; this tool uses ESDU 91022)")
    print(f"   P1                    : {P1 / PSI:7.1f} psia  ({P1 / 1e6:.3f} MPa)")
    print(f"   T1                    : {T1:7.1f} K")
    print(f"   Psat(T1), this tool   : {Psat / PSI:7.1f} psia  ({Psat / 1e6:.3f} MPa)")
    print(f"   supercharge, this tool: {super_psi:7.1f} psi")
    print(f"   supercharge, paper    : {REF_SUPERCHARGE_PSI:7.1f} psi")
    err = (super_psi - REF_SUPERCHARGE_PSI) / REF_SUPERCHARGE_PSI * 100
    print(f"   -> agreement          : {err:+.1f} % on supercharge")
    print(f"      (implies Psat agrees with REFPROP to "
          f"{abs(super_psi - REF_SUPERCHARGE_PSI) * PSI / Psat * 100:.2f} %)")

    # -- 2. Single- to two-phase transition -------------------------------
    print("\n2. TRANSITION OUT OF SINGLE-PHASE FLOW")
    print("   predicted: flashing cannot begin until P2 falls below Psat,")
    print(f"              i.e. dP > supercharge = {super_psi:.0f} psi")
    print(f"   observed : SPI model valid up to dP ~ {REF_SPI_UPPER_PSI:.0f} psi")
    print("   -> the small delay is consistent with the metastable superheat")
    print("      the paper describes: real flow does not flash the instant it")
    print("      crosses the saturation line.")

    # -- 3. Effective discharge coefficient vs dP -------------------------
    print("\n3. EFFECTIVE DISCHARGE COEFFICIENT")
    print("   Cd_eff = mdot / (A sqrt(2 rho dP))   [paper's definition, Cd = 1]")
    print(f"   predicted Cd_eff = {CD_SPI:.2f} * G_model / G_SPI\n")
    print(f"   {'dP [psi]':>9} {'P2 [MPa]':>9} {'SPI':>8} {'HEM':>8} {'Dyer':>8} {'kappa':>8}")
    print("   " + "-" * 56)
    for dP_psi in (50, 100, 166, 200, 250, 300, 330, 400, 500, 600):
        dP = dP_psi * PSI
        P2 = P1 - dP
        if P2 <= 0:
            continue
        r = mass_flux(table, up, P2, FlowModel.DYER, L_over_D=L_OVER_D)
        k = "inf" if math.isinf(r.kappa) else f"{r.kappa:.3f}"
        mark = "  <-- paper" if dP_psi == REF_CD_EFF_DP_PSI else ""
        print(f"   {dP_psi:9.0f} {P2 / 1e6:9.3f} {CD_SPI:8.3f} "
              f"{CD_SPI * r.G_hem / r.G_spi:8.3f} {CD_SPI * r.G / r.G_spi:8.3f} "
              f"{k:>8}{mark}")

    # -- 4. Quantitative comparison at the paper's stated point -----------
    P2 = P1 - REF_CD_EFF_DP_PSI * PSI
    r = mass_flux(table, up, P2, FlowModel.DYER, L_over_D=L_OVER_D)
    preds = {
        "SPI": CD_SPI,
        "HEM": CD_SPI * r.G_hem / r.G_spi,
        "Dyer (NHNE)": CD_SPI * r.G / r.G_spi,
    }
    print(f"\n4. COMPARISON AT dP = {REF_CD_EFF_DP_PSI:.0f} psi "
          f"(P2 = {P2 / 1e6:.3f} MPa)")
    print(f"   measured Cd_eff ~ {REF_CD_EFF:.2f} (paper states 'approximately 0.6')\n")
    print(f"   {'model':>14} {'Cd_eff':>9} {'error':>9} {'mdot [g/s]':>12}")
    print("   " + "-" * 47)
    for name, cd in preds.items():
        e = (cd - REF_CD_EFF) / REF_CD_EFF * 100
        print(f"   {name:>14} {cd:9.3f} {e:+8.1f}% {cd * A * r.G_spi * 1e3:12.2f}")
    print(f"   {'measured':>14} {REF_CD_EFF:9.3f} {'--':>9} "
          f"{REF_CD_EFF * A * r.G_spi * 1e3:12.2f}")

    # -- 5. Model selection guidance --------------------------------------
    model, text = recommend_model(L_OVER_D)
    print("\n5. MODEL SELECTION")
    print(f"   this tool recommends : {model.value}")
    print(f"   reason               : {text}")
    best = min(preds, key=lambda k: abs(preds[k] - REF_CD_EFF))
    print(f"   closest to measured  : {best} "
          f"({(preds[best] - REF_CD_EFF) / REF_CD_EFF * 100:+.1f} %)")

    print("\n6. INTERPRETATION")
    print("   HEM under-predicts, which is what theory requires: the paper states")
    print("   equilibrium 'gives a lower-bound estimate for the critical mass flow")
    print("   rate', because real flashing is delayed by metastability. Seeing HEM")
    print("   land below the data, and SPI well above it, is the expected ordering.")
    print("   For this long orifice (L/D = 12.3) HEM is the better predictor, which")
    print("   matches both the paper's reasoning and this tool's own L/D guidance.")
    print("=" * 74)

    return preds, r


if __name__ == "__main__":
    main()
