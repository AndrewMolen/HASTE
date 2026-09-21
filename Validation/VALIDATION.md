# Validation status, roadmap, and known limitations

Read this before using the tool for hardware decisions.

**Bottom line:** the injector flow models have now been checked against **real
measured N2O cold-flow data** (§1.4) and land where theory says they should.
The tool has still never been compared against a static fire or a real HRAP
run, and the *motor* side — regression, blowdown, chamber — remains unvalidated
against hardware. Trust the injector models to a few percent for orifices
similar to the tested ones; do not yet trust predicted burn time, O/F history,
or impulse without your own firing data.

---

## 1. What was built, and how it was verified

### Workflow followed

1. **Read HRAP's actual source** rather than assuming its behaviour — the MATLAB
   `util/*.m`, the Python `hrap/*.py`, the `HRAP.mlapp` GUI internals, and the
   theory PDF that ships in the repo. This established: HRAP's injector is SPI
   only; its N2O properties are ESDU 91022; its CEA tables come from RPA; its
   regression law uses oxidiser flux; and its exact unit and state-flag
   conventions.
2. **Implemented the fluid properties first** and cross-validated them against
   an independent reference EOS before building anything on top.
3. **Implemented the three flow models**, then checked each against the physical
   limits it must satisfy rather than against expected output values.
4. **Built the burn simulation** to HRAP's formulation so that SPI mode is a
   like-for-like reproduction.
5. **Built the sizing solver** on top, then the GUI, then HRAP import.
6. Every stage got tests before moving on; two genuine physics bugs and two
   robustness bugs were caught this way (listed in §1.3).

### Testing performed — 102 tests, all passing

```bash
python -m pytest tests/ -q
```

| Group | What is actually checked |
|---|---|
| **Fluid properties** | ESDU vs published saturated N2O at 20 °C (P, ρ_l, ρ_v to 1–2 %); critical point exact; monotonicity of the saturation curve; `T_sat` inverts `P_sat`; latent heat decreases toward critical; `s_fg = h_fg/T` exactly |
| **Cross-backend** | ESDU curve fits vs CoolProp's reference EOS over −20…+30 °C: pressure 0.1 %, ρ_l <1 %, ρ_v <3 %, h_fg <2 %, **s_fg 0.4 %**. HEM mass flux agrees to **0.05 %** between the two |
| **Flow-model limits** | SPI is exactly √(2ρΔP); SPI > Dyer > HEM for a saturated feed; κ = 1 when saturated, making Dyer the exact mean; Dyer → SPI when the feed cannot flash; more subcooling raises κ; L/D → 0 gives SPI and large L/D gives HEM; HEM chokes and plateaus; critical pressure ratio 0.72; no flow when downstream ≥ upstream; flux monotone in chamber pressure; fast precomputed curve matches the direct solver to 0.2 % |
| **HRAP parity** | SPI reproduces HRAP's `dLoss = K/(NA)²`, `ṁ = √(2ρΔP/dLoss)` to machine precision; liquid-mass split, regression law and c* formula match HRAP's; constant-O/F matches `const_OF.m` |
| **HRAP import** | Unit factors match HRAP's initialiser (incl. its `psi = 101325/14.696`); imperial and metric; both tank-volume modes; tank state as temperature or pressure; fill as mass or percentage; nozzle as ratio or exit diameter; C/K/F/R round-trip; export→import round-trip; propellant `.mat` with transposed CEA grids |
| **HRAP CSV** | Unit conversion; 13- and 15-column variants; recovery of the true injector ΔP; rejects non-HRAP files |
| **Comparison** | Identical runs → ~0 % error; an injected 10 % perturbation shows up as one |
| **Conservation** | Integrated oxidiser and fuel flow match tank/grain depletion to 2 %; port opens monotonically; tank stays on the saturation curve; chamber always below tank |
| **Failure modes** | Liquid-full and vapour-only tanks, out-of-range temperature, ports that don't fit, sub-minimum holes, n ≥ 1, degenerate designs |

Two internal consistency results worth noting, because they were not designed
in and could have failed:

- The **two chamber models converge** — quasi-steady and HRAP's transient ODE
  end at 32.04 vs 32.06 bar with total impulse within 3 %.
- The **two property backends agree on HEM flux to 0.05 %**, even though ESDU
  carries no entropy data and the entropy had to be reconstructed
  thermodynamically. That is the strongest evidence the HEM branch is right.

### 1.4 External validation against measured NASA/Stanford data

Source: B. S. Waxman, J. E. Zimmerman, B. J. Cantwell (Stanford) and
G. G. Zilliac (NASA Ames), *Mass Flow Rate and Isolation Characteristics of
Injectors for Use with Self-Pressurizing Oxidizers in Hybrid Rockets*,
NTRS 20190001326. Reproduce with:

```bash
python examples/validate_waxman.py
```

The paper implements the same three models and its equations confirm the
implementations line-for-line — SPI (Eqn. 2), HEM (Eqns. 3–5), and Dyer/NHNE
with `κ = √((P₁−P₂)/(Pᵥ−P₂))` (Eqns. 8–9), identical to what is coded here.

**Test case** — injector 3: straight hole, D = 1.50 mm, L = 18.4 mm
(L/D = 12.3), rounded inlet, N2O at P₁ = 704 psia, T₁ = 280 K, supercharged
169 psi, measured single-phase Cd = 0.77.

| Check | Paper | This tool | Result |
|---|---|---|---|
| Supercharge (pins Psat via REFPROP) | 169 psi | 166 psi | **−2.0 %**, implying Psat agrees to **0.63 %** |
| SPI valid up to | ΔP ≈ 200 psi | onset at ΔP = 166 psi | correct direction — real flow flashes *late* (metastable) |
| Cd_eff at ΔP = 330 psi | ≈ 0.60 | HEM 0.545 / Dyer 0.677 / SPI 0.770 | **HEM −9.1 %**, Dyer +12.8 %, SPI +28.3 % |
| Critical (choked) flow observed | yes | yes | ✓ |

**What this establishes:**

- The measurement is **bracketed by SPI above and HEM below**, which is the
  ordering the physics demands. A result outside that bracket would have meant
  a coding error.
- **HEM under-predicts by 9 %** — and it must. The paper states equilibrium
  "gives a lower-bound estimate for the critical mass flow rate" because real
  flashing is delayed by metastable superheat. Landing just below the data is
  the correct behaviour, not a defect.
- **SPI over-predicts by 28 %.** This is the paper's central practical point,
  and it is exactly why HRAP's SPI-only liquid model will over-predict oxidiser
  flow for a flashing injector.
- **The tool's own L/D guidance is confirmed by the data.** It recommends HEM
  for L/D > 10; at L/D = 12.3, HEM is indeed the closest model. That guidance
  was written from the literature before this comparison was run.

**Limits of this check.** The reference numbers are quoted from the paper's
prose and figure captions, not a data table (the full dataset is in an appendix
of an earlier paper by the same authors, not to hand), and Cd_eff is stated as
"approximately 0.6". So this is good to a few percent, not three digits. It
covers **one injector at one operating point**; the paper spans 500+ tests
across five injectors, D = 0.79–1.93 mm, L/D = 9.5–23.4, and supercharges from
41 to 500 psi. **Every tested orifice was long** (L/D ≥ 9.5) — there is still no
external check in the thin-plate regime (L/D < 5) where Dyer is the recommended
model and most practical injector plates actually sit.

Locked in as 11 tests in `tests/test_external_validation.py`.

### 1.5 Bugs found and fixed during development

These are listed because they indicate the class of error that testing catches
— and, by implication, what might still be lurking in untested areas.

1. **HEM returned zero flux above ~30 bar.** `h₁ − h₂` in a flashing expansion
   is a few kJ/kg out of ~50, so independently-fitted enthalpy and entropy
   disagreed by more than the signal. Fixed by deriving entropy *from* the ESDU
   enthalpy fit via `T ds = dh − v dP`.
2. **Choked plateau wobbled** because the scalar HEM solver anchored its
   pressure grid at `P₂`. The plateau is physically flat; now the grid is
   anchored independently of `P₂`.
3. **Transient chamber never settled** — chamber gas mass was seeded at ~0
   instead of HRAP's `1.225 × free volume`, stretching the ignition transient
   across the whole burn.
4. **An infeasible design-point guess aborted objectives that never use it**,
   and a target that drives ΔP to ~0 returned a converged-looking but
   meaningless area. Both now handled explicitly.

---

## 2. What to do to actually trust it

Ordered by value per unit effort.

### Tier A — physical testing (only you can do this)

**A1. Cold-flow the injector plate. Highest value single action.**
Water first: cheap, safe, and gives you the geometric discharge coefficient
directly (`Cd = ṁ / (A√(2ρΔP))`). Then N2O if you can do it safely: this is the
*only* way to tell which of SPI/HEM/Dyer describes your actual plate. Sweep
several ΔP values and, if possible, two feed temperatures.
*Send me the ṁ / ΔP / T data and I will fit Cd, overlay all three models, and
tell you which one your hardware follows.*

**A2. Static fire with instrumentation.** Minimum useful set: tank pressure,
tank temperature, chamber pressure, thrust, and propellant masses before and
after. Time-align everything to a common trigger.

**A3. Measure your own regression coefficients.** Pre- and post-fire port
diameter and grain mass. Two firings at meaningfully different oxidiser flux
give you `a` and `n` by fitting `ṙ = a·G_ox^n`. **See §3.1 — this matters more
than everything else combined.**

**A4. Weigh the N2O load and the residual.** Blowdown ends with a significant
vapour residual; confirming it validates the tank model independently of
combustion.

### Tier B — analysis I can do now, with no new data

- **B1. Timestep convergence study.** Confirm results are converged at your `dt`
  and report the error at each. The integrator is forward Euler (matching HRAP);
  this is currently unquantified and is the most likely source of silent error.
- **B2. Monte Carlo uncertainty propagation.** Give me your input uncertainties
  (±Cd, ±a, ±n, ±fill mass, ±temperature) and I'll return a distribution on hole
  diameter, O/F and impulse instead of a single number. Given §3.1 this is
  probably the most useful thing I can produce without new data.
- **B3. Lock in a reference-case regression test** so future changes cannot
  silently move the answers.
- **B4. Grid-independence check on the HEM solver** (currently 600 points) and
  on the saturation table.
- ~~**B5. Find and digest published N2O injector data.**~~ **Done** — see §1.4.
  The remaining gap is the thin-plate regime (L/D < 5), which the NASA dataset
  does not cover. Worth hunting for a second dataset there, since that is where
  most practical injector plates sit and where Dyer is the recommended model.

### Tier C — needs one thing from you, then I can finish

- **C1. Real HRAP cross-check.** Export a CSV from your HRAP install for a motor
  you care about, and I'll run the comparison and chase any discrepancy. This is
  now a 3-click workflow in the GUI. *This is the obvious next step.*
- **C2. Tailored sensitivity study** on your actual motor rather than my demo.

### Tier D — missing physics I can implement on request

Listed in §3.3 with the size of the error each one is hiding.

---

## 3. Notes, warnings, and known inaccuracies

### 3.1 The regression coefficients — what the sensitivity actually means

**Correction to an earlier version of this document.** The sensitivity table
below is a property of the **inverse problem this tool solves**, not evidence
that HRAP's coefficients are wrong. HRAP runs *forwards* — fixed geometry,
predict O/F — and in that direction the same coefficient errors are benign:

| +10 % error | Forward (HRAP's direction): resulting O/F | Inverse (sizing to an O/F target): resulting area |
|---|---:|---:|
| coefficient `a` | −8.7 % | **+148 %** |
| exponent `n` | −33 % | **+10456 %** |

The amplification is created by the inversion. To hit a target O/F you must
solve `ṁ_ox ∝ [ … ]^(1/(1−n))`; for paraffin `n ≈ 0.68`, so that exponent is
**3.13** and a 10 % error in `a` becomes 35 % in required flow. A second
amplification follows: a larger injector raises chamber pressure, which cuts
the injector ΔP, which demands more area again. HRAP never does this inversion,
so this was never an HRAP problem.

**This is now fixed.** Two objectives were added that size the orifice from the
feed state alone and never touch the regression law:

| Objective | `a` +10 % | `n` +10 % |
|---|---:|---:|
| `burn_average` (O/F target) | +148 % | +10456 % |
| **`mdot_ox`** (target oxidiser flow) | **+1.0 %** | **+1.8 %** |
| **`chamber_pressure`** (target Pc) | **−1.4 %** | **−2.2 %** |

The physical design barely changes — sizing the demo motor for 1.53 kg/s gives
a 1.697 mm hole against 1.699 mm from the O/F objective. You get the *same
injector*, arrived at in a way that is roughly 5000× less sensitive to
coefficient error. The tool now emits an `ILL-CONDITIONED OBJECTIVE` warning,
and reports `1/(1−n)`, whenever an O/F objective is used.

**Recommended workflow:** size with `mdot_ox` or `chamber_pressure`, then run
forwards and read off the O/F you actually get. Use the O/F objectives only
once `a` and `n` are measured on your own motor.

### 3.1b Why the coefficients still can't be transplanted

None of the above means the shipped coefficients are usable as-is. HRAP's own
theory document is explicit (§II.B, eq. 11):

> the values a, n and m are experimentally determined constants which depend on
> the propellant being used, **and the oxidizer injection system**. Injection
> systems which induce a vortex in the oxidizer flow substantially increase
> regression rates compared to simple axial injection…

So `a` and `n` are **not a fuel property**. They encode the fuel *and* the
injector *and* the test article, fitted over one flux range. HRAP also says the
constant-O/F model exists precisely "in cases where ballistic coefficients for
the propellant are unknown", to be replaced "once sufficient testing is
conducted" — which is why every motor config HRAP ships uses constant-O/F
rather than the shipped coefficient sets.

A concrete illustration, using only coefficients read from HRAP's own files:

| ṙ [mm/s] at G = | 50 | 100 | 200 | 350 | 500 |
|---|---:|---:|---:|---:|---:|
| HTPB (a=0.198, n=0.325) | 0.71 | 0.88 | 1.11 | 1.33 | 1.49 |
| 50/50 HTPB-paraffin (a=0.1146, n=0.5036) | 0.82 | 1.17 | 1.65 | 2.19 | 2.62 |
| Paraffin (a=0.0304, n=0.681) | 0.44 | 0.70 | 1.12 | 1.64 | 2.09 |

Pure paraffin comes out **slower than the 50/50 blend across the entire range**
(ratio 0.53 → 0.80), and no faster than plain HTPB below G ≈ 200. Taken at face
value that is backwards. The likely explanation is not an error in HRAP but
exactly what its documentation warns about: these entries come from different
motors with different injectors and different fitted flux ranges, so they are
not comparable to each other and none of them is a generic "paraffin" curve.

**Comparing `a` values across different `n` values is meaningless anyway** —
only ṙ at a stated G is comparable. That is the correct way to sanity-check any
coefficient set you are handed, including your own.

### 3.1c Sensitivity table (inverse problem, O/F objective)

Measured sensitivity of the sized injector area to a **+10 %** change in each
input, for the demo motor (24 holes, Dyer, target O/F 8.27):

| Input +10 % | Sized Cd·A | Hole diameter | Leverage |
|---|---:|---:|---:|
| **Regression exponent n** | **+10456 %** | **+927 %** | **1046×** |
| Fuel density ρ | +201 % | +74 % | 20× |
| Grain length | +201 % | +74 % | 20× |
| Target O/F | +157 % | +60 % | 16× |
| **Regression coefficient a** | **+148 %** | **+58 %** | **15×** |
| Initial port diameter | −10.6 % | −5.5 % | 1.1× |
| Throat diameter | −10.4 % | −5.3 % | 1.0× |
| Tank temperature (+5 °C) | −10.0 % | −5.1 % | 1.0× |
| c\* efficiency | +4.1 % | +2.0 % | 0.4× |
| Discharge coefficient Cd | 0 % | −4.7 % | — |
| **Flow model: SPI instead of Dyer** | **−21 %** | −11 % | — |
| **Flow model: HEM instead of Dyer** | **+37 %** | +17 % | — |

This table applies **only to the O/F-based objectives**. With `mdot_ox` or
`chamber_pressure` the top five rows collapse to a couple of percent (§3.1).

Practical consequences:

- **Use a flow-based objective unless you measured `a` and `n` yourself.** That
  removes the amplification entirely and costs nothing in design quality.
- **HRAP's shipped paraffin values are a starting point, not your motor** — see
  §3.1b. Measure your own; they depend on your injector as well as your fuel.
- Cd does **not** change the required effective area `Cd·A` (that is what gets
  solved for) — it changes the hole diameter you must drill to achieve it.
- Once a flow-based objective is used, the **injector flow model becomes the
  dominant modelling choice again** (SPI −21 %, HEM +37 % on area), which is the
  question this tool was built to answer.

### 3.2 Warnings about interpreting results

- **The `Dyer (L/D weighted)` model is a heuristic I wrote, not a published
  model.** It exists to expose the regime trend and to make the L/D limits
  testable. **Do not use it for design.** Use `Dyer`.
- **A converged answer is not necessarily a sensible one.** If the target is
  unreachable, the solver drives chamber pressure onto tank pressure and returns
  a huge area with near-zero ΔP. It now flags this as `DEGENERATE RESULT`, but
  **check the ΔP margin on every run.**
- **`Cd = 0.7` is a placeholder, not a prediction.** It is a direct multiplier on
  mass flow. Measure it (A1).
- **Constant-O/F mode cannot size an injector** — O/F is pinned by definition, so
  any area satisfies it. The tool refuses rather than pretending. Use it only to
  reproduce an HRAP run.
- **ESDU correlations are valid −90 °C to +36 °C.** N2O's critical point is
  36.4 °C. A hot day on the pad genuinely approaches this, where properties move
  fast and the model stops. The tool errors rather than extrapolating.
- **Burn ends at liquid depletion by default.** The vapour-phase tail is
  modelled but not included unless you ask for it. Real impulse includes some of
  that tail.
- **The dP > 15–20 % rule is a rule of thumb, not a stability model.** Meeting it
  does not guarantee stability; missing it does not guarantee instability. There
  is no combustion-instability model in this tool.

### 3.3 Physics not modelled — and how much it could cost you

| Not modelled | Likely effect | Can I add it? |
|---|---|---|
| **Feed system between tank and injector** (plumbing, valve, bends, manifold) — upstream state is taken *at the tank* | Over-predicts ΔP across the injector, hence over-predicts flow. Can be several bar on a long/narrow run | Yes — needs your line sizes and Cv |
| **Vents** (HRAP has None/External/Internal) | Oxidiser flow and blowdown rate both differ. Flagged on import, not simulated | Yes, straightforward |
| **Cd variation** with Reynolds number, cavitation, and inlet geometry | Cd is constant; real Cd shifts through the burn | Partly — correlations exist but are geometry-specific |
| **Injector element type** (showerhead / impinging / swirl) | Affects both Cd and mixing efficiency; the latter feeds c\* efficiency | Only empirically, from your data |
| **Tank heat transfer** — blowdown is adiabatic, with liquid and vapour in thermal equilibrium | Real tanks draw heat from walls, so pressure decays slower than predicted. Predicted burn time is likely conservative. HRAP shares this assumption | Yes, with a wall model |
| **Non-equilibrium bubble dynamics** | Dyer's κ is a correlation, not physics. Real flashing depends on nucleation-site density and residence time | Only with a much heavier model |
| **Paraffin entrainment** — the liquid-layer physics that makes paraffin fast-regressing | The `a·G^n` power law is empirical and may not extrapolate outside the flux range it was fitted in | No — this is why you must measure a and n |
| **Throat erosion** | Chamber pressure decays over a long burn; impulse over-predicted | Yes, with an erosion rate |
| **Grain end-face regression** | Slightly under-predicts fuel flow. HRAP explicitly assumes this too | Yes |
| **Complex port geometry** — only N identical circular ports | Multi-port web is a nominal equivalent-area figure, not true bolt-circle geometry. HRAP's Python build does arbitrary cross-sections | Yes, with effort |
| **c\* efficiency as a constant** | Real η_c\* varies with L\*, O/F and port geometry | Yes, as a curve |
| **Ignition transient / igniter mass** | First ~100 ms of the transient run is not meaningful | Yes |
| **Combustion instability** | No model at all; only the static ΔP heuristic | No — outside scope |
| **Mass properties / CG** | Not computed. HRAP does this | Yes, easily |
| **Nozzle losses** beyond a lumped efficiency (divergence, boundary layer, two-phase) | Absorbed into `noz_eff`, so thrust is only as good as that guess | Partly |

### 3.4 Numerical caveats

- **Forward Euler integration** (matching HRAP), so accuracy is first-order in
  `dt`. **This is not yet quantified — see B1.** Default `dt` is 5 ms for sizing
  and whatever the HRAP config specifies on import (often 1 ms).
- **Quasi-steady vs transient chamber differ by ~3 % on impulse** in the case I
  checked. Quasi-steady is the default for sizing (robust, no ignition
  transient); transient is for HRAP comparison.
- **The HEM solver uses a 600-point pressure grid** with local refinement, and
  saturation properties come from a 3000-point interpolated table. Both are far
  below the correlation's own uncertainty, but neither has a formal convergence
  study.
- **The saturation table caches on temperature quantised to 1 mK.**

---

## 4. Suggested order of work

1. **C1** — export one HRAP CSV and run the comparison. Cheapest possible
   external check, and it either builds confidence or finds a real bug.
2. **A1** — water cold-flow for Cd. Cheap, safe, removes the largest single
   arbitrary multiplier.
3. **B1 + B2** — convergence and uncertainty propagation, so you know what the
   error bars actually are before committing to hardware.
4. **A3** — measure `a` and `n` from your own firings. Per §3.1 this is what
   actually determines whether the tool is right.
5. **A1 with N2O** — discriminate between SPI/HEM/Dyer on real hardware.
6. Revisit §3.3 and add whichever missing physics the data says matters.
