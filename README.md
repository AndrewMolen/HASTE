# HASTE

*N2O / paraffin hybrid — injector sizing tool*

Sizes a multi-hole orifice plate for a liquid nitrous oxide injector on a
N2O/paraffin hybrid rocket motor, modelling the two-phase flashing flow that
makes N2O injectors behave unlike ordinary incompressible orifices, and
tracking an O/F target across the **whole burn** including tank blowdown.

Built to interoperate with [HRAP](https://github.com/rnickel1/HRAP_Source)
(Hybrid Rocket Analysis Program).

![burn summary](examples/output/burn_summary.png)

---

## Quick start

```bash
pip install numpy scipy matplotlib
```

```bash
python -m n2o_injector
```

### Run it like a normal app

To get a desktop icon instead of a terminal command:

```bash
powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1
```

That adds **N2O Injector Sizing Tool** to the Desktop and Start Menu (so it also
appears in Windows search and can be pinned to the taskbar). It launches via
`pythonw.exe`, so no console window appears. `-Uninstall` removes both.

You can also just double-click **`run_app.pyw`** directly — Windows opens `.pyw`
files with `pythonw`. If startup fails (usually a missing package), it shows an
error dialog and writes `startup_error.log` rather than doing nothing.

The icon is generated, not hand-drawn — rebuild it with
`python tools/make_icon.py`.

### Standalone executable (no Python needed)

```bash
python -m PyInstaller tools/n2o_injector.spec --noconfirm
```

Takes ~2.5 minutes and produces two flavours in `dist/`:

| Build | Size | Launch | Use when |
|---|---:|---:|---|
| `dist/N2O Injector Sizing Tool/` (folder) | 445 MB | **~5 s** | day-to-day use *(recommended)* |
| `dist/N2O Injector Sizing Tool.exe` (single file) | 156 MB | ~12 s | handing to someone else |

The one-file build is smaller and portable, but unpacks the whole bundle to a
temp directory on **every** launch, which is where its extra startup time goes.
The one-folder build has nothing to unpack. First launch of either is slower
(~18 s) while matplotlib builds its font cache; it is fast from then on.

Point the desktop shortcut at the standalone build with:

```bash
powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1 -UseExe
```

**Verify a build** — this runs a real sizing calculation, not just a launch:

```bash
"dist\N2O Injector Sizing Tool\N2O Injector Sizing Tool.exe" --self-test
```

It should report `SELF TEST: PASS` with `24 holes x 1.495 mm, mean O/F 7.89`,
matching the worked example in the User Guide. Add `--quiet` to skip the dialog
and just write `self_test.log`.

The `build/` directory is intermediate and can be deleted; `dist/` is the
output.

CoolProp is optional (`pip install CoolProp`) and adds a second N2O property
backend. A scripted example that writes every export and both figures:

```bash
python examples/demo_sizing.py path/to/HRAP/propellant_configs/Paraffin.mat
```

Tests:

```bash
python -m pytest tests/ -q
```

**New here?** Read the [User Guide](User%20Guide/USER_GUIDE.md) — background
physics, every input field, and how to read the output.

A worked example of the whole workflow, end to end on a real motor, is in
[Trade Study/V2.2_injector_trade_study.pdf](Trade%20Study/V2.2_injector_trade_study.pdf) —
oxidiser flow against O/F and specific impulse for the V2.2 hybrid, with every
sensitivity run and the plate drawing it concludes with. Regenerate it with:

```bash
python examples/v22_report.py
```

Both the guide and the [validation report](Validation/VALIDATION.md) are also
available as Word/PDF with typeset equations —
[USER_GUIDE.docx](User%20Guide/USER_GUIDE.docx),
[VALIDATION.docx](Validation/VALIDATION.docx) — rebuildable with:

```bash
python tools/build_docx.py
```

---

## What it computes

**Three injector flow models, plus the Dyer blend**

| Model | Assumption | When it applies |
|---|---|---|
| **SPI** | liquid stays liquid through the orifice, `G = √(2ρΔP)` | feed well subcooled |
| **HEM** | liquid and vapour reach equilibrium instantly; isentropic expansion, flux maximised over exit pressure (i.e. choked) | long orifices, L/D ≳ 10 |
| **Dyer (NHNE)** | weighted blend, `κ = √((P₁−P₂)/(P_sat−P₂))` | thin-plate orifices — the practical standard |
| **Dyer (L/D weighted)** | heuristic blend driven by L/D | regime exploration only, clearly flagged in the code |

For a **saturated (self-pressurising) feed**, `P₁ = P_sat` so `κ = 1` exactly
and Dyer is the 50/50 average of SPI and HEM — visible as the flat purple line
in `examples/output/model_comparison.png`.

**Sizing across the burn.** A drilled plate has one fixed area, but the
operating point moves continuously as the tank blows down and the port opens.
Five objectives are offered:

- `design_point` — match a target O/F at one instant (closed-form).
- `burn_average` — match the mass-averaged O/F over the burn.
- `least_squares` — best fit to a time-varying target O/F profile.
- `mdot_ox` — match a target oxidiser mass flow *(recommended)*.
- `chamber_pressure` — match a target chamber pressure *(recommended)*.

The last two never touch the regression law and are far better conditioned —
see [Choosing a sizing objective](#choosing-a-sizing-objective) below.

**Reported margins.** Injector ΔP as a fraction of chamber pressure (flagged
against a configurable 15–20 % minimum for feed-coupled stability), choked
fraction of the burn, and validity warnings when the operating point falls
outside the selected model's range.

**Three chamber models, one equation.** `quasi-steady` solves
`Pc = ṁ c*/(Cd_n A_t)` as a fixed point — it is the *steady solution* of the
chamber ODE, is independent of timestep, and is what sizing should use.
`transient` integrates `dP/P = dṁ_g/m_g − dV/V` in ratio form with
sub-stepping, which keeps the ignition ramp a property of the chamber rather
than of the timestep. `transient-hrap` keeps HRAP's own forward-Euler
discretisation for bit-level comparison against an HRAP run. Since the first is
the steady limit of the others, they must agree away from ignition and
tail-off — that agreement is a self-check, not a coincidence.

---

## From sized plate to manufactured plate

The **Plate drawing** tab turns the sized plate into a document a shop can work
from. Holes are placed on concentric bolt circles with alternate rings staggered
by half a pitch, which is both how plates are actually drilled and what
maximises the web between neighbouring holes.

The sheet is A4 landscape at a standard scale (1:1 where the plate fits), with
a plan view on the inlet face, an edge view, the plate OD dimensioned, a ring
table (bolt-circle ⌀, hole count, pitch, stagger), a hole coordinate table, a
title block and manufacturing notes. Three exports:

| Export | Format | For |
|---|---|---|
| **Save sheet** | PDF / PNG / SVG | printing and review; PDF prints to scale at 100 % |
| **Export DXF** | DXF R12, mm, full scale | CAD and CAM — hole circles usable directly as drill targets |
| **Hole table** | CSV | DRO, CMM or a CAM package that wants a point list |

The DXF puts geometry on named layers — `PLATE_OUTLINE`, `HOLES`,
`HOLE_CENTRES`, `CENTRELINES`, `SECTION`, `ANNOTATION` — so switching the last
two off leaves just the plate outline and the holes, ready to extrude.

All three come from the same `PlateLayout` object that draws the on-screen
sheet, so the drawing and the CAD file cannot disagree about where a hole is;
the tests assert that hole-for-hole. Scripted use:

```python
from n2o_injector import layout_from_plate, write_dxf, write_hole_table_csv
from n2o_injector.plots import save_plate_drawing

layout = layout_from_plate(result.plate, plate_d_mm=85.9)
save_plate_drawing("plate.pdf", layout)
write_dxf("plate.dxf", layout)
write_hole_table_csv("plate.csv", layout)
```

Two numbers on the sheet decide whether the plate is makeable: **min web**
(edge-to-edge between the closest two holes — negative means they intersect,
and the drawing says so in red) and **edge margin**. The notes also quote the
flow-area sensitivity to hole diameter, because area goes as *d*² — on a
⌀1.381 hole, +0.02 mm is +2.9 % oxidiser flow, which is the tolerance argument
to have with the machinist before the plate is cut, not after.

---

## HRAP compatibility

Your brief asked what HRAP does internally and whether this tool should feed it
or cross-validate it. I read HRAP's source and theory document directly rather
than assuming. The findings:

**HRAP's liquid injector model is SPI, and only SPI.** From its theory
document (eqs. 1–2) it defines `dLoss = K/(NA)²` with `K = 1/Cd²`, then
`ṁ = √(2ρΔP/dLoss)`, which reduces exactly to `ṁ = Cd·N·A·√(2ρ_l·ΔP)`. The
same expression appears in both implementations — `HRAP - Matlab/util/tank.m`
and `HRAP - Python/hrap/tank.py` (`inj_liq_model → 'Incompressible'`). HRAP
implements **no HEM and no Dyer model**. Its vapour-phase model is a separate
compressible real-gas relation (γ = 1.31, R = 188.91), which this tool also
reproduces for the post-liquid tail.

*Consequence:* for the same geometry, HRAP will predict **higher** oxidiser
flow than this tool in Dyer or HEM mode. In the demo case that is 1.72 kg/s
(SPI) vs 1.53 kg/s (Dyer) vs 1.11 kg/s (HEM) — a 55 % spread. Selecting `SPI`
here reproduces HRAP exactly; this is verified by a test against the
closed-form equation.

**N2O properties.** HRAP MATLAB uses ESDU 91022 curve fits (Beaton,
*Thermophysical Properties of Nitrous Oxide*, ESDU, 1991 — ref. [4] in its
theory document). Those exact correlations are reproduced coefficient-for-
coefficient in `properties.py`, so fluid properties match HRAP MATLAB
identically. HRAP's newer Python build instead uses CoolProp; that is offered
as a second backend for parity with it.

ESDU carries no entropy data, which HEM needs. Rather than integrating `cp`
independently, entropy is derived *from* the ESDU enthalpy fit via `T ds =
dh − v dP` along the saturation line. This matters: `h₁ − h₂` in HEM is a small
residual (a few kJ/kg out of ~50 kJ/kg), and mutually inconsistent `h` and `s`
correlations drive the computed flux to zero. The reconstruction is validated
against CoolProp's reference EOS — HEM mass flux agrees to **0.05 %** between
the two independent backends.

**Regression law.** Matches HRAP's `shift_OF`:
`ṙ [m/s] = 0.001·a·G_ox^n·L^m`, with `G_ox` the *oxidiser* flux in kg/m²/s,
`L` in m, and `a` in mm/s. HRAP propellant `.mat` files load directly
(`Propellant.from_hrap_mat`), including their CEA/RPA tables for `c*`.

**Export.** `export_hrap_mat()` writes an HRAP MATLAB motor config carrying
the sized `inj_D`, `inj_N` and `inj_Cd`. Field names, struct layout and unit
strings were taken from HRAP's own shipped `motor_configs/*.mat` and the unit
drop-downs inside `HRAP.mlapp`, so only values HRAP accepts are emitted.

So the answer to your question is **both**: the tool produces an injector
definition HRAP can import, *and* it independently reproduces HRAP's O/F-vs-time
prediction when run in SPI mode — which is the cross-validation path, since any
difference then isolates to the O/F and blowdown logic rather than the injector
model.

---

## Cross-referencing against HRAP

Three steps, from the **HRAP cross-reference** box in the GUI:

1. **Import HRAP motor config (.mat)** — populates every input field from an
   HRAP motor configuration, and switches this tool to SPI + `transient-hrap`
   so it is modelling what HRAP models, with HRAP's own discretisation.
2. **Load HRAP output (.csv)** — the file HRAP writes from its *Export CSV*
   button.
3. **Run comparison** — overlays both runs and prints a per-channel error table.

Scripted equivalent:

```python
from n2o_injector import (get_backend, SaturationTable, simulate,
                          load_hrap_motor, load_hrap_output,
                          compare_to_hrap, format_comparison)

props, table = get_backend("esdu"), None
imported = load_hrap_motor("motor_configs/example_98mm.mat")
table = SaturationTable(props)
burn = simulate(imported.config, props, table)
hrap = load_hrap_output("output/HRAP_output.csv")
print(format_comparison(compare_to_hrap(burn, hrap), "SPI"))
```

The importer resolves everything HRAP's own initialiser does — imperial and
metric units, tank volume given either directly or as length × diameter, tank
state given as either temperature or pressure (inverted through the saturation
curve), oxidiser load given as either mass or fill percentage, nozzle given as
either expansion ratio or exit diameter, and the percentage-stored efficiencies.
It also finds the propellant `.mat` the config references even when the stored
absolute path is from another machine.

**Things it will tell you about rather than silently absorb:** vents (not
modelled here, so oxidiser flow will differ), a missing propellant table (c*
falls back to a constant, so chamber pressure will not match), and a forced
burn time.

### Two things worth knowing about HRAP files

**Every motor config HRAP ships uses `reg_model = 'Constant OF'`,** not the
shifting-O/F regression law. In that mode fuel flow is *defined* as
`mdot_ox / const_OF`, so O/F is pinned no matter what the injector does. The
tool implements this mode (it's needed for imports to reproduce anything), but
**injector sizing is meaningless in it** — any orifice area gives the same O/F —
so `size_injector` refuses with an explanation rather than silently failing to
converge. Switch to the shifting model to size hardware.

**HRAP's CSV column "Injector Pressure Drop (kPa)" is not the injector pressure
drop.** In `tank.m` the stored `x.dP` is `Pv(T_new) − P_tank_old` — the change
in *tank* pressure over one timestep, which the blowdown model reuses as a
running average. It is negative and of order 0.01 kPa rather than the tens of
bar an injector drop would be. The importer detects this, recomputes the real
drop as `P_tank − P_chamber` from the two pressure columns, and flags it.

---

## Answers to the open items in your brief

**1. Reference documents.** I did not need them for the physics: HRAP's own
theory PDF ships in the repository and supplied the injector, blowdown and
combustion formulations, and it cites ESDU 91022 for N2O properties, which I
implemented from the coefficients in HRAP's `NOX.m`.

**2. HRAP link.** Found and used: `github.com/rnickel1/HRAP_Source`. The theory
document is `HRAP - Matlab/sources/Theory and Application of the Hybrid Rocket
Analysis Program (HRAP).pdf`.

**3. Fixed vs. profile O/F target.** Both are supported, so this did not need
to block. A constant target is the default; a time-varying profile is entered
as `time, O/F` pairs and selected with the `least_squares` objective.

**4. Paraffin regression coefficients.** Defaulted to HRAP's own shipped
`Paraffin.mat` values — **a = 0.0304, n = 0.681, m = 0**, ρ = 900 kg/m³,
optimum O/F 8.27 — and left fully editable.

> ⚠️ **Worth checking before you trust a design.** HRAP's paraffin `a = 0.0304`
> gives ~1.1 mm/s at G = 200 kg/m²/s, which is on the low side compared with
> commonly cited paraffin data (paraffin is usually quoted as regressing
> roughly 3× faster than HTPB, and HRAP's own HTPB entry is a = 0.198,
> n = 0.325). A low `a` forces a high oxidiser flow to reach a given O/F, which
> is why the demo case shows `G_ox` starting around 1350 kg/m²/s. **Substitute
> your own static-fire-derived a and n if you have them** — this single input
> moves the sized orifice area more than the choice of flow model does.

---

## Validation status

102 tests pass (`python -m pytest tests/ -q`), covering:

- **Properties** — ESDU vs published saturated N2O values at 20 °C (P = 50.6 bar,
  ρ_l = 786, ρ_v = 158 kg/m³) to within 1–2 %; critical point exact;
  monotonicity; `T_sat` inverts `P_sat`; and ESDU vs CoolProp's reference EOS
  across −20…+30 °C.
- **Limiting cases** — SPI is exactly `√(2ρΔP)`; SPI > Dyer > HEM for a
  saturated feed; `κ = 1` when saturated and Dyer is then the exact mean of the
  branches; Dyer collapses to SPI when the feed cannot flash; the L/D-weighted
  blend → SPI as L/D → 0 and → HEM as L/D grows (the limits your brief asked
  for); HEM chokes and plateaus, with a critical pressure ratio of 0.72.
- **HRAP parity** — SPI reproduces HRAP's closed-form injector equation to
  machine precision; the liquid-mass, regression and `c*` formulas match HRAP's;
  propellant `.mat` round-trips including transposed CEA grids; the exported
  motor config round-trips with HRAP-legal unit strings.
- **End-to-end** — oxidiser and fuel mass conservation to 2 %; monotone port
  growth; tank tracks the saturation curve; chamber always below tank pressure;
  sizing hits the burn-average target to 2 %.
- **HRAP import** — unit factors checked against HRAP's own initialiser
  (including its `psi = 101325/14.696` convention); imperial and metric configs;
  both tank-volume, tank-state, fill and nozzle-definition modes; a
  temperature round-trip through C/K/F/R; export→import round-trip; the
  constant-O/F model matching `const_OF.m`; CSV unit conversion, the 13- and
  15-column variants, and recovery of the true injector ΔP. The comparison
  itself is checked both ways: identical runs must give ~0 % error, and a known
  10 % perturbation must show up as one.

### External validation against measured data

The injector models have been checked against **real N2O cold-flow measurements**
— Waxman, Zimmerman, Cantwell (Stanford) & Zilliac (NASA Ames), NTRS 20190001326.

```bash
python examples/validate_waxman.py
```

For their injector 3 (D = 1.50 mm, L/D = 12.3, rounded inlet; N2O at 704 psia,
280 K, 169 psi supercharge, measured Cd = 0.77):

| | Measured | This tool | Error |
|---|---:|---:|---:|
| Saturation pressure (via supercharge) | 169 psi | 166 psi | **0.63 %** on Psat |
| Cd_eff at ΔP = 330 psi — **HEM** | ≈ 0.60 | 0.545 | **−9.1 %** |
| Cd_eff at ΔP = 330 psi — Dyer | ≈ 0.60 | 0.677 | +12.8 % |
| Cd_eff at ΔP = 330 psi — SPI | ≈ 0.60 | 0.770 | +28.3 % |

The measurement is **bracketed by SPI above and HEM below**, exactly as the
physics requires — the paper notes equilibrium "gives a lower-bound estimate for
the critical mass flow rate" because real flashing is delayed by metastability.
The tool's own L/D guidance (recommend HEM above L/D 10) picks the model that
actually fits best, having been written from the literature before this
comparison was run.

### What is *still* not validated

- **The thin-plate regime (L/D < 5)** — every orifice in the NASA dataset was
  long (L/D ≥ 9.5), yet most practical plates are thin. Biggest remaining gap.
- **Everything downstream of the injector** — regression, blowdown, chamber
  pressure, thrust. No static-fire comparison exists.
- **A real HRAP run.** Still the cheapest remaining check.
- `Cd = 0.7` remains a placeholder default, not a prediction. Cold-flow your
  own plate.

**➜ See [VALIDATION.md](VALIDATION.md)** for the full validation roadmap, the
measured input-sensitivity table, and every known limitation.

### Choosing a sizing objective

The O/F-based objectives must invert the regression law, `ṁ_ox ∝ [ … ]^(1/(1−n))`.
For paraffin that exponent is ~3.1, so a 10 % error in the coefficient `a`
becomes ~150 % in sized area. Two flow-based objectives avoid this entirely:

| Objective | `a` +10 % | `n` +10 % | Uses regression law? |
|---|---:|---:|---|
| `burn_average` / `design_point` / `least_squares` | +148 % | +10456 % | yes — inverted |
| **`mdot_ox`** | **+1.0 %** | **+1.8 %** | no |
| **`chamber_pressure`** | **−1.4 %** | **−2.2 %** | no |

They give essentially the same hardware (1.697 mm vs 1.699 mm on the demo
motor), just arrived at robustly. **Size with `mdot_ox` or `chamber_pressure`,
then run forwards and read off the O/F**, unless you have measured `a` and `n`
on your own motor. The tool warns when an ill-conditioned objective is used.

---

## Layout

```
n2o_injector/
  properties.py   ESDU 91022 + CoolProp backends, entropy reconstruction
  injector.py     SPI / HEM / Dyer, choking, precomputed operating curve
  propellant.py   regression law, CEA tables, HRAP propellant loader
  motor.py        grain, tank blowdown, chamber, burn simulation
  sizing.py       O/F back-solve, CdA solver, plate realisation
  hrap_io.py      HRAP motor-config + output-CSV import, run comparison
  report.py       design report, HRAP export, JSON/CSV export
  drawing.py      plate hole layout, DXF R12 writer, hole-table CSV
  plots.py        figures and the A4 drawing sheet (shared by GUI and scripts)
  gui.py          Tkinter interface
tests/            140 validation tests
examples/         scripted demo, the V2.2 trade study, generated outputs
Trade Study/      V2.2 oxidiser-flow trade study (PDF) + the plate it recommends
```

## Notes on the numerics

The HEM branch is the expensive part: it maximises mass flux over exit
pressure, and the chamber-pressure solve calls it many times per timestep.
Two structural choices keep a full sizing run interactive (~2 s rather than
~2 min): the isentropic flux curve is computed **once per timestep** and the
HEM branch read off as its suffix maximum (`OrificeCurve`), and the CEA table
interpolation uses plain Python scalar arithmetic instead of NumPy's very slow
scalar path.
