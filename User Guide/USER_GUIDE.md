# User Guide

How to run the tool, and the background needed to interpret what it tells you.

- **[Part 1 — Background](#part-1--background-physics-you-need)** — the physics. Read once.
- **[Part 2 — Running it](#part-2--running-it)** — every input field.
- **[Part 3 — Reading results](#part-3--reading-the-results)** — plots, report, warnings.
- **[Part 4 — Worked example](#part-4--worked-example)**
- **[Part 5 — HRAP cross-reference](#part-5--cross-referencing-against-hrap)**
- **[Part 6 — Sanity checks](#part-6--sanity-checks-and-common-mistakes)**

---

## Five-minute first run

```bash
cd "C:\Users\andre\OneDrive\Desktop\claude\n2o-injector-sizing"
```

```bash
python -m n2o_injector
```

The defaults describe a plausible ~13 kN·s paraffin motor. Press **Compute
injector sizing** and you get an orifice plate, six plots, and a report. Then
come back and read Part 1 so you know whether to believe it.

---

# Part 1 — Background physics you need

## 1.1 Why an N2O injector is not a water orifice

For an ordinary liquid, flow through an orifice follows the textbook relation

> **ṁ = Cd · A · √(2 ρ ΔP)**

Push harder (bigger ΔP), get more flow, forever. This is the **SPI** model
(Single-Phase Incompressible), and for kerosene or water it is fine.

Nitrous oxide breaks it. At room temperature N2O has a vapour pressure of about
**50 bar** — it is stored as a *saturated liquid*, meaning liquid sitting right
at the edge of boiling. As it accelerates through an orifice its static pressure
drops. The moment that local pressure falls below the vapour pressure, the
liquid starts **flashing to vapour** inside the hole.

Vapour is roughly 5× less dense than the liquid. So the fluid passing the throat
is now a frothy two-phase mixture, and it carries far less mass than the SPI
equation predicts. Use SPI on a saturated N2O injector and **you will
over-predict flow** — by 28% in the one case where we have measured data, and by
more at higher pressure drops.

That single fact is the reason this tool exists.

## 1.2 Saturated vs. supercharged (subcooled) feed

Two ways to run an N2O feed system:

**Self-pressurising (saturated).** The tank is just N2O. Its own vapour pressure
pushes the liquid out. Simple, no pressurant, no regulator — which is why most
amateur and student hybrids do it. Tank pressure is then set *entirely* by
temperature: warm day, high pressure; cold day, low. As the tank empties it
cools by evaporation, so pressure decays through the burn ("blowdown").

**Supercharged (subcooled).** You add helium or nitrogen on top to push the
liquid pressure *above* its vapour pressure. The margin `P₁ − P_sat` is the
**supercharge**. This keeps the liquid further from flashing, so flow behaves
more like ordinary SPI, and it holds tank pressure up as the tank drains.

Why it matters here: **supercharge is the single biggest lever on which flow
model applies.** With zero supercharge (saturated), the liquid flashes almost
immediately and two-phase effects dominate. With a lot of supercharge, the flow
may stay liquid all the way through and plain SPI is correct.

In the tool: leave **Supercharge P** blank for a self-pressurising tank (the
common case). Enter an absolute pressure to model a supercharged one.

## 1.3 The three flow models

| Model | What it assumes | Predicts | Use when |
|---|---|---|---|
| **SPI** | no flashing; liquid stays liquid | **highest** flow | strongly subcooled feed |
| **HEM** | liquid and vapour reach equilibrium *instantly* | **lowest** flow | long orifices, L/D ≳ 10 |
| **Dyer (NHNE)** | weighted blend of the two | between | thin plates — the practical standard |

The truth sits between SPI and HEM, and **which end it sits nearer depends on
how long the fluid spends inside the hole.**

- Very short hole → fluid exits before bubbles have time to grow → behaves
  SPI-like.
- Long hole → bubbles have time to grow, flow reaches equilibrium → HEM-like.

**Dyer's model** captures this with a weighting parameter

> **κ = √( (P₁ − P₂) / (P_v − P₂) )**,  ṁ = (κ/(1+κ))·ṁ_SPI + (1/(1+κ))·ṁ_HEM

For a **saturated feed** `P₁ = P_v`, so **κ = 1 exactly** and Dyer is the plain
50/50 average of SPI and HEM. That is not a coincidence or a fudge — it falls
out of the algebra, and you will see it as a flat line at κ = 1 on the Model
comparison tab.

More supercharge → larger κ → the blend leans toward SPI. That is the model
correctly reflecting the physics of §1.2.

> **A fourth option, `Dyer (L/D weighted)`, exists in the dropdown. It is a
> heuristic written to explore the L/D trend and to make the limiting cases
> testable. It is not a published model. Do not design with it.**

## 1.4 Choking (critical flow)

Ordinary orifice: lower the downstream pressure, always get more flow.

Flashing orifice: below some **critical back pressure**, flow *stops
responding* — you can drop chamber pressure further and mass flow does not
change. This is analogous to choking in a gas nozzle and is called **critical
flow**. It happens because the two-phase mixture reaches its own sonic
condition inside the hole.

This is genuinely useful. A choked injector **isolates the feed system from the
chamber**: pressure oscillations in the combustion chamber cannot travel back
upstream and modulate the flow. That is the main mechanism for avoiding
feed-coupled combustion instability, and it is the subject of half the NASA
paper this tool is validated against.

The tool reports whether you are choked and at what pressure. For saturated N2O
at 20 °C it predicts choking below about **36 bar** chamber pressure (a critical
pressure ratio near 0.72).

## 1.5 Discharge coefficient, Cd

`Cd` lumps together everything the ideal equation ignores — flow separation at
the sharp inlet edge, the vena contracta, friction. It multiplies straight
through to mass flow, so **an error in Cd is an error of the same size in your
answer.**

Rough expectations, from the NASA data:

- **Square-edged** drilled hole: lower Cd — that dataset found the square-edged
  orifice passed about **20% less** critical flow than rounded or chamfered.
- **Chamfered or rounded** inlet: noticeably higher, and the two are nearly
  identical. **A simple 45° chamfer buys you almost all of the benefit of a
  proper radius** — a cheap and worthwhile machining change.
- Their measured single-phase values ran roughly **0.6 to 0.8**, and Cd
  *decreased* with increasing hole diameter.

**The 0.70 default in this tool is a placeholder, not a prediction.** Cold-flow
your actual plate if the number matters. Water is fine for a first pass and is
much safer than N2O.

## 1.6 Orifice L/D — why plate thickness is an input

**L/D = plate thickness ÷ hole diameter.** It decides how long the fluid is
inside the hole, and therefore how far toward equilibrium it gets:

| L/D | Regime | Recommended |
|---|---|---|
| < 1 | very short — little time to flash | Dyer (leans SPI) |
| 1–5 | thin plate — most real injectors | **Dyer** |
| 5–10 | transitional | Dyer, cross-check HEM |
| > 10 | long — approaches equilibrium | **HEM** |

The tool prints a recommendation based on your L/D and warns if you have picked
a model outside its sensible range.

> **Validation caveat.** The measured data this tool has been checked against
> used **long** orifices only (L/D 9.5–23.4), where HEM won. The thin-plate
> regime — where most real plates sit — has **not** been externally validated
> here. See `VALIDATION.md` §1.4.

## 1.7 The motor side: O/F, regression, and mass flux

**O/F** is the oxidiser-to-fuel mass flow ratio. It sets combustion temperature
and c*, so it drives performance. Too high (oxidiser-rich) and you waste
oxidiser and run hot; too low and you leave energy unreleased.

In a hybrid you **cannot set O/F directly.** You choose the oxidiser flow; the
fuel flow is whatever the grain decides to give you. The grain surface regresses
at a rate that depends on the oxidiser mass flux down the port:

> **ṙ [mm/s] = a · G_ox^n · L^m**,  where **G_ox = ṁ_ox / A_port**

- `a`, `n`, `m` — empirical **ballistic coefficients**
- `G_ox` — oxidiser mass flux, kg/m²/s (note: *oxidiser only*, not total)
- `L` — grain length in metres (usually `m = 0`, so it drops out)

**The consequence that trips people up:** as the burn proceeds the port opens
up, so for the same oxidiser flow `G_ox` falls, so regression slows, so fuel
flow drops — and **O/F drifts upward through the burn.** This is normal and
unavoidable with a single-port grain. You will see it in the O/F plot. A fixed
orifice cannot hold O/F constant; you are choosing where in the drift to sit.

### `a` and `n` are not fuel properties

HRAP's own theory document is explicit that these coefficients depend on

> "the propellant being used, **and the oxidizer injection system**. Injection
> systems which induce a vortex in the oxidizer flow substantially increase
> regression rates compared to simple axial injection…"

So a coefficient pair fitted on someone else's motor with someone else's
injector will not transfer to yours. The defaults (`a = 0.0304, n = 0.681`) come
from HRAP's shipped paraffin config and are a **starting point only**. Measure
your own from static fires: pre- and post-fire port diameter and grain mass,
across at least two firings at different flux, then fit `ṙ = a·G_ox^n`.

Also: comparing `a` values between two coefficient sets with different `n` is
meaningless. Only `ṙ` evaluated at a stated `G` is comparable.

## 1.8 Blowdown

For a self-pressurising tank, pulling liquid out means the remaining liquid must
boil to fill the space. Boiling absorbs latent heat, so **the tank cools**, so
its vapour pressure drops. Feed pressure therefore decays continuously through
the burn — typically 50 bar down to high-30s over a few seconds.

Consequences you will see in the plots:

- Oxidiser flow decreases through the burn.
- Thrust decreases through the burn (a hybrid blowdown curve is not flat).
- The burn ends when liquid runs out, leaving a **significant vapour residual**
  still in the tank. That residual is real mass you loaded and did not usefully
  burn — in the demo case about 1.3 kg of 7 kg. Do not size your tank assuming
  you get all of it.

## 1.9 Injector pressure drop and stability margin

**ΔP_inj = P_tank − P_chamber**, usually quoted as a fraction of chamber
pressure.

If ΔP is small, the injector barely resists flow, and a pressure bump in the
chamber pushes back into the feed line, changing the flow, which changes the
chamber pressure — a feedback loop that can grow into **feed-coupled combustion
instability**. Motors have been destroyed by this.

The conventional guard is to keep **ΔP ≥ 15–20% of chamber pressure**. The tool
defaults to 20% and flags anything below.

Caveats worth holding onto:

- This is a **rule of thumb, not a stability model.** There is no combustion
  instability model in this tool. Meeting 20% does not guarantee stability.
- **A choked injector is isolated regardless** (§1.4), which is a stronger
  argument than the ΔP rule.
- Very *high* ΔP is not free either — you are throwing away tank pressure as
  heat. If ΔP is 60% of Pc, a larger injector would buy you chamber pressure and
  performance.

---

# Part 2 — Running it

Launch with `python -m n2o_injector`. Inputs are on the left, results on the
right. **Hover any label for a tooltip.** Buttons at the bottom-left never
scroll away.

## 2.1 Oxidiser tank / feed

| Field | Default | Notes |
|---|---|---|
| Tank volume | 12.0 L | Internal volume. |
| Fill temperature | 20 °C | **Sets feed pressure** for a saturated tank. Use the temperature you actually expect on the pad. |
| Oxidiser mass | 7.0 kg | Must leave both liquid and vapour present — the tool rejects a liquid-full or vapour-only fill. |
| Supercharge P | *(blank)* | Blank = self-pressurising. Enter absolute bar for a pressurant-fed system. |
| Ambient pressure | 1.01325 bar | Back pressure for thrust. |

**Fill temperature is not a minor input.** N2O vapour pressure roughly doubles
between 0 °C and 30 °C. Sizing for 20 °C and flying at 30 °C gives noticeably
more flow. Run both.

Also note the tool refuses temperatures above **36 °C** — N2O's critical point
is 36.4 °C, above which there is no liquid at all. A hot pad genuinely
approaches this.

You will often hit a second limit before that one. Near the critical point the
saturated *liquid* density collapses, so a fill that fits comfortably at 20 °C
may not fit at all when warm. The default 7 kg in 12 L is accepted at 35 °C but
rejected as **liquid-full at 36 °C** — the tank physically cannot hold that mass
as a saturated mixture any more. If you get that error on a hot-day run, the
answer is a bigger tank or less oxidiser, not a software workaround.

## 2.2 Injector

| Field | Default | Notes |
|---|---|---|
| Flow model | Dyer | See §1.3. Dyer for thin plates, HEM for L/D > 10, SPI only to match HRAP. |
| Discharge coeff Cd | 0.70 | **Placeholder.** Measure it (§1.5). |
| Plate thickness | 3.0 mm | Sets L/D with hole diameter. |
| Hole count | 24 | Held fixed if solving for diameter. |
| Hole diameter | 1.60 mm | Held fixed if solving for count. |
| Min drillable dia | 0.80 mm | Manufacturing floor; flagged if violated. |
| L/D reference | 5.0 | Only used by the L/D-weighted heuristic. Ignore otherwise. |
| Plate diameter | 0 mm | Plate OD, used by the drawing only (§3.4). 0 = use the grain outer diameter. |

**Solve for: diameter / count.** Pick which one you're free to change. "Diameter"
fixes hole count and gives you an exact area (but a non-standard drill size).
"Count" fixes the diameter to a drill you own and rounds the count to an
integer, reporting the resulting area error.

## 2.3 Fuel grain

| Field | Default | Notes |
|---|---|---|
| Grain length | 450 mm | |
| Initial port dia | 38 mm | Starting bore. |
| Grain outer dia | 92 mm | Sets available web. |
| Number of ports | 1 | Multi-port uses N identical circular ports. |

> Multi-port geometry is approximate — the web is reported as a nominal
> equivalent-area value, not true bolt-circle geometry. Single port is exact.

## 2.4 Propellant / regression

Pick a **Preset** (Paraffin, HTPB/Paraffin 50/50, HTPB) to load HRAP's
coefficients, or **Load HRAP propellant (.mat)** to pull in a full CEA table —
strongly preferred, since otherwise c* is a single constant.

| Field | Default | Notes |
|---|---|---|
| Regression a | 0.0304 mm/s | **Replace with your own measured value.** §1.7 |
| Regression n | 0.681 | Must be < 1. The most leveraged input in the tool. |
| Regression m | 0.0 | Length exponent; 0 to ignore. |
| Fuel density | 900 kg/m³ | |
| c* (if no table) | 1550 m/s | Only used when no CEA table is loaded. |
| c* efficiency | 0.95 | Combustion efficiency. 0.90–0.95 typical. |
| Regression model | shifting | `shifting` = the a/n/m law. `constant_OF` = pins O/F (HRAP's mode; **cannot size an injector**). |

## 2.5 Nozzle

Throat diameter (32 mm), expansion ratio (4.0), nozzle Cd (0.95), nozzle
efficiency (0.95). Throat diameter sets chamber pressure for a given total mass
flow — it is the other half of the design, and if chamber pressure comes out
wrong the throat is usually the thing to change.

## 2.6 Sizing target — the most important choice

| Objective | Uses regression law? | Robust? |
|---|---|---|
| `burn_average` | yes, **inverted** | ✗ |
| `design_point` | yes, inverted | ✗ |
| `least_squares` | yes, inverted | ✗ |
| **`mdot_ox`** | no | ✓ |
| **`chamber_pressure`** | no | ✓ |

The O/F-based objectives have to solve backwards through the regression law,
`ṁ_ox ∝ [ … ]^(1/(1−n))`. With `n = 0.681` that exponent is **3.13**, so a 10%
error in `a` becomes about **150%** in sized area. Measured:

| Objective | `a` +10% | `n` +10% |
|---|---:|---:|
| `burn_average` | +148% | +10456% |
| `mdot_ox` | **+1.0%** | **+1.8%** |
| `chamber_pressure` | **−1.4%** | **−2.2%** |

**Unless you have measured `a` and `n` on your own motor, use `mdot_ox` or
`chamber_pressure`, then read off the O/F you get.** You land on essentially the
same hardware (1.697 mm vs 1.699 mm in the demo) but the answer no longer hinges
on borrowed coefficients. The tool warns `ILL-CONDITIONED OBJECTIVE` if you use
an O/F target with a large exponent.

How to pick a number for these: you usually have a thrust or total-impulse
target. Total mass flow ≈ thrust / (Isp · g₀); oxidiser flow is roughly
`OF/(1+OF)` of that. Or work from chamber pressure directly if your hardware is
pressure-limited.

Other fields: **Min dP margin** (20%, §1.9), **Design time** (for
`design_point`), and **O/F profile** — one `time, O/F` pair per line for a
time-varying target with `least_squares`.

## 2.7 Numerics

**Timestep** 10 ms, **Max sim time** 60 s, **N2O properties** (`esdu` = HRAP
MATLAB parity; `coolprop` = reference EOS).

**Chamber model** — three choices, and they are not three theories:

| Mode | What it does | Use it for |
|---|---|---|
| `quasi-steady` | Solves `Pc = ṁ c*/(Cd_n A_t)` as a fixed point each step. This is the *steady solution* of the transient equation. | **Sizing.** No timestep sensitivity, no assumed initial `Pc`, no chamber-volume guess. |
| `transient` | Integrates `dP/P = dṁ_g/m_g − dV/V` in ratio form with sub-stepping. | The ignition ramp and tail-off. |
| `transient-hrap` | HRAP's own forward-Euler discretisation of the same equation. | Reproducing an HRAP run exactly — its numerics as well as its physics. |

Because quasi-steady is the steady solution of the transient equation, the two
**must** agree away from ignition and tail-off. If they don't, one is being
integrated badly rather than telling you something physical.

> **Always refine the timestep before believing a transient number.** Halve
> `dt` and confirm the answer does not move; the tool does not do this for you.
> `quasi-steady` is insensitive to `dt` by construction, so the check only
> really bites on the transient modes.
>
> `transient-hrap` deliberately keeps HRAP's dt-dependence, including an
> ignition ramp whose *duration scales with the timestep*. Don't read startup
> numbers off it — that is what `transient` is for.

**Chamber volume** matters only to the transient modes: 0 means "port volume
only", which ignores your pre- and post-chamber. Setting the real free volume
both improves the ramp and relaxes the timestep you need.

**What the ignition ramp is and isn't.** The transient mode models the chamber
*filling* — gas accumulating until the nozzle can pass it. It does not model
igniter energy, flame spread, or oxidiser pooling before light. So it gives you
the idealised fill shape and its timescale; it will not predict a hard start.

---

# Part 3 — Reading the results

## 3.1 The six plots

1. **Mass flow rate** — oxidiser, fuel, total. All should decay smoothly as the
   tank blows down. Sharp kinks mean something is wrong.
2. **O/F ratio** — achieved vs target. **Expect an upward drift** (§1.7). Judge
   the *range*, not just the mean.
3. **Tank and chamber pressure** — both decay; the gap between them is your
   injector ΔP. Chamber must always sit below tank.
4. **Injector ΔP margin** — with the red minimum line. Must stay above it. This
   is your stability-margin plot.
5. **Thrust** — decaying, as a blowdown hybrid should be. Total impulse in the
   title.
6. **Port growth and G_ox** — port opens, flux falls. Watch that the port does
   not approach the grain OD (burn-through).

## 3.2 The report tab

- **Recommended injector** — hole count, diameter, total and effective area,
  L/D, and the rounding error from realising an integer hole count.
- **Model and assumptions** — everything needed to reproduce the run. Include
  this if you circulate results.
- **Design point (t = 0)** — the instantaneous condition at ignition.
- **Burn summary** — mean O/F, burn time, propellant consumed, impulse, Isp.
- **Margins and flags** — ΔP margins, choked fraction, convergence.
- **HRAP cross-check** — the `inj_N` / `inj_D` / `inj_Cd` values to type into
  HRAP, plus a reminder that HRAP's SPI-only model will read higher.

## 3.3 Model comparison tab

Press **Compare SPI / HEM / Dyer**. Shows mass flux vs chamber pressure for all
three models at your feed condition, plus κ.

Read it for two things:

- **The spread between SPI and HEM at your operating pressure.** That is your
  model uncertainty. If it is large, the choice of model dominates your design
  and you should cold-flow to settle it.
- **Where HEM flattens** — that is your choking point (§1.4). Chamber pressure
  below it means the injector is isolating the feed system.

## 3.4 The plate drawing tab

This is the tab you send to a machine shop. It renders the sized plate as an A4
landscape drawing sheet and exports it three ways.

**What is on the sheet**

| Part of the sheet | What it tells you |
|---|---|
| Plan view | The plate as seen **from the inlet (upstream) face**, at a standard scale (1:1 wherever the plate fits the sheet). |
| Ring data | Each bolt circle: diameter, hole count, angular pitch, and the stagger applied to it. |
| Edge view | Plate thickness with the holes projected — a visual check that L/D is what you meant. |
| Hole table | Every hole as X/Y from the plate centre, plus polar R and angle. |
| Notes | Datum, tolerance sensitivity, Cd assumption, min web, edge margin. |
| Title block | Project, part no, material, scale, date, flow model and Cd used. |

**Two fields above the drawing.** *Part no* and *Material* fill the title block
and name the exported files; both are optional. *Plate diameter* (in the
injector input panel) sets the plate OD — leave it at 0 and the grain outer
diameter is used, which is the usual envelope.

**Hole placement.** Holes go on concentric bolt circles, with the count on each
ring proportional to its circumference and alternate rings rotated by half a
pitch. That staggering is not cosmetic: it is what keeps the web between rings
open, and it is how a plate is actually drilled. The tool does *not* optimise
the spray pattern — impingement, film cooling and atomisation are not modelled
anywhere in this program, so treat the layout as a manufacturable default, not
an injector design.

**The three exports**

- **Save sheet** — PDF, PNG or SVG. Use PDF and **print at 100 %**; the sheet
  carries a 50 mm check bar so the shop can confirm the print was not scaled
  before measuring anything off it.
- **Export DXF** — DXF R12, full scale, millimetres, origin at the plate
  centre. Geometry sits on named layers (`PLATE_OUTLINE`, `HOLES`,
  `HOLE_CENTRES`, `CENTRELINES`, `SECTION`, `ANNOTATION`); turn off the last
  two and you have just the outline and the holes to extrude or to use as drill
  targets. R12 is deliberate — it is the revision every CAD and CAM package
  still reads.
- **Hole table .csv** — the coordinate list for a DRO, a CMM or a CAM package
  that prefers points to geometry. The header states the datum and units,
  because a bare coordinate list is how plates get drilled mirrored.

**Two numbers decide whether it can be made.** *Min web* is the smallest
edge-to-edge gap between any two holes; if it goes negative the holes intersect
and the sheet says so in red. *Edge margin* is the gap from the outermost hole
to the rim. If either is marginal, reduce the hole count (and increase the
diameter) or use a larger plate.

**The tolerance argument, in advance.** Flow area goes as *d*², so the sheet
computes what a hole-diameter error costs you: on a ⌀1.381 mm hole, +0.02 mm is
+2.9 % area — and, since the injector is the flow-metering element, ~+2.9 %
oxidiser flow and a shifted O/F. Twenty holes drilled 0.02 mm oversize is not a
rounding error, it is a different motor. Decide the tolerance before the plate
is cut.

## 3.5 Warning decoder

| Message | Meaning | Do |
|---|---|---|
| `DEGENERATE RESULT: injector pressure drop has collapsed…` | The target is unreachable; the solver pushed chamber pressure onto tank pressure. Result is **not a design**. | Enlarge the throat, lower the target, or re-check `a`/`n`. |
| `ILL-CONDITIONED OBJECTIVE…` | O/F objective with a big `1/(1−n)`. | Switch to `mdot_ox` or `chamber_pressure` (§2.6). |
| `minimum injector pressure drop is X%…below target` | Weak feed isolation. | Reduce injector area, or raise tank temperature. |
| `SPI with a saturated feed…will over-predict` | Using SPI where flashing is certain. | Use Dyer, unless deliberately matching HRAP. |
| `HEM at L/D = X: orifice too short…` | HEM on a thin plate under-predicts. | Use Dyer. |
| `Dyer at L/D = X: long orifices approach equilibrium` | Long hole. | Cross-check HEM. |
| `SPI and HEM differ by more than 3x…` | Huge model uncertainty at this point. | Treat as provisional; cold-flow. |
| `instantaneous O/F deviates…by up to X%` | Large drift across the burn. | Decide whether the ends are acceptable. |
| `injector pressure drop is very high (X%)` | Over-restrictive; wasting tank pressure. | Consider a larger injector. |
| `rounding N holes to M changes area by X%` | Integer hole count. | Adjust hole diameter. |
| `oxidiser flux peaks at X kg/m^2/s, above the ~700…` | `a·G^n` is being extrapolated past the range the correlation was fitted over. | Open the grain ports — `G_ox` is set by port area, not by the injector. Or accept it and say so. |
| `oxidiser flux peaks at only X…below the ~50…` | Flux too low for the power law; radiation matters. | Smaller ports or more flow. |
| `fuel web burned through` | Grain ran out before oxidiser. | Thicker web or shorter burn. |
| `liquid oxidiser exhausted at t = …` | Normal end of a blowdown burn. | None — expected. |

---

# Part 4 — Worked example

Say you want roughly **3 kN of thrust** from a paraffin motor, self-pressurising
tank at 20 °C, and you can drill 1.6 mm holes.

**1. Estimate the flow you need.** At Isp ≈ 210 s, total ṁ ≈ 3000/(210·9.81)
≈ 1.46 kg/s. At O/F ≈ 8, oxidiser is 8/9 of that ≈ **1.3 kg/s**.

**2. Use the robust objective.** Objective `mdot_ox`, target **1.3 kg/s**. No
reliance on borrowed regression coefficients.

**3. Enter geometry.** Grain 450 mm long, 38 mm port, 92 mm OD. Throat 32 mm.
Plate 3 mm, Cd 0.70. Solve for **diameter**, hole count 24.

**4. Compute.** This exact case gives:

```
plate            : 24 holes x 1.495 mm   (L/D 2.01)
mdot_ox(0)       : 1.3000 kg/s           (target 1.30)
thrust (peak)    : 3001 N
mean O/F         : 7.89   (range 7.30 - 8.24)
burn time        : 4.68 s
total impulse    : 12865 N.s             Isp 205.4 s
min dP margin    : 61.6 %                (ok)
port at burnout  : 60.9 mm               (grain OD 92.0 mm)
ox used / loaded : 5.67 / 7.00 kg
warnings         : none
```

Thrust came out at 3001 N against the 3 kN we asked for, which is the objective
doing its job. Isp landed at 205 s rather than the 210 s assumed in step 1 —
that is the hand-estimate being slightly optimistic, not an error.

> Run with no CEA table loaded, so `c*` is the constant 1550 m/s × 0.95. Load
> the matching HRAP propellant `.mat` and Isp rises a few seconds. Always load
> the table if you have it.

Now check it:

- Is **ΔP margin** above 20% for the whole burn? *(61.6% — comfortably, though
  high enough that a larger injector would buy chamber pressure)*
- Is **mean O/F** near the propellant's optimum (8.27 for HRAP paraffin)?
  *(7.89, drifting 7.30 → 8.24 — slightly fuel-rich on average)*
- Is **burn time** what you wanted? *(4.68 s)*
- Is the **port** clear of the grain OD at burnout? *(60.9 of 92 mm — yes)*
- Did you get all your oxidiser? *(5.67 of 7.00 kg — the rest is vapour
  residual, §1.8)*
- Is **L/D** in a regime where your chosen model is valid? *(2.01 — thin-plate,
  so Dyer is right, but note this is the regime with **no external validation**)*

**5. Iterate on the right knob.**

| Symptom | Fix |
|---|---|
| O/F too high | Smaller injector, or larger port/longer grain |
| O/F too low | Larger injector, or smaller port |
| ΔP margin too low | Smaller injector, or raise tank temperature |
| Chamber pressure too high | Larger throat |
| Burn too short | More oxidiser, smaller injector |

**6. Re-run with the model you did not choose.** If Dyer and HEM give you
meaningfully different plates, that gap is your real uncertainty — and the
argument for cold-flow testing.

---

# Part 5 — Cross-referencing against HRAP

Three buttons, in order:

1. **Import HRAP motor config (.mat)** — fills every field and switches to SPI +
   transient chamber, matching what HRAP models.
2. **Load HRAP output (.csv)** — from HRAP's *Export CSV* button.
3. **Run comparison** — overlay plus a per-channel RMS/max/bias table.

Two things about HRAP files worth knowing:

- **Every motor config HRAP ships uses `Constant OF`**, where fuel flow is
  *defined* as `ṁ_ox/const_OF`. O/F is then pinned regardless of injector area,
  so sizing is meaningless — the tool refuses and explains.
- **HRAP's CSV column "Injector Pressure Drop" is mislabelled.** It holds the
  per-timestep *tank* pressure change (~0.01 kPa, negative), not the injector
  drop. The importer detects this and recomputes the real value from the
  pressure columns.

Expect SPI mode to reproduce HRAP closely. If it doesn't, that's a real finding
worth chasing — the injector model is identical, so any difference isolates to
blowdown or O/F logic.

---

# Part 6 — Sanity checks and common mistakes

**Before believing any run:**

- Is chamber pressure **below** tank pressure everywhere? (The tool enforces it,
  but if it is *just* below, you are in the degenerate regime.)
- Is O/F within the CEA table range? Outside it, c* is clamped.
- Does the port stay clear of the grain OD?
- Is `G_ox` plausible? Paraffin hybrids are commonly designed around a few
  hundred kg/m²/s. If you see well over a thousand, question the port size and
  the regression coefficients before trusting the answer.
- Does the burn end from **liquid exhaustion** (normal) or **web burn-through**
  (usually a design problem)?

**Common mistakes:**

1. **Trusting the default Cd.** It is a placeholder. It multiplies your answer.
2. **Using borrowed `a` and `n` with an O/F objective.** The two worst inputs
   feeding the most fragile solver. Use `mdot_ox` instead.
3. **Sizing at one temperature.** Run your cold and hot day cases.
4. **Reading mean O/F only.** Look at the range; the ends of the burn are where
   you run oxidiser-rich.
5. **Assuming you get all your oxidiser.** Blowdown leaves a vapour residual.
6. **Using SPI because it is familiar.** It over-predicts for a saturated feed —
   by 28% against measured data.
7. **Designing to exactly 20% ΔP.** It is a rule of thumb, not a guarantee, and
   there is no instability model here.
8. **Treating any of this as validated for flight.** See `VALIDATION.md`. The
   injector models are checked against real data for long orifices; the motor
   side is not validated against hardware at all.

---

## Further reading

- `README.md` — what the tool does, HRAP compatibility.
- `VALIDATION.md` — what is and is not validated, sensitivities, limitations.
- `examples/validate_waxman.py` — the external validation, reproducible.
- Waxman, Zimmerman, Cantwell & Zilliac, *Mass Flow Rate and Isolation
  Characteristics of Injectors for Use with Self-Pressurizing Oxidizers in
  Hybrid Rockets*, NTRS 20190001326 — the reference for §1.1–1.5.
- HRAP's own theory document, in the
  [HRAP repository](https://github.com/rnickel1/HRAP_Source).
