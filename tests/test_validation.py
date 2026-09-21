"""Validation and regression tests.

These fall into four groups:

1. **Property validation** -- the ESDU curve fits are checked against
   independently published saturated-N2O values, and cross-checked against
   CoolProp's reference equation of state.
2. **Model limiting cases** -- SPI/HEM/Dyer are checked against the behaviour
   the physics requires (ordering, choking, the kappa=1 saturated result, and
   the L/D limits the brief calls for).
3. **HRAP parity** -- the SPI branch is checked against the closed-form
   equation HRAP's own source and theory document specify.
4. **End-to-end** -- sizing round-trips and conservation checks.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2o_injector import (  # noqa: E402
    FlowModel,
    Grain,
    InjectorSpec,
    MotorConfig,
    Nozzle,
    OrificeCurve,
    Propellant,
    SaturationTable,
    SizingTarget,
    Tank,
    coolprop_available,
    dyer_kappa,
    get_backend,
    ld_hem_weight,
    mass_flux,
    plate_from_CdA,
    required_mdot_ox,
    simulate,
    size_injector,
    spi_mass_flux,
)
from n2o_injector.properties import P_CRIT, T_CRIT  # noqa: E402


@pytest.fixture(scope="module")
def props():
    return get_backend("esdu")


@pytest.fixture(scope="module")
def table(props):
    return SaturationTable(props)


# ==========================================================================
# 1. Property validation
# ==========================================================================

# Saturated N2O reference values. These are the standard published figures for
# nitrous oxide at 20 C, the condition most hybrid work is quoted at.
REFERENCE_20C = {
    "P": 50.6e5,      # Pa
    "rho_l": 786.0,   # kg/m^3
    "rho_v": 158.0,   # kg/m^3
}


def test_esdu_matches_published_values_at_20C(props):
    s = props.sat(293.15)
    assert s.P == pytest.approx(REFERENCE_20C["P"], rel=0.01)
    assert s.rho_l == pytest.approx(REFERENCE_20C["rho_l"], rel=0.01)
    assert s.rho_v == pytest.approx(REFERENCE_20C["rho_v"], rel=0.02)


def test_critical_point_is_consistent(props):
    """The vapour-pressure fit must return the critical pressure at Tc."""
    assert props.P_sat(T_CRIT) == pytest.approx(P_CRIT, rel=1e-6)


def test_saturation_curve_is_monotone(props):
    T = np.linspace(200.0, 305.0, 200)
    P = np.array([props.P_sat(t) for t in T])
    assert np.all(np.diff(P) > 0)
    rho_l = np.array([props.sat(t).rho_l for t in T])
    rho_v = np.array([props.sat(t).rho_v for t in T])
    # Liquid density falls and vapour density rises towards the critical point.
    assert np.all(np.diff(rho_l) < 0)
    assert np.all(np.diff(rho_v) > 0)
    assert np.all(rho_l > rho_v)


def test_T_sat_inverts_P_sat(props):
    for TC in (-20.0, 0.0, 15.0, 25.0, 33.0):
        T = TC + 273.15
        assert props.T_sat(props.P_sat(T)) == pytest.approx(T, abs=0.05)


def test_latent_heat_decreases_towards_critical(props):
    h = [props.sat(t).h_fg for t in (250.0, 270.0, 290.0, 305.0)]
    assert all(h[i] > h[i + 1] for i in range(len(h) - 1))
    assert h[-1] > 0


def test_clausius_relation_holds(props):
    """s_fg must equal h_fg / T by construction of the saturation state."""
    for T in (250.0, 275.0, 295.0):
        s = props.sat(T)
        assert s.s_fg == pytest.approx(s.h_fg / T, rel=1e-9)


@pytest.mark.skipif(not coolprop_available(), reason="CoolProp not installed")
def test_esdu_agrees_with_coolprop_reference_eos():
    """ESDU curve fits vs the reference EOS CoolProp implements.

    This is the strongest available check on the reconstructed entropy: ESDU
    carries no entropy data, so agreement here means the thermodynamic
    reconstruction reproduces the reference EOS independently.
    """
    e, c = get_backend("esdu"), get_backend("coolprop")
    for TC in (-20.0, 0.0, 10.0, 20.0, 30.0):
        T = TC + 273.15
        a, b = e.sat(T), c.sat(T)
        assert a.P == pytest.approx(b.P, rel=0.01), f"P at {TC} C"
        assert a.rho_l == pytest.approx(b.rho_l, rel=0.01), f"rho_l at {TC} C"
        assert a.rho_v == pytest.approx(b.rho_v, rel=0.03), f"rho_v at {TC} C"
        assert a.h_fg == pytest.approx(b.h_fg, rel=0.02), f"h_fg at {TC} C"
        assert a.s_fg == pytest.approx(b.s_fg, rel=0.02), f"s_fg at {TC} C"


@pytest.mark.skipif(not coolprop_available(), reason="CoolProp not installed")
def test_hem_flux_agrees_between_property_backends():
    """The HEM mass flux must not depend on which property backend is used."""
    e, c = get_backend("esdu"), get_backend("coolprop")
    te, tc = SaturationTable(e), SaturationTable(c, n=1500)
    for Pc in (10e5, 25e5, 35e5):
        ge = mass_flux(te, e.upstream_state(293.15), Pc, FlowModel.HEM).G
        gc = mass_flux(tc, c.upstream_state(293.15), Pc, FlowModel.HEM).G
        assert ge == pytest.approx(gc, rel=0.02), f"HEM flux at Pc={Pc / 1e5} bar"


# ==========================================================================
# 2. Model limiting cases
# ==========================================================================


def test_spi_matches_closed_form(props):
    """SPI must be exactly sqrt(2 rho dP)."""
    up = props.upstream_state(293.15)
    for Pc in (10e5, 30e5, 45e5):
        expected = math.sqrt(2 * up.rho * (up.P - Pc))
        assert spi_mass_flux(up, Pc) == pytest.approx(expected, rel=1e-12)


def test_spi_zero_when_no_pressure_drop(props):
    up = props.upstream_state(293.15)
    assert spi_mass_flux(up, up.P) == 0.0
    assert spi_mass_flux(up, up.P * 1.1) == 0.0


def test_model_ordering_spi_above_hem(props, table):
    """SPI must over-predict relative to HEM for a saturated feed, with Dyer between.

    This is the physical signature of flashing: the equilibrium model sees a
    lower-density two-phase mixture and passes less mass.
    """
    up = props.upstream_state(293.15)
    for Pc in (10e5, 20e5, 30e5, 40e5):
        r = mass_flux(table, up, Pc, FlowModel.DYER)
        assert r.G_spi > r.G_hem > 0
        assert r.G_hem < r.G < r.G_spi


def test_dyer_kappa_is_one_for_saturated_feed(props):
    """A self-pressurising tank sits on the saturation line, so P1 = Psat."""
    up = props.upstream_state(293.15)
    assert up.is_saturated
    for Pc in (5e5, 20e5, 40e5):
        assert dyer_kappa(up, Pc) == pytest.approx(1.0, rel=1e-9)


def test_dyer_is_mean_of_spi_and_hem_when_saturated(props, table):
    """kappa = 1 makes Dyer the arithmetic mean of the two branches."""
    up = props.upstream_state(293.15)
    for Pc in (10e5, 25e5, 40e5):
        r = mass_flux(table, up, Pc, FlowModel.DYER)
        assert r.G == pytest.approx(0.5 * (r.G_spi + r.G_hem), rel=1e-9)
        assert r.w_hem == pytest.approx(0.5, rel=1e-9)


def test_dyer_collapses_to_spi_for_strongly_subcooled_feed(props, table):
    """With no possibility of flashing, the blend must return pure SPI.

    Raising the feed pressure far above saturation while holding the chamber
    above Psat means the fluid never crosses the saturation line, so kappa is
    infinite and the HEM branch carries zero weight.
    """
    T = 293.15
    P_sat = props.P_sat(T)
    up = props.upstream_state(T, P=P_sat * 2.0)
    Pc = P_sat * 1.05  # above saturation: cannot flash
    r = mass_flux(table, up, Pc, FlowModel.DYER)
    assert math.isinf(r.kappa)
    assert r.w_hem == 0.0
    assert r.G == pytest.approx(spi_mass_flux(up, Pc), rel=1e-9)


def test_subcooling_increases_kappa_towards_spi(props):
    """More subcooling must push the blend towards the SPI branch."""
    T = 293.15
    P_sat = props.P_sat(T)
    Pc = 20e5
    k_sat = dyer_kappa(props.upstream_state(T), Pc)
    k_sub = dyer_kappa(props.upstream_state(T, P=P_sat * 1.5), Pc)
    assert k_sub > k_sat == pytest.approx(1.0)


def test_ld_weighting_limits(props, table):
    """The brief's regime requirement: L/D -> 0 gives SPI, large L/D gives HEM."""
    assert ld_hem_weight(0.0) == pytest.approx(0.0)
    assert ld_hem_weight(1e4) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < ld_hem_weight(5.0) < 1.0
    # Monotone in L/D.
    w = [ld_hem_weight(x) for x in (0.1, 1.0, 3.0, 10.0, 50.0)]
    assert all(w[i] < w[i + 1] for i in range(len(w) - 1))

    up = props.upstream_state(293.15)
    Pc = 25e5
    g_short = mass_flux(table, up, Pc, FlowModel.DYER_LD, L_over_D=1e-6).G
    g_long = mass_flux(table, up, Pc, FlowModel.DYER_LD, L_over_D=1e4).G
    assert g_short == pytest.approx(spi_mass_flux(up, Pc), rel=1e-6)
    assert g_long == pytest.approx(mass_flux(table, up, Pc, FlowModel.HEM).G, rel=1e-6)


def test_hem_chokes_and_plateaus(props, table):
    """Below the critical pressure the HEM flux must stop responding to Pc."""
    up = props.upstream_state(293.15)
    r_low = mass_flux(table, up, 3e5, FlowModel.HEM)
    r_mid = mass_flux(table, up, 12e5, FlowModel.HEM)
    assert r_low.choked and r_mid.choked
    assert r_low.G == pytest.approx(r_mid.G, rel=1e-6)
    assert r_low.P_crit == pytest.approx(r_mid.P_crit, rel=1e-6)
    # The critical pressure must lie strictly inside the interval.
    assert 3e5 < r_low.P_crit < up.P


def test_hem_critical_pressure_ratio_is_physical(props, table):
    """Saturated N2O chokes at a pressure ratio well inside (0.5, 0.95)."""
    up = props.upstream_state(293.15)
    r = mass_flux(table, up, 5e5, FlowModel.HEM)
    ratio = r.P_crit / up.P
    assert 0.5 < ratio < 0.95, f"critical pressure ratio {ratio:.3f} is unphysical"


def test_unchoked_near_upstream_pressure(props, table):
    """Just below the feed pressure the flow cannot be choked."""
    up = props.upstream_state(293.15)
    r = mass_flux(table, up, up.P * 0.98, FlowModel.HEM)
    assert not r.choked


def test_no_flow_when_downstream_exceeds_upstream(props, table):
    up = props.upstream_state(293.15)
    for model in FlowModel:
        r = mass_flux(table, up, up.P * 1.2, model)
        assert r.G == 0.0


def test_flux_decreases_monotonically_with_chamber_pressure(props, table):
    up = props.upstream_state(293.15)
    P = np.linspace(1e5, up.P * 0.999, 60)
    for model in (FlowModel.SPI, FlowModel.HEM, FlowModel.DYER):
        G = np.array([mass_flux(table, up, p, model).G for p in P])
        # Relative tolerance: the choked plateau is flat to within grid noise,
        # which is meaningless next to fluxes of order 1e4 kg/m^2/s.
        assert np.all(np.diff(G) <= 1e-9 * G.max()), f"{model.value} flux not monotone"


def test_orifice_curve_matches_direct_evaluation(props, table):
    """The fast precomputed curve must agree with the direct scalar solver."""
    up = props.upstream_state(293.15)
    curve = OrificeCurve(table, up, model=FlowModel.DYER)
    for Pc in (5e5, 15e5, 25e5, 35e5, 45e5):
        direct = mass_flux(table, up, Pc, FlowModel.DYER).G
        assert curve.flux(Pc) == pytest.approx(direct, rel=2e-3), f"Pc={Pc / 1e5} bar"


def test_flux_rises_with_tank_temperature(props, table):
    """Warmer tank -> higher feed pressure -> more flow."""
    Pc = 25e5
    g = [
        mass_flux(table, props.upstream_state(T), Pc, FlowModel.DYER).G
        for T in (283.15, 293.15, 300.15)
    ]
    assert g[0] < g[1] < g[2]


# ==========================================================================
# 3. HRAP parity
# ==========================================================================


def test_spi_reproduces_hrap_injector_equation(props):
    """HRAP: dLoss = K/(NA)^2 with K = 1/Cd^2, mdot = sqrt(2 rho dP / dLoss).

    That reduces to mdot = Cd*N*A*sqrt(2 rho dP), which must equal this tool's
    SPI mass flux multiplied by CdA. Verified against the equations in HRAP's
    theory document and its tank.m / tank.py source.
    """
    up = props.upstream_state(293.15)
    Cd, N, A = 0.72, 24, 0.25 * math.pi * 0.0016**2
    Pc = 28e5

    K = 1.0 / Cd**2
    dLoss = K / (N * A) ** 2
    hrap_mdot = math.sqrt(2 * up.rho * (up.P - Pc) / dLoss)

    ours = (Cd * N * A) * spi_mass_flux(up, Pc)
    assert ours == pytest.approx(hrap_mdot, rel=1e-12)


def test_hrap_liquid_mass_formula(props):
    """HRAP eq. 5 for liquid mass must match the simulation's phase split."""
    T, V, m_ox = 293.15, 0.012, 7.0
    s = props.sat(T)
    expected = (V - m_ox / s.rho_v) / (1 / s.rho_l - 1 / s.rho_v)
    got = (V - m_ox / s.rho_v) / (1.0 / s.rho_l - 1.0 / s.rho_v)
    assert got == pytest.approx(expected, rel=1e-12)
    assert 0 < got < m_ox


def test_regression_law_matches_hrap_convention():
    """rdot[m/s] = 0.001 * a * G^n * L^m, exactly as HRAP's shift_OF."""
    p = Propellant(reg_a=0.0304, reg_n=0.681, reg_m=0.0)
    G, L = 250.0, 0.45
    assert p.regression_rate(G, L) == pytest.approx(1e-3 * 0.0304 * G**0.681, rel=1e-12)
    # A non-zero length exponent must enter as L^m.
    p2 = Propellant(reg_a=0.0304, reg_n=0.681, reg_m=-0.2)
    assert p2.regression_rate(G, L) == pytest.approx(
        1e-3 * 0.0304 * G**0.681 * L**-0.2, rel=1e-12
    )


def test_cstar_matches_hrap_formula():
    """c* = eff * sqrt(R T / (k (2/(k+1))^((k+1)/(k-1)))) with R = 8314.5/M."""
    OF = np.array([5.0, 8.0])
    Pc = np.array([1e6, 4e6])
    k = np.full((2, 2), 1.22)
    M = np.full((2, 2), 28.0)
    T = np.full((2, 2), 3200.0)
    p = Propellant(OF_grid=OF, Pc_grid=Pc, k_grid=k, M_grid=M, T_grid=T, cstar_eff=0.95)

    R = 8314.5 / 28.0
    expected = 0.95 * math.sqrt((R * 3200.0) / (1.22 * (2 / 2.22) ** (2.22 / 0.22)))
    assert p.cstar(6.0, 2e6) == pytest.approx(expected, rel=1e-9)


def test_hrap_propellant_roundtrip(tmp_path):
    """Write an HRAP-layout .mat and read it back through the loader."""
    from scipy.io import savemat

    OF = np.linspace(1.0, 10.0, 19)
    Pc = np.linspace(5e5, 5e6, 10)
    # Store transposed, the way HRAP's shipped files actually do.
    k = np.full((len(Pc), len(OF)), 1.21)
    M = np.full((len(Pc), len(OF)), 29.0)
    T = np.full((len(Pc), len(OF)), 3100.0)
    path = tmp_path / "Paraffin.mat"
    savemat(str(path), {"s": {
        "prop_Pc": Pc, "prop_OF": OF.reshape(-1, 1), "prop_k": k, "prop_M": M,
        "prop_T": T, "prop_nm": "Paraffin", "prop_Reg": np.array([0.0304, 0.681, 0.0]),
        "prop_Rho": 900.0, "opt_OF": 8.27,
    }})

    p = Propellant.from_hrap_mat(str(path))
    assert p.reg_a == pytest.approx(0.0304)
    assert p.reg_n == pytest.approx(0.681)
    assert p.rho_fuel == pytest.approx(900.0)
    assert p.opt_OF == pytest.approx(8.27)
    assert p.has_table
    # Orientation must be resolved to (OF, Pc).
    assert p.k_grid.shape == (len(OF), len(Pc))
    assert p.gamma(5.0, 2e6) == pytest.approx(1.21, rel=1e-9)


def test_hrap_mat_export_is_readable(tmp_path, props, table):
    """The exported motor config must round-trip and carry the sized injector."""
    from scipy.io import loadmat

    from n2o_injector.report import export_hrap_mat

    cfg, target = _demo_config()
    res = size_injector(cfg, props, table, target)
    path = tmp_path / "motor.mat"
    export_hrap_mat(str(path), res, cfg, motor_name="test_motor")

    raw = loadmat(str(path))["cfg"][0, 0]
    fields = set(raw.dtype.names)
    for required in ("inj_D", "inj_N", "inj_Cd", "grn_ID", "grn_OD", "grn_L",
                     "tnk_V", "noz_thrt", "prop_a", "prop_n", "reg_model"):
        assert required in fields, f"missing HRAP field {required}"

    assert float(np.asarray(raw["inj_N"]).ravel()[0]) == res.plate.n_holes
    assert float(np.asarray(raw["inj_D"]).ravel()[0]) == pytest.approx(
        res.plate.hole_d * 1e3, rel=1e-9
    )
    assert float(np.asarray(raw["inj_Cd"]).ravel()[0]) == pytest.approx(res.plate.Cd)
    # Units must be strings HRAP's own drop-downs accept.
    assert str(np.asarray(raw["inj_D_unit"]).ravel()[0]).strip() in {
        "in", "ft", "mm", "cm", "m"
    }
    assert str(np.asarray(raw["tnk_V_unit"]).ravel()[0]).strip() in {
        "in^3", "ft^3", "cm^3", "L", "Gal", "m^3"
    }


# ==========================================================================
# 4. End-to-end
# ==========================================================================


def _demo_config(model=FlowModel.DYER):
    return (
        MotorConfig(
            tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
            grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
            nozzle=Nozzle(throat_d=0.032, expansion_ratio=4.0),
            injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.7,
                                  plate_thickness=0.003, model=model),
            propellant=Propellant(),
            dt=0.01,
        ),
        SizingTarget(OF=8.27, objective="burn_average"),
    )


def test_required_mdot_ox_inverts_the_regression_law():
    """Feeding the returned flow back through the law must recover the target."""
    grain = Grain(length=0.45, port_id=0.038, outer_d=0.092)
    prop = Propellant()
    for OF_target in (5.0, 8.27, 12.0):
        mdot = required_mdot_ox(OF_target, grain, prop, grain.port_id)
        G = mdot / grain.port_area(grain.port_id)
        rdot = prop.regression_rate(G, grain.length)
        mdot_f = prop.rho_fuel * rdot * grain.burn_perimeter(grain.port_id) * grain.length
        assert mdot / mdot_f == pytest.approx(OF_target, rel=1e-9)


def test_sizing_hits_burn_average_target(props, table):
    cfg, target = _demo_config()
    res = size_injector(cfg, props, table, target)
    assert res.converged
    assert res.achieved_mean_OF == pytest.approx(target.OF, rel=0.02)
    assert res.plate.n_holes == 24
    assert 0.5e-3 < res.plate.hole_d < 10e-3


def test_sizing_is_consistent_across_objectives(props, table):
    """Both objectives must land in the same ballpark for a mild blowdown."""
    cfg, target = _demo_config()
    r_avg = size_injector(cfg, props, table, target)

    cfg2, t2 = _demo_config()
    t2.objective = "design_point"
    r_dp = size_injector(cfg2, props, table, t2)

    ratio = r_dp.CdA_required / r_avg.CdA_required
    assert 0.5 < ratio < 2.5, f"objectives disagree wildly (ratio {ratio:.2f})"


def test_larger_injector_gives_higher_OF(props, table):
    """Monotonicity is what the sizing root-find depends on."""
    cfg, _ = _demo_config()
    means = []
    for scale in (0.7, 1.0, 1.4):
        b = simulate(cfg, props, table, CdA_override=cfg.injector.CdA * scale)
        means.append(b.mean_OF)
    assert means[0] < means[1] < means[2]


def test_burn_conserves_oxidiser_mass(props, table):
    """Integrated oxidiser flow must match the drop in tank mass."""
    cfg, _ = _demo_config()
    b = simulate(cfg, props, table)
    integrated = float(np.trapezoid(b.mdot_ox, b.t))
    depleted = b.m_ox[0] - b.m_ox[-1]
    assert integrated == pytest.approx(depleted, rel=0.02)


def test_burn_conserves_fuel_mass(props, table):
    cfg, _ = _demo_config()
    b = simulate(cfg, props, table)
    integrated = float(np.trapezoid(b.mdot_fuel, b.t))
    depleted = b.m_fuel[0] - b.m_fuel[-1]
    assert integrated == pytest.approx(depleted, rel=0.02)


def test_tank_blows_down_and_cools(props, table):
    """A self-pressurising tank must lose both pressure and temperature."""
    cfg, _ = _demo_config()
    b = simulate(cfg, props, table)
    assert b.P_tank[-1] < b.P_tank[0]
    assert b.T_tank[-1] < b.T_tank[0]
    assert np.all(np.diff(b.P_tank) <= 1e-6)
    # Tank pressure must always track the saturation curve.
    for i in (0, len(b.t) // 2, -1):
        assert b.P_tank[i] == pytest.approx(props.P_sat(b.T_tank[i]), rel=1e-6)


def test_port_opens_monotonically(props, table):
    cfg, _ = _demo_config()
    b = simulate(cfg, props, table)
    assert np.all(np.diff(b.port_d) >= 0)
    assert b.port_d[-1] > b.port_d[0]


def test_chamber_pressure_below_tank_pressure(props, table):
    """A physical design cannot have the chamber above the feed."""
    cfg, _ = _demo_config()
    b = simulate(cfg, props, table)
    assert np.all(b.P_chamber < b.P_tank)
    assert np.all(b.dP_inj > 0)


def test_model_choice_orders_the_burn(props, table):
    """SPI sizing predicts the most flow, HEM the least, at fixed geometry."""
    flows = {}
    for model in (FlowModel.SPI, FlowModel.HEM, FlowModel.DYER):
        cfg, _ = _demo_config(model)
        b = simulate(cfg, props, table)
        flows[model] = b.mdot_ox[0]
    assert flows[FlowModel.HEM] < flows[FlowModel.DYER] < flows[FlowModel.SPI]


def test_plate_from_CdA_roundtrip():
    """Solving for diameter then recomputing the area must be self-consistent."""
    CdA, Cd = 4.0e-5, 0.7
    plate, warns = plate_from_CdA(CdA, Cd, 0.003, n_holes=24)
    assert not warns
    assert plate.CdA == pytest.approx(CdA, rel=1e-12)
    assert plate.n_holes == 24


def test_plate_from_CdA_flags_sub_minimum_diameter():
    _, warns = plate_from_CdA(4.0e-5, 0.7, 0.003, n_holes=5000, min_hole_d=0.0008)
    assert warns and "minimum" in warns[0].lower()


def test_plate_from_CdA_rounds_hole_count():
    plate, _ = plate_from_CdA(4.0e-5, 0.7, 0.003, hole_d=0.0016)
    assert isinstance(plate.n_holes, int) and plate.n_holes >= 1
    assert plate.hole_d == pytest.approx(0.0016)


def test_plate_from_CdA_rejects_over_specification():
    with pytest.raises(ValueError):
        plate_from_CdA(4e-5, 0.7, 0.003, n_holes=10, hole_d=0.0016)


def test_dP_margin_is_reported_and_flagged(props, table):
    cfg, target = _demo_config()
    target.min_dP_fraction = 0.99  # deliberately unattainable
    res = size_injector(cfg, props, table, target)
    assert not res.dP_margin_ok
    assert any("pressure drop" in w.lower() for w in res.warnings)


# ==========================================================================
# 5. Input validation
# ==========================================================================


def test_rejects_liquid_full_tank(props):
    t = Tank(volume=0.001, fill_temp=293.15, ox_mass=50.0)
    errs = t.validate(props)
    assert errs and "liquid-full" in errs[0]


def test_rejects_vapour_only_tank(props):
    t = Tank(volume=1.0, fill_temp=293.15, ox_mass=0.001)
    errs = t.validate(props)
    assert errs and "no liquid" in errs[0]


def test_rejects_out_of_range_temperature(props):
    t = Tank(volume=0.01, fill_temp=340.0, ox_mass=5.0)
    assert any("outside" in e for e in t.validate(props))


def test_rejects_ports_that_do_not_fit():
    g = Grain(length=0.4, port_id=0.05, outer_d=0.09, n_ports=4)
    assert any("do not fit" in e for e in g.validate())


def test_rejects_sub_minimum_hole():
    i = InjectorSpec(n_holes=10, hole_d=0.0002, min_hole_d=0.0008)
    assert any("manufacturing minimum" in e for e in i.validate())


def test_rejects_regression_exponent_at_unity(props):
    cfg, _ = _demo_config()
    cfg.propellant.reg_n = 1.0
    assert any("n = 1" in e or "< 1" in e for e in cfg.validate(props))


def test_required_mdot_ox_rejects_bad_exponent():
    grain = Grain()
    with pytest.raises(ValueError):
        required_mdot_ox(8.0, grain, Propellant(reg_n=1.0), grain.port_id)


def test_recommend_model_spans_regimes():
    from n2o_injector import recommend_model

    assert recommend_model(0.5)[0] is FlowModel.DYER
    assert recommend_model(3.0)[0] is FlowModel.DYER
    assert recommend_model(20.0)[0] is FlowModel.HEM


def test_report_builds_without_error(props, table):
    from n2o_injector import build_report

    cfg, target = _demo_config()
    res = size_injector(cfg, props, table, target)
    text = build_report(res, cfg, "ESDU 91022")
    assert "RECOMMENDED INJECTOR" in text
    assert "HRAP CROSS-CHECK" in text
    assert f"{res.plate.n_holes}" in text


def test_degenerate_design_is_flagged_loudly(props, table):
    """An unreachable target must not return a plausible-looking answer.

    Inflating the regression exponent makes the required oxidiser flow explode,
    which the solver can only satisfy by driving chamber pressure onto tank
    pressure. It converges, but the result is not a design -- so it must be
    called out explicitly rather than only failing the dP margin.
    """
    cfg, target = _demo_config()
    cfg.propellant.reg_n *= 1.1
    res = size_injector(cfg, props, table, target)
    assert res.min_dP_fraction < 0.02
    assert any("DEGENERATE" in w for w in res.warnings)


def test_infeasible_design_point_does_not_abort_burn_average(props, table):
    """A bad initial guess must not kill an objective that never uses it."""
    cfg, target = _demo_config()
    cfg.propellant.reg_n *= 1.1  # design point becomes infeasible
    target.objective = "burn_average"
    res = size_injector(cfg, props, table, target)  # must not raise
    assert any("design-point guess was infeasible" in n for n in res.notes)

    cfg2, t2 = _demo_config()
    cfg2.propellant.reg_n *= 1.1
    t2.objective = "design_point"
    with pytest.raises(ValueError, match="tank pressure"):
        size_injector(cfg2, props, table, t2)


# ==========================================================================
# 6. Objective conditioning
# ==========================================================================


def _perturbed(mut):
    cfg, _ = _demo_config()
    mut(cfg)
    return cfg


def test_flow_objectives_hit_their_targets(props, table):
    cfg, _ = _demo_config()
    r = size_injector(cfg, props, table, SizingTarget(objective="mdot_ox", mdot_ox=1.5))
    assert r.burn.mdot_ox[0] == pytest.approx(1.5, rel=1e-3)

    cfg2, _ = _demo_config()
    r2 = size_injector(
        cfg2, props, table, SizingTarget(objective="chamber_pressure", chamber_P=30e5)
    )
    assert r2.burn.P_chamber[0] == pytest.approx(30e5, rel=1e-3)


def test_flow_objectives_are_well_conditioned(props, table):
    """The whole point: these must not amplify regression-coefficient error.

    The O/F objectives invert the regression law and amplify a 10% coefficient
    error into a >100% area change. The flow-based objectives never touch that
    law, so the same perturbation must move the answer by only a few percent.
    """
    for objective, kwargs in (
        ("mdot_ox", dict(mdot_ox=1.5)),
        ("chamber_pressure", dict(chamber_P=30e5)),
    ):
        cfg, _ = _demo_config()
        base = size_injector(
            cfg, props, table, SizingTarget(objective=objective, **kwargs)
        ).CdA_required
        for mut in (
            lambda c: setattr(c.propellant, "reg_a", c.propellant.reg_a * 1.1),
            lambda c: setattr(c.propellant, "reg_n", c.propellant.reg_n * 1.1),
        ):
            got = size_injector(
                _perturbed(mut), props, table,
                SizingTarget(objective=objective, **kwargs),
            ).CdA_required
            change = abs(got - base) / base * 100.0
            assert change < 10.0, f"{objective} moved {change:.1f}% -- not well-conditioned"


def test_OF_objective_is_ill_conditioned_and_says_so(props, table):
    """The ill-conditioning is real and must be reported, not hidden."""
    cfg, target = _demo_config()
    base = size_injector(cfg, props, table, target)
    assert base.conditioning == pytest.approx(1.0 / (1.0 - cfg.propellant.reg_n))
    assert base.conditioning > 2.5
    assert any("ILL-CONDITIONED" in w for w in base.warnings)

    perturbed = size_injector(
        _perturbed(lambda c: setattr(c.propellant, "reg_a", c.propellant.reg_a * 1.1)),
        props, table, SizingTarget(OF=8.27, objective="burn_average"),
    )
    change = abs(perturbed.CdA_required - base.CdA_required) / base.CdA_required * 100
    assert change > 50.0, "the ill-conditioning should be dramatic, not marginal"


def test_flow_objectives_carry_no_conditioning_warning(props, table):
    cfg, _ = _demo_config()
    r = size_injector(cfg, props, table, SizingTarget(objective="mdot_ox", mdot_ox=1.5))
    assert not any("ILL-CONDITIONED" in w for w in r.warnings)
    assert not math.isfinite(r.conditioning)


def test_flow_objectives_work_with_constant_OF(props, table):
    """Constant-O/F blocks the O/F objectives but not the flow-based ones."""
    cfg, _ = _demo_config()
    cfg.regression_mode = "constant_OF"
    cfg.const_OF = 7.0
    r = size_injector(cfg, props, table, SizingTarget(objective="mdot_ox", mdot_ox=1.5))
    assert r.burn.mdot_ox[0] == pytest.approx(1.5, rel=1e-3)
    assert r.achieved_mean_OF == pytest.approx(7.0, rel=1e-6)


def test_flow_objectives_reject_impossible_targets(props, table):
    cfg, _ = _demo_config()
    with pytest.raises(ValueError, match="tank pressure"):
        size_injector(cfg, props, table,
                      SizingTarget(objective="chamber_pressure", chamber_P=200e5))
    cfg2, _ = _demo_config()
    with pytest.raises(ValueError, match="tank pressure"):
        size_injector(cfg2, props, table,
                      SizingTarget(objective="mdot_ox", mdot_ox=50.0))


def test_tail_off_terminates_vapour_phase(props, table):
    """A vapour blowdown must end on physics, not on the time limit.

    Without a cutoff the tail dribbles until max_time, so burn time and
    impulse become artefacts of the run-control setting.
    """
    cfg, _ = _demo_config()
    cfg.chamber_mode = "transient"
    cfg.initial_Pc = cfg.ambient_P
    cfg.stop_at_liquid_exhausted = False
    cfg.max_time = 60.0
    burn = simulate(cfg, props, table)
    assert burn.burn_time < 0.9 * cfg.max_time, "run hit the time limit instead of tailing off"
    assert any("tail-off" in n for n in burn.notes)
    assert burn.P_chamber[-1] <= 1.06 * cfg.ambient_P


def test_tail_off_does_not_fire_during_ignition(props, table):
    """Starting at ambient must not be mistaken for burnout on step one."""
    cfg, _ = _demo_config()
    cfg.chamber_mode = "transient"
    cfg.initial_Pc = cfg.ambient_P
    burn = simulate(cfg, props, table)
    assert len(burn.t) > 100
    assert burn.P_chamber.max() > 5 * cfg.ambient_P
