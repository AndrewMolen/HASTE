"""Tests for HRAP import, cross-referencing, and the constant-O/F model.

The unit-conversion tests use hand-checked values from HRAP's own shipped
motor configurations, so they verify the importer against real files rather
than against its own assumptions.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from n2o_injector import (  # noqa: E402
    FlowModel,
    Grain,
    InjectorSpec,
    MotorConfig,
    Nozzle,
    Propellant,
    SaturationTable,
    SizingTarget,
    Tank,
    get_backend,
    simulate,
    size_injector,
)
from n2o_injector.hrap_io import (  # noqa: E402
    DENSITY,
    LENGTH,
    MASS,
    PRESSURE,
    VOLUME,
    compare_to_hrap,
    format_comparison,
    load_hrap_motor,
    load_hrap_output,
)
from n2o_injector.motor import fuel_flow  # noqa: E402
from n2o_injector.report import export_hrap_mat  # noqa: E402

HRAP_CSV_HEADER = (
    "Time (s),Thrust (N),Oxidizer Tank Pressure (kPa),"
    "Combustion Chamber Pressure (kPa),Injector Pressure Drop (kPa),"
    "Oxidizer Mass (kg),Fuel Mass (kg),Oxidizer Mass Flow Rate (kg/s),"
    "Fuel Mass Flow Rate (kg/s),Exhaust Mass Flow Rate (kg/s),"
    "Oxidizer to Fuel Ratio,Grain ID (cm),Regression Rate (mm/s)"
)


@pytest.fixture(scope="module")
def props():
    return get_backend("esdu")


@pytest.fixture(scope="module")
def table(props):
    return SaturationTable(props)


# ==========================================================================
# Unit tables
# ==========================================================================


def test_unit_tables_match_hrap():
    """Factors taken verbatim from HRAP.mlapp's run_sim initialiser."""
    assert LENGTH["in"] == 0.0254
    assert LENGTH["ft"] == 0.3048
    assert LENGTH["mm"] == 0.001
    assert VOLUME["cm^3"] == pytest.approx(1e-6)
    assert VOLUME["L"] == 0.001
    assert VOLUME["Gal"] == 0.00378541
    assert MASS["lbm"] == 0.453592
    assert MASS["oz"] == 0.0283495
    assert DENSITY["g/cm^3"] == 1000.0
    # HRAP defines psi as 101325/14.696, not the SI 6894.757.
    assert PRESSURE["psi"] == pytest.approx(101325.0 / 14.696)
    assert PRESSURE["atm"] == 101325.0
    assert PRESSURE["psf"] == pytest.approx(PRESSURE["psi"] * 144.0)


# ==========================================================================
# Motor config import
# ==========================================================================


def _write_cfg(tmp_path, name="motor.mat", **overrides):
    """Build a minimal but valid HRAP motor config."""
    cfg = {
        "mtr_nm": "test_motor",
        "tnk_V": 12000.0, "tnk_V_unit": "cm^3", "tnk_V_state": 0,
        "tnk_L": 0.0, "tnk_L_unit": "in", "tnk_D": 0.0, "tnk_D_unit": "in",
        "tnk_dd": "Starting Tank Temperature", "tnk_cond": 293.15, "T_tnk_unit": "K",
        "fill_dd": "Starting Oxidizer Mass", "fill": 7.0, "fill_unit": "kg",
        "cmbr_V_state": 1, "cmbr_V": 0.0, "cmbr_V_unit": "cm^3",
        "P_cmbr": 1.0, "P_cmbr_unit": "atm", "Pa": 1.0, "Pa_unit": "atm",
        "noz_thrt": 32.0, "noz_thrt_unit": "mm",
        "noz_def": "Nozzle Expansion Ratio", "noz_ex": 4.0, "noz_ex_unit": "mm",
        "noz_eff": 95.0, "noz_Cd": 0.95,
        "grn_ID": 38.0, "grn_ID_unit": "mm",
        "grn_OD": 92.0, "grn_OD_unit": "mm",
        "grn_L": 450.0, "grn_L_unit": "mm",
        "prop_file": "", "prop_nm": "TestProp",
        "prop_rho": 900.0, "prop_rho_unit": "kg/m^3",
        "prop_a": 0.0304, "prop_n": 0.681, "prop_m": 0.0,
        "const_OF": 8.27, "cstar_eff": 95.0, "reg_model": "Shifting OF",
        "inj_D": 1.6, "inj_D_unit": "mm", "inj_N": 24.0, "inj_Cd": 0.7,
        "vnt_state": "None", "vnt_D": 0.0, "vnt_D_unit": "mm", "vnt_Cd": 0.75,
        "mp_state": 0, "t_max": 30.0, "t_burn": 0.0, "dt": 0.001,
    }
    cfg.update(overrides)
    path = tmp_path / name
    savemat(str(path), {"cfg": cfg}, oned_as="row")
    return str(path)


def test_import_basic_config(tmp_path):
    im = load_hrap_motor(_write_cfg(tmp_path))
    c = im.config
    assert im.motor_name == "test_motor"
    assert c.tank.volume == pytest.approx(0.012)
    assert c.tank.ox_mass == pytest.approx(7.0)
    assert c.tank.fill_temp == pytest.approx(293.15)
    assert c.grain.port_id == pytest.approx(0.038)
    assert c.grain.outer_d == pytest.approx(0.092)
    assert c.grain.length == pytest.approx(0.450)
    assert c.nozzle.throat_d == pytest.approx(0.032)
    assert c.nozzle.expansion_ratio == pytest.approx(4.0)
    assert c.injector.n_holes == 24
    assert c.injector.hole_d == pytest.approx(0.0016)
    assert c.injector.Cd == pytest.approx(0.7)


def test_percentages_are_converted(tmp_path):
    """HRAP stores nozzle and c* efficiency as percentages."""
    im = load_hrap_motor(_write_cfg(tmp_path, noz_eff=97.0, cstar_eff=88.0))
    assert im.config.nozzle.efficiency == pytest.approx(0.97)
    assert im.config.propellant.cstar_eff == pytest.approx(0.88)


def test_imperial_units_convert(tmp_path):
    im = load_hrap_motor(_write_cfg(
        tmp_path, grn_ID=2.1875, grn_ID_unit="in", grn_OD=3.375, grn_OD_unit="in",
        grn_L=16.946, grn_L_unit="in", noz_thrt=1.0, noz_thrt_unit="in",
        inj_D=0.25, inj_D_unit="in", fill=46.74, fill_unit="oz",
    ))
    c = im.config
    assert c.grain.port_id == pytest.approx(2.1875 * 0.0254)
    assert c.grain.outer_d == pytest.approx(3.375 * 0.0254)
    assert c.nozzle.throat_d == pytest.approx(0.0254)
    assert c.injector.hole_d == pytest.approx(0.25 * 0.0254)
    assert c.tank.ox_mass == pytest.approx(46.74 * 0.0283495)


def test_tank_volume_from_dimensions(tmp_path):
    """tnk_V_state == 1 means volume comes from length x diameter."""
    im = load_hrap_motor(_write_cfg(
        tmp_path, tnk_V_state=1, tnk_L=1000.0, tnk_L_unit="mm",
        tnk_D=100.0, tnk_D_unit="mm",
    ))
    assert im.config.tank.volume == pytest.approx(1.0 * 0.25 * math.pi * 0.1**2)
    assert im.info["tank_volume_from"] == "length x diameter"


def test_nozzle_exit_diameter_becomes_expansion_ratio(tmp_path):
    im = load_hrap_motor(_write_cfg(
        tmp_path, noz_def="Nozzle Exit Diameter", noz_ex=1.125, noz_ex_unit="in",
        noz_thrt=0.375, noz_thrt_unit="in",
    ))
    assert im.config.nozzle.expansion_ratio == pytest.approx((1.125 / 0.375) ** 2)


def test_tank_pressure_inverts_to_temperature(tmp_path, props):
    """'Starting Tank Pressure' must be inverted through the saturation curve."""
    im = load_hrap_motor(_write_cfg(
        tmp_path, tnk_dd="Starting Tank Pressure", tnk_cond=50.0, T_tnk_unit="Bar",
    ))
    T = im.config.tank.fill_temp
    assert props.P_sat(T) == pytest.approx(50e5, rel=1e-4)


def test_fill_percentage_resolves_to_mass(tmp_path, props):
    """HRAP: m_ox = f*V*rho_l + (1-f)*V*rho_v at the fill temperature."""
    im = load_hrap_motor(_write_cfg(
        tmp_path, fill_dd="Tank Fill Percentage", fill=95.0, fill_unit="%",
    ))
    sat = props.sat(293.15)
    expected = 0.95 * 0.012 * sat.rho_l + 0.05 * 0.012 * sat.rho_v
    assert im.config.tank.ox_mass == pytest.approx(expected, rel=1e-6)


def test_temperature_units(tmp_path):
    for value, unit, expected in [
        (20.0, "C", 293.15),
        (293.15, "K", 293.15),
        (68.0, "F", 293.15),
        (527.67, "R", 293.15),
    ]:
        im = load_hrap_motor(_write_cfg(tmp_path, tnk_cond=value, T_tnk_unit=unit))
        assert im.config.tank.fill_temp == pytest.approx(expected, abs=0.02), unit


def test_constant_OF_model_is_detected(tmp_path):
    im = load_hrap_motor(_write_cfg(tmp_path, reg_model="Constant OF", const_OF=6.71))
    assert im.config.regression_mode == "constant_OF"
    assert im.config.const_OF == pytest.approx(6.71)


def test_import_defaults_to_hrap_models(tmp_path):
    """Matching HRAP means SPI flow and the transient chamber ODE."""
    im = load_hrap_motor(_write_cfg(tmp_path), match_hrap_models=True)
    assert im.config.injector.model is FlowModel.SPI
    assert im.config.chamber_mode == "transient-hrap"

    im2 = load_hrap_motor(_write_cfg(tmp_path), match_hrap_models=False)
    assert im2.config.injector.model is FlowModel.DYER
    assert any("Dyer" in w for w in im2.warnings)


def test_vent_is_flagged_not_silently_ignored(tmp_path):
    im = load_hrap_motor(_write_cfg(tmp_path, vnt_state="External", vnt_D=0.7))
    assert any("vent" in w.lower() for w in im.warnings)


def test_missing_propellant_is_flagged(tmp_path):
    im = load_hrap_motor(_write_cfg(tmp_path, prop_nm="NoSuchFuel"))
    assert not im.config.propellant.has_table
    assert any("propellant" in w.lower() for w in im.warnings)
    assert any("constant c*" in w.lower() for w in im.warnings)


def test_rejects_propellant_config_as_motor(tmp_path):
    """A propellant .mat must give a clear error, not a confusing one."""
    path = tmp_path / "Paraffin.mat"
    savemat(str(path), {"s": {
        "prop_Pc": np.linspace(5e5, 5e6, 10),
        "prop_OF": np.linspace(1, 10, 19).reshape(-1, 1),
        "prop_k": np.full((10, 19), 1.2),
        "prop_M": np.full((10, 19), 29.0),
        "prop_T": np.full((10, 19), 3000.0),
        "prop_nm": "Paraffin", "prop_Rho": 900.0, "opt_OF": 8.27,
    }})
    with pytest.raises(ValueError, match="not an HRAP motor configuration"):
        load_hrap_motor(str(path))


def test_imported_config_validates_and_simulates(tmp_path, props, table):
    im = load_hrap_motor(_write_cfg(tmp_path))
    assert im.config.validate(props) == []
    burn = simulate(im.config, props, table)
    assert len(burn.t) > 10
    assert burn.total_impulse > 0


def test_export_import_roundtrip(tmp_path, props, table):
    """A config exported for HRAP must import back to the same motor."""
    cfg = MotorConfig(
        tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
        grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
        nozzle=Nozzle(throat_d=0.032, expansion_ratio=4.0, Cd=0.95, efficiency=0.95),
        injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.7, plate_thickness=0.003),
        propellant=Propellant(),
        dt=0.01,
    )
    res = size_injector(cfg, props, table, SizingTarget(OF=8.27, objective="burn_average"))

    path = str(tmp_path / "exported.mat")
    export_hrap_mat(path, res, cfg, motor_name="roundtrip")
    back = load_hrap_motor(path)

    assert back.motor_name == "roundtrip"
    assert back.config.tank.volume == pytest.approx(cfg.tank.volume, rel=1e-9)
    assert back.config.tank.ox_mass == pytest.approx(cfg.tank.ox_mass, rel=1e-9)
    assert back.config.grain.length == pytest.approx(cfg.grain.length, rel=1e-9)
    assert back.config.grain.port_id == pytest.approx(cfg.grain.port_id, rel=1e-9)
    assert back.config.nozzle.throat_d == pytest.approx(cfg.nozzle.throat_d, rel=1e-9)
    assert back.config.nozzle.efficiency == pytest.approx(cfg.nozzle.efficiency, rel=1e-9)
    assert back.config.injector.n_holes == res.plate.n_holes
    assert back.config.injector.hole_d == pytest.approx(res.plate.hole_d, rel=1e-9)
    assert back.config.injector.Cd == pytest.approx(res.plate.Cd, rel=1e-9)


# ==========================================================================
# Constant-O/F regression
# ==========================================================================


def test_constant_OF_pins_the_ratio():
    cfg = MotorConfig(regression_mode="constant_OF", const_OF=6.5)
    for mdot_ox in (0.5, 1.5, 3.0):
        rdot, mdot_f, OF = fuel_flow(cfg, cfg.grain.port_id, mdot_ox)
        assert OF == pytest.approx(6.5)
        assert mdot_f == pytest.approx(mdot_ox / 6.5)
        assert rdot > 0


def test_constant_OF_regression_rate_matches_hrap():
    """HRAP const_OF.m: rdot = mdot_f / (rho * pi * ID * L)."""
    cfg = MotorConfig(regression_mode="constant_OF", const_OF=6.5)
    d = cfg.grain.port_id
    rdot, mdot_f, _ = fuel_flow(cfg, d, 2.0)
    expected = mdot_f / (cfg.propellant.rho_fuel * math.pi * d * cfg.grain.length)
    assert rdot == pytest.approx(expected, rel=1e-12)


def test_constant_OF_burn_holds_target(props, table):
    cfg = MotorConfig(
        tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
        grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
        nozzle=Nozzle(throat_d=0.032),
        injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.7),
        propellant=Propellant(), dt=0.01,
        regression_mode="constant_OF", const_OF=6.71,
    )
    burn = simulate(cfg, props, table)
    ok = np.isfinite(burn.OF)
    assert np.allclose(burn.OF[ok], 6.71, rtol=1e-9)
    assert burn.mean_OF == pytest.approx(6.71, rel=1e-6)


def test_sizing_rejects_constant_OF(props, table):
    """O/F is pinned regardless of area, so sizing must fail loudly."""
    cfg = MotorConfig(
        tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
        grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
        nozzle=Nozzle(throat_d=0.032),
        injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.7),
        propellant=Propellant(), dt=0.01,
        regression_mode="constant_OF", const_OF=8.0,
    )
    with pytest.raises(ValueError, match="constant-O/F"):
        size_injector(cfg, props, table, SizingTarget(OF=8.0))


def test_shifting_mode_unaffected(props, table):
    """The default path must behave exactly as before constant-O/F was added."""
    cfg = MotorConfig(
        tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
        grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
        nozzle=Nozzle(throat_d=0.032),
        injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.7),
        propellant=Propellant(), dt=0.01,
    )
    assert cfg.regression_mode == "shifting"
    d = cfg.grain.port_id
    rdot, mdot_f, OF = fuel_flow(cfg, d, 1.5)
    G = 1.5 / cfg.grain.port_area(d)
    assert rdot == pytest.approx(cfg.propellant.regression_rate(G, cfg.grain.length))
    assert OF == pytest.approx(1.5 / mdot_f)


# ==========================================================================
# Output CSV import
# ==========================================================================


def _write_csv(tmp_path, burn, scale=None, name="HRAP_output.csv"):
    scale = scale or {}
    path = tmp_path / name
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HRAP_CSV_HEADER + "\n")
        for i in range(len(burn.t)):
            row = [
                burn.t[i],
                burn.thrust[i] * scale.get("thrust", 1.0),
                burn.P_tank[i] / 1e3,
                burn.P_chamber[i] / 1e3 * scale.get("P_chamber", 1.0),
                -0.008,  # HRAP's mislabelled column: tank dP per timestep
                burn.m_ox[i], burn.m_fuel[i],
                burn.mdot_ox[i] * scale.get("mdot_ox", 1.0),
                burn.mdot_fuel[i], 0.0, burn.OF[i],
                burn.port_d[i] * 100.0, 0.5,
            ]
            f.write(",".join(f"{v:g}" for v in row) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def demo_burn(props, table):
    cfg = MotorConfig(
        tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
        grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
        nozzle=Nozzle(throat_d=0.032),
        injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.7, model=FlowModel.SPI),
        propellant=Propellant(), dt=0.01,
    )
    return simulate(cfg, props, table)


def test_csv_units_are_converted(tmp_path, demo_burn):
    """kPa -> Pa, cm -> m, mm/s -> m/s."""
    run = load_hrap_output(_write_csv(tmp_path, demo_burn))
    assert run.channels["P_tank"][0] == pytest.approx(demo_burn.P_tank[0], rel=1e-5)
    assert run.channels["port_d"][0] == pytest.approx(demo_burn.port_d[0], rel=1e-5)
    assert run.channels["rdot"][0] == pytest.approx(5e-4, rel=1e-9)
    assert run.t[0] == pytest.approx(demo_burn.t[0])


def test_csv_recomputes_true_injector_dP(tmp_path, demo_burn):
    """HRAP's dP column is the tank pressure step, so it must be recomputed."""
    run = load_hrap_output(_write_csv(tmp_path, demo_burn))
    assert "dP_inj" in run.channels
    expected = demo_burn.P_tank[0] - demo_burn.P_chamber[0]
    assert run.channels["dP_inj"][0] == pytest.approx(expected, rel=1e-5)
    # The original column is preserved separately, and the discrepancy flagged.
    assert run.channels["dP_reported"][0] == pytest.approx(-8.0)
    assert any("tank pressure change" in w for w in run.warnings)


def test_csv_rejects_non_hrap_file(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like an HRAP output"):
        load_hrap_output(str(path))


def test_csv_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hrap_output(str(path))


def test_csv_handles_mass_property_columns(tmp_path, demo_burn):
    """The 15-column variant adds motor mass and CG."""
    header = HRAP_CSV_HEADER.replace(
        "Fuel Mass (kg),", "Fuel Mass (kg),Total Motor Mass (kg),"
    ) + ",Motor Center of Mass (cm)"
    path = tmp_path / "mp.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for i in range(len(demo_burn.t)):
            f.write(",".join(f"{v:g}" for v in [
                demo_burn.t[i], demo_burn.thrust[i], demo_burn.P_tank[i] / 1e3,
                demo_burn.P_chamber[i] / 1e3, -0.008, demo_burn.m_ox[i],
                demo_burn.m_fuel[i], 20.0, demo_burn.mdot_ox[i],
                demo_burn.mdot_fuel[i], 0.0, demo_burn.OF[i],
                demo_burn.port_d[i] * 100.0, 0.5, 55.0,
            ]) + "\n")
    run = load_hrap_output(str(path))
    assert run.has("m_total") and run.has("cg")
    assert run.channels["m_total"][0] == pytest.approx(20.0)
    assert run.channels["cg"][0] == pytest.approx(0.55)
    # The extra columns must not shift the others.
    assert run.channels["P_tank"][0] == pytest.approx(demo_burn.P_tank[0], rel=1e-5)


# ==========================================================================
# Comparison
# ==========================================================================


def test_identical_runs_compare_to_zero_error(tmp_path, demo_burn):
    run = load_hrap_output(_write_csv(tmp_path, demo_burn))
    cmp = compare_to_hrap(demo_burn, run)
    assert cmp.channels
    for c in cmp.channels:
        assert c.rms_pct < 0.05, f"{c.label} should match itself"
    assert cmp.impulse_ours == pytest.approx(cmp.impulse_hrap, rel=1e-3)


def test_comparison_detects_injected_error(tmp_path, demo_burn):
    """A known perturbation must show up at roughly its true size."""
    run = load_hrap_output(
        _write_csv(tmp_path, demo_burn, scale={"mdot_ox": 1.10}, name="scaled.csv")
    )
    cmp = compare_to_hrap(demo_burn, run)
    by_key = {c.key: c for c in cmp.channels}
    assert by_key["mdot_ox"].rms_pct > 5.0
    assert by_key["mdot_ox"].bias_pct < 0  # ours is lower than the inflated HRAP
    # Untouched channels must stay clean.
    assert by_key["P_tank"].rms_pct < 0.05


def test_comparison_handles_mismatched_burn_times(tmp_path, demo_burn, props, table):
    """Runs of different length must compare over their overlap and say so."""
    run = load_hrap_output(_write_csv(tmp_path, demo_burn))
    n = len(demo_burn.t) // 2
    truncated = type(demo_burn)(
        **{k: (v[:n] if isinstance(v, np.ndarray) else v)
           for k, v in demo_burn.__dict__.items()}
    )
    cmp = compare_to_hrap(truncated, run)
    assert cmp.overlap == pytest.approx(float(truncated.t[-1]), rel=1e-6)
    assert any("burn times differ" in w for w in cmp.warnings)


def test_format_comparison_renders(tmp_path, demo_burn):
    run = load_hrap_output(_write_csv(tmp_path, demo_burn))
    text = format_comparison(compare_to_hrap(demo_burn, run), "SPI")
    assert "CROSS-REFERENCE vs HRAP" in text
    assert "total impulse" in text
    assert "oxidiser mass flow" in text
