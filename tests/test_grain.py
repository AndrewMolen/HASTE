"""Sector-port grain geometry, the run-valve cutoff, and the grain drawing."""

from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from n2o_injector import SaturationTable, get_backend, simulate  # noqa: E402
from n2o_injector.config_io import load_config, save_config  # noqa: E402
from n2o_injector.grain_drawing import (  # noqa: E402
    grain_layout,
    section_bands,
    write_grain_dxf,
)
from n2o_injector.grain_geometry import (  # noqa: E402
    balanced_sector,
    sector_errors,
    sector_outlines,
    sector_port,
)
from n2o_injector.motor import Grain, MotorConfig, Nozzle, Tank  # noqa: E402
from n2o_injector.sizing import SizingTarget, analyse_geometry  # noqa: E402

OD = 0.085852
AREA = 5 * math.pi / 4 * 0.01778**2  # the V2.2 five-port grain's port area


@pytest.fixture(scope="module")
def props():
    return get_backend("esdu")


@pytest.fixture(scope="module")
def table(props):
    return SaturationTable(props)


def sector_grain(**kw) -> Grain:
    b = balanced_sector(OD, AREA)
    g = Grain(length=0.1778, port_id=b["centre_d"], outer_d=OD, layout="sector",
              ring_web=b["ring_web"], spoke_web=b["spoke_web"], wall_web=b["wall_web"])
    return dataclasses.replace(g, **kw)


def _polygon(port, n=4000):
    """Densely sampled outline of one sector port (mm)."""
    r1, a10, a11 = port["inner"]
    r2, a20, a21 = port["outer"]
    ta = np.radians(np.linspace(a20, a21, n))
    tb = np.radians(np.linspace(a11, a10, n))
    x = np.r_[r2 * np.cos(ta), r1 * np.cos(tb)]
    y = np.r_[r2 * np.sin(ta), r1 * np.sin(tb)]
    return x, y


def _shoelace(x, y):
    a = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    p = np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])))
    return a, p


# --------------------------------------------------------------- geometry


def test_sector_port_area_and_perimeter_match_the_outline():
    port = next(sector_outlines(85.852, 14.0, 17.0, 17.0, 4, 8.5))
    a, p = sector_port(7.0 + 17.0, 42.926 - 8.5, 8.5, 4)
    a_num, p_num = _shoelace(*_polygon(port))
    assert a == pytest.approx(a_num, rel=1e-5)
    assert p == pytest.approx(p_num, rel=1e-5)


def test_balanced_sector_keeps_area_balances_webs_and_hydraulic_diameter():
    b = balanced_sector(OD, AREA)
    g = sector_grain()
    assert g.port_area(g.port_id) == pytest.approx(AREA, rel=1e-9)
    assert b["ring_web"] == pytest.approx(2 * b["wall_web"])
    assert b["spoke_web"] == pytest.approx(2 * b["wall_web"])
    r1 = 0.5 * b["centre_d"] + b["ring_web"]
    a, p = sector_port(r1, 0.5 * OD - b["wall_web"], 0.5 * b["spoke_web"], 4)
    assert 4 * a / p == pytest.approx(b["centre_d"], rel=1e-6)
    # The V2.2 numbers quoted to the user.
    assert b["wall_web"] * 1e3 == pytest.approx(8.764, abs=1e-3)
    assert b["centre_d"] * 1e3 == pytest.approx(14.135, abs=1e-3)


def test_regressed_area_follows_steiner_exactly():
    """Offset area from the closed form vs a brute-force distance field."""
    from scipy.spatial import cKDTree

    from matplotlib.path import Path

    g = sector_grain()
    e = 0.005  # 5 mm of regression, short of the 8.76 mm burnout
    d = g.port_id + 2 * e
    mm = 1e3
    port = next(sector_outlines(OD * mm, g.port_id * mm, g.ring_web * mm,
                                g.spoke_web * mm, 4, g.wall_web * mm))
    x, y = _polygon(port, 1500)
    # Densify every edge, including the two straight sides the arcs leave out.
    xs, ys = np.r_[x, x[0]], np.r_[y, y[0]]
    seg = [np.column_stack([xs[i] + (xs[i + 1] - xs[i]) * t, ys[i] + (ys[i + 1] - ys[i]) * t])
           for i in range(len(x)) for t in [np.linspace(0, 1, 2 + int(
               math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) / 0.01), endpoint=False)]]
    tree = cKDTree(np.vstack(seg))

    h, E = 0.04, e * mm
    X, Y = np.meshgrid(np.arange(x.min() - E, x.max() + E, h) + h / 2,
                       np.arange(y.min() - E, y.max() + E, h) + h / 2)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    dist, _ = tree.query(pts)
    burned = Path(np.column_stack([x, y])).contains_points(pts) | (dist <= E)

    # Whole grain = centre circle + 4 identical sectors, each offset by e.
    sector_mm2 = (g.port_area(d) - 0.25 * math.pi * d * d) / 4 * mm**2
    assert burned.sum() * h * h == pytest.approx(sector_mm2, rel=2e-3)


def test_round_grain_geometry_unchanged():
    g = Grain(length=0.2, port_id=0.02, outer_d=0.09, n_ports=5)
    d = 0.03
    assert g.port_area(d) == pytest.approx(5 * math.pi / 4 * d * d)
    assert g.burn_perimeter(d) == pytest.approx(5 * math.pi * d)
    assert g.fuel_volume(d) == pytest.approx(math.pi / 4 * (0.09**2 - 5 * d * d) * 0.2)
    assert not g.web_is_exact


def test_sector_validation_catches_impossible_layouts():
    assert sector_errors(0.086, 0.014, 0.03, 0.017, 0.02, 4)  # no radial depth
    assert sector_errors(0.086, 0.004, 0.002, 0.03, 0.005, 4)  # spokes too thick
    assert not sector_grain().validate()
    assert Grain(layout="hexagon").validate()


# ------------------------------------------------------------- simulation


def _cfg(grain, valve=None, reg=(0.0781, 0.545)) -> MotorConfig:
    cfg = MotorConfig(
        tank=Tank(volume=0.0122, fill_temp=293.15, ox_mass=4.48),
        grain=grain, nozzle=Nozzle(throat_d=0.03048, expansion_ratio=4.0),
        dt=0.005, stop_at_liquid_exhausted=False, valve_close_t=valve,
    )
    cfg.injector.n_holes, cfg.injector.hole_d = 40, 0.0012
    cfg.injector.plate_thickness = 0.0127
    cfg.propellant.reg_a, cfg.propellant.reg_n = reg
    return cfg


def test_valve_closes_the_burn_on_time(props, table):
    b = simulate(_cfg(sector_grain(), valve=3.0), props, table)
    assert b.burn_time == pytest.approx(3.0, abs=0.006)
    assert any("run valve closed" in n for n in b.notes)
    assert b.m_ox[-1] > 0.5  # oxidiser left behind in the tank


def test_cutoff_uses_less_web_than_a_full_blowdown(props, table):
    g = sector_grain()
    full = simulate(_cfg(g), props, table)
    cut = simulate(_cfg(g, valve=3.8), props, table)
    assert g.regression(cut.port_d[-1]) < g.regression(full.port_d[-1])
    assert g.regression(cut.port_d[-1]) < g.web


def test_sector_grain_stops_at_the_wall_web(props, table):
    # Fastest candidate fit: burns through long before the tank empties.
    g = sector_grain()
    b = simulate(_cfg(g, reg=(0.169, 0.600)), props, table)
    assert any("burned through" in n for n in b.notes)
    assert g.regression(b.port_d[-1]) == pytest.approx(g.web, abs=5e-4)


def test_sizing_reports_web_use_and_burn_through(props, table):
    g = sector_grain()
    ok = analyse_geometry(_cfg(g, valve=3.8), props, table, SizingTarget())
    assert 0.5 < ok.web_used_frac < 0.9
    assert not any("BURN-THROUGH" in w for w in ok.warnings)
    bad = analyse_geometry(_cfg(g, reg=(0.169, 0.600)), props, table, SizingTarget())
    assert any("BURN-THROUGH" in w for w in bad.warnings)


def test_ports_smaller_than_throat_are_flagged(props, table):
    g = Grain(length=0.1778, port_id=0.012, outer_d=OD, n_ports=5)  # 565 mm^2 < 730
    res = analyse_geometry(_cfg(g, valve=1.0), props, table, SizingTarget())
    assert any("PORTS SMALLER THAN THROAT" in w for w in res.warnings)


# ------------------------------------------------------------ persistence


def test_config_round_trip_keeps_sector_grain_and_valve(tmp_path):
    cfg = _cfg(sector_grain(), valve=3.8)
    path = save_config(str(tmp_path / "c.json"), cfg)
    back, _, _ = load_config(path)
    assert back.grain == cfg.grain
    assert back.valve_close_t == 3.8


def test_old_configs_load_as_round_with_no_valve():
    here = os.path.join(os.path.dirname(__file__), "..", "configs",
                        "V2.2_recommended_mdot1.0_HEM.json")
    cfg, _, _ = load_config(here)
    assert cfg.grain.layout == "round"
    assert cfg.valve_close_t is None


# ---------------------------------------------------------------- drawing


def test_grain_dxf_has_true_arcs_for_every_sector(tmp_path):
    lay = grain_layout(sector_grain(), Nozzle(throat_d=0.03048))
    path = write_grain_dxf(str(tmp_path / "g.dxf"), lay)
    text = open(path, encoding="ascii").read()
    assert text.count("\nARC\n") == 2 * 4
    assert text.count("\nCIRCLE\n") == 2  # case + centre port
    for layer in ("GRAIN_OUTLINE", "PORTS", "SECTION", "ANNOTATION"):
        assert layer in text
    assert lay.port_throat_ratio == pytest.approx(AREA / (math.pi / 4 * 0.03048**2))


def test_round_grain_section_cuts_through_a_port():
    lay = grain_layout(Grain(length=0.1778, port_id=0.01778, outer_d=OD, n_ports=5))
    bands = section_bands(lay)
    assert len(bands) == 1
    lo, hi = bands[0]
    assert hi - lo == pytest.approx(17.78)
    assert lay.wall_web_actual == pytest.approx(7.00, abs=0.01)


@pytest.mark.parametrize("ext", [".png", ".pdf"])
def test_grain_sheet_renders(tmp_path, ext):
    from n2o_injector.plots import save_grain_drawing

    for g in (sector_grain(), Grain(length=0.1778, port_id=0.01778, outer_d=OD, n_ports=5)):
        p = save_grain_drawing(str(tmp_path / f"g_{g.layout}{ext}"), grain_layout(g), dpi=80)
        assert os.path.getsize(p) > 5000
