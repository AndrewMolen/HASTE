"""Tests for the orifice-plate drawing, DXF and hole table.

The drawing is a manufacturing document: if it disagrees with the sized plate,
or the DXF disagrees with the drawing, someone drills the wrong plate. These
tests pin the three things that would cause that -- hole count, hole position
and hole diameter -- in every output the tool produces.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from n2o_injector.drawing import (  # noqa: E402
    DrawingMeta,
    hole_table,
    layout_from_plate,
    plate_layout,
    ring_positions,
    suggested_filename,
    write_dxf,
    write_hole_table_csv,
)
from n2o_injector.plots import build_plate_drawing, save_plate_drawing  # noqa: E402
from n2o_injector.sizing import OrificePlate  # noqa: E402

# The plate sized for the V2.2 motor: 20 holes on an 85.9 mm plate.
V22 = dict(n_holes=20, hole_d_mm=1.381, plate_d_mm=85.9, thickness_mm=12.70, Cd=0.70)


def _v22():
    return plate_layout(**V22)


# --------------------------------------------------------------- geometry


def test_layout_places_every_hole():
    lay = _v22()
    assert lay.n_holes == 20
    assert sum(r.count for r in lay.rings) + (1 if lay.centre_hole else 0) == 20


def test_holes_lie_on_their_bolt_circles():
    lay = _v22()
    radii = {round(r.radius, 6) for r in lay.rings}
    for x, y in lay.holes:
        r = round(math.hypot(x, y), 6)
        assert r in radii or r == 0.0


def test_holes_are_equally_spaced_on_each_ring():
    lay = _v22()
    for ring in lay.rings:
        on_ring = sorted(
            (math.degrees(math.atan2(y, x)) + 360) % 360
            for x, y in lay.holes
            if abs(math.hypot(x, y) - ring.radius) < 1e-9
        )
        steps = [b - a for a, b in zip(on_ring, on_ring[1:])]
        assert all(abs(s - ring.pitch_deg) < 1e-6 for s in steps)


def test_every_hole_fits_inside_the_plate():
    lay = _v22()
    assert lay.edge_margin > 0
    assert lay.min_web > 0
    assert lay.feasible


def test_alternate_rings_are_staggered():
    """Staggering is what keeps the web between rings open."""
    lay = _v22()
    assert lay.rings[0].phase == 0.0
    assert lay.rings[1].phase == pytest.approx(math.pi / lay.rings[1].count)


def test_overlapping_holes_are_flagged_not_hidden():
    lay = plate_layout(n_holes=60, hole_d_mm=6.0, plate_d_mm=40.0, thickness_mm=3.0)
    assert lay.min_web < 0
    assert not lay.feasible


def test_single_hole_goes_on_centre():
    lay = plate_layout(n_holes=1, hole_d_mm=2.0, plate_d_mm=40.0, thickness_mm=3.0)
    assert lay.holes == [(0.0, 0.0)]
    assert lay.centre_hole


def test_area_matches_the_sized_plate():
    plate = OrificePlate(n_holes=20, hole_d=1.381e-3, Cd=0.70, plate_thickness=12.7e-3)
    lay = layout_from_plate(plate, plate_d_mm=85.9)
    assert lay.total_area == pytest.approx(plate.total_area * 1e6, rel=1e-9)
    assert lay.L_over_D == pytest.approx(plate.L_over_D, rel=1e-9)


def test_plate_diameter_falls_back_to_the_grain():
    plate = OrificePlate(n_holes=8, hole_d=1.5e-3, Cd=0.7, plate_thickness=3e-3)
    assert layout_from_plate(plate, plate_d_mm=0, grain_od_mm=60.0).plate_d == 60.0


def test_area_sensitivity_follows_d_squared():
    lay = _v22()
    d = lay.hole_d
    assert lay.area_sensitivity(0.02) == pytest.approx(
        100 * ((d + 0.02) ** 2 / d**2 - 1), rel=1e-12
    )


def test_ring_positions_is_deterministic():
    a = ring_positions(20, 42.95, 0.6905)
    b = ring_positions(20, 42.95, 0.6905)
    assert a[1] == b[1]


# ------------------------------------------------------------- hole table


def test_hole_table_round_trips_to_cartesian():
    lay = _v22()
    for row, (x, y) in zip(hole_table(lay), lay.holes):
        assert row["radius_mm"] * math.cos(math.radians(row["angle_deg"])) == pytest.approx(x, abs=1e-9)
        assert row["radius_mm"] * math.sin(math.radians(row["angle_deg"])) == pytest.approx(y, abs=1e-9)
        assert row["diameter_mm"] == lay.hole_d


def test_hole_table_csv_has_one_row_per_hole(tmp_path):
    lay = _v22()
    p = write_hole_table_csv(str(tmp_path / "holes.csv"), lay, DrawingMeta())
    lines = [ln for ln in open(p, encoding="utf-8").read().splitlines() if ln]
    data = [ln for ln in lines if not ln.startswith("#")]
    assert len(data) == lay.n_holes + 1  # header + holes
    assert data[0].startswith("hole,ring,x_mm,y_mm")
    assert all(ln.endswith("THRU") for ln in data[1:])


def test_hole_table_csv_states_its_datum(tmp_path):
    """A coordinate list without a stated datum is a machining error."""
    text = open(write_hole_table_csv(str(tmp_path / "h.csv"), _v22()), encoding="utf-8").read()
    assert "Origin at plate centre" in text
    assert "units: mm" in text


# -------------------------------------------------------------------- DXF


def _dxf_pairs(path):
    raw = open(path, encoding="ascii").read().splitlines()
    return list(zip(raw[0::2], raw[1::2]))


def test_dxf_structure_is_well_formed(tmp_path):
    p = write_dxf(str(tmp_path / "plate.dxf"), _v22(), DrawingMeta())
    pairs = _dxf_pairs(p)
    codes = [(c.strip(), v.strip()) for c, v in pairs]
    assert ("0", "SECTION") in codes
    assert ("2", "HEADER") in codes
    assert ("2", "TABLES") in codes
    assert ("2", "ENTITIES") in codes
    assert codes[-1] == ("0", "EOF")
    assert codes.count(("0", "SECTION")) == codes.count(("0", "ENDSEC")) == 3
    assert ("1", "AC1009") in codes  # R12
    # LTYPE must be declared before the LAYER table that references it, and
    # every table before the entities that reference them.
    assert codes.index(("2", "LTYPE")) < codes.index(("2", "LAYER"))
    assert codes.index(("2", "STYLE")) < codes.index(("2", "ENTITIES"))
    assert codes.count(("0", "TABLE")) == codes.count(("0", "ENDTAB")) == 3


def test_dxf_text_references_a_defined_style(tmp_path):
    p = write_dxf(str(tmp_path / "plate.dxf"), _v22(), DrawingMeta())
    codes = [(c.strip(), v.strip()) for c, v in _dxf_pairs(p)]
    n_text = codes.count(("0", "TEXT"))
    assert n_text > 0
    assert codes.count(("7", "STANDARD")) == n_text
    assert ("2", "STANDARD") in codes  # the STYLE table entry itself


def test_dxf_circles_match_the_layout(tmp_path):
    lay = _v22()
    p = write_dxf(str(tmp_path / "plate.dxf"), lay)
    pairs = [(c.strip(), v.strip()) for c, v in _dxf_pairs(p)]

    circles, i = [], 0
    while i < len(pairs):
        if pairs[i] == ("0", "CIRCLE"):
            ent = {}
            j = i + 1
            while j < len(pairs) and pairs[j][0] != "0":
                ent[pairs[j][0]] = pairs[j][1]
                j += 1
            circles.append(ent)
            i = j
        else:
            i += 1

    holes = [c for c in circles if c["8"] == "HOLES"]
    assert len(holes) == lay.n_holes
    assert all(float(c["40"]) == pytest.approx(lay.hole_d / 2, rel=1e-9) for c in holes)

    outline = [c for c in circles if c["8"] == "PLATE_OUTLINE"]
    assert len(outline) == 1
    assert float(outline[0]["40"]) == pytest.approx(lay.plate_r, rel=1e-9)

    # Positions, to the micron, against the layout itself.
    got = sorted((round(float(c["10"]), 6), round(float(c["20"]), 6)) for c in holes)
    want = sorted((round(x, 6), round(y, 6)) for x, y in lay.holes)
    assert got == want

    assert len([c for c in circles if c["8"] == "CENTRELINES"]) == len(lay.rings)


def test_dxf_is_plain_ascii(tmp_path):
    """R12 has no Unicode; the diameter symbol must become the %%c control code."""
    p = write_dxf(str(tmp_path / "plate.dxf"), _v22(), DrawingMeta(project="ø-test"))
    raw = open(p, "rb").read()
    raw.decode("ascii")  # raises if any byte is non-ASCII
    assert b"%%c" in raw


def test_dxf_layers_let_annotation_be_switched_off(tmp_path):
    lay = _v22()
    full = open(write_dxf(str(tmp_path / "a.dxf"), lay), encoding="ascii").read()
    bare = open(write_dxf(str(tmp_path / "b.dxf"), lay, include_text=False,
                          include_section=False), encoding="ascii").read()
    assert "ANNOTATION" in full and "SECTION" in full
    assert len(bare) < len(full)
    # The geometry that matters survives either way.
    assert bare.count("CIRCLE") == full.count("CIRCLE")


def test_suggested_filename_states_the_design():
    name = suggested_filename(_v22(), ".dxf", part_no="INJ 001")
    assert name.startswith("INJ_001_20x1.381mm")
    assert name.endswith(".dxf")


# ----------------------------------------------------------------- sheet


def test_drawing_sheet_renders(tmp_path):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11.69, 8.27))
    ax = build_plate_drawing(fig, _v22(), DrawingMeta(project="test"))
    assert ax.get_xlim() == (0, 297.0)
    assert ax.get_ylim() == (0, 210.0)


@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg"])
def test_sheet_saves_in_every_offered_format(tmp_path, ext):
    p = save_plate_drawing(str(tmp_path / ("sheet" + ext)), _v22(), DrawingMeta(), dpi=72)
    assert os.path.getsize(p) > 2000


def test_sheet_survives_an_infeasible_plate(tmp_path):
    """The warning belongs on the drawing, so it must still render."""
    lay = plate_layout(n_holes=60, hole_d_mm=6.0, plate_d_mm=40.0, thickness_mm=3.0)
    p = save_plate_drawing(str(tmp_path / "bad.png"), lay, DrawingMeta(), dpi=72)
    assert os.path.getsize(p) > 2000


def test_sheet_handles_a_large_hole_count(tmp_path):
    """More holes than the table can show must not overflow the sheet."""
    lay = plate_layout(n_holes=96, hole_d_mm=0.9, plate_d_mm=120.0, thickness_mm=3.0)
    p = save_plate_drawing(str(tmp_path / "many.png"), lay, DrawingMeta(), dpi=72)
    assert os.path.getsize(p) > 2000
