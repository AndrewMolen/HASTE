"""Save and reload a complete tool configuration.

HRAP's ``.mat`` format cannot represent everything this tool models -- it has
no multi-port grain, no injector flow-model selection and no plate thickness --
so round-tripping a design through it would silently drop exactly the fields
that matter most here. This module stores the full configuration as JSON
instead.

The CEA table is referenced by path rather than embedded: the tables are large,
and keeping the reference means an updated propellant file is picked up on the
next load. If the path no longer resolves, the loader falls back to the stored
constant ``c*`` and says so, rather than failing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from .injector import FlowModel
from .motor import Grain, InjectorSpec, MotorConfig, Nozzle, Tank
from .propellant import Propellant
from .sizing import SizingTarget

SCHEMA = "n2o-injector-config/1"


def save_config(
    path: str,
    cfg: MotorConfig,
    target: SizingTarget | None = None,
    propellant_file: str = "",
    notes: str = "",
) -> str:
    """Write ``cfg`` (and optionally the sizing target) to a JSON file."""
    p = cfg.propellant
    data = {
        "schema": SCHEMA,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "tank": {
            "volume_L": cfg.tank.volume * 1e3,
            "fill_temp_C": cfg.tank.fill_temp - 273.15,
            "ox_mass_kg": cfg.tank.ox_mass,
            "supercharge_bar": (
                None if cfg.tank.supercharge_P is None else cfg.tank.supercharge_P / 1e5
            ),
        },
        "grain": {
            "length_mm": cfg.grain.length * 1e3,
            "port_id_mm": cfg.grain.port_id * 1e3,
            "outer_d_mm": cfg.grain.outer_d * 1e3,
            "n_ports": cfg.grain.n_ports,
        },
        "nozzle": {
            "throat_d_mm": cfg.nozzle.throat_d * 1e3,
            "expansion_ratio": cfg.nozzle.expansion_ratio,
            "Cd": cfg.nozzle.Cd,
            "efficiency": cfg.nozzle.efficiency,
        },
        "injector": {
            "n_holes": cfg.injector.n_holes,
            "hole_d_mm": cfg.injector.hole_d * 1e3,
            "Cd": cfg.injector.Cd,
            "plate_thickness_mm": cfg.injector.plate_thickness * 1e3,
            "min_hole_d_mm": cfg.injector.min_hole_d * 1e3,
            "model": cfg.injector.model.value,
            "ld_ref": cfg.injector.ld_ref,
        },
        "propellant": {
            "name": p.name,
            "reg_a": p.reg_a,
            "reg_n": p.reg_n,
            "reg_m": p.reg_m,
            "rho_fuel": p.rho_fuel,
            "opt_OF": p.opt_OF,
            "cstar_const": p.cstar_const,
            "cstar_eff": p.cstar_eff,
            # Referenced, not embedded -- see module docstring.
            "cea_table_file": propellant_file,
        },
        "run": {
            "ambient_bar": cfg.ambient_P / 1e5,
            "dt_ms": cfg.dt * 1e3,
            "max_time_s": cfg.max_time,
            "chamber_mode": cfg.chamber_mode,
            "chamber_volume_L": cfg.chamber_volume * 1e3,
            "regression_mode": cfg.regression_mode,
            "stop_at_liquid_exhausted": cfg.stop_at_liquid_exhausted,
            "tail_cutoff_frac": cfg.tail_cutoff_frac,
            "const_OF": cfg.const_OF,
            "initial_Pc_bar": (
                None if cfg.initial_Pc is None else cfg.initial_Pc / 1e5
            ),
        },
    }
    if target is not None:
        data["target"] = {
            "objective": target.objective,
            "OF": target.OF,
            "mdot_ox": target.mdot_ox,
            "chamber_P_bar": target.chamber_P / 1e5,
            "min_dP_fraction": target.min_dP_fraction,
            "design_time": target.design_time,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_config(path: str) -> tuple[MotorConfig, SizingTarget, list[str]]:
    """Read a saved configuration.

    Returns ``(config, target, warnings)``. Warnings are non-fatal: a missing
    CEA table or an unrecognised flow model degrades gracefully rather than
    raising, so an otherwise-good design still loads.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    schema = data.get("schema", "")
    if not schema.startswith("n2o-injector-config/"):
        raise ValueError(
            f"{os.path.basename(path)} is not a tool configuration file "
            f"(schema {schema!r}). To import an HRAP motor use the HRAP "
            "cross-reference panel instead."
        )

    warnings: list[str] = []
    t, g, n, i, pr, run = (
        data["tank"], data["grain"], data["nozzle"],
        data["injector"], data["propellant"], data["run"],
    )

    # Propellant: prefer the referenced CEA table, fall back to constants.
    cea = pr.get("cea_table_file") or ""
    prop: Propellant | None = None
    if cea:
        candidates = [cea, os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(cea))]
        for c in candidates:
            if os.path.isfile(c):
                try:
                    prop = Propellant.from_hrap_mat(c, cstar_eff=pr["cstar_eff"])
                    break
                except Exception as exc:
                    warnings.append(f"could not load CEA table {c}: {exc}")
        if prop is None:
            warnings.append(
                f"CEA table '{cea}' not found; using the stored constant c* of "
                f"{pr['cstar_const']:.0f} m/s. Chamber pressure and thrust will be "
                "less accurate, and O/F outside the table range will not be flagged."
            )
    if prop is None:
        prop = Propellant()
        prop.cstar_const = pr["cstar_const"]
    prop.name = pr["name"]
    prop.reg_a, prop.reg_n, prop.reg_m = pr["reg_a"], pr["reg_n"], pr["reg_m"]
    prop.rho_fuel, prop.opt_OF = pr["rho_fuel"], pr["opt_OF"]
    prop.cstar_eff = pr["cstar_eff"]

    try:
        model = FlowModel.from_str(i["model"])
    except ValueError:
        model = FlowModel.DYER
        warnings.append(f"unknown flow model {i['model']!r}; defaulting to Dyer")

    sup = t.get("supercharge_bar")
    cfg = MotorConfig(
        tank=Tank(volume=t["volume_L"] / 1e3,
                  fill_temp=t["fill_temp_C"] + 273.15,
                  ox_mass=t["ox_mass_kg"],
                  supercharge_P=None if sup is None else sup * 1e5),
        grain=Grain(length=g["length_mm"] / 1e3, port_id=g["port_id_mm"] / 1e3,
                    outer_d=g["outer_d_mm"] / 1e3, n_ports=int(g["n_ports"])),
        nozzle=Nozzle(throat_d=n["throat_d_mm"] / 1e3,
                      expansion_ratio=n["expansion_ratio"],
                      Cd=n["Cd"], efficiency=n["efficiency"]),
        injector=InjectorSpec(n_holes=int(i["n_holes"]), hole_d=i["hole_d_mm"] / 1e3,
                              Cd=i["Cd"],
                              plate_thickness=i["plate_thickness_mm"] / 1e3,
                              min_hole_d=i["min_hole_d_mm"] / 1e3,
                              model=model, ld_ref=i["ld_ref"]),
        propellant=prop,
        ambient_P=run["ambient_bar"] * 1e5,
        dt=run["dt_ms"] / 1e3,
        max_time=run["max_time_s"],
        chamber_mode=run["chamber_mode"],
        chamber_volume=run["chamber_volume_L"] / 1e3,
        regression_mode=run["regression_mode"],
        const_OF=run["const_OF"],
        initial_Pc=(None if run.get("initial_Pc_bar") is None
                    else run["initial_Pc_bar"] * 1e5),
        stop_at_liquid_exhausted=run.get("stop_at_liquid_exhausted", True),
        tail_cutoff_frac=run.get("tail_cutoff_frac", 1.05),
    )

    td = data.get("target", {})
    target = SizingTarget(
        objective=td.get("objective", "burn_average"),
        OF=td.get("OF", prop.opt_OF),
        mdot_ox=td.get("mdot_ox", 1.0),
        chamber_P=td.get("chamber_P_bar", 30.0) * 1e5,
        min_dP_fraction=td.get("min_dP_fraction", 0.20),
        design_time=td.get("design_time", 0.0),
    )
    return cfg, target, warnings
