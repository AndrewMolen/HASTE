"""Scripted example: size an injector and export everything.

Run with::

    python examples/demo_sizing.py [path/to/Paraffin.mat]

Passing HRAP's ``Paraffin.mat`` uses its CEA table for ``c*``; without it the
tool falls back to the constant-``c*`` model.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

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
    build_report,
    export_hrap_mat,
    export_injector_json,
    export_timeseries_csv,
    get_backend,
    simulate,
    size_injector,
)
from n2o_injector.plots import build_comparison_figure, build_results_figure  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main(prop_mat: str | None = None):
    os.makedirs(OUT, exist_ok=True)

    props = get_backend("esdu")
    table = SaturationTable(props)

    propellant = (
        Propellant.from_hrap_mat(prop_mat) if prop_mat else Propellant()
    )

    cfg = MotorConfig(
        tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
        grain=Grain(length=0.45, port_id=0.038, outer_d=0.092, n_ports=1),
        nozzle=Nozzle(throat_d=0.032, expansion_ratio=4.0),
        injector=InjectorSpec(
            n_holes=24, hole_d=0.0016, Cd=0.70,
            plate_thickness=0.003, model=FlowModel.DYER,
        ),
        propellant=propellant,
        dt=0.005,
    )
    target = SizingTarget(OF=propellant.opt_OF, objective="burn_average",
                          min_dP_fraction=0.20)

    result = size_injector(cfg, props, table, target, fix="n_holes")

    report = build_report(result, cfg, props.name)
    print(report)

    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    export_hrap_mat(os.path.join(OUT, "motor_hrap.mat"), result, cfg,
                    motor_name="demo_motor", propellant_file=prop_mat or "")
    export_injector_json(os.path.join(OUT, "injector.json"), result, cfg)
    export_timeseries_csv(os.path.join(OUT, "timeseries.csv"), result)

    fig = Figure(figsize=(13, 10), dpi=110, layout="constrained")
    build_results_figure(fig, result, cfg)
    fig.savefig(os.path.join(OUT, "burn_summary.png"))

    up = props.upstream_state(cfg.tank.fill_temp)
    curve = OrificeCurve(table, up, model=FlowModel.DYER,
                         L_over_D=result.plate.L_over_D)
    fig2 = Figure(figsize=(11, 8), dpi=110, layout="constrained")
    build_comparison_figure(fig2, curve, up)
    fig2.savefig(os.path.join(OUT, "model_comparison.png"))

    # Same geometry, three models -- shows how much the model choice matters.
    print("\nSame plate, different flow models (initial oxidiser flow):")
    for model in (FlowModel.SPI, FlowModel.HEM, FlowModel.DYER):
        cfg.injector.model = model
        b = simulate(cfg, props, table, CdA_override=result.plate.CdA)
        print(f"  {model.value:<22} mdot_ox(0) = {b.mdot_ox[0]:.3f} kg/s   "
              f"mean O/F = {b.mean_OF:.2f}   burn = {b.burn_time:.2f} s")

    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
