"""N2O / paraffin hybrid rocket injector sizing tool.

Public API:

>>> from n2o_injector import get_backend, SaturationTable, size_injector
>>> props = get_backend("esdu")
>>> table = SaturationTable(props)
"""

from .injector import (
    FlowModel,
    FluxResult,
    OrificeCurve,
    SaturationTable,
    check_validity,
    dyer_kappa,
    hem_mass_flux,
    ld_hem_weight,
    mass_flow,
    mass_flux,
    recommend_model,
    required_CdA,
    spi_mass_flux,
)
from .drawing import (
    DrawingMeta,
    PlateLayout,
    Ring,
    default_meta,
    export_all,
    hole_table,
    layout_from_plate,
    plate_layout,
    ring_positions,
    suggested_filename,
    write_dxf,
    write_hole_table_csv,
)
from .hrap_io import (
    HrapRun,
    ImportedMotor,
    RunComparison,
    compare_to_hrap,
    format_comparison,
    load_hrap_motor,
    load_hrap_output,
)
from .motor import (
    BurnResult,
    Grain,
    InjectorSpec,
    MotorConfig,
    Nozzle,
    Tank,
    fuel_flow,
    required_mdot_ox,
    simulate,
)
from .propellant import FUEL_PRESETS, PARAFFIN_DEFAULTS, Propellant
from .properties import (
    CoolPropProperties,
    ESDUProperties,
    FluidState,
    SatState,
    coolprop_available,
    get_backend,
)
from .report import (
    build_report,
    export_hrap_mat,
    export_injector_json,
    export_timeseries_csv,
)
from .sizing import OrificePlate, SizingResult, SizingTarget, plate_from_CdA, size_injector

__version__ = "1.0.0"

__all__ = [
    "FlowModel", "FluxResult", "OrificeCurve", "SaturationTable", "check_validity",
    "dyer_kappa", "hem_mass_flux", "ld_hem_weight", "mass_flow", "mass_flux",
    "recommend_model", "required_CdA", "spi_mass_flux",
    "DrawingMeta", "PlateLayout", "Ring", "default_meta", "export_all", "hole_table",
    "layout_from_plate", "plate_layout", "ring_positions", "suggested_filename",
    "write_dxf", "write_hole_table_csv",
    "BurnResult", "Grain", "InjectorSpec", "MotorConfig", "Nozzle", "Tank",
    "fuel_flow", "required_mdot_ox", "simulate",
    "HrapRun", "ImportedMotor", "RunComparison", "compare_to_hrap",
    "format_comparison", "load_hrap_motor", "load_hrap_output",
    "FUEL_PRESETS", "PARAFFIN_DEFAULTS", "Propellant",
    "CoolPropProperties", "ESDUProperties", "FluidState", "SatState",
    "coolprop_available", "get_backend",
    "build_report", "export_hrap_mat", "export_injector_json", "export_timeseries_csv",
    "OrificePlate", "SizingResult", "SizingTarget", "plate_from_CdA", "size_injector",
    "__version__",
]
