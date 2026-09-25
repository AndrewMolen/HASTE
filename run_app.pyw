"""Double-clickable launcher for the injector sizing tool.

A ``.pyw`` file is opened by ``pythonw.exe`` on Windows, so this starts the GUI
with no console window. It can be double-clicked directly, pinned, or pointed
at by a desktop shortcut.

The startup is wrapped in a handler because a windowless process that fails has
nowhere to print: without this, a missing dependency or a syntax error would
just make nothing happen at all. Instead the error is shown in a dialog and
written to ``startup_error.log`` next to this file.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

#: True when running from a PyInstaller bundle rather than the source tree.
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    # __file__ points inside the bundle (a temp extraction directory for the
    # one-file build), which is the wrong place to import from or to write to.
    # Anchor on the executable instead.
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))


def _log_path() -> str:
    """Somewhere writable to record a startup failure.

    Next to the executable normally, but that may be read-only if the app was
    installed under Program Files, so fall back to the temp directory.
    """
    candidate = os.path.join(ROOT, "startup_error.log")
    try:
        with open(candidate, "a", encoding="utf-8"):
            pass
        return candidate
    except OSError:
        return os.path.join(tempfile.gettempdir(), "n2o_injector_startup_error.log")


LOG = _log_path()


def _report(title: str, message: str) -> None:
    """Show a message box, falling back to a log file if even that fails."""
    try:
        with open(LOG, "w", encoding="utf-8") as f:
            f.write(message)
    except OSError:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message[:3000])
        root.destroy()
    except Exception:
        # No Tk at all: fall back to the Win32 dialog directly.
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message[:3000], title, 0x10)
        except Exception:
            pass


def self_test() -> int:
    """Run a real sizing calculation and report the result.

    Invoked with ``--self-test``. This exists because a frozen GUI that merely
    opens proves very little -- the numerics live behind a button. This
    exercises the full path (properties, injector models, blowdown, solver) so
    a broken bundle is caught without anyone having to click anything.

    Results go to a file because a windowed build has no stdout.
    """
    out = os.path.join(ROOT, "self_test.log")
    lines: list[str] = []
    ok = True
    try:
        from n2o_injector import (
            Grain, InjectorSpec, MotorConfig, Nozzle, Propellant,
            SaturationTable, SizingTarget, Tank, get_backend, size_injector,
        )

        props = get_backend("esdu")
        table = SaturationTable(props)
        sat = props.sat(293.15)
        lines.append(f"N2O at 20 C: Psat={sat.P / 1e5:.2f} bar, rho_l={sat.rho_l:.1f} kg/m3")

        cfg = MotorConfig(
            tank=Tank(volume=0.012, fill_temp=293.15, ox_mass=7.0),
            grain=Grain(length=0.45, port_id=0.038, outer_d=0.092),
            nozzle=Nozzle(throat_d=0.032, expansion_ratio=4.0),
            injector=InjectorSpec(n_holes=24, hole_d=0.0016, Cd=0.70,
                                  plate_thickness=0.003),
            propellant=Propellant(), dt=0.01,
        )
        res = size_injector(cfg, props, table,
                            SizingTarget(objective="mdot_ox", mdot_ox=1.3))
        lines.append(
            f"sizing: {res.plate.n_holes} holes x {res.plate.hole_d * 1e3:.3f} mm, "
            f"mean O/F {res.achieved_mean_OF:.2f}, "
            f"impulse {res.burn.total_impulse:.0f} N.s"
        )

        # Guard the numbers, not just the absence of an exception.
        assert abs(sat.P / 1e5 - 50.60) < 0.5, "saturation pressure wrong"
        assert abs(res.burn.mdot_ox[0] - 1.3) < 0.01, "solver missed its target"
        assert 1.0e-3 < res.plate.hole_d < 3.0e-3, "hole diameter implausible"

        import matplotlib
        lines.append(f"matplotlib {matplotlib.__version__} backend={matplotlib.get_backend()}")

        # The drawing exports pull in the Agg canvas, the PDF writer and the
        # font cache -- all things a frozen bundle can be missing -- so they
        # are exercised here rather than discovered by a user at the shop.
        import tempfile

        from n2o_injector import layout_from_plate, write_dxf, write_hole_table_csv
        from n2o_injector.plots import save_plate_drawing

        lay = layout_from_plate(res.plate, grain_od_mm=cfg.grain.outer_d * 1e3)
        assert lay.n_holes == res.plate.n_holes, "drawing lost a hole"
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("plate.png", "plate.pdf"):
                p = os.path.join(tmp, name)
                save_plate_drawing(p, lay)
                assert os.path.getsize(p) > 2000, f"{name} came out empty"
            write_dxf(os.path.join(tmp, "plate.dxf"), lay)
            write_hole_table_csv(os.path.join(tmp, "plate.csv"), lay)
        lines.append(
            f"drawing: {lay.n_holes} holes on {len(lay.rings)} ring(s), "
            f"min web {lay.min_web:.2f} mm, DXF/CSV/PNG/PDF written"
        )
    except Exception:
        ok = False
        lines.append("FAILED\n" + traceback.format_exc())

    lines.insert(0, "SELF TEST: " + ("PASS" if ok else "FAIL"))
    lines.insert(1, f"frozen={FROZEN}  root={ROOT}")
    text = "\n".join(lines)
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass
    # --quiet suppresses the dialog so the test can run unattended.
    if "--quiet" not in sys.argv[1:]:
        _report("HASTE - self test", text)
    return 0 if ok else 1


def main() -> int:

    # Running from source: put the project root on the path so the package
    # imports regardless of where the shortcut was invoked from. When frozen,
    # the package is inside the bundle and sys.path is already correct --
    # prepending the executable's directory would only risk shadowing it.
    if not FROZEN:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        try:
            os.chdir(ROOT)
        except OSError:
            pass

    # Checked after the path setup above so it works from source too.
    if "--self-test" in sys.argv[1:]:
        return self_test()

    try:
        from n2o_injector.gui import main as gui_main
    except Exception:
        _report(
            "HASTE - startup failed",
            "The application could not start.\n\n"
            "This usually means a required package is missing. From a terminal:\n\n"
            "    pip install numpy scipy matplotlib\n\n"
            "Full details:\n\n" + traceback.format_exc(),
        )
        return 1

    try:
        gui_main()
    except Exception:
        _report(
            "HASTE - error",
            "The application stopped unexpectedly.\n\n" + traceback.format_exc(),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
