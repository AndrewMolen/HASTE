# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone N2O Injector Sizing Tool.

Builds two flavours from one analysis pass:

* ``dist/N2O Injector Sizing Tool/`` -- one-folder build. Starts in about a
  second because nothing has to be unpacked at launch. Best for day-to-day use.
* ``dist/N2O Injector Sizing Tool.exe`` -- single-file build. One portable
  file, but every launch extracts the whole bundle to a temp directory first,
  which with numpy/scipy/matplotlib is noticeably slower.

Build both:
    python -m PyInstaller tools/n2o_injector.spec --noconfirm

The excludes matter more than usual here: matplotlib pulls in every backend it
can find, and scipy/numpy ship large test suites. Dropping them roughly halves
the bundle without touching anything the GUI uses (it only ever needs TkAgg).
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "assets", "n2o_injector.ico")
APP_NAME = "N2O Injector Sizing Tool"

# CoolProp is the optional second N2O property backend. It ships data files
# that will not be found unless they are collected explicitly.
try:
    import CoolProp  # noqa: F401

    coolprop_datas = collect_data_files("CoolProp")
    coolprop_hidden = collect_submodules("CoolProp")
except ImportError:
    coolprop_datas, coolprop_hidden = [], []

hiddenimports = [
    # The application package itself. run_app.pyw imports it lazily, inside a
    # function and inside a try block, so PyInstaller's static analysis does
    # not follow it -- without this the bundle builds cleanly but ships no
    # application code at all and fails at launch.
    *collect_submodules("n2o_injector"),
    # The on-screen backend; PyInstaller cannot infer it from the imports.
    "matplotlib.backends.backend_tkagg",
    # Saving the plate drawing picks a writer backend by file extension at
    # call time -- a dynamic import static analysis cannot see. Without these
    # the GUI runs but "Save sheet" fails on .pdf and .svg in the frozen build
    # only, which is the worst place to find out.
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
    "PIL._tkinter_finder",
    # scipy.io.loadmat is used for HRAP .mat interop and is imported lazily
    # inside functions, so static analysis misses it.
    "scipy.io",
    "scipy.io.matlab",
    "scipy.optimize",
    "scipy.interpolate",
    "scipy.special",
] + coolprop_hidden

excludes = [
    # Alternative GUI toolkits matplotlib probes for but we never use.
    "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "gtk", "cairo",
    # Notebook/plotting extras.
    "IPython", "jupyter", "notebook", "nbconvert", "nbformat", "zmq",
    "pandas", "sphinx", "docutils", "pytest", "_pytest", "pyflakes",
    # NOTE: do NOT exclude numpy.testing. It looks like dead weight, but
    # scipy's array_api_compat imports it at runtime while loading
    # scipy.linalg, so excluding it makes every scipy import fail with
    # "No module named 'numpy.testing'" -- and only at launch, never at build.
    # Document tooling used only by tools/build_docx.py, not by the app.
    "docx", "fitz", "pymupdf", "win32com", "pypdf",
]

a = Analysis(
    [os.path.join(ROOT, "run_app.pyw")],
    pathex=[ROOT],
    binaries=[],
    datas=[(ICON, "assets")] + coolprop_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # Ship the on-screen backend plus the three used to write drawings,
        # rather than every backend matplotlib has.
        "matplotlib": {"backends": ["TkAgg", "Agg", "PDF", "SVG"]},
    },
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# --- one-folder build (fast launch) ----------------------------------------
exe_dir = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
    icon=ICON,
)

coll = COLLECT(
    exe_dir,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# --- one-file build (portable) ---------------------------------------------
exe_onefile = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon=ICON,
)
