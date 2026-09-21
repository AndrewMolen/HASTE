"""Tkinter GUI for the N2O injector sizing tool.

Layout is a scrollable input column on the left and a tabbed results panel on
the right (plots, report, and the raw operating-point sweep).  Sizing runs on a
worker thread so the window stays responsive; results are marshalled back to
the Tk main loop via ``after``.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .hrap_io import (
    compare_to_hrap,
    format_comparison,
    load_hrap_motor,
    load_hrap_output,
)
from .injector import FlowModel, OrificeCurve, SaturationTable
from .motor import Grain, InjectorSpec, MotorConfig, Nozzle, Tank, simulate
from .plots import (
    build_comparison_figure,
    build_hrap_overlay_figure,
    build_injector_figure,
    build_results_figure,
)
from .propellant import FUEL_PRESETS, Propellant
from .properties import coolprop_available, get_backend
from .report import (
    build_report,
    export_hrap_mat,
    export_injector_json,
    export_timeseries_csv,
)
from .config_io import load_config, save_config
from .sizing import SizingTarget, analyse_geometry, size_injector

BAR = 1e5


class Field:
    """One labelled entry box bound to a variable, with unit and tooltip."""

    def __init__(self, parent, row, label, default, unit="", tip=""):
        self.var = tk.StringVar(value=str(default))
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=(6, 4), pady=1)
        self.entry = ttk.Entry(parent, textvariable=self.var, width=11)
        self.entry.grid(row=row, column=1, sticky="ew", pady=1)
        ttk.Label(parent, text=unit, foreground="#666").grid(
            row=row, column=2, sticky="w", padx=(4, 6)
        )
        if tip:
            _Tooltip(lbl, tip)
            _Tooltip(self.entry, tip)

    def get(self, cast=float):
        raw = self.var.get().strip()
        if raw == "":
            raise ValueError("value is empty")
        return cast(raw)

    def set(self, value):
        self.var.set(str(value))


class _Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.win = widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _):
        if self.win:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            self.win, text=self.text, background="#ffffe0", relief="solid",
            borderwidth=1, wraplength=340, justify="left", padding=4,
        ).pack()

    def _hide(self, _):
        if self.win:
            self.win.destroy()
            self.win = None


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        master.title("N2O / Paraffin Hybrid -- Injector Sizing Tool")
        master.geometry("1500x950")

        self._queue: queue.Queue = queue.Queue()
        self._result = None
        self._cfg = None
        self._props_name = ""
        self._table = None
        self._props = None
        self._prop_file = ""
        self._hrap_run = None
        self._chamber_volume = 0.0
        self._initial_Pc = None

        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ UI

    def _build(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=0)
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_inputs(left)
        self._build_results(right)

    def _build_inputs(self, parent):
        # Actions and status are pinned to the bottom, outside the scrollable
        # region: "Compute" is the primary action and must never be scrolled
        # off-screen on a short window.
        footer = ttk.Frame(parent)
        footer.pack(side="bottom", fill="x")
        self._build_actions(footer)

        canvas = tk.Canvas(parent, width=340, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all(
            "<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
        )

        def section(title):
            f = ttk.LabelFrame(inner, text=title, padding=(0, 4))
            f.pack(fill="x", padx=6, pady=4)
            f.columnconfigure(1, weight=1)
            return f

        # ---- oxidiser / feed ----
        s = section("Oxidiser tank / feed")
        self.f_tankV = Field(s, 0, "Tank volume", 12.0, "L",
                             "Internal volume of the oxidiser tank.")
        self.f_tankT = Field(s, 1, "Fill temperature", 20.0, "C",
                             "Tank temperature at ignition. For a self-pressurising "
                             "tank this sets the feed pressure via the N2O saturation curve.")
        self.f_oxm = Field(s, 2, "Oxidiser mass", 7.0, "kg",
                           "Mass of N2O loaded. Must lie between the saturated vapour "
                           "and saturated liquid density limits for the tank volume.")
        self.f_super = Field(s, 3, "Supercharge P", "", "bar",
                             "Optional absolute feed pressure if the tank is "
                             "pressure-fed above saturation. Leave blank for a "
                             "self-pressurising (saturated) tank.")
        self.f_amb = Field(s, 4, "Ambient pressure", 1.01325, "bar", "Ambient back pressure.")

        # ---- injector ----
        s = section("Injector")
        self.model_var = tk.StringVar(value=FlowModel.DYER.value)
        ttk.Label(s, text="Flow model").grid(row=0, column=0, sticky="w", padx=(6, 4))
        cb = ttk.Combobox(s, textvariable=self.model_var, state="readonly", width=18,
                          values=[m.value for m in FlowModel])
        cb.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=1)
        _Tooltip(cb, "SPI: incompressible liquid (what HRAP uses).\n"
                     "HEM: full thermodynamic equilibrium, best for long orifices.\n"
                     "Dyer: the standard NHNE blend for thin-plate orifices.\n"
                     "Dyer (L/D weighted): heuristic regime-exploration variant.")

        self.f_cd = Field(s, 1, "Discharge coeff Cd", 0.70, "",
                          "Orifice discharge coefficient. 0.6-0.8 is typical for a "
                          "sharp-edged drilled plate; measure it by cold flow if you can.")
        self.f_thick = Field(s, 2, "Plate thickness", 3.0, "mm",
                             "Sets the orifice length, and hence L/D, which decides "
                             "which flow model is physically appropriate.")
        self.f_nholes = Field(s, 3, "Hole count", 24, "",
                              "Held fixed when solving for hole diameter.")
        self.f_holed = Field(s, 4, "Hole diameter", 1.60, "mm",
                             "Held fixed when solving for hole count.")
        self.f_dmin = Field(s, 5, "Min drillable dia", 0.80, "mm",
                            "Manufacturing minimum; the tool flags designs below it.")
        self.f_plated = Field(s, 7, "Plate diameter", 0.0, "mm",
                             "Outer diameter of the orifice plate, used only for the "
                             "Plate layout drawing. Leave 0 to use the grain outer "
                             "diameter.")
        self.f_ldref = Field(s, 6, "L/D reference", 5.0, "",
                             "Only used by the L/D-weighted Dyer variant.")

        self.fix_var = tk.StringVar(value="n_holes")
        ttk.Label(s, text="Solve for").grid(row=7, column=0, sticky="w", padx=(6, 4))
        fr = ttk.Frame(s)
        fr.grid(row=7, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(fr, text="diameter", variable=self.fix_var,
                        value="n_holes").pack(side="left")
        ttk.Radiobutton(fr, text="count", variable=self.fix_var,
                        value="hole_d").pack(side="left")

        # ---- grain ----
        s = section("Fuel grain")
        self.f_grainL = Field(s, 0, "Grain length", 450.0, "mm", "Axial length of the grain.")
        self.f_portID = Field(s, 1, "Initial port dia", 38.0, "mm", "Starting port diameter.")
        self.f_grainOD = Field(s, 2, "Grain outer dia", 92.0, "mm",
                               "Outer diameter; sets the available web.")
        self.f_nports = Field(s, 3, "Number of ports", 1, "",
                              "Identical circular ports. >1 uses total port area and "
                              "total burning perimeter.")

        s = section("Propellant / regression")
        self.fuel_var = tk.StringVar(value="Paraffin")
        ttk.Label(s, text="Preset").grid(row=0, column=0, sticky="w", padx=(6, 4))
        fcb = ttk.Combobox(s, textvariable=self.fuel_var, state="readonly", width=18,
                           values=list(FUEL_PRESETS))
        fcb.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=1)
        fcb.bind("<<ComboboxSelected>>", self._apply_preset)

        self.reg_mode_var = tk.StringVar(value="shifting")
        ttk.Label(s, text="Regression model").grid(row=9, column=0, sticky="w", padx=(6, 4))
        rcb = ttk.Combobox(s, textvariable=self.reg_mode_var, state="readonly", width=18,
                           values=["shifting", "constant_OF"])
        rcb.grid(row=9, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=1)
        _Tooltip(rcb, "shifting: rdot = a*G_ox^n*L^m (HRAP's 'Shifting OF'). "
                      "Required for injector sizing.\n"
                      "constant_OF: fuel flow pinned to hold a fixed O/F "
                      "(HRAP's 'Constant OF'). Used to reproduce an HRAP run; "
                      "sizing is meaningless in this mode since O/F is fixed "
                      "regardless of injector area.")
        self.f_constOF = Field(s, 10, "Constant O/F", 8.27, "",
                               "Only used by the constant_OF regression model.")

        self.f_rega = Field(s, 1, "Regression a", 0.0304, "mm/s",
                            "rdot[mm/s] = a * G_ox^n * L^m, with G_ox in kg/m^2/s "
                            "and L in m. This is HRAP's convention.")
        self.f_regn = Field(s, 2, "Regression n", 0.681, "", "Mass-flux exponent. Must be < 1.")
        self.f_regm = Field(s, 3, "Regression m", 0.0, "", "Grain-length exponent (0 to ignore).")
        self.f_rhof = Field(s, 4, "Fuel density", 900.0, "kg/m3", "Solid fuel density.")
        self.f_cstar = Field(s, 5, "c* (if no table)", 1550.0, "m/s",
                             "Used only when no HRAP propellant table is loaded.")
        self.f_cstareff = Field(s, 6, "c* efficiency", 0.95, "", "Combustion efficiency.")

        bf = ttk.Frame(s)
        bf.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 2), padx=6)
        ttk.Button(bf, text="Load HRAP propellant (.mat)",
                   command=self._load_prop).pack(fill="x")
        self.prop_lbl = ttk.Label(s, text="no CEA table loaded (constant c*)",
                                  foreground="#a60", wraplength=310)
        self.prop_lbl.grid(row=8, column=0, columnspan=3, sticky="w", padx=6)

        # ---- nozzle ----
        s = section("Nozzle")
        self.f_thrt = Field(s, 0, "Throat diameter", 32.0, "mm", "Nozzle throat diameter.")
        self.f_er = Field(s, 1, "Expansion ratio", 4.0, "", "Ae/At.")
        self.f_nozcd = Field(s, 2, "Nozzle Cd", 0.95, "", "Nozzle discharge coefficient.")
        self.f_nozeff = Field(s, 3, "Nozzle efficiency", 0.95, "", "Thrust correction factor.")

        # ---- target ----
        s = section("Sizing target")
        self.obj_var = tk.StringVar(value="burn_average")
        ttk.Label(s, text="Objective").grid(row=0, column=0, sticky="w", padx=(6, 4))
        ocb = ttk.Combobox(s, textvariable=self.obj_var, state="readonly", width=18,
                           values=["burn_average", "design_point", "least_squares",
                                   "mdot_ox", "chamber_pressure"])
        ocb.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=1)
        _Tooltip(ocb,
                 "O/F-based (invert the regression law -- ill-conditioned if a and n "
                 "are literature values):\n"
                 "  burn_average: match mass-averaged O/F over the burn.\n"
                 "  design_point: match O/F exactly at the design instant.\n"
                 "  least_squares: best fit to a target O/F profile.\n\n"
                 "Flow-based (never touch the regression law -- well-conditioned, "
                 "recommended unless you measured a and n yourself):\n"
                 "  mdot_ox: size for a target oxidiser mass flow.\n"
                 "  chamber_pressure: size for a target chamber pressure.\n"
                 "Then read off the O/F you actually get.")

        self.f_ofTarget = Field(s, 1, "Target O/F", 8.27, "",
                                "Used by the O/F-based objectives.")
        self.f_mdotTarget = Field(s, 2, "Target mdot_ox", 1.50, "kg/s",
                                  "Used by the mdot_ox objective. Well-conditioned: "
                                  "insensitive to regression-coefficient error.")
        self.f_PcTarget = Field(s, 3, "Target chamber P", 30.0, "bar",
                                "Used by the chamber_pressure objective. Also "
                                "well-conditioned.")
        self.f_dPmin = Field(s, 4, "Min dP margin", 20.0, "% of Pc",
                             "Injector pressure drop below this fraction of chamber "
                             "pressure is flagged as a feed-coupling risk.")
        self.f_designT = Field(s, 5, "Design time", 0.0, "s",
                               "Instant used by the design_point objective.")

        ttk.Label(s, text="O/F profile").grid(row=6, column=0, sticky="nw", padx=(6, 4))
        self.profile_txt = tk.Text(s, height=3, width=22)
        self.profile_txt.grid(row=6, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=2)
        _Tooltip(self.profile_txt,
                 "Optional time-varying target, one 'time, OF' pair per line, e.g.\n"
                 "0, 7.5\n2, 8.3\n5, 8.8\nLeave blank to use the constant target above.")

        # ---- numerics ----
        s = section("Numerics")
        self.f_dt = Field(s, 0, "Timestep", 10.0, "ms", "Integration timestep.")
        self.f_tmax = Field(s, 1, "Max sim time", 60.0, "s", "Safety limit.")
        self.prop_backend = tk.StringVar(value="esdu")
        ttk.Label(s, text="N2O properties").grid(row=2, column=0, sticky="w", padx=(6, 4))
        vals = ["esdu"] + (["coolprop"] if coolprop_available() else [])
        pcb = ttk.Combobox(s, textvariable=self.prop_backend, state="readonly",
                           width=18, values=vals)
        pcb.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=1)
        _Tooltip(pcb, "esdu: ESDU 91022 curve fits -- identical to HRAP MATLAB.\n"
                      "coolprop: reference EOS -- matches HRAP's Python build.")
        self.vapour_var = tk.BooleanVar(value=False)
        vch = ttk.Checkbutton(s, text="Include vapour phase", variable=self.vapour_var)
        vch.grid(row=4, column=0, columnspan=3, sticky="w", padx=(6, 4), pady=(2, 0))
        _Tooltip(vch,
                 "Off (default): the run stops the instant liquid oxidiser is gone. "
                 "Cleaner for sizing, because the vapour tail would otherwise skew a "
                 "burn-averaged O/F.\n\nOn: keeps going through vapour-phase blowdown "
                 "until chamber pressure decays to ambient, the way HRAP does. Turn this "
                 "on to see the liquid-to-vapour transition and the tail-off.")

        self.chamber_var = tk.StringVar(value="quasi-steady")
        ttk.Label(s, text="Chamber model").grid(row=3, column=0, sticky="w", padx=(6, 4))
        ccb = ttk.Combobox(s, textvariable=self.chamber_var, state="readonly", width=18,
                           values=["quasi-steady", "transient"])
        ccb.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(0, 6), pady=1)
        _Tooltip(ccb, "quasi-steady: solves Pc from the nozzle relation each step "
                      "(robust, recommended for sizing).\n"
                      "transient: HRAP's chamber ODE, for direct HRAP comparison.")

    def _build_actions(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(2, 0))
        af = ttk.Frame(parent)
        af.pack(fill="x", padx=6, pady=6)

        self.run_btn = ttk.Button(af, text="Compute injector sizing", command=self._run)
        self.run_btn.pack(fill="x", pady=2)
        self.sim_btn = ttk.Button(
            af, text="Simulate geometry as-built (no sizing)", command=self._simulate_only
        )
        self.sim_btn.pack(fill="x", pady=2)
        _Tooltip(self.sim_btn,
                 "Run the hole count / diameter / Cd exactly as entered and report what "
                 "the motor does. Nothing is solved for, so the resulting O/F is whatever "
                 "the geometry gives.\n\nThis is the question HRAP answers, so use it when "
                 "cross-referencing against an HRAP run.")
        ttk.Button(af, text="Compare SPI / HEM / Dyer",
                   command=self._compare).pack(fill="x", pady=2)

        imp = ttk.LabelFrame(af, text="HRAP cross-reference", padding=3)
        imp.pack(fill="x", pady=(6, 0))
        ttk.Button(imp, text="1. Import HRAP motor config (.mat)",
                   command=self._import_motor).pack(fill="x", pady=1)
        ttk.Button(imp, text="2. Load HRAP output (.csv)",
                   command=self._import_output).pack(fill="x", pady=1)
        self.cmp_btn = ttk.Button(imp, text="3. Run comparison",
                                  command=self._run_comparison, state="disabled")
        self.cmp_btn.pack(fill="x", pady=1)
        self.hrap_lbl = ttk.Label(imp, text="no HRAP run loaded",
                                  foreground="#a60", wraplength=300)
        self.hrap_lbl.pack(fill="x")

        cfgf = ttk.LabelFrame(af, text="Configuration", padding=3)
        cfgf.pack(fill="x", pady=(6, 0))
        ttk.Button(cfgf, text="Save config", width=13,
                   command=self._save_config).grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ttk.Button(cfgf, text="Load config", width=13,
                   command=self._load_config).grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        cfgf.columnconfigure(0, weight=1)
        cfgf.columnconfigure(1, weight=1)

        exp = ttk.LabelFrame(af, text="Export", padding=3)
        exp.pack(fill="x", pady=(6, 0))
        for i, (label, kind) in enumerate(
            [("Report .txt", "txt"), ("HRAP .mat", "mat"),
             ("Injector .json", "json"), ("History .csv", "csv")]
        ):
            ttk.Button(exp, text=label, width=13,
                       command=lambda k=kind: self._export(k)).grid(
                row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
        exp.columnconfigure(0, weight=1)
        exp.columnconfigure(1, weight=1)

        self.status = ttk.Label(parent, text="Ready.", foreground="#060", wraplength=320)
        self.status.pack(fill="x", padx=6, pady=(0, 6))

    def _build_results(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        self.nb = nb

        # plots
        pf = ttk.Frame(nb)
        nb.add(pf, text="Plots")
        self.fig = Figure(figsize=(10, 8), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, pf).update()

        # report
        rf = ttk.Frame(nb)
        nb.add(rf, text="Report")
        self.report_txt = tk.Text(rf, wrap="none", font=("Consolas", 9))
        ys = ttk.Scrollbar(rf, orient="vertical", command=self.report_txt.yview)
        xs = ttk.Scrollbar(rf, orient="horizontal", command=self.report_txt.xview)
        self.report_txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.report_txt.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        rf.rowconfigure(0, weight=1)
        rf.columnconfigure(0, weight=1)

        # model sweep
        sf = ttk.Frame(nb)
        nb.add(sf, text="Model comparison")
        self.sweep_fig = Figure(figsize=(10, 8), dpi=100, layout="constrained")
        self.sweep_canvas = FigureCanvasTkAgg(self.sweep_fig, master=sf)
        self.sweep_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.sweep_canvas, sf).update()

        # injector plate layout
        pf = ttk.Frame(nb)
        nb.add(pf, text="Plate layout")
        self.plate_fig = Figure(figsize=(8, 8), dpi=100, layout="constrained")
        self.plate_canvas = FigureCanvasTkAgg(self.plate_fig, master=pf)
        self.plate_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.plate_canvas, pf).update()

        # HRAP cross-reference
        hf = ttk.Frame(nb)
        nb.add(hf, text="HRAP cross-reference")
        hpane = ttk.PanedWindow(hf, orient="vertical")
        hpane.pack(fill="both", expand=True)

        top = ttk.Frame(hpane)
        hpane.add(top, weight=3)
        self.hrap_fig = Figure(figsize=(10, 7), dpi=100, layout="constrained")
        self.hrap_canvas = FigureCanvasTkAgg(self.hrap_fig, master=top)
        self.hrap_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.hrap_canvas, top).update()

        bot = ttk.Frame(hpane)
        hpane.add(bot, weight=1)
        self.hrap_txt = tk.Text(bot, wrap="none", font=("Consolas", 9), height=14)
        hys = ttk.Scrollbar(bot, orient="vertical", command=self.hrap_txt.yview)
        self.hrap_txt.configure(yscrollcommand=hys.set)
        self.hrap_txt.pack(side="left", fill="both", expand=True)
        hys.pack(side="right", fill="y")

    # -------------------------------------------------------------- helpers

    def _apply_preset(self, _=None):
        p = FUEL_PRESETS[self.fuel_var.get()]
        self.f_rega.set(p["reg_a"])
        self.f_regn.set(p["reg_n"])
        self.f_regm.set(p["reg_m"])
        self.f_rhof.set(p["rho_fuel"])
        self.f_ofTarget.set(p["opt_OF"])

    def _load_prop(self):
        path = filedialog.askopenfilename(
            title="Select HRAP propellant configuration",
            filetypes=[("HRAP propellant", "*.mat"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            prop = Propellant.from_hrap_mat(path, cstar_eff=self.f_cstareff.get())
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self._prop_file = path
        self._loaded_prop = prop
        self.f_rega.set(prop.reg_a)
        self.f_regn.set(prop.reg_n)
        self.f_regm.set(prop.reg_m)
        self.f_rhof.set(prop.rho_fuel)
        self.f_ofTarget.set(prop.opt_OF)
        self.prop_lbl.configure(
            text=f"loaded '{prop.name}': CEA table "
                 f"{len(prop.OF_grid)}x{len(prop.Pc_grid)} (O/F {prop.OF_grid[0]:g}-"
                 f"{prop.OF_grid[-1]:g}, Pc {prop.Pc_grid[0] / BAR:g}-"
                 f"{prop.Pc_grid[-1] / BAR:g} bar)",
            foreground="#060",
        )

    # ------------------------------------------------------- HRAP import

    def _apply_config(self, cfg):
        """Push an imported MotorConfig back into every input widget."""
        self.f_tankV.set(round(cfg.tank.volume * 1e3, 6))
        self.f_tankT.set(round(cfg.tank.fill_temp - 273.15, 4))
        self.f_oxm.set(round(cfg.tank.ox_mass, 6))
        self.f_super.set("" if cfg.tank.supercharge_P is None
                         else round(cfg.tank.supercharge_P / BAR, 4))
        self.f_amb.set(round(cfg.ambient_P / BAR, 6))

        self.model_var.set(cfg.injector.model.value)
        self.f_cd.set(round(cfg.injector.Cd, 5))
        self.f_thick.set(round(cfg.injector.plate_thickness * 1e3, 4))
        self.f_nholes.set(cfg.injector.n_holes)
        self.f_holed.set(round(cfg.injector.hole_d * 1e3, 5))

        self.f_grainL.set(round(cfg.grain.length * 1e3, 4))
        self.f_portID.set(round(cfg.grain.port_id * 1e3, 4))
        self.f_grainOD.set(round(cfg.grain.outer_d * 1e3, 4))
        self.f_nports.set(cfg.grain.n_ports)

        self.f_rega.set(cfg.propellant.reg_a)
        self.f_regn.set(cfg.propellant.reg_n)
        self.f_regm.set(cfg.propellant.reg_m)
        self.f_rhof.set(round(cfg.propellant.rho_fuel, 4))
        self.f_cstareff.set(round(cfg.propellant.cstar_eff, 5))
        self.reg_mode_var.set(cfg.regression_mode)
        self.f_constOF.set(round(cfg.const_OF, 5))

        self.f_thrt.set(round(cfg.nozzle.throat_d * 1e3, 4))
        self.f_er.set(round(cfg.nozzle.expansion_ratio, 5))
        self.f_nozcd.set(round(cfg.nozzle.Cd, 5))
        self.f_nozeff.set(round(cfg.nozzle.efficiency, 5))

        self.f_dt.set(round(cfg.dt * 1e3, 5))
        self.f_tmax.set(cfg.max_time)
        self.chamber_var.set(cfg.chamber_mode)
        self.vapour_var.set(not cfg.stop_at_liquid_exhausted)

        if cfg.regression_mode == "shifting":
            self.f_ofTarget.set(round(cfg.propellant.opt_OF, 4))
        else:
            self.f_ofTarget.set(round(cfg.const_OF, 4))

        # Carried through _collect but not exposed as editable fields.
        self._chamber_volume = cfg.chamber_volume
        self._initial_Pc = cfg.initial_Pc

    def _import_motor(self):
        path = filedialog.askopenfilename(
            title="Select HRAP motor configuration",
            filetypes=[("HRAP motor config", "*.mat"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            imported = load_hrap_motor(path, match_hrap_models=True)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return

        self._apply_config(imported.config)
        prop = imported.config.propellant
        self._loaded_prop = prop
        self._prop_file = imported.propellant_file
        if prop.has_table:
            self.prop_lbl.configure(
                text=f"loaded '{prop.name}': CEA table "
                     f"{len(prop.OF_grid)}x{len(prop.Pc_grid)}",
                foreground="#060",
            )
        else:
            self.prop_lbl.configure(
                text=f"'{prop.name}': no CEA table found -- constant c* in use",
                foreground="#a60",
            )

        detail = "\n".join(f"  {k}: {v}" for k, v in imported.info.items())
        msg = f"Imported '{imported.motor_name}'.\n\n{detail}"
        if imported.warnings:
            msg += "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in imported.warnings)
        msg += (
            "\n\nThe injector model was set to SPI and the chamber to transient, "
            "which is what HRAP itself uses. Run the sizing or comparison now."
        )
        messagebox.showinfo("HRAP motor imported", msg)
        self.status.configure(
            text=f"Imported HRAP motor '{imported.motor_name}'.", foreground="#060"
        )
        self._maybe_enable_compare()

    def _import_output(self):
        path = filedialog.askopenfilename(
            title="Select HRAP simulation output CSV",
            filetypes=[("HRAP output", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            run = load_hrap_output(path)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        self._hrap_run = run
        self.hrap_lbl.configure(
            text=f"loaded {run.source}: {len(run.t)} rows, {run.burn_time:.2f} s, "
                 f"{len(run.channels)} channels",
            foreground="#060",
        )
        if run.warnings:
            messagebox.showwarning("Loaded with notes", "\n\n".join(run.warnings))
        self._maybe_enable_compare()

    def _maybe_enable_compare(self):
        state = "normal" if getattr(self, "_hrap_run", None) is not None else "disabled"
        self.cmp_btn.configure(state=state)

    def _run_comparison(self):
        run = getattr(self, "_hrap_run", None)
        if run is None:
            messagebox.showinfo("No HRAP run", "Load an HRAP output CSV first.")
            return
        try:
            cfg, _ = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        props, table = self._backend()
        errs = cfg.validate(props)
        if errs:
            messagebox.showerror("Invalid configuration", "\n\n".join(errs))
            return

        self.status.configure(text="Running comparison...", foreground="#a60")
        self.update_idletasks()
        try:
            burn = simulate(cfg, props, table)
            cmp = compare_to_hrap(burn, run)
        except Exception as exc:
            messagebox.showerror("Comparison failed", str(exc))
            self.status.configure(text=f"Comparison failed: {exc}", foreground="#a00")
            return

        build_hrap_overlay_figure(self.hrap_fig, burn, run, cfg.injector.model.value)
        self.hrap_canvas.draw()
        self.hrap_txt.delete("1.0", "end")
        self.hrap_txt.insert("1.0", format_comparison(cmp, cfg.injector.model.value))
        self.nb.select(3)

        worst = max((c.rms_pct for c in cmp.channels), default=float("nan"))
        self.status.configure(
            text=f"Comparison done. Worst channel RMS {worst:.1f}% "
                 f"({cfg.injector.model.value} model).",
            foreground="#060" if worst < 10 else "#a60",
        )

    def _parse_profile(self):
        raw = self.profile_txt.get("1.0", "end").strip()
        if not raw:
            return None, None
        ts, ofs = [], []
        for i, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"O/F profile line {i}: expected 'time, OF', got {line!r}")
            ts.append(float(parts[0]))
            ofs.append(float(parts[1]))
        if len(ts) < 2:
            raise ValueError("O/F profile needs at least two points")
        order = np.argsort(ts)
        return np.asarray(ts)[order], np.asarray(ofs)[order]

    def _collect(self):
        """Read every input field into a MotorConfig + SizingTarget."""
        prop = getattr(self, "_loaded_prop", None)
        if prop is None:
            prop = Propellant()
        prop.reg_a = self.f_rega.get()
        prop.reg_n = self.f_regn.get()
        prop.reg_m = self.f_regm.get()
        prop.rho_fuel = self.f_rhof.get()
        prop.cstar_const = self.f_cstar.get()
        prop.cstar_eff = self.f_cstareff.get()
        if not prop.has_table:
            prop.name = self.fuel_var.get()

        sup = self.f_super.var.get().strip()
        tank = Tank(
            volume=self.f_tankV.get() / 1e3,
            fill_temp=self.f_tankT.get() + 273.15,
            ox_mass=self.f_oxm.get(),
            supercharge_P=(float(sup) * BAR if sup else None),
        )
        grain = Grain(
            length=self.f_grainL.get() / 1e3,
            port_id=self.f_portID.get() / 1e3,
            outer_d=self.f_grainOD.get() / 1e3,
            n_ports=self.f_nports.get(int),
        )
        noz = Nozzle(
            throat_d=self.f_thrt.get() / 1e3,
            expansion_ratio=self.f_er.get(),
            Cd=self.f_nozcd.get(),
            efficiency=self.f_nozeff.get(),
        )
        inj = InjectorSpec(
            n_holes=self.f_nholes.get(int),
            hole_d=self.f_holed.get() / 1e3,
            Cd=self.f_cd.get(),
            plate_thickness=self.f_thick.get() / 1e3,
            min_hole_d=self.f_dmin.get() / 1e3,
            model=FlowModel.from_str(self.model_var.get()),
            ld_ref=self.f_ldref.get(),
        )
        cfg = MotorConfig(
            tank=tank, grain=grain, nozzle=noz, injector=inj, propellant=prop,
            ambient_P=self.f_amb.get() * BAR,
            dt=self.f_dt.get() / 1e3,
            max_time=self.f_tmax.get(),
            chamber_mode=self.chamber_var.get(),
            stop_at_liquid_exhausted=not self.vapour_var.get(),
            regression_mode=self.reg_mode_var.get(),
            const_OF=self.f_constOF.get(),
            chamber_volume=getattr(self, "_chamber_volume", 0.0),
            initial_Pc=getattr(self, "_initial_Pc", None),
        )

        pt, pof = self._parse_profile()
        target = SizingTarget(
            OF=self.f_ofTarget.get(),
            mdot_ox=self.f_mdotTarget.get(),
            chamber_P=self.f_PcTarget.get() * BAR,
            profile_t=pt,
            profile_OF=pof,
            objective=self.obj_var.get(),
            design_time=self.f_designT.get(),
            min_dP_fraction=self.f_dPmin.get() / 100.0,
        )
        return cfg, target

    def _backend(self):
        name = self.prop_backend.get()
        if self._props is None or self._props_name != name:
            self._props = get_backend(name)
            self._table = SaturationTable(self._props)
            self._props_name = name
        return self._props, self._table

    # ------------------------------------------------------------- actions

    def _run(self):
        try:
            cfg, target = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        props, table = self._backend()
        errs = cfg.validate(props)
        if errs:
            messagebox.showerror("Invalid configuration", "\n\n".join(f"- {e}" for e in errs))
            return

        self.run_btn.configure(state="disabled")
        self.status.configure(text="Sizing...", foreground="#a60")

        def work():
            try:
                res = size_injector(cfg, props, table, target, fix=self.fix_var.get())
                self._queue.put(("ok", res, cfg))
            except Exception as exc:
                self._queue.put(("err", exc, traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _save_config(self):
        """Persist every input field so a design can be reopened later."""
        try:
            cfg, target = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="motor_config.json",
            filetypes=[("Tool configuration", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            save_config(path, cfg, target, propellant_file=self._prop_file)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.configure(text=f"Configuration saved to {path}", foreground="#060")

    def _load_config(self):
        path = filedialog.askopenfilename(
            title="Load tool configuration",
            filetypes=[("Tool configuration", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            cfg, target, warns = load_config(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self._apply_config(cfg)
        self._loaded_prop = cfg.propellant
        self.obj_var.set(target.objective)
        self.f_ofTarget.set(round(target.OF, 5))
        self.f_mdotTarget.set(round(target.mdot_ox, 5))
        self.f_PcTarget.set(round(target.chamber_P / BAR, 5))
        self.f_dPmin.set(round(target.min_dP_fraction * 100, 4))
        self.f_designT.set(target.design_time)

        if cfg.propellant.has_table:
            self.prop_lbl.configure(
                text=f"loaded '{cfg.propellant.name}': CEA table "
                     f"{len(cfg.propellant.OF_grid)}x{len(cfg.propellant.Pc_grid)}",
                foreground="#060")
        else:
            self.prop_lbl.configure(
                text=f"'{cfg.propellant.name}': constant c* in use",
                foreground="#a60")
        if warns:
            messagebox.showwarning("Loaded with notes", "\n\n".join(warns))
        self.status.configure(text=f"Loaded {path}", foreground="#060")

    def _simulate_only(self):
        """Forward-run the entered geometry, the way HRAP does."""
        try:
            cfg, target = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        props, table = self._backend()
        errs = cfg.validate(props)
        if errs:
            messagebox.showerror("Invalid configuration", "\n\n".join(f"- {e}" for e in errs))
            return

        self.run_btn.configure(state="disabled")
        self.sim_btn.configure(state="disabled")
        self.status.configure(text="Simulating as-built geometry...", foreground="#a60")

        def work():
            try:
                res = analyse_geometry(cfg, props, table, target)
                self._queue.put(("ok", res, cfg))
            except Exception as exc:
                self._queue.put(("err", exc, traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _compare(self):
        """Overlay the three flow models at fixed geometry."""
        try:
            cfg, _ = self._collect()
        except Exception as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        props, table = self._backend()
        up = props.upstream_state(cfg.tank.fill_temp, cfg.tank.supercharge_P)

        curve = OrificeCurve(table, up, model=FlowModel.DYER,
                             L_over_D=cfg.injector.L_over_D, ld_ref=cfg.injector.ld_ref)
        build_comparison_figure(self.sweep_fig, curve, up)
        self.sweep_canvas.draw()
        self.nb.select(2)
        self.status.configure(text="Model comparison updated.", foreground="#060")

    def _poll(self):
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return

        self.run_btn.configure(state="normal")
        self.sim_btn.configure(state="normal")
        if item[0] == "err":
            exc, tb = item[1], item[2]
            self.status.configure(text=f"Failed: {exc}", foreground="#a00")
            messagebox.showerror("Sizing failed", f"{exc}\n\n{tb[-1500:]}")
        else:
            _, res, cfg = item
            self._result, self._cfg = res, cfg
            self._show(res, cfg)
        self.after(100, self._poll)

    def _show(self, res, cfg):
        self.report_txt.delete("1.0", "end")
        self.report_txt.insert("1.0", build_report(res, cfg, self._props_name))

        build_results_figure(self.fig, res, cfg)
        self.canvas.draw()

        try:
            pd_mm = self.f_plated.get()
        except Exception:
            pd_mm = 0.0
        build_injector_figure(self.plate_fig, res.plate, plate_d_mm=pd_mm,
                              grain_od_mm=cfg.grain.outer_d * 1e3)
        self.plate_canvas.draw()

        flag = "" if res.dP_margin_ok else "  [dP margin LOW]"
        colour = "#060" if (res.dP_margin_ok and not res.warnings) else "#a60"
        self.status.configure(
            text=f"{res.plate.n_holes} holes x {res.plate.hole_d * 1e3:.3f} mm, "
                 f"mean O/F {res.achieved_mean_OF:.2f}{flag}",
            foreground=colour,
        )
        self.nb.select(0)

    def _export(self, kind):
        if self._result is None:
            messagebox.showinfo("Nothing to export", "Run a sizing computation first.")
            return
        ext = {"txt": ".txt", "mat": ".mat", "json": ".json", "csv": ".csv"}[kind]
        path = filedialog.asksaveasfilename(
            defaultextension=ext, filetypes=[(kind.upper(), f"*{ext}"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            if kind == "txt":
                with open(path, "w", encoding="utf-8") as f:
                    f.write(build_report(self._result, self._cfg, self._props_name))
            elif kind == "mat":
                export_hrap_mat(path, self._result, self._cfg,
                                motor_name=self._cfg.propellant.name.replace(" ", "_"),
                                propellant_file=self._prop_file)
            elif kind == "json":
                export_injector_json(path, self._result, self._cfg)
            else:
                export_timeseries_csv(path, self._result)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.configure(text=f"Exported to {path}", foreground="#060")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
