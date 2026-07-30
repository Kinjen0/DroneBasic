"""
SeaDronesSee Auto GUI panel.

Isolated from the interactive multi-frame labeling UI. Can be embedded in a
Toplevel or any parent frame.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Dict, Any

from .config import SeaDronesSeeConfig
from .runner import SeaDronesSeeRunner


class SeaDronesSeePanel(ttk.Frame):
    """Controls + live stats for the SDS auto-labeled A/RED pipeline."""

    def __init__(
        self,
        master,
        initial_config: Optional[SeaDronesSeeConfig] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.cfg = initial_config or SeaDronesSeeConfig.default()
        self.runner = SeaDronesSeeRunner(self.cfg)
        self.runner.on_stats = self._on_stats
        self._poll_job = None
        self._build()
        self._start_poller()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        pad = {"padx": 6, "pady": 4}

        title = ttk.Label(
            self,
            text="SeaDronesSee Auto A/RED  (COCO bbox labels · no human multi-frame labeling)",
            font=("TkDefaultFont", 12, "bold"),
        )
        title.pack(anchor="w", **pad)

        # --- Dataset ---
        ds = ttk.LabelFrame(self, text="Dataset")
        ds.pack(fill="x", **pad)

        row = ttk.Frame(ds)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Root:").pack(side="left")
        self.root_var = tk.StringVar(value=self.cfg.dataset_root)
        ttk.Entry(row, textvariable=self.root_var, width=50).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse…", command=self._browse_root).pack(side="left")

        row2 = ttk.Frame(ds)
        row2.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2, text="Split:").pack(side="left")
        self.split_var = tk.StringVar(value=self.cfg.split)
        for s in ("train", "val", "both"):
            ttk.Radiobutton(row2, text=s, value=s, variable=self.split_var).pack(side="left", padx=4)

        row3 = ttk.Frame(ds)
        row3.pack(fill="x", padx=4, pady=2)
        ttk.Label(row3, text="Max images (empty=all):").pack(side="left")
        self.max_images_var = tk.StringVar(
            value="" if self.cfg.max_images is None else str(self.cfg.max_images)
        )
        ttk.Entry(row3, textvariable=self.max_images_var, width=8).pack(side="left", padx=4)
        ttk.Label(row3, text="Max tiles (empty=all):").pack(side="left", padx=(12, 0))
        self.max_tiles_var = tk.StringVar(
            value="" if self.cfg.max_tiles is None else str(self.cfg.max_tiles)
        )
        ttk.Entry(row3, textvariable=self.max_tiles_var, width=10).pack(side="left", padx=4)

        # --- Tiling ---
        til = ttk.LabelFrame(self, text="Tiling")
        til.pack(fill="x", **pad)
        r = ttk.Frame(til)
        r.pack(fill="x", padx=4, pady=2)
        ttk.Label(r, text="Tile W:").pack(side="left")
        self.tw_var = tk.StringVar(value=str(self.cfg.tiling.tile_width))
        ttk.Entry(r, textvariable=self.tw_var, width=6).pack(side="left", padx=2)
        ttk.Label(r, text="H:").pack(side="left")
        self.th_var = tk.StringVar(value=str(self.cfg.tiling.tile_height))
        ttk.Entry(r, textvariable=self.th_var, width=6).pack(side="left", padx=2)
        ttk.Label(r, text="Stride X:").pack(side="left", padx=(10, 0))
        self.sx_var = tk.StringVar(value=str(self.cfg.tiling.stride_x or self.cfg.tiling.tile_width))
        ttk.Entry(r, textvariable=self.sx_var, width=6).pack(side="left", padx=2)
        ttk.Label(r, text="Y:").pack(side="left")
        self.sy_var = tk.StringVar(value=str(self.cfg.tiling.stride_y or self.cfg.tiling.tile_height))
        ttk.Entry(r, textvariable=self.sy_var, width=6).pack(side="left", padx=2)
        ttk.Button(r, text="16×16", command=lambda: self._set_tile(16)).pack(side="left", padx=6)
        ttk.Button(r, text="32×32", command=lambda: self._set_tile(32)).pack(side="left")

        # --- Features ---
        feat = ttk.LabelFrame(self, text="Features")
        feat.pack(fill="x", **pad)
        self.mode_var = tk.StringVar(value=self.cfg.feature_mode)
        rf = ttk.Frame(feat)
        rf.pack(fill="x", padx=4, pady=2)
        ttk.Radiobutton(rf, text="Raw pixels", value="raw", variable=self.mode_var,
                        command=self._toggle_mode).pack(side="left")
        ttk.Radiobutton(rf, text="DINOv3", value="dino", variable=self.mode_var,
                        command=self._toggle_mode).pack(side="left", padx=8)

        self.dino_row = ttk.Frame(feat)
        self.dino_row.pack(fill="x", padx=4, pady=2)
        ttk.Label(self.dino_row, text="Model:").pack(side="left")
        self.dino_model_var = tk.StringVar(value=self.cfg.features.model_name)
        ttk.Entry(self.dino_row, textvariable=self.dino_model_var, width=48).pack(
            side="left", fill="x", expand=True, padx=4
        )

        self.raw_row = ttk.Frame(feat)
        self.raw_row.pack(fill="x", padx=4, pady=2)
        self.raw_l2_var = tk.BooleanVar(value=self.cfg.raw_features.l2_normalize)
        self.raw_gray_var = tk.BooleanVar(value=self.cfg.raw_features.grayscale)
        ttk.Checkbutton(self.raw_row, text="L2 normalize", variable=self.raw_l2_var).pack(side="left")
        ttk.Checkbutton(self.raw_row, text="Grayscale", variable=self.raw_gray_var).pack(side="left", padx=8)
        self._toggle_mode()

        # --- A_RED ---
        ared = ttk.LabelFrame(self, text="A/RED")
        ared.pack(fill="x", **pad)
        ar = ttk.Frame(ared)
        ar.pack(fill="x", padx=4, pady=2)
        ttk.Label(ar, text="κ (kappa):").pack(side="left")
        self.kappa_var = tk.StringVar(value=str(self.cfg.ared.kappa))
        ttk.Entry(ar, textvariable=self.kappa_var, width=8).pack(side="left", padx=4)
        ttk.Label(ar, text="Buffer:").pack(side="left", padx=(10, 0))
        self.buf_var = tk.StringVar(value=str(self.cfg.ared.l_buf_size))
        ttk.Entry(ar, textvariable=self.buf_var, width=8).pack(side="left", padx=4)

        # --- Paths ---
        paths = ttk.LabelFrame(self, text="Storage")
        paths.pack(fill="x", **pad)
        pr = ttk.Frame(paths)
        pr.pack(fill="x", padx=4, pady=2)
        ttk.Label(pr, text="Annotation DB:").pack(side="left")
        self.db_var = tk.StringVar(value=self.cfg.tile_annotations_db)
        ttk.Entry(pr, textvariable=self.db_var, width=40).pack(side="left", fill="x", expand=True, padx=4)

        # --- Controls ---
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", **pad)
        ttk.Button(ctrl, text="Start", command=self._start).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Pause", command=self._pause).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Resume", command=self._resume).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Stop", command=self._stop).pack(side="left", padx=2)
        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(ctrl, text="Save ARED Model", command=self._save_model).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Load ARED Model", command=self._load_model).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Reset ARED (cold)", command=self._reset_model).pack(side="left", padx=2)

        # --- Stats ---
        stats_f = ttk.LabelFrame(self, text="Live stats")
        stats_f.pack(fill="both", expand=True, **pad)
        self.stats_text = tk.Text(stats_f, height=14, wrap="word")
        self.stats_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.stats_text.insert("end", "Idle. Configure and press Start.\n")
        self.stats_text.configure(state="disabled")

        note = ttk.Label(
            self,
            text=(
                "Labels come from COCO boxes (object tile → category/relevant; else water). "
                "Metrics write to runs/ like the interactive pipeline. "
                "Uses seadronesee_tile_annotations.db by default (not the interactive drone DB)."
            ),
            wraplength=720,
            justify="left",
        )
        note.pack(anchor="w", **pad)

    def _toggle_mode(self) -> None:
        mode = self.mode_var.get()
        # Enable/disable rows visually
        state_dino = "normal" if mode == "dino" else "disabled"
        state_raw = "normal" if mode == "raw" else "disabled"
        for child in self.dino_row.winfo_children():
            try:
                child.configure(state=state_dino)
            except tk.TclError:
                pass
        for child in self.raw_row.winfo_children():
            try:
                child.configure(state=state_raw)
            except tk.TclError:
                pass

    def _set_tile(self, n: int) -> None:
        self.tw_var.set(str(n))
        self.th_var.set(str(n))
        self.sx_var.set(str(n))
        self.sy_var.set(str(n))

    def _browse_root(self) -> None:
        d = filedialog.askdirectory(title="SeaDronesSee processed export root")
        if d:
            self.root_var.set(d)

    def _parse_optional_int(self, s: str) -> Optional[int]:
        s = (s or "").strip()
        if not s:
            return None
        return int(s)

    def _read_config_from_ui(self) -> SeaDronesSeeConfig:
        cfg = SeaDronesSeeConfig.default()
        # copy nested from current then override
        cfg = self.cfg
        cfg.dataset_root = self.root_var.get().strip() or cfg.dataset_root
        cfg.split = self.split_var.get().strip() or "train"
        cfg.max_images = self._parse_optional_int(self.max_images_var.get())
        cfg.max_tiles = self._parse_optional_int(self.max_tiles_var.get())
        tw = max(1, int(float(self.tw_var.get())))
        th = max(1, int(float(self.th_var.get())))
        sx = max(1, int(float(self.sx_var.get())))
        sy = max(1, int(float(self.sy_var.get())))
        cfg.tiling.tile_width = tw
        cfg.tiling.tile_height = th
        cfg.tiling.stride_x = sx
        cfg.tiling.stride_y = sy
        cfg.tiling.overlap_x = max(0, tw - sx)
        cfg.tiling.overlap_y = max(0, th - sy)
        cfg.feature_mode = self.mode_var.get().strip() or "dino"
        cfg.features.model_name = self.dino_model_var.get().strip() or cfg.features.model_name
        cfg.raw_features.l2_normalize = bool(self.raw_l2_var.get())
        cfg.raw_features.grayscale = bool(self.raw_gray_var.get())
        cfg.ared.kappa = float(self.kappa_var.get())
        cfg.ared.l_buf_size = int(float(self.buf_var.get()))
        cfg.tile_annotations_db = self.db_var.get().strip() or cfg.tile_annotations_db
        return cfg

    def _start(self) -> None:
        if self.runner.is_running():
            messagebox.showinfo("Running", "A SeaDronesSee run is already in progress.")
            return
        try:
            cfg = self._read_config_from_ui()
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return
        root = Path(cfg.dataset_root)
        if not root.is_dir():
            messagebox.showerror("Dataset", f"Dataset root not found:\n{root}")
            return
        self.cfg = cfg
        self.runner.update_config(cfg)
        self._append_stats(f"Starting… mode={cfg.feature_mode} tile={cfg.tiling.tile_width}x{cfg.tiling.tile_height} split={cfg.split}\n")
        try:
            self.runner.start()
        except Exception as e:
            messagebox.showerror("Start failed", str(e))

    def _pause(self) -> None:
        self.runner.pause()

    def _resume(self) -> None:
        self.runner.resume()

    def _stop(self) -> None:
        self.runner.stop()

    def _save_model(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            title="Save ARED model state",
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self.runner.save_ared_state(path)
            messagebox.showinfo("Saved", f"Saved A_RED state to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Load ARED model state",
            filetypes=[("Pickle", "*.pkl"), ("All", "*.*")],
        )
        if not path:
            return
        if self.runner.is_running():
            messagebox.showwarning("Busy", "Stop the current run before loading a model.")
            return
        try:
            self.runner.load_ared_state(path)
            messagebox.showinfo("Loaded", f"Loaded A_RED state from:\n{path}\nIt will be used on next Start.")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def _reset_model(self) -> None:
        if self.runner.is_running():
            messagebox.showwarning("Busy", "Stop the run first.")
            return
        self.runner.reset_ared()
        self._append_stats("A_RED reset to cold-start.\n")

    def _on_stats(self, stats: Dict[str, Any]) -> None:
        # Called from worker thread — schedule UI update
        try:
            self.after(0, lambda s=stats: self._render_stats(s))
        except Exception:
            pass

    def _render_stats(self, stats: Dict[str, Any]) -> None:
        lines = [
            f"status: {stats.get('status')}",
            f"feature_mode: {stats.get('feature_mode')}  dim: {stats.get('feature_dim')}",
            f"current: {stats.get('current_video')}",
            f"images_done: {stats.get('images_done')}  frames_read: {stats.get('frames_read')}",
            f"tiles_processed: {stats.get('tiles_processed')}",
            f"ared_queries: {stats.get('ared_queries')}  clusters: {stats.get('ared_clusters')}  known_labels: {stats.get('ared_known_labels')}",
            f"gt_positives: {stats.get('gt_positives')}  gt_negatives: {stats.get('gt_negatives')}",
            f"metrics: {stats.get('metrics_last_line') or ''}",
            f"run_dir: {stats.get('metrics_run_dir') or ''}",
        ]
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("end", "\n".join(lines) + "\n")
        self.stats_text.configure(state="disabled")

    def _append_stats(self, text: str) -> None:
        self.stats_text.configure(state="normal")
        self.stats_text.insert("end", text)
        self.stats_text.see("end")
        self.stats_text.configure(state="disabled")

    def _start_poller(self) -> None:
        def _tick():
            if self.runner and self.runner.stats:
                # light refresh even without callbacks
                st = self.runner.stats
                if st.get("status") in ("running", "paused", "finished", "stopped", "error"):
                    self._render_stats(st)
            self._poll_job = self.after(500, _tick)

        self._poll_job = self.after(500, _tick)

    def destroy(self) -> None:
        try:
            if self._poll_job is not None:
                self.after_cancel(self._poll_job)
        except Exception:
            pass
        try:
            if self.runner.is_running():
                self.runner.stop(join_timeout=2.0)
        except Exception:
            pass
        super().destroy()


def open_seadronesee_window(parent=None, initial_config: Optional[SeaDronesSeeConfig] = None) -> tk.Toplevel:
    """Open a standalone Toplevel with the SDS panel (safe for interactive MainWindow)."""
    if parent is None:
        win = tk.Toplevel()
    else:
        win = tk.Toplevel(parent)
    win.title("SeaDronesSee Auto A/RED")
    win.geometry("820x720")
    win.minsize(640, 520)
    panel = SeaDronesSeePanel(win, initial_config=initial_config)
    panel.pack(fill="both", expand=True)
    win._sds_panel = panel  # type: ignore[attr-defined]

    def _on_close():
        try:
            panel.destroy()
        except Exception:
            pass
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _on_close)
    return win
