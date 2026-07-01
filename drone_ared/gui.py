"""
GUI module - Main control window + resizable labeling dialog.

This is the primary user interface. All major actions (load video, start/pause,
change parameters, save/load model & labels, etc.) are available from the GUI.

Labeling dialog requirements implemented:
- Fully resizable (user can shrink or grow to always fit on screen)
- Large image area that refits on resize (Canvas + PIL)
- Existing classes shown in a Listbox (filterable)
- **Double-click** (or Assign button / Return) on existing class assigns immediately
- New class: simple Entry + Checkbutton("Relevant") + Create & Assign
- Very keyboard friendly for labeling hundreds of tiles
- Cache hits are silent (no dialog)

Design is deliberately split into small methods with lots of comments for future expansion
(e.g. adding a "history" panel, class merge UI, export buttons, theming, etc.).
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .config import PipelineConfig, GUIConfig
from .pipeline import DroneAREDController, LabelRequest
from .label_store import PersistentLabelStore
from .ared_adapter import AREDAdapter


class LabelingDialog(tk.Toplevel):
    """
    The critical high-volume labeling UI.

    - Resizable main window (user drags edges).
    - Canvas shows the tile image and scales nicely on <Configure>.
    - Left: list of existing classes (double-click to assign).
    - Filter box above the list.
    - Bottom-right: New class entry + relevant checkbox.
    """

    def __init__(self, master, request: LabelRequest, known_classes: List[str],
                 on_assign: callable, class_counts: Optional[Dict[str, int]] = None):
        super().__init__(master)
        self.title("Review Queried Tile - A/RED Drone  [Persistent - reposition me once!]")
        self.current_req = request
        self.on_assign = on_assign  # notification callback (refresh etc.), we do set_result ourselves
        self.class_counts = class_counts or {}
        self.known_classes = sorted(set(known_classes))

        # Larger default + fully resizable for high-res displays and comfort during long labeling sessions
        self.geometry("1050x820")
        self.minsize(650, 520)
        self.resizable(True, True)

        # Use StringVar for reliable .get() on the new class entry
        self.new_var = tk.StringVar()

        self._current_img_tk: Optional[ImageTk.PhotoImage] = None
        self._filtered_classes: List[str] = list(self.known_classes)

        self._build_ui()
        self._load_and_show_image()
        self._refresh_class_list()
        self._update_info()

        # Keyboard bindings for power users labeling many tiles
        self.bind("<Return>", lambda e: self._assign_selected())
        self.bind("<Escape>", lambda e: self._assign_as_background())
        self.bind("<Control-n>", lambda e: self.new_entry.focus_set())

        # Focus the list so double-click / arrows work immediately
        self.after(150, lambda: self.class_list.focus_set())

    def _build_ui(self):
        # Top info bar - use variable so it can be updated for each new tile in persistent window
        info = ttk.Frame(self)
        info.pack(fill="x", padx=8, pady=4)

        self.info_var = tk.StringVar(value="Loading tile info...")
        ttk.Label(info, textvariable=self.info_var, font=("TkDefaultFont", 12)).pack(side="left")

        ttk.Label(info, text="  (Double-click class or press Enter to assign. Resize me!)", foreground="gray").pack(side="right")

        # Main area: image (left) + classes (right)
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        # --- IMAGE (resizable canvas) ---
        img_frame = ttk.LabelFrame(main, text="Tile Image (resize window to fit)")
        img_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(img_frame, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Fit controls for comfort (very useful when labeling hundreds of tiles)
        fit_bar = ttk.Frame(img_frame)
        fit_bar.pack(fill="x")
        ttk.Button(fit_bar, text="Fit to Window", command=lambda: (self.update_idletasks(), self._display_image_on_canvas())).pack(side="left", padx=2)
        ttk.Button(fit_bar, text="Larger View", command=lambda: (self.update_idletasks(), self._zoom(1.2))).pack(side="left")
        ttk.Button(fit_bar, text="Smaller View", command=lambda: (self.update_idletasks(), self._zoom(0.8))).pack(side="left")

        # --- CLASSES SIDE ---
        right = ttk.Frame(main)
        right.pack(side="right", fill="y", padx=(8, 0))

        ttk.Label(right, text="Existing Classes (double-click or select + Assign)").pack(anchor="w")

        # Filter
        filter_frame = ttk.Frame(right)
        filter_frame.pack(fill="x", pady=2)
        ttk.Label(filter_frame, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", self._on_filter_changed)
        ttk.Entry(filter_frame, textvariable=self.filter_var, width=22).pack(side="left", fill="x", expand=True)

        # Class list - bigger font for readability
        list_frame = ttk.Frame(right)
        list_frame.pack(fill="both", expand=True)

        self.class_list = tk.Listbox(list_frame, height=18, exportselection=False, font=("TkDefaultFont", 13))
        self.class_list.pack(side="left", fill="both", expand=True)
        self.class_list.bind("<Double-Button-1>", self._on_double_click)
        self.class_list.bind("<Return>", lambda e: self._assign_selected())

        yscroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_list.yview)
        yscroll.pack(side="right", fill="y")
        self.class_list.configure(yscrollcommand=yscroll.set)

        # Assign button for selected
        ttk.Button(right, text="Assign Selected (Enter / Double-click)", command=self._assign_selected).pack(fill="x", pady=4)

        # --- NEW CLASS ---
        new_frame = ttk.LabelFrame(right, text="New Class")
        new_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(new_frame, text="Class name:").pack(anchor="w")
        self.new_entry = ttk.Entry(new_frame, textvariable=self.new_var, font=("TkDefaultFont", 13))
        self.new_entry.pack(fill="x", pady=2)
        self.new_entry.bind("<Return>", lambda e: self._create_and_assign())

        self.relevant_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(new_frame, text="Relevant (interesting / anomaly)", variable=self.relevant_var).pack(anchor="w")

        ttk.Button(new_frame, text="Create & Assign", command=self._create_and_assign).pack(fill="x", pady=4)

        # Bottom quick actions
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=6)

        ttk.Button(bottom, text="Mark as Background / Irrelevant", command=self._assign_as_background).pack(side="left")
        ttk.Button(bottom, text="Close Window (recreates on next query)", command=self._close_window).pack(side="right")

        # Status for the dialog
        self.dialog_status_var = tk.StringVar(value="Choose from list (double-click or button) or type new class + Create & Assign")
        ttk.Label(self, textvariable=self.dialog_status_var, relief="sunken", font=("TkDefaultFont", 11)).pack(fill="x", padx=8, pady=4)

        # Make the new entry easy to reach
        self.after(200, lambda: self.new_entry.focus_set() if not self.known_classes else None)

        # Initial status
        self.dialog_status_var.set("Select an existing class (double-click or button) or type a new name + Create & Assign. Window will stay open.")

    def _load_and_show_image(self):
        try:
            req = getattr(self, 'current_req', None) or getattr(self, 'request', None)
            tile_obj = getattr(req, 'tile', None) if req else None
            img = getattr(tile_obj, 'image', None) if tile_obj else None
            if isinstance(img, Image.Image):
                self._original_pil = img.copy()
            else:
                self._original_pil = Image.new("RGB", (224, 224), "gray")
        except Exception:
            self._original_pil = Image.new("RGB", (224, 224), "gray")

        # Delay the first display to ensure canvas has real size after layout
        self.after(80, lambda: (self.update_idletasks(), self._display_image_on_canvas(), self._update_info()))

    def _display_image_on_canvas(self, target_max=None):
        """Fit the tile image nicely inside the canvas while preserving aspect ratio.
        Called on resize and via the Fit button. This is the 'fit to box' system.
        """
        if not hasattr(self, "_original_pil"):
            return
        # Force Tk to compute current sizes (common issue with winfo_* right after layout changes or button clicks)
        self.update_idletasks()
        self.canvas.update_idletasks()
        cw = max(120, self.canvas.winfo_width())
        ch = max(120, self.canvas.winfo_height())
        if cw < 20 or ch < 20:
            cw, ch = 700, 520

        if target_max is None:
            max_w, max_h = cw - 12, ch - 12
        else:
            max_w, max_h = target_max

        img = self._original_pil.copy()
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self._current_img_tk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._current_img_tk, anchor="center")

    def _zoom(self, factor: float):
        """Simple zoom relative to current view (re-computes from original)."""
        if not hasattr(self, "_original_pil"):
            return
        self.update_idletasks()
        self.canvas.update_idletasks()
        cw = max(120, self.canvas.winfo_width())
        ch = max(120, self.canvas.winfo_height())
        # Approximate new target size
        new_w = int(cw * factor)
        new_h = int(ch * factor)
        self._display_image_on_canvas((new_w, new_h))

    def _on_canvas_resize(self, event):
        # Redraw scaled image when user resizes the window
        self.after(10, lambda: self._display_image_on_canvas())

    def _update_info(self):
        """Update the dynamic tile info (frame, row, col, global) for the current req.
        This must be called when a new tile/query is loaded into the persistent window.
        """
        req = getattr(self, 'current_req', None) or getattr(self, 'request', None)
        if not req:
            self.info_var.set("No current tile")
            return
        meta = getattr(req, 'meta', {}) or {}
        tile = getattr(req, 'tile', None)
        gidx = getattr(tile, 'global_idx', '?') if tile else '?'
        text = (f"Frame {meta.get('frame', '?')} | "
                f"Tile r{meta.get('row', '?')} c{meta.get('col', '?')} | "
                f"Global #{gidx}")
        self.info_var.set(text)

    # ---------------- Class list management ----------------
    def _refresh_class_list(self, filter_text: str = ""):
        self.class_list.delete(0, "end")
        self._filtered_classes = []
        f = filter_text.lower().strip()

        for cls in self.known_classes:
            if f and f not in cls.lower():
                continue
            count = self.class_counts.get(cls, 0)
            # Show count only when > 0. (0) was confusing; it meant "seen this many times so far in the store".
            # New or first-seen classes legitimately start at 0 until labeled.
            display = f"{cls}  ({count})" if count > 0 else cls
            self.class_list.insert("end", display)
            self._filtered_classes.append(cls)

    def _on_filter_changed(self, *_):
        self._refresh_class_list(self.filter_var.get())

    def _get_selected_class(self) -> Optional[str]:
        sel = self.class_list.curselection()
        if not sel:
            return None
        idx = sel[0]
        if 0 <= idx < len(self._filtered_classes):
            return self._filtered_classes[idx]
        return None

    # ---------------- Assignment actions (persistent window version) ----------------
    def _assign(self, label: str, relevant: bool):
        """Submit the chosen label for the current pending ARED query.
        Does NOT destroy the window (persistent for user repositioning).
        """
        print(f"[GUI Dialog] User assigned label '{label}' (relevant={relevant}) for current tile.")
        if getattr(self, 'current_req', None):
            self.current_req.set_result(label, relevant)
            self.current_req = None
        if self.on_assign:
            # on_assign here is the notification callback from main (for refresh etc.)
            try:
                self.on_assign(label, relevant)
            except Exception:
                pass
        self._prepare_for_next(label, relevant)

    def _prepare_for_next(self, last_label: str = "", last_relevant: bool = False):
        """Keep the window open, update status, clear for next query.
        IMPORTANT: This window only receives new tiles when A/RED decides to query.
        Non-query tiles are handled automatically by A/RED internally (no label needed).
        """
        self.new_var.set("")  # clear for next
        self._refresh_class_list(self.filter_var.get())
        msg = f"Submitted: {last_label} (relevant={last_relevant}). Waiting for next A/RED query (window stays positioned for you)."
        self.dialog_status_var.set(msg)
        self.info_var.set(f"Last labeled: {last_label} | Waiting for next A/RED query...")
        # Keep the last image for reference; it will be replaced when next A/RED query arrives.
        self.lift()

    def _assign_selected(self):
        label = self._get_selected_class()
        if label is None:
            # No selection: don't auto-create here. Direct user to the Create section
            # or click the explicit "Create & Assign" button. This prevents accidental
            # advances when double-clicking whitespace.
            self.new_entry.focus_set()
            return
        rel = self.relevant_var.get()
        self._assign(label, rel)

    def _create_and_assign(self):
        name = self.new_var.get().strip()
        if not name:
            messagebox.showwarning("New Class", "Please enter a class name.")
            return
        rel = self.relevant_var.get()

        # Give immediate visual feedback in *this* dialog:
        # add the new class to the list right away so user sees it was accepted.
        if name not in self.known_classes:
            self.known_classes.append(name)
            self.class_counts[name] = self.class_counts.get(name, 0)
            self._refresh_class_list(self.filter_var.get())
            try:
                idx = self._filtered_classes.index(name)
                self.class_list.selection_clear(0, "end")
                self.class_list.selection_set(idx)
                self.class_list.see(idx)
            except ValueError:
                pass

        # Clear entry 
        self.new_var.set("")

        self._assign(name, rel)

    def _assign_as_background(self):
        self._assign("__BACKGROUND__", False)

    def _close_window(self):
        # User explicitly wants to close (will be recreated on next query if needed)
        # Satisfy the pending req so the worker does not hang forever waiting for a label.
        if getattr(self, 'current_req', None):
            print("[GUI Dialog] Window closed without assign - satisfying worker with __BACKGROUND__ to avoid freeze.")
            try:
                self.current_req.set_result("__BACKGROUND__", False)
            except Exception:
                pass
            self.current_req = None
        self.destroy()

    def _on_double_click(self, event):
        # Only act on actual selection. Double-clicking empty space or whitespace
        # in the list should NOT advance the tile or auto-create from the entry.
        # Use the "Assign Selected" button or Enter (when a class is selected),
        # or the dedicated Create & Assign for new classes.
        label = self._get_selected_class()
        if label is not None:
            rel = self.relevant_var.get()
            self._assign(label, rel)
        # else: ignore accidental double-clicks on empty list area

    def set_current_request(self, req, known_classes=None, class_counts=None):
        """Update this persistent window for a new A/RED query without destroying/recreating it.
        User can keep the window in a convenient screen position.
        """
        self.current_req = req
        if known_classes is not None:
            self.known_classes = sorted(set(known_classes))
        if class_counts is not None:
            self.class_counts = class_counts or {}
        self._load_and_show_image()
        filt = self.filter_var.get() if hasattr(self, 'filter_var') else ""
        self._refresh_class_list(filt)
        self._update_info()
        # Force immediate refresh of image for the new tile (after layout update)
        self.after(20, lambda: (self.update_idletasks(), self._display_image_on_canvas()))
        self.dialog_status_var.set("New A/RED query. Label this tile (select or create), then Assign.")
        self.lift()
        self.focus_force()


class MainWindow:
    """
    The main application window. Everything controllable from here.
    """

    def __init__(self, root: tk.Tk, initial_config: Optional[PipelineConfig] = None):
        self.root = root
        self.root.title("Drone A/RED - Tiling + DINO + A_REDIN (High-Volume Labeling)")

        # --- High DPI / large text support for high-resolution displays ---
        # This makes text, buttons, listboxes etc. readable on 4K/ retina / high-dpi screens.
        # You can tune the scaling factor. 1.5-2.0 is common for high-res.
        try:
            self.root.tk.call('tk', 'scaling', 1.8)
        except Exception:
            pass

        self.root.geometry("1200x780")
        self.root.minsize(950, 600)

        self.config = initial_config or PipelineConfig.default()
        self.controller = DroneAREDController(self.config)
        self.label_store: Optional[PersistentLabelStore] = None
        self._stats_job = None
        self._pending_label_request: Optional[LabelRequest] = None
        self.discovered_classes: set = set()  # labels we have assigned in this run (for immediate UI feedback)
        self._last_queried_global = -1

        self._build_ui()
        self._start_stat_poller()

        # Register for status from worker
        self.controller.on_stats = self._on_worker_stats

    def _build_ui(self):
        # --- Make UI elements larger and more readable on high-res displays ---
        style = ttk.Style()
        big_font = ("TkDefaultFont", 13)
        bigger_font = ("TkDefaultFont", 14)
        style.configure(".", font=big_font)
        style.configure("TButton", font=bigger_font, padding=6)
        style.configure("TLabel", font=big_font)
        style.configure("TCheckbutton", font=big_font)
        style.configure("TLabelframe.Label", font=bigger_font)

        # Top menu (expandable later)
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Video(s)...", command=self._load_videos)
        file_menu.add_command(label="Save Label Cache", command=self._save_label_cache)
        file_menu.add_command(label="Load Label Cache", command=self._load_label_cache)
        file_menu.add_separator()
        file_menu.add_command(label="Save ARED Model State", command=self._save_ared_state)
        file_menu.add_command(label="Load ARED Model State", command=self._load_ared_state)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Control bar
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=6, pady=4)

        ttk.Button(ctrl, text="Load Videos", command=self._load_videos).pack(side="left", padx=2)
        self.start_btn = ttk.Button(ctrl, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=2)
        ttk.Button(ctrl, text="Pause", command=self.controller.pause).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Resume", command=self.controller.resume).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Stop", command=self.controller.stop).pack(side="left", padx=2)

        # Quick model controls
        ttk.Button(ctrl, text="Save ARED Model", command=self._save_ared_state).pack(side="left", padx=8)
        ttk.Button(ctrl, text="Load ARED Model", command=self._load_ared_state).pack(side="left")

        # Status line
        self.status_var = tk.StringVar(value="Ready. Load videos and press Start.")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken").pack(fill="x", padx=6, pady=2)

        # Main content: left params, center stats + classes, right preview stub
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        # Parameters (many are live for next run)
        param_frame = ttk.LabelFrame(body, text="Parameters (applied on next Start)")
        param_frame.pack(side="left", fill="y", padx=(0, 6))

        self._add_param_row(param_frame, "Tile W (px, uniform)", "tile_w", self.config.tiling.tile_width)
        self._add_param_row(param_frame, "Tile H (px, uniform)", "tile_h", self.config.tiling.tile_height)
        self._add_param_row(param_frame, "Frame stride (every Nth)", "frame_stride", self.config.tiling.frame_stride)
        self._add_param_row(param_frame, "Kappa (higher = MORE queries)", "kappa", self.config.ared.kappa, is_float=True)
        self._add_param_row(param_frame, "Buffer size", "buf_size", self.config.ared.l_buf_size)
        self._add_param_row(param_frame, "Cache threshold (L2)", "cache_thresh", self.config.label_cache.auto_label_threshold, is_float=True)
        self._add_param_row(param_frame, "DINO model name", "dino_model", self.config.features.model_name, is_str=True)

        ttk.Checkbutton(param_frame, text="Use label cache", variable=tk.BooleanVar(value=self.config.label_cache.enabled)).pack(anchor="w", pady=2)

        # Right side: stats + discovered classes
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        stats_frame = ttk.LabelFrame(right, text="Live Stats")
        stats_frame.pack(fill="x")

        self.stats_text = tk.Text(stats_frame, height=6, width=60, state="disabled")
        self.stats_text.pack(fill="x", padx=4, pady=4)

        classes_frame = ttk.LabelFrame(right, text="Discovered Classes (from current + cache)")
        classes_frame.pack(fill="both", expand=True, pady=(6, 0))

        self.class_listbox = tk.Listbox(classes_frame, height=10, font=("TkDefaultFont", 13))
        self.class_listbox.pack(fill="both", expand=True, side="left")
        ysb = ttk.Scrollbar(classes_frame, orient="vertical", command=self.class_listbox.yview)
        ysb.pack(side="right", fill="y")
        self.class_listbox.configure(yscrollcommand=ysb.set)

        # Preview area (simple for now)
        preview_frame = ttk.LabelFrame(body, text="Preview (last processed frame - stub)")
        preview_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.preview_label = ttk.Label(preview_frame, text="(Preview will show last frame + tile highlights in future)")
        self.preview_label.pack(expand=True)

    def _add_param_row(self, parent, label, attr, default, is_float=False, is_str=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=26).pack(side="left")
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(row, textvariable=var, width=20, font=("TkDefaultFont", 12))
        entry.pack(side="left")
        # Store for later read-back
        setattr(self, f"_{attr}_var", var)

    def _read_params_into_config(self):
        try:
            self.config.tiling.tile_width = int(getattr(self, "_tile_w_var").get())
            self.config.tiling.tile_height = int(getattr(self, "_tile_h_var").get())
            self.config.tiling.frame_stride = int(getattr(self, "_frame_stride_var").get())
            self.config.ared.kappa = float(getattr(self, "_kappa_var").get())
            self.config.ared.l_buf_size = int(getattr(self, "_buf_size_var").get())
            self.config.label_cache.auto_label_threshold = float(getattr(self, "_cache_thresh_var").get())
            self.config.features.model_name = getattr(self, "_dino_model_var").get()
        except Exception as e:
            messagebox.showerror("Params", f"Bad parameter value: {e}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _load_videos(self):
        files = filedialog.askopenfilenames(
            title="Select drone videos",
            filetypes=[("MP4 videos", "*.MP4 *.mp4"), ("All files", "*.*")]
        )
        if files:
            self.config.video_paths = list(files)
            self.status_var.set(f"Loaded {len(files)} video(s). Ready to Start.")

    def _start(self):
        self._read_params_into_config()

        # Prepare label store
        if self.config.label_cache.enabled:
            self.label_store = PersistentLabelStore(
                db_path=self.config.label_cache.db_path,
                auto_label_threshold=self.config.label_cache.auto_label_threshold,
            )
            self.controller.set_label_store(self.label_store)
        else:
            self.label_store = None

        self.controller.update_config(self.config)
        self.controller.start()

        self.start_btn.config(state="disabled")
        self.status_var.set("Processing... (use Pause / Stop)")

    def _save_label_cache(self):
        if self.label_store:
            self.label_store.save()
            messagebox.showinfo("Label Cache", "Label cache saved.")
        else:
            messagebox.showwarning("Label Cache", "No active label store.")

    def _load_label_cache(self):
        path = filedialog.askopenfilename(title="Load label cache", filetypes=[("Pickle", "*.pkl")])
        if path:
            self.label_store = PersistentLabelStore(db_path=path)
            self.controller.set_label_store(self.label_store)
            self._refresh_class_list()
            messagebox.showinfo("Label Cache", f"Loaded cache with {len(self.label_store)} entries.")

    def _save_ared_state(self):
        if not self.controller.ared_adapter:
            messagebox.showwarning("ARED State", "No active ARED session.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pkl", title="Save ARED model state")
        if path:
            self.controller.ared_adapter.save_state(path)
            messagebox.showinfo("ARED State", "Model state saved.")

    def _load_ared_state(self):
        path = filedialog.askopenfilename(title="Load ARED model state", filetypes=[("Pickle", "*.pkl")])
        if path and self.controller.ared_adapter:
            self.controller.ared_adapter.load_state(path, label_lookup=self._label_lookup_from_store)
            self._refresh_class_list()
            messagebox.showinfo("ARED State", "Model state loaded and replayed.")

    def _label_lookup_from_store(self, emb):
        if self.label_store:
            return self.label_store.lookup(emb) or ("__UNKNOWN__", False)
        return "__UNKNOWN__", False

    # ------------------------------------------------------------------
    # Label request handling (called from worker via queue)
    # ------------------------------------------------------------------
    def _poll_label_requests(self):
        """Called periodically from the Tk main loop.
        Uses a single persistent labeling window so the user can position it once.
        """
        try:
            req: LabelRequest = self.controller.label_request_queue.get_nowait()
        except queue.Empty:
            return

        print(f"[GUI] Received label REQUEST from ARED for tile global={getattr(getattr(req,'tile',None),'global_idx','?')} meta={getattr(req,'meta',{})}")
        self._pending_label_request = req

        # Guard against duplicate req for the exact same tile (shouldn't happen, but prevents "same tile twice")
        tile = getattr(req, 'tile', None)
        gidx = getattr(tile, 'global_idx', -1) if tile else -1
        if gidx != -1 and getattr(self, '_last_queried_global', -1) == gidx:
            # Already handled this tile's query; ignore stale/duplicate
            # CRITICAL: still satisfy the req or the worker thread will block forever on wait()
            print("[GUI]   -> Duplicate request for same global_idx, ignoring (but satisfying req to unblock worker).")
            try:
                req.set_result("__DUPLICATE__", False)
            except Exception:
                pass
            return
        self._last_queried_global = gidx

        # Gather current known classes from both store and ARED
        classes = []
        counts = {}
        if self.label_store:
            classes.extend(self.label_store.get_all_labels())
            counts.update(self.label_store.get_class_counts())
        if self.controller.ared_adapter:
            classes.extend(self.controller.ared_adapter.get_known_labels())
        # Include GUI-assigned ones immediately (so new classes appear in the very next dialog)
        for lbl in getattr(self, 'discovered_classes', []):
            if lbl not in classes:
                classes.append(lbl)
            if lbl not in counts:
                counts[lbl] = counts.get(lbl, 0)

        classes = sorted(set(classes))

        def _assign_cb(label: str, relevant: bool):
            # This is now purely notification / UI update.
            # The dialog itself calls set_result on the req it holds.
            print(f"[GUI] Label SUBMITTED from dialog: '{label}' (relevant={relevant})")
            self._pending_label_request = None
            self.discovered_classes.add(label)
            self._refresh_class_list()
            self.status_var.set(f"Last label assigned: {label} (relevant={relevant})")
            print("[GUI] Dialog back to WAITING state for next A/RED query (worker continues processing non-queried tiles in background).")
            # After submit, the window is prepared with waiting status; it will only update again on next A/RED query req

        # Persistent window: create once, then update for new requests
        if not hasattr(self, '_labeling_win') or not self._labeling_win.winfo_exists():
            print("[GUI] Creating new persistent LabelingDialog for this query.")
            self._labeling_win = LabelingDialog(
                self.root,
                req,
                known_classes=classes,
                on_assign=_assign_cb,
                class_counts=counts,
            )
        else:
            print("[GUI] Updating existing persistent LabelingDialog with new query tile.")
            self._labeling_win.set_current_request(req, known_classes=classes, class_counts=counts)

    def _refresh_class_list(self):
        self.class_listbox.delete(0, "end")
        counts = {}
        if self.label_store:
            counts.update(self.label_store.get_class_counts())
        all_labels = set(counts.keys())
        if self.controller.ared_adapter:
            all_labels.update(self.controller.ared_adapter.get_known_labels())
        # Include ones we just assigned in the GUI (for immediate visibility on next queries)
        all_labels.update(getattr(self, 'discovered_classes', set()))

        for lbl in sorted(all_labels):
            c = counts.get(lbl, 0)
            display = f"{lbl} ({c})" if c > 0 else lbl
            self.class_listbox.insert("end", display)

    # ------------------------------------------------------------------
    # Stats & polling
    # ------------------------------------------------------------------
    def _start_stat_poller(self):
        def poll():
            try:
                self._poll_label_requests()
                if self.controller.stats:
                    self._update_stats_display(self.controller.stats)
                # Periodically refresh the discovered classes list so counts update after labeling
                if self.controller.stats and int(self.controller.stats.get("tiles_processed", 0)) % 5 == 0:
                    self._refresh_class_list()
            except Exception as e:
                print(f"[GUI] ERROR in stat/label poll (this would previously freeze polling!): {e}")
                import traceback
                traceback.print_exc()
            # ALWAYS reschedule even on error, otherwise worker can block forever on next query
            self._stats_job = self.root.after(80, poll)
        self._stats_job = self.root.after(120, poll)

    def _on_worker_stats(self, stats: Dict):
        # This may be called from worker thread; marshal to main thread
        self.root.after(0, lambda: self._update_stats_display(stats))

    def _update_stats_display(self, stats: Dict[str, Any]):
        text = (
            f"Status: {stats.get('status', '?')}   Video: {stats.get('current_video', '')}\n"
            f"Frames: {stats.get('frames_read', 0)}   Tiles: {stats.get('tiles_processed', 0)}\n"
            f"User labels needed: {stats.get('user_queries', 0)}   "
            f"Cache auto-labels: {stats.get('cache_hits', 0)}\n"
            f"ARED clusters: {stats.get('ared_clusters', '?')}   Known labels: {stats.get('ared_known_labels', '?')}"
        )
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

        if stats.get("tiles_processed", 0) % 15 == 0:
            self._refresh_class_list()
