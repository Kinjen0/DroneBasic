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
        self.title("Review Queried Tile - A/RED Drone")
        self.request = request
        self.on_assign = on_assign
        self.class_counts = class_counts or {}
        self.known_classes = sorted(set(known_classes))

        # Make the whole dialog resizable and set a reasonable default
        self.geometry("920x720")
        self.minsize(600, 450)
        self.resizable(True, True)

        self._current_img_tk: Optional[ImageTk.PhotoImage] = None
        self._filtered_classes: List[str] = list(self.known_classes)

        self._build_ui()
        self._load_and_show_image()
        self._refresh_class_list()

        # Keyboard bindings for power users labeling many tiles
        self.bind("<Return>", lambda e: self._assign_selected())
        self.bind("<Escape>", lambda e: self._assign_as_background())
        self.bind("<Control-n>", lambda e: self.new_entry.focus_set())

        # Focus the list so double-click / arrows work immediately
        self.after(150, lambda: self.class_list.focus_set())

    def _build_ui(self):
        # Top info bar
        info = ttk.Frame(self)
        info.pack(fill="x", padx=8, pady=4)

        meta = self.request.meta
        info_text = (f"Frame {meta.get('frame', '?')} | "
                     f"Tile r{meta.get('row', '?')} c{meta.get('col', '?')} | "
                     f"Global #{self.request.tile.global_idx if hasattr(self.request.tile, 'global_idx') else '?'}")
        ttk.Label(info, text=info_text, font=("TkDefaultFont", 10)).pack(side="left")

        ttk.Label(info, text="  (Double-click class or press Enter to assign)", foreground="gray").pack(side="right")

        # Main area: image (left) + classes (right)
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=4)

        # --- IMAGE (resizable canvas) ---
        img_frame = ttk.LabelFrame(main, text="Tile Image (resize window to fit)")
        img_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(img_frame, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

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

        # Class list
        list_frame = ttk.Frame(right)
        list_frame.pack(fill="both", expand=True)

        self.class_list = tk.Listbox(list_frame, height=18, exportselection=False)
        self.class_list.pack(side="left", fill="both", expand=True)
        self.class_list.bind("<Double-Button-1>", self._on_double_click)
        self.class_list.bind("<Return>", lambda e: self._assign_selected())

        yscroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_list.yview)
        yscroll.pack(side="right", fill="y")
        self.class_list.configure(yscrollcommand=yscroll.set)

        # Assign button for selected
        ttk.Button(right, text="Assign Selected (Enter)", command=self._assign_selected).pack(fill="x", pady=4)

        # --- NEW CLASS ---
        new_frame = ttk.LabelFrame(right, text="New Class")
        new_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(new_frame, text="Class name:").pack(anchor="w")
        self.new_entry = ttk.Entry(new_frame)
        self.new_entry.pack(fill="x", pady=2)
        self.new_entry.bind("<Return>", lambda e: self._create_and_assign())

        self.relevant_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(new_frame, text="Relevant (interesting / anomaly)", variable=self.relevant_var).pack(anchor="w")

        ttk.Button(new_frame, text="Create & Assign", command=self._create_and_assign).pack(fill="x", pady=4)

        # Bottom quick actions
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=6)

        ttk.Button(bottom, text="Mark as Background / Irrelevant", command=self._assign_as_background).pack(side="left")
        ttk.Button(bottom, text="Cancel (treat as background)", command=self._cancel).pack(side="right")

        # Make the new entry easy to reach
        self.after(200, lambda: self.new_entry.focus_set() if not self.known_classes else None)

    def _load_and_show_image(self):
        try:
            img = self.request.tile.image if hasattr(self.request.tile, "image") else self.request.tile
            if isinstance(img, Image.Image):
                self._original_pil = img.copy()
            else:
                self._original_pil = Image.new("RGB", (224, 224), "gray")
        except Exception:
            self._original_pil = Image.new("RGB", (224, 224), "gray")

        self._display_image_on_canvas()

    def _display_image_on_canvas(self):
        if not hasattr(self, "_original_pil"):
            return
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            cw, ch = 640, 480

        # Fit while preserving aspect
        img = self._original_pil.copy()
        img.thumbnail((cw - 10, ch - 10), Image.Resampling.LANCZOS)
        self._current_img_tk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._current_img_tk, anchor="center")

    def _on_canvas_resize(self, event):
        # Redraw scaled image when user resizes the window
        self._display_image_on_canvas()

    # ---------------- Class list management ----------------
    def _refresh_class_list(self, filter_text: str = ""):
        self.class_list.delete(0, "end")
        self._filtered_classes = []
        f = filter_text.lower().strip()

        for cls in self.known_classes:
            if f and f not in cls.lower():
                continue
            count = self.class_counts.get(cls, 0)
            display = f"{cls}  ({count})"
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

    # ---------------- Assignment actions ----------------
    def _assign(self, label: str, relevant: bool):
        self.on_assign(label, relevant)
        self.destroy()

    def _assign_selected(self):
        label = self._get_selected_class()
        if label is None:
            # If nothing selected, treat as new-class creation using the entry
            self._create_and_assign()
            return
        rel = self.relevant_var.get()
        self._assign(label, rel)

    def _create_and_assign(self):
        name = self.new_entry.get().strip()
        if not name:
            messagebox.showwarning("New Class", "Please enter a class name.")
            return
        rel = self.relevant_var.get()
        self._assign(name, rel)

    def _assign_as_background(self):
        self._assign("__BACKGROUND__", False)

    def _cancel(self):
        # Treat cancel as background so ARED can continue
        self._assign("__BACKGROUND__", False)

    def _on_double_click(self, event):
        self._assign_selected()


class MainWindow:
    """
    The main application window. Everything controllable from here.
    """

    def __init__(self, root: tk.Tk, initial_config: Optional[PipelineConfig] = None):
        self.root = root
        self.root.title("Drone A/RED - Tiling + DINO + A_REDIN")
        self.root.geometry("1150x720")
        self.root.minsize(900, 550)

        self.config = initial_config or PipelineConfig.default()
        self.controller = DroneAREDController(self.config)
        self.label_store: Optional[PersistentLabelStore] = None
        self._stats_job = None
        self._pending_label_request: Optional[LabelRequest] = None

        self._build_ui()
        self._start_stat_poller()

        # Register for status from worker
        self.controller.on_stats = self._on_worker_stats

    def _build_ui(self):
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

        self._add_param_row(param_frame, "Tile W", "tile_w", self.config.tiling.tile_width)
        self._add_param_row(param_frame, "Tile H", "tile_h", self.config.tiling.tile_height)
        self._add_param_row(param_frame, "Frame stride", "frame_stride", self.config.tiling.frame_stride)
        self._add_param_row(param_frame, "Kappa (lower = more queries)", "kappa", self.config.ared.kappa, is_float=True)
        self._add_param_row(param_frame, "Buffer size", "buf_size", self.config.ared.l_buf_size)
        self._add_param_row(param_frame, "Cache threshold", "cache_thresh", self.config.label_cache.auto_label_threshold, is_float=True)
        self._add_param_row(param_frame, "DINO model", "dino_model", self.config.features.model_name, is_str=True)

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

        self.class_listbox = tk.Listbox(classes_frame, height=10)
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
        ttk.Label(row, text=label, width=24).pack(side="left")
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(row, textvariable=var, width=18)
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
        """Called periodically from the Tk main loop."""
        try:
            req: LabelRequest = self.controller.label_request_queue.get_nowait()
        except queue.Empty:
            return

        self._pending_label_request = req

        # Gather current known classes from both store and ARED
        classes = []
        counts = {}
        if self.label_store:
            classes.extend(self.label_store.get_all_labels())
            counts.update(self.label_store.get_class_counts())
        if self.controller.ared_adapter:
            classes.extend(self.controller.ared_adapter.get_known_labels())

        classes = sorted(set(classes))

        def _assign_cb(label: str, relevant: bool):
            req.set_result(label, relevant)
            self._pending_label_request = None
            # Refresh class list after new label
            self._refresh_class_list()

        # Show the dialog (modal-ish but non-blocking for the rest of GUI)
        LabelingDialog(
            self.root,
            req,
            known_classes=classes,
            on_assign=_assign_cb,
            class_counts=counts,
        )

    def _refresh_class_list(self):
        self.class_listbox.delete(0, "end")
        all_labels = set()
        if self.label_store:
            all_labels.update(self.label_store.get_all_labels())
        if self.controller.ared_adapter:
            all_labels.update(self.controller.ared_adapter.get_known_labels())
        for lbl in sorted(all_labels):
            self.class_listbox.insert("end", lbl)

    # ------------------------------------------------------------------
    # Stats & polling
    # ------------------------------------------------------------------
    def _start_stat_poller(self):
        def poll():
            self._poll_label_requests()
            if self.controller.stats:
                self._update_stats_display(self.controller.stats)
            self._stats_job = self.root.after(80, poll)
        self._stats_job = self.root.after(120, poll)

    def _on_worker_stats(self, stats: Dict):
        # This may be called from worker thread; marshal to main thread
        self.root.after(0, lambda: self._update_stats_display(stats))

    def _update_stats_display(self, stats: Dict[str, Any]):
        text = (
            f"Status: {stats.get('status', '?')}   Video: {stats.get('current_video', '')}\n"
            f"Frames read: {stats.get('frames_read', 0)}   "
            f"Tiles processed: {stats.get('tiles_processed', 0)}\n"
            f"Queries to user: {stats.get('queries', 0)}   "
            f"Cache hits: {stats.get('cache_hits', 0)}\n"
            f"ARED clusters: {getattr(self.controller.ared_adapter, 'ared', None) and len(getattr(self.controller.ared_adapter.ared, 'subspace_partition', type('x',(),{'cluster_dict':{}})()).cluster_dict) or '?'}"
        )
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

        if stats.get("tiles_processed", 0) % 15 == 0:
            self._refresh_class_list()
