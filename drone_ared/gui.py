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
from .tile_database import TileAnnotationDB, extract_tile_from_video  # exact identity + re-extract helper


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
                 on_assign: callable, class_counts: Optional[Dict[str, int]] = None,
                 ui_scale: float = 1.6, class_relevance: Optional[Dict[str, bool]] = None):
        super().__init__(master)
        self.title("Review Queried Tile - A/RED Drone  [Persistent - reposition me once!]")
        self.current_req = request
        self.on_assign = on_assign  # notification callback (refresh etc.), we do set_result ourselves
        self.class_counts = class_counts or {}
        self.known_classes = sorted(set(known_classes))
        self.ui_scale = float(ui_scale) if ui_scale else 1.6
        self._zoom_level = 1.0
        self.class_relevance: dict[str, bool] = dict(class_relevance or {})  # class -> relevant (set at creation)

        # Larger default + fully resizable for high-res displays and comfort during long labeling sessions
        # Scale the dialog size with ui_scale for better defaults on large screens
        base_w, base_h = 1050, 820
        scaled_w = int(base_w * min(self.ui_scale, 2.5))
        scaled_h = int(base_h * min(self.ui_scale, 2.5))
        self.geometry(f"{scaled_w}x{scaled_h}")
        self.minsize(int(650 * min(self.ui_scale, 2.0)), int(520 * min(self.ui_scale, 2.0)))
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
        # Compute scaled sizes once (ui_scale passed from MainWindow config)
        s = self.ui_scale
        fs = int(12 * s)
        fs_big = int(13 * s)
        fs_bigger = int(14 * s)
        pady_s = int(3 * s)
        padx_s = int(4 * s)

        # Configure ttk styles here so that themed widgets (especially Checkbutton, Labels)
        # get the proper scaled fonts. Passing font= directly to ttk.Checkbutton (and some
        # other ttk widgets) raises "unknown option -font".
        style = ttk.Style()
        style.configure("TLabel", font=("TkDefaultFont", fs))
        style.configure("TCheckbutton", font=("TkDefaultFont", fs))
        style.configure("TButton", font=("TkDefaultFont", fs_big))

        # Top info bar - use variable so it can be updated for each new tile in persistent window
        info = ttk.Frame(self)
        info.pack(fill="x", padx=int(8 * s), pady=int(4 * s))

        self.info_var = tk.StringVar(value="Loading tile info...")
        ttk.Label(info, textvariable=self.info_var).pack(side="left")

        ttk.Label(info, text="  (Double-click class or press Enter to assign. Resize me!)", foreground="gray").pack(side="right")

        # Main area: image (left) + classes (right)
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=int(8*s), pady=int(4*s))

        # --- IMAGE (resizable canvas with zoom + scroll support) ---
        img_frame = ttk.LabelFrame(main, text="Tile Image (resize window to fit; use zoom buttons + scrollbars when enlarged)")
        img_frame.pack(side="left", fill="both", expand=True)

        # Use a container + grid for canvas + scrollbars so "Larger View" can overflow and be pannable
        canvas_container = ttk.Frame(img_frame)
        canvas_container.pack(fill="both", expand=True)
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.rowconfigure(1, weight=0)
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.columnconfigure(1, weight=0)

        self.canvas = tk.Canvas(canvas_container, bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Mouse wheel scrolling for the zoomed tile image (cross platform)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)       # Windows / mac
        self.canvas.bind("<Button-4>", self._on_mousewheel)         # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)         # Linux scroll down

        # Fit/zoom controls - now functional with scrollregion for >fit renders
        fit_bar = ttk.Frame(img_frame)
        fit_bar.pack(fill="x")
        ttk.Button(fit_bar, text="Fit to Window", command=lambda: (
            self.update_idletasks(),
            setattr(self, '_zoom_level', 1.0),
            self._display_image_on_canvas(),
            self.after(5, getattr(self, '_center_view', lambda: None))
        )).pack(side="left", padx=int(3*s))
        ttk.Button(fit_bar, text="Larger View", command=lambda: (self.update_idletasks(), self._zoom(1.25))).pack(side="left", padx=int(2*s))
        ttk.Button(fit_bar, text="Smaller View", command=lambda: (self.update_idletasks(), self._zoom(0.8))).pack(side="left", padx=int(2*s))

        # --- CLASSES SIDE ---
        right = ttk.Frame(main)
        right.pack(side="right", fill="y", padx=(int(8*s), 0))

        ttk.Label(right, text="Existing Classes (double-click or select + Assign)").pack(anchor="w")

        # Filter
        filter_frame = ttk.Frame(right)
        filter_frame.pack(fill="x", pady=pady_s)
        ttk.Label(filter_frame, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", self._on_filter_changed)
        ttk.Entry(filter_frame, textvariable=self.filter_var, width=int(22 * min(s, 1.8))).pack(side="left", fill="x", expand=True)

        # Class list - scaled font for readability (tk.Listbox accepts font directly)
        list_frame = ttk.Frame(right)
        list_frame.pack(fill="both", expand=True)

        self.class_list = tk.Listbox(list_frame, height=18, exportselection=False, font=("TkDefaultFont", fs_big))
        self.class_list.pack(side="left", fill="both", expand=True)
        self.class_list.bind("<Double-Button-1>", self._on_double_click)
        self.class_list.bind("<Return>", lambda e: self._assign_selected())

        yscroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_list.yview)
        yscroll.pack(side="right", fill="y")
        self.class_list.configure(yscrollcommand=yscroll.set)

        # Assign button for selected
        ttk.Button(right, text="Assign Selected (Enter / Double-click)", command=self._assign_selected).pack(fill="x", pady=pady_s)

        # --- NEW CLASS ---
        new_frame = ttk.LabelFrame(right, text="New Class")
        new_frame.pack(fill="x", pady=(int(10*s), 0))

        ttk.Label(new_frame, text="Class name:").pack(anchor="w")
        self.new_entry = ttk.Entry(new_frame, textvariable=self.new_var, font=("TkDefaultFont", fs_big))
        self.new_entry.pack(fill="x", pady=pady_s)
        self.new_entry.bind("<Return>", lambda e: self._create_and_assign())

        self.relevant_var = tk.BooleanVar(value=False)
        # NOTE: Do NOT pass font= to ttk.Checkbutton. It does not support the option.
        # We configured the "TCheckbutton" style above instead.
        ttk.Checkbutton(new_frame, text="Relevant (interesting / anomaly)", variable=self.relevant_var).pack(anchor="w")

        ttk.Button(new_frame, text="Create & Assign", command=self._create_and_assign).pack(fill="x", pady=pady_s)

        # Bottom quick actions
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=int(8*s), pady=pady_s)

        ttk.Button(bottom, text="Mark as Background / Irrelevant", command=self._assign_as_background).pack(side="left")
        ttk.Button(bottom, text="Close Window (recreates on next query)", command=self._close_window).pack(side="right")

        # Status for the dialog
        self.dialog_status_var = tk.StringVar(value="Choose from list (double-click or button) or type new class + Create & Assign")
        ttk.Label(self, textvariable=self.dialog_status_var, relief="sunken").pack(fill="x", padx=int(8*s), pady=pady_s)

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
        self.after(80, lambda: (self.update_idletasks(), self._display_image_on_canvas(), self._update_info(), self.after(10, getattr(self, '_center_view', lambda: None))))

    def _display_image_on_canvas(self, target_max=None):
        """Fit the tile image nicely inside the canvas while preserving aspect ratio.
        Supports zoom_level >1.0 (larger render + scrollbars for panning details).
        Called on resize, fit, and zoom buttons.
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

        if target_max is not None:
            max_w, max_h = target_max
        else:
            z = getattr(self, '_zoom_level', 1.0)
            # Base fit size from canvas, multiplied by zoom level (>1 = "zoomed in" / higher res render)
            max_w = int((cw - 12) * z)
            max_h = int((ch - 12) * z)

        # Use resize (not thumbnail) so we can *upscale* for zoom-in (magnification) beyond native tile resolution.
        # thumbnail never enlarges beyond original pixels; resize + LANCZOS does (interpolated).
        orig_w, orig_h = self._original_pil.size
        if orig_w > 0 and orig_h > 0 and max_w > 0 and max_h > 0:
            ratio = min(max_w / orig_w, max_h / orig_h)
            new_w = max(1, int(orig_w * ratio))
            new_h = max(1, int(orig_h * ratio))
            img = self._original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            img = self._original_pil.copy()
            img.thumbnail((max_w or 100, max_h or 100), Image.Resampling.LANCZOS)

        self._current_img_tk = ImageTk.PhotoImage(img)

        disp_w, disp_h = img.size
        self.canvas.delete("all")

        # When the rendered image is larger than the view, draw at origin so scrollbars can pan it.
        # When smaller, center it within the canvas area.
        if disp_w <= cw and disp_h <= ch:
            x = (cw - disp_w) // 2
            y = (ch - disp_h) // 2
            self.canvas.create_image(x, y, image=self._current_img_tk, anchor="nw")
            self.canvas.config(scrollregion=(0, 0, cw, ch))
        else:
            self.canvas.create_image(0, 0, image=self._current_img_tk, anchor="nw")
            self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        # Ensure layout and scrollbars update after image/region change
        self.canvas.update_idletasks()

    def _zoom(self, factor: float):
        """Adjust zoom level (multiplier on fit-to-canvas size) and redraw.
        >1.0 renders larger version of the tile (more detail) using scrollbars to pan.
        """
        if not hasattr(self, "_original_pil"):
            return
        current = getattr(self, '_zoom_level', 1.0)
        self._zoom_level = max(0.2, min(8.0, current * factor))  # clamp reasonable range
        self.update_idletasks()
        self.canvas.update_idletasks()
        self._display_image_on_canvas()  # uses _zoom_level internally
        # Re-center the view after zoom so "larger" magnifies around the middle instead of jumping to corner
        self.after(5, self._center_view)

    def _center_view(self):
        """Center the current scroll view over the image content (for natural zoom in/out around center)."""
        try:
            sr = self.canvas.cget("scrollregion")
            if not sr:
                return
            coords = [int(float(x)) for x in sr.split()]
            if len(coords) != 4:
                return
            _, _, iw, ih = coords
            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())
            # For content larger than viewport, scroll so image center aligns with view center
            if iw > cw:
                frac_x = max(0.0, (iw / 2.0 - cw / 2.0) / iw)
                self.canvas.xview_moveto(frac_x)
            else:
                self.canvas.xview_moveto(0.0)
            if ih > ch:
                frac_y = max(0.0, (ih / 2.0 - ch / 2.0) / ih)
                self.canvas.yview_moveto(frac_y)
            else:
                self.canvas.yview_moveto(0.0)
        except Exception:
            # Non-fatal; view centering is best-effort
            pass

    def _on_mousewheel(self, event):
        """Scroll the canvas view with the mouse wheel when the tile is zoomed larger than the view."""
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", 0)
        # Shift + wheel or horizontal scroll -> x axis
        if getattr(event, "state", 0) & 0x0001 or num in (6, 7):  # shift mask or horiz buttons
            if num == 7 or delta < 0:
                self.canvas.xview_scroll(1, "units")
            else:
                self.canvas.xview_scroll(-1, "units")
        else:
            # Vertical
            if num == 5 or delta < 0:
                self.canvas.yview_scroll(1, "units")
            elif num == 4 or delta > 0:
                self.canvas.yview_scroll(-1, "units")

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
            if self.class_relevance.get(cls, False):
                display += " [relevant]"
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
        # For existing classes, use the relevance that was set when the class was created.
        # The checkbox only affects *new* class creation.
        rel = self.class_relevance.get(label, False)
        self._assign(label, rel)

    def _create_and_assign(self):
        name = self.new_var.get().strip()
        if not name:
            messagebox.showwarning("New Class", "Please enter a class name.")
            return
        # Checkbox determines relevance *only* for newly created classes.
        checkbox_rel = self.relevant_var.get()

        # Give immediate visual feedback in *this* dialog:
        # add the new class to the list right away so user sees it was accepted.
        if name not in self.known_classes:
            self.known_classes.append(name)
            self.class_counts[name] = self.class_counts.get(name, 0)
            self.class_relevance[name] = checkbox_rel  # record the relevance decided at creation
            rel = checkbox_rel
            self._refresh_class_list(self.filter_var.get())
            try:
                idx = self._filtered_classes.index(name)
                self.class_list.selection_clear(0, "end")
                self.class_list.selection_set(idx)
                self.class_list.see(idx)
            except ValueError:
                pass
        else:
            # Typing an existing name into "Create" — use the class's established relevance
            rel = self.class_relevance.get(name, checkbox_rel)

        # Clear entry 
        self.new_var.set("")

        self._assign(name, rel)

    def _assign_as_background(self):
        self.class_relevance["__BACKGROUND__"] = False
        self._assign("__BACKGROUND__", False)

    def _close_window(self):
        # User explicitly wants to close (will be recreated on next query if needed)
        # Satisfy the pending req so the worker does not hang forever waiting for a label.
        if getattr(self, 'current_req', None):
            print("[GUI Dialog] Window closed without assign - satisfying worker with __BACKGROUND__ to avoid freeze.")
            self.class_relevance["__BACKGROUND__"] = False
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
            # For existing classes, use the relevance that was set when the class was created.
            # The checkbox only affects *new* class creation.
            rel = self.class_relevance.get(label, False)
            self._assign(label, rel)
        # else: ignore accidental double-clicks on empty list area

    def set_current_request(self, req, known_classes=None, class_counts=None, class_relevance=None):
        """Update this persistent window for a new A/RED query without destroying/recreating it.
        User can keep the window in a convenient screen position.
        """
        self.current_req = req
        if known_classes is not None:
            self.known_classes = sorted(set(known_classes))
        if class_counts is not None:
            self.class_counts = class_counts or {}
        if class_relevance is not None:
            # merge without overwriting ones we just created in this dialog session
            for k, v in class_relevance.items():
                if k not in self.class_relevance:
                    self.class_relevance[k] = v
        self._load_and_show_image()
        filt = self.filter_var.get() if hasattr(self, 'filter_var') else ""
        self._refresh_class_list(filt)
        self._update_info()
        # Force immediate refresh of image for the new tile (after layout update)
        self.after(20, lambda: (self.update_idletasks(), self._display_image_on_canvas(), self.after(10, getattr(self, '_center_view', lambda: None))))
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

        self.config = initial_config or PipelineConfig.default()
        gui_cfg = self.config.gui
        self.ui_scale = float(getattr(gui_cfg, 'ui_scale', 1.6))

        # --- High DPI / large text support for high-resolution displays ---
        # Bound to ui_scale in config (default 1.6 for readability on modern/large screens).
        # Change in config or GUIConfig.ui_scale and restart for different resolution scaling.
        try:
            self.root.tk.call('tk', 'scaling', self.ui_scale)
        except Exception:
            pass

        # Scale initial window size (clamped)
        base_w, base_h = 1200, 780
        w = int(base_w * min(self.ui_scale, 2.2))
        h = int(base_h * min(self.ui_scale, 2.2))
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(int(950 * min(self.ui_scale, 1.8)), int(600 * min(self.ui_scale, 1.8)))

        self.controller = DroneAREDController(self.config)
        self.label_store: Optional[PersistentLabelStore] = None
        self.tile_db: Optional[TileAnnotationDB] = None   # NEW: exact (video, frame, tile) labels
        self.edit_mode: bool = False
        self._stats_job = None
        self._pending_label_request: Optional[LabelRequest] = None
        self.discovered_classes: set = set()  # labels we have assigned in this run (for immediate UI feedback)
        self.class_relevance: dict[str, bool] = {}  # class name -> is_relevant (set at creation time)
        self._last_queried_global = -1

        self._build_ui()
        self._start_stat_poller()

        # Register for status from worker
        self.controller.on_stats = self._on_worker_stats

    def _build_ui(self):
        # --- Make UI elements larger and more readable on high-res displays ---
        # Uses self.ui_scale (from config) for fonts, paddings, etc. Change ui_scale for your display.
        style = ttk.Style()
        s = self.ui_scale
        base = int(11 * s)
        big_font = ("TkDefaultFont", base)
        bigger_font = ("TkDefaultFont", int(base + 2))
        pad = max(4, int(6 * s))
        style.configure(".", font=big_font)
        style.configure("TButton", font=bigger_font, padding=pad)
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
        ctrl.pack(fill="x", padx=int(6 * self.ui_scale), pady=int(4 * self.ui_scale))

        s = self.ui_scale
        ttk.Button(ctrl, text="Load Videos", command=self._load_videos).pack(side="left", padx=int(2*s))
        self.start_btn = ttk.Button(ctrl, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=int(2*s))
        ttk.Button(ctrl, text="Pause", command=self.controller.pause).pack(side="left", padx=int(2*s))
        ttk.Button(ctrl, text="Resume", command=self.controller.resume).pack(side="left", padx=int(2*s))
        ttk.Button(ctrl, text="Stop", command=self._stop).pack(side="left", padx=int(2*s))

        # Quick model controls
        ttk.Button(ctrl, text="Save ARED Model", command=self._save_ared_state).pack(side="left", padx=int(8*s))
        ttk.Button(ctrl, text="Load ARED Model", command=self._load_ared_state).pack(side="left")

        # Status line
        self.status_var = tk.StringVar(value="Ready. Load videos and press Start.")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken").pack(fill="x", padx=int(6*self.ui_scale), pady=int(2*self.ui_scale))

        # Main content: left params, center stats + classes, right preview stub
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=int(6*self.ui_scale), pady=int(4*self.ui_scale))

        # Parameters (many are live for next run)
        param_frame = ttk.LabelFrame(body, text="Parameters (applied on next Start)")
        param_frame.pack(side="left", fill="y", padx=(0, int(6*self.ui_scale)))

        self._add_param_row(param_frame, "Tile W (px, uniform)", "tile_w", self.config.tiling.tile_width)
        self._add_param_row(param_frame, "Tile H (px, uniform)", "tile_h", self.config.tiling.tile_height)
        self._add_param_row(param_frame, "Frame stride (every Nth)", "frame_stride", self.config.tiling.frame_stride)
        self._add_param_row(param_frame, "Kappa (higher = MORE queries)", "kappa", self.config.ared.kappa, is_float=True)
        self._add_param_row(param_frame, "Buffer size", "buf_size", self.config.ared.l_buf_size)
        self._add_param_row(param_frame, "Cache threshold (L2)", "cache_thresh", self.config.label_cache.auto_label_threshold, is_float=True)
        self._add_param_row(param_frame, "DINO model name", "dino_model", self.config.features.model_name, is_str=True)

        s = self.ui_scale
        ttk.Checkbutton(param_frame, text="Use label cache (similarity)", variable=tk.BooleanVar(value=self.config.label_cache.enabled)).pack(anchor="w", pady=int(2*s))

        # NEW: Exact tile annotation DB controls
        ttk.Separator(param_frame, orient="horizontal").pack(fill="x", pady=int(4*s))
        ttk.Label(param_frame, text="Exact Tile Labels DB (identity by video+frame+pos)").pack(anchor="w")
        self._add_param_row(param_frame, "Annotation DB path", "tile_ann_db", self.config.tile_annotations.db_path, is_str=True)
        self.edit_mode_var = tk.BooleanVar(value=self.config.tile_annotations.edit_mode_default)
        cb = ttk.Checkbutton(param_frame, text="EDIT MODE: force GUI even on known exact tiles (for corrections)",
                             variable=self.edit_mode_var)
        cb.pack(anchor="w", pady=int(3*s))
        # Live update if controller already running
        def _sync_edit_mode(*_):
            self.edit_mode = self.edit_mode_var.get()
            if hasattr(self, "controller"):
                self.controller.set_edit_mode(self.edit_mode)
        self.edit_mode_var.trace_add("write", _sync_edit_mode)

        # Data Augmentation (DINO rotations)
        self.aug_var = tk.BooleanVar(value=getattr(self.config.ared, "data_augmentation_enabled", False))
        ttk.Checkbutton(param_frame, text="Data Augmentation (rotate labeled tiles 3× + DINO re-embed)",
                        variable=self.aug_var).pack(anchor="w", pady=int(3*s))

        # Label Only mode (for building reference datasets for metrics)
        self.label_only_var = tk.BooleanVar(value=getattr(self.config.tile_annotations, "label_only_default", False))
        ttk.Checkbutton(param_frame, text="Label Only Mode (no A/RED, no DINO — pure labeling for metrics)",
                        variable=self.label_only_var).pack(anchor="w", pady=int(3*s))
        ttk.Button(param_frame, text="Review / Edit Past Labels...", command=self._open_review_window).pack(fill="x", pady=int(3*s))
        ttk.Button(param_frame, text="Save Annotation DB Now", command=self._save_tile_annotations).pack(fill="x")
        ttk.Button(param_frame, text="Load different Annotation DB...", command=self._load_tile_annotations).pack(fill="x")

        # Right side: stats + discovered classes
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=int(4*s))

        stats_frame = ttk.LabelFrame(right, text="Live Stats")
        stats_frame.pack(fill="x")

        self.stats_text = tk.Text(stats_frame, height=6, width=int(60 * min(s, 1.5)), state="disabled", font=("TkDefaultFont", int(11*s)))
        self.stats_text.pack(fill="x", padx=4, pady=4)

        # --- Metrics box (Query Precision + Relevant Recall as defined in the A/RED papers) ---
        # See IJSC_2026-1.pdf and SPIE_IVSP_2026.pdf for exact definitions.
        metrics_frame = ttk.LabelFrame(right, text="Metrics (Query Precision / Relevant Recall)")
        metrics_frame.pack(fill="x", pady=(int(6*s), 0))

        self.metrics_text = tk.Text(metrics_frame, height=5, width=int(60 * min(s, 1.5)), state="disabled",
                                    font=("TkDefaultFont", int(11*s)), bg="#f8f8f8")
        self.metrics_text.pack(fill="x", padx=4, pady=4)
        self.metrics_text.insert("1.0", "Click 'Compute from DB (last video)' after a run.\n"
                                        "QP / RR as defined in IJSC_2026-1.pdf & SPIE_IVSP_2026.pdf")
        self.metrics_text.config(state="disabled")

        btn_row = ttk.Frame(metrics_frame)
        btn_row.pack(fill="x", padx=4, pady=2)
        ttk.Button(btn_row, text="Compute from DB (last video)", command=self._compute_metrics_from_db).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self._clear_metrics_display).pack(side="left", padx=4)

        self._last_metrics: Dict[str, Any] = {}

        classes_frame = ttk.LabelFrame(right, text="Discovered Classes (from current + cache)")
        classes_frame.pack(fill="both", expand=True, pady=(int(6*s), 0))

        self.class_listbox = tk.Listbox(classes_frame, height=10, font=("TkDefaultFont", int(13 * s)))
        self.class_listbox.pack(fill="both", expand=True, side="left")
        ysb = ttk.Scrollbar(classes_frame, orient="vertical", command=self.class_listbox.yview)
        ysb.pack(side="right", fill="y")
        self.class_listbox.configure(yscrollcommand=ysb.set)

        # Preview area (simple for now)
        preview_frame = ttk.LabelFrame(body, text="Preview (last processed frame - stub)")
        preview_frame.pack(side="left", fill="both", expand=True, padx=(int(6 * self.ui_scale), 0))
        self.preview_label = ttk.Label(preview_frame, text="(Preview will show last frame + tile highlights in future)")
        self.preview_label.pack(expand=True)

    def _add_param_row(self, parent, label, attr, default, is_float=False, is_str=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=max(1, int(2 * self.ui_scale)))
        fs = int(11 * self.ui_scale)
        # Do not pass font to ttk.Label (can cause "unknown option -font" on some ttk versions).
        # The global style (configured with ui_scale in _build_ui) handles it.
        ttk.Label(row, text=label, width=int(26 * min(self.ui_scale, 1.5))).pack(side="left")
        var = tk.StringVar(value=str(default))
        entry = ttk.Entry(row, textvariable=var, width=int(20 * min(self.ui_scale, 1.5)), font=("TkDefaultFont", fs))
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

            # NEW exact annotation DB
            self.config.tile_annotations.db_path = getattr(self, "_tile_ann_db_var").get()
            self.config.tile_annotations.edit_mode_default = self.edit_mode_var.get()
            self.config.ared.data_augmentation_enabled = self.aug_var.get()
            self.config.tile_annotations.label_only_default = self.label_only_var.get()
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

        # Prepare label store (embedding similarity)
        if self.config.label_cache.enabled:
            self.label_store = PersistentLabelStore(
                db_path=self.config.label_cache.db_path,
                auto_label_threshold=self.config.label_cache.auto_label_threshold,
            )
            self.controller.set_label_store(self.label_store)
        else:
            self.label_store = None

        # Prepare exact tile annotation DB (NEW - primary for persistent exact labels + editing)
        if getattr(self.config.tile_annotations, 'enabled', True):
            ann_path = self.config.tile_annotations.db_path or "drone_tile_annotations.db"
            self.tile_db = TileAnnotationDB(db_path=ann_path)
            self.controller.set_tile_database(self.tile_db)
        else:
            self.tile_db = None
            self.controller.set_tile_database(None)

        # Edit mode (force re-label of known exact tiles for corrections)
        self.edit_mode = self.edit_mode_var.get()
        self.controller.set_edit_mode(self.edit_mode)

        # Data augmentation flag
        self.config.ared.data_augmentation_enabled = self.aug_var.get()

        # Label Only mode
        label_only = self.label_only_var.get()
        self.controller.set_label_only_mode(label_only)

        # If label only, we can skip heavy DINO loading in the controller (it will check the flag)
        self.controller.update_config(self.config)
        self.controller.start()

        self.start_btn.config(state="disabled")
        if getattr(self.controller, 'label_only_mode', False):
            self.status_var.set("Label Only Mode — labeling every tile (no A/RED). Use Pause/Stop when done.")
        else:
            self.status_var.set("Processing... (use Pause / Stop)")

    def _stop(self):
        """Stop the worker and re-enable Start so the user can restart without restarting the whole program."""
        self.controller.stop()
        self.start_btn.config(state="normal")
        self.status_var.set("Stopped. You can change parameters/videos and press Start again.")
        if self.controller.stats:
            self._update_stats_display(self.controller.stats)

    def _save_label_cache(self):
        if self.label_store:
            self.label_store.save()
            messagebox.showinfo("Label Cache", "Label cache saved.")
        else:
            messagebox.showwarning("Label Cache", "No active label store.")

    def _save_tile_annotations(self):
        """NEW: force flush of the exact annotation DB (sqlite writes immediately on set, but explicit is nice)."""
        if self.tile_db:
            try:
                self.tile_db.conn.commit()
                count = len(self.tile_db)
                messagebox.showinfo("Tile Annotations", f"Annotation DB saved. Total entries: {count}")
            except Exception as e:
                messagebox.showerror("Tile Annotations", f"Save failed: {e}")
        else:
            messagebox.showwarning("Tile Annotations", "No annotation DB active.")

    # ------------------------------------------------------------------
    # Metrics display (Query Precision + Relevant Recall)
    # References the exact definitions in IJSC_2026-1.pdf and SPIE_IVSP_2026.pdf
    # ------------------------------------------------------------------
    def _compute_metrics_from_db(self, video_name: Optional[str] = None):
        """Compute and display metrics using the current TileAnnotationDB + any logged A/RED queries."""
        if not self.controller or not self.tile_db:
            self._display_metrics_error("Need an active TileAnnotationDB (run Label Only or A/RED with DB enabled).")
            return

        if video_name is None:
            # Try last processed video
            video_name = self.controller.stats.get("current_video", "")
            if not video_name and self.config.video_paths:
                video_name = Path(self.config.video_paths[-1]).name

        if not video_name:
            self._display_metrics_error("No video name available. Select a video or run processing first.")
            return

        try:
            result = self.controller.compute_metrics_for_video(video_name)
            if "error" in result:
                self._display_metrics_error(result["error"])
                return

            self._last_metrics = result
            self._refresh_metrics_display(result)
        except Exception as e:
            self._display_metrics_error(str(e))

    def _clear_metrics_display(self):
        self._last_metrics = {}
        if hasattr(self, 'metrics_text'):
            self.metrics_text.config(state="normal")
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("1.0", "No metrics computed yet.\nRun A/RED (with DB labels) or use the button.")
            self.metrics_text.config(state="disabled")

    def _display_metrics_error(self, msg: str):
        if hasattr(self, 'metrics_text'):
            self.metrics_text.config(state="normal")
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("1.0", f"Error: {msg}")
            self.metrics_text.config(state="disabled")

    def _refresh_metrics_display(self, result: Dict[str, Any]):
        if not hasattr(self, 'metrics_text'):
            return
        lines = [
            f"Video: {result.get('video', '?')}",
            f"Query Precision (QP): {result.get('query_precision', 0):.4f}",
            f"Relevant Recall (RR): {result.get('relevant_recall', 0):.4f}",
            f"Queries: {result.get('n_actual_queries', 0)} / {result.get('total_points', 0)}",
            f"vs Random ≈ {result.get('baseline_random_query_precision_approx', 0):.4f}",
            result.get('summary', '')
        ]
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(lines))
        self.metrics_text.config(state="disabled")

    def _update_metrics_on_finish(self):
        """Called when a run finishes. Tries to auto-compute using the last video + DB."""
        if getattr(self.controller, 'label_only_mode', False):
            # In pure label-only we have no A/RED queries, so just show stats
            self._refresh_metrics_display({
                "video": self.controller.stats.get("current_video", "?"),
                "query_precision": 0.0,
                "relevant_recall": 0.0,
                "n_actual_queries": 0,
                "total_points": self.stats.get("tiles_processed", 0),
                "summary": "Label Only run — no A/RED queries. Use 'Compute from DB' after labeling."
            })
            return

        # Normal A/RED run — try to compute real metrics
        self._compute_metrics_from_db()

    def _load_tile_annotations(self):
        path = filedialog.askopenfilename(title="Load tile annotation DB", filetypes=[("SQLite DB", "*.db"), ("All", "*.*")])
        if not path:
            return
        try:
            if self.tile_db:
                try:
                    self.tile_db.close()
                except Exception:
                    pass
            self.tile_db = TileAnnotationDB(db_path=path)
            self.controller.set_tile_database(self.tile_db)
            self.config.tile_annotations.db_path = path
            messagebox.showinfo("Tile Annotations", f"Loaded DB with {len(self.tile_db)} entries.")
        except Exception as e:
            messagebox.showerror("Load", str(e))

    # ------------------------------------------------------------------
    # NEW: Review / Edit past exact labels (works across runs and stride changes)
    # ------------------------------------------------------------------
    def _open_review_window(self):
        if not self.tile_db:
            # Try to open/create one from current config
            try:
                path = getattr(self.config.tile_annotations, 'db_path', 'drone_tile_annotations.db')
                self.tile_db = TileAnnotationDB(db_path=path)
                self.controller.set_tile_database(self.tile_db)
            except Exception as e:
                messagebox.showerror("Review", f"Could not open annotation DB: {e}")
                return

        videos = self.tile_db.list_videos()
        if not videos:
            messagebox.showinfo("Review", "No annotations saved yet. Label some tiles while running A/RED (or load a previous DB).")
            return

        LabelReviewWindow(self.root, self.tile_db, ui_scale=self.ui_scale)

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
        if not path:
            return
        # Allow loading even before first Start (create adapter if necessary so we can warm-start)
        if not self.controller.ared_adapter:
            self.controller.ared_adapter = AREDAdapter(self.config.ared)
            if self.label_store:
                self.controller.ared_adapter.set_label_store(self.label_store)
        self.controller.ared_adapter.load_state(path, label_lookup=self._label_lookup_from_store)
        # Re-apply feature extractor so data augmentation can still work after re-init inside load
        if hasattr(self.controller, "_last_feature_extractor") and self.controller._last_feature_extractor:
            self.controller.ared_adapter.set_feature_extractor(self.controller._last_feature_extractor)
        self._refresh_class_list()
        # reflect the loaded clusters in the live stats
        if self.controller.ared_adapter:
            self.controller.stats["ared_clusters"] = self.controller.ared_adapter.num_clusters
            self.controller.stats["ared_known_labels"] = self.controller.ared_adapter.num_known_labels
            self._update_stats_display(self.controller.stats)
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

        # Build relevance map (source of truth is our class_relevance; seed from store if possible)
        class_relevance = dict(getattr(self, 'class_relevance', {}))
        for lbl in classes:
            if lbl not in class_relevance:
                seeded = False
                if self.label_store:
                    rel = self.label_store.get_class_relevance(lbl)
                    if rel is not None:
                        class_relevance[lbl] = rel
                        seeded = True
                if not seeded:
                    class_relevance[lbl] = False

        def _assign_cb(label: str, relevant: bool):
            # This is now purely notification / UI update.
            # The dialog itself calls set_result on the req it holds.
            print(f"[GUI] Label SUBMITTED from dialog: '{label}' (relevant={relevant})")
            self._pending_label_request = None
            self.discovered_classes.add(label)
            self.class_relevance[label] = relevant  # remember the relevance decided at (or for) this class
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
                ui_scale=getattr(self, 'ui_scale', 1.6),
                class_relevance=class_relevance,
            )
        else:
            print("[GUI] Updating existing persistent LabelingDialog with new query tile.")
            self._labeling_win.set_current_request(req, known_classes=classes, class_counts=counts, class_relevance=class_relevance)

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

        # Ensure we have a relevance entry (seed from store when possible)
        for lbl in all_labels:
            if lbl not in self.class_relevance:
                seeded = False
                if self.label_store:
                    rel = self.label_store.get_class_relevance(lbl)
                    if rel is not None:
                        self.class_relevance[lbl] = rel
                        seeded = True
                if not seeded:
                    self.class_relevance[lbl] = False

        for lbl in sorted(all_labels):
            c = counts.get(lbl, 0)
            display = f"{lbl} ({c})" if c > 0 else lbl
            if self.class_relevance.get(lbl, False):
                display += " [R]"
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

        # Re-enable Start after a run ends so user can restart / load new videos without restarting the program
        status = stats.get("status", "")
        if hasattr(self, "start_btn"):
            if status in ("stopped", "finished", "error", "idle"):
                self.start_btn.config(state="normal")
            elif status in ("running", "paused"):
                self.start_btn.config(state="disabled")

        # Auto-show metrics box when processing finishes (at end of video)
        if status in ("finished", "stopped"):
            self._update_metrics_on_finish()


# =============================================================================
# LabelReviewWindow - browse + correct previous exact labels (no stored pixels)
# =============================================================================

class LabelReviewWindow(tk.Toplevel):
    """
    Standalone window for reviewing and editing past tile labels stored in the
    exact TileAnnotationDB.

    - Select video (from those that have annotations)
    - Browse annotations (list + prev/next)
    - Re-extracts the tile image live from the original video file (using stored
      frame + crop position). No images are saved to disk.
    - Same quick labeling UX: list of classes or create new + relevant checkbox.
    - "Save Change" writes the (possibly corrected) label back to the DB.
    """

    def __init__(self, master, tile_db: "TileAnnotationDB", ui_scale: float = 1.6):
        super().__init__(master)
        self.tile_db = tile_db
        self.ui_scale = float(ui_scale) if ui_scale else 1.6
        self.current_ann: Optional[Dict] = None
        self.current_img: Optional[Image.Image] = None
        self._zoom = 1.0

        self.title("Review & Edit Past Tile Labels - Exact DB")
        base = int(1100 * min(self.ui_scale, 2.0))
        self.geometry(f"{base}x{int(base*0.72)}")
        self.minsize(800, 520)
        self.resizable(True, True)

        self._build()
        self._load_videos()
        self.after(80, self._center_initial)

    def _build(self):
        s = self.ui_scale
        fs = int(11 * s)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=int(8*s), pady=int(4*s))

        ttk.Label(top, text="Video:").pack(side="left")
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(top, textvariable=self.video_var, width=60, state="readonly")
        self.video_combo.pack(side="left", padx=4)
        self.video_combo.bind("<<ComboboxSelected>>", lambda e: self._load_annotations_for_current_video())

        ttk.Button(top, text="Reload List", command=self._load_videos).pack(side="left", padx=4)

        # Main split: left list of annotations, center image + info, right edit controls
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=int(6*s), pady=int(4*s))

        # Left: list of annotations for the video
        left = ttk.LabelFrame(main, text="Labeled tiles in this video (click or use arrows)")
        left.pack(side="left", fill="both", expand=False, padx=(0, int(6*s)))

        self.ann_list = tk.Listbox(left, width=42, height=18, font=("TkDefaultFont", fs))
        self.ann_list.pack(fill="both", expand=True, side="left")
        ysb = ttk.Scrollbar(left, orient="vertical", command=self.ann_list.yview)
        ysb.pack(side="right", fill="y")
        self.ann_list.configure(yscrollcommand=ysb.set)
        self.ann_list.bind("<<ListboxSelect>>", self._on_list_select)

        nav = ttk.Frame(left)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀ Prev", command=self._prev).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Next ▶", command=self._next).pack(side="left", expand=True, fill="x")

        # Center: image display
        center = ttk.LabelFrame(main, text="Tile (re-extracted from source video on demand)")
        center.pack(side="left", fill="both", expand=True, padx=int(4*s))

        self.canvas = tk.Canvas(center, bg="#222", width=520, height=420)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)

        self.info_var = tk.StringVar(value="Select an annotation from the list on the left.")
        ttk.Label(center, textvariable=self.info_var, relief="sunken").pack(fill="x", padx=4, pady=2)

        zf = ttk.Frame(center)
        zf.pack(fill="x")
        ttk.Button(zf, text="Zoom -", command=lambda: self._zoom_delta(-0.2)).pack(side="left")
        ttk.Button(zf, text="Zoom +", command=lambda: self._zoom_delta(0.2)).pack(side="left")
        ttk.Button(zf, text="Fit", command=self._display_image).pack(side="left")

        # Right: editing (re-uses spirit of the main labeling dialog)
        right = ttk.LabelFrame(main, text="Edit label for this tile")
        right.pack(side="left", fill="y", padx=(int(4*s), 0))

        ttk.Label(right, text="Current / New label:").pack(anchor="w", padx=4, pady=(4,0))
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(right, textvariable=self.label_var, width=28)
        self.label_entry.pack(fill="x", padx=4)

        self.rel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="Relevant (interesting / anomaly)", variable=self.rel_var).pack(anchor="w", padx=4, pady=4)

        ttk.Button(right, text="Save Change to DB", command=self._save_current_edit).pack(fill="x", padx=4, pady=6)
        ttk.Button(right, text="Mark Background", command=lambda: self._quick_assign("__BACKGROUND__", False)).pack(fill="x", padx=4)
        ttk.Button(right, text="Delete this annotation", command=self._delete_current).pack(fill="x", padx=4, pady=(2,8))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(right, text="Tip: Changes are immediately usable.\nRe-run A/RED (normal mode) to auto-apply.").pack(anchor="w", padx=4)

        self.status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status_var, relief="sunken").pack(fill="x", side="bottom", pady=4)

    def _load_videos(self):
        vids = self.tile_db.list_videos()
        self.video_combo['values'] = vids
        if vids:
            self.video_var.set(vids[0])
            self._load_annotations_for_current_video()

    def _load_annotations_for_current_video(self):
        v = self.video_var.get()
        if not v:
            return
        self.annotations = self.tile_db.get_annotations_for_video(v)
        self.ann_list.delete(0, "end")
        for i, a in enumerate(self.annotations):
            rel_mark = " [R]" if a["relevant"] else ""
            txt = f"f{a['abs_frame']:06d} r{a['tile_row']}c{a['tile_col']}  {a['label']}{rel_mark}"
            self.ann_list.insert("end", txt)
        if self.annotations:
            self.ann_list.selection_set(0)
            self._show_annotation(0)

    def _on_list_select(self, event=None):
        sel = self.ann_list.curselection()
        if sel:
            self._show_annotation(sel[0])

    def _show_annotation(self, idx: int):
        if not (0 <= idx < len(getattr(self, 'annotations', []))):
            return
        ann = self.annotations[idx]
        self.current_ann = ann

        # Build bbox from stored crop or fall back to grid calc
        cx = ann.get("crop_x", ann["tile_col"] * ann["tile_width"])
        cy = ann.get("crop_y", ann["tile_row"] * ann["tile_height"])
        tw, th = ann["tile_width"], ann["tile_height"]
        bbox = (cx, cy, cx + tw, cy + th)

        img = extract_tile_from_video(ann["video_path"], ann["abs_frame"], bbox)
        self.current_img = img

        self.label_var.set(ann["label"])
        self.rel_var.set(ann["relevant"])

        name = Path(ann["video_path"]).name
        self.info_var.set(f"{name}  frame={ann['abs_frame']}  pos=({ann['tile_row']},{ann['tile_col']})  size={tw}x{th}")
        self._display_image()
        self.status_var.set("Loaded. Edit above and click Save Change.")

    def _display_image(self):
        if not self.current_img:
            self.canvas.delete("all")
            self.canvas.create_text(200, 100, text="(Could not re-extract tile image from video)", fill="orange")
            return
        self.canvas.delete("all")
        cw = max(100, self.canvas.winfo_width() or 520)
        ch = max(100, self.canvas.winfo_height() or 380)
        z = max(0.1, min(6.0, self._zoom))

        orig_w, orig_h = self.current_img.size
        ratio = min((cw - 10) * z / orig_w, (ch - 10) * z / orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))
        disp = self.current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._tkimg = ImageTk.PhotoImage(disp)
        self.canvas.create_image(cw//2, ch//2, image=self._tkimg, anchor="center")

    def _zoom_delta(self, d: float):
        self._zoom = max(0.2, min(5.0, self._zoom + d))
        self._display_image()

    def _prev(self):
        sel = self.ann_list.curselection()
        idx = sel[0] - 1 if sel else 0
        if idx < 0:
            idx = len(self.annotations) - 1
        self.ann_list.selection_clear(0, "end")
        self.ann_list.selection_set(idx)
        self.ann_list.see(idx)
        self._show_annotation(idx)

    def _next(self):
        sel = self.ann_list.curselection()
        idx = (sel[0] + 1) if sel else 0
        if idx >= len(self.annotations):
            idx = 0
        self.ann_list.selection_clear(0, "end")
        self.ann_list.selection_set(idx)
        self.ann_list.see(idx)
        self._show_annotation(idx)

    def _save_current_edit(self):
        if not self.current_ann:
            return
        ann = self.current_ann
        new_label = self.label_var.get().strip() or "__UNLABELED__"
        new_rel = self.rel_var.get()

        try:
            self.tile_db.set_annotation(
                ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                ann["tile_width"], ann["tile_height"],
                new_label, new_rel,
                embedding=None,  # we don't re-embed here; original emb if present stays
                crop_x=ann.get("crop_x"), crop_y=ann.get("crop_y")
            )
            self.status_var.set(f"Saved: {new_label} (rel={new_rel})")
            # Refresh list
            self._load_annotations_for_current_video()
        except Exception as e:
            messagebox.showerror("Save", str(e))

    def _quick_assign(self, label: str, rel: bool):
        self.label_var.set(label)
        self.rel_var.set(rel)
        self._save_current_edit()

    def _delete_current(self):
        if not self.current_ann:
            return
        if not messagebox.askyesno("Delete", "Remove this annotation from the DB?"):
            return
        ann = self.current_ann
        try:
            self.tile_db.delete_annotation(
                ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                ann["tile_width"], ann["tile_height"]
            )
            self.status_var.set("Deleted.")
            self._load_annotations_for_current_video()
        except Exception as e:
            messagebox.showerror("Delete", str(e))

    def _center_initial(self):
        try:
            self.update_idletasks()
        except Exception:
            pass
