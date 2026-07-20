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
from typing import Optional, List, Dict, Any, Tuple

from .config import PipelineConfig, GUIConfig
from .pipeline import DroneAREDController, LabelRequest
from .label_store import PersistentLabelStore
from .ared_adapter import AREDAdapter
from .tile_database import TileAnnotationDB, extract_tile_from_video
from .annotation_manager import AnnotationManager
from .annotation_domain import TileKey, AnnotationFilter
from .tile_database import TileAnnotationDB
from .gui_bulk_dialog import BulkLabelOpsDialog  # modular bulk editing UI
from .logutil import vprint, set_terminal_logging
# GridTiler is imported lazily inside the browser (to avoid pulling numpy/PIL deps at gui module load
# if not already present from other paths).


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
                 ui_scale: float = 1.6, class_relevance: Optional[Dict[str, bool]] = None,
                 allow_skip: bool = False,
                 current_label: Optional[str] = None,
                 current_relevant: bool = False):
        """High-volume labeling dialog, now enhanced for efficient 'sparse' labeling sessions.

        New parameters for the alternative labeling mode (efficient relevant-focused labeling):
        - allow_skip: When True (Label Only / sparse mode), shows a prominent "Skip / Move On"
          button. Clicking it means "do not assign a label to this tile right now".
          This enables quickly labeling only relevant class instances (e.g. every person)
          while skipping background tiles to save time on thousands of samples.
          Skipped tiles stay unlabeled in the DB and will re-appear on resume (unless labeled later).
        - current_label / current_relevant: When resuming in edit mode on an already-labeled tile,
          pre-fill so the user sees what was previously chosen and can correct it or skip.

        Skip is deliberately distinct from "Mark as Background":
        - Background = explicit "__BACKGROUND__", relevant=False (still a label).
        - Skip = no label written at all for this pass.

        All paths still respect the global edit_mode for forcing re-labeling of known tiles.
        """
        super().__init__(master)
        self.title("Review Queried Tile - A/RED Drone  [Persistent - reposition me once!]")
        self.current_req = request
        self.on_assign = on_assign
        self.class_counts = class_counts or {}
        self.known_classes = sorted(set(known_classes))
        self.ui_scale = float(ui_scale) if ui_scale else 1.6
        self._zoom_level = 1.0
        self.class_relevance: dict[str, bool] = dict(class_relevance or {})

        # New for sparse/resume labeling mode
        self.allow_skip = bool(allow_skip)
        self.current_label = current_label
        self.current_relevant = bool(current_relevant)

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

        # If we were given a current label (edit/resume case in sparse label mode),
        # try to pre-select it and set the relevant checkbox state.
        if self.current_label:
            self._apply_current_label_prefill()

        # Keyboard bindings for power users (↑↓ class select, 1-9 assign, etc.)
        self._bind_shortcuts()

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

        ttk.Label(
            info,
            text="  (↑↓ select class, 1-9 assign, Enter assign, B=background. Resize me!)",
            foreground="gray",
        ).pack(side="right")

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
        self._filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=int(22 * min(s, 1.8)))
        self._filter_entry.pack(side="left", fill="x", expand=True)

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

        # "Skip / Move On" button - only shown in sparse / Label Only efficient mode.
        if getattr(self, 'allow_skip', False):
            ttk.Button(bottom, text="Skip / Move On (no label)", command=self._skip_current).pack(side="left", padx=8)

        ttk.Button(bottom, text="Close Window (recreates on next query)", command=self._close_window).pack(side="right")

        # Navigation controls for Label Only mode (back/forward/jump)
        # Added so the user can move around the video if they miss a tile or want
        # to jump to a specific frame while doing sparse relevant-only labeling.
        if getattr(self, 'allow_skip', False):
            nav = ttk.Frame(self)
            nav.pack(fill="x", padx=int(8*s), pady=(2, int(4*s)))

            ttk.Button(nav, text="◀ Prev", command=self._nav_prev).pack(side="left")
            ttk.Button(nav, text="Next ▶", command=self._nav_next).pack(side="left", padx=4)

            ttk.Label(nav, text="Jump frame:").pack(side="left")
            self.jump_var = tk.StringVar()
            self._jump_entry = ttk.Entry(nav, textvariable=self.jump_var, width=7)
            self._jump_entry.pack(side="left")
            self._jump_entry.bind("<Return>", lambda ev: self._nav_jump())
            ttk.Button(nav, text="Go", command=self._nav_jump).pack(side="left", padx=2)
            ttk.Label(nav, text="(Esc=Skip)", foreground="#666").pack(side="left", padx=6)

        # Status for the dialog
        self.dialog_status_var = tk.StringVar(value="Choose from list (double-click or button) or type new class + Create & Assign")
        ttk.Label(self, textvariable=self.dialog_status_var, relief="sunken").pack(fill="x", padx=int(8*s), pady=pady_s)

        # Make the new entry easy to reach
        self.after(200, lambda: self.new_entry.focus_set() if not self.known_classes else None)

        # Initial status
        self.dialog_status_var.set(
            "↑↓ to select class, 1-9 / Enter to assign, B=background. "
            "Or type a new name + Create & Assign. Window stays open."
        )

    # ---------------- Keyboard power-user shortcuts ----------------
    def _focus_in_widget(self, widget) -> bool:
        focus = self.focus_get()
        if focus is None or widget is None:
            return False
        current = focus
        while current is not None:
            if current == widget:
                return True
            try:
                current = current.master
            except AttributeError:
                break
        return False

    def _is_typing_focus(self) -> bool:
        """True when focus is in a text entry (don't steal digits / arrows)."""
        return (
            self._focus_in_widget(getattr(self, "new_entry", None))
            or self._focus_in_widget(getattr(self, "_filter_entry", None))
            or self._focus_in_widget(getattr(self, "_jump_entry", None))
        )

    def _bind_shortcuts(self):
        """Bind query-dialog keys. Left/Right reserved for Label-Only nav when allow_skip."""
        # Escape: skip in sparse mode, else background
        if getattr(self, "allow_skip", False):
            self.bind("<Escape>", lambda e: self._skip_current() if not self._is_typing_focus() else None)
            self.bind("<Left>", lambda e: self._nav_prev() if not self._is_typing_focus() else None)
            self.bind("<Right>", lambda e: self._nav_next() if not self._is_typing_focus() else None)
        else:
            self.bind("<Escape>", lambda e: self._assign_as_background() if not self._is_typing_focus() else None)

        self.bind("<Return>", lambda e: self._assign_selected() if not self._is_typing_focus() else None)
        self.bind("<Control-n>", lambda e: self.new_entry.focus_set())
        self.bind("<Control-f>", lambda e: self._focus_filter())
        self.bind("<slash>", lambda e: self._focus_filter() if not self._is_typing_focus() else None)
        self.bind("<Up>", lambda e: self._nav_class_list(-1) if not self._is_typing_focus() else None)
        self.bind("<Down>", lambda e: self._nav_class_list(1) if not self._is_typing_focus() else None)
        self.bind("<b>", lambda e: self._assign_as_background() if not self._is_typing_focus() else None)
        self.bind("<B>", lambda e: self._assign_as_background() if not self._is_typing_focus() else None)
        for i in range(1, 10):
            self.bind(f"<Key-{i}>", lambda e, n=i: self._assign_by_number(n))

    def _focus_filter(self):
        entry = getattr(self, "_filter_entry", None)
        if entry is not None:
            entry.focus_set()

    def _nav_class_list(self, delta: int):
        """Move selection in the filtered class list without assigning."""
        if not self._filtered_classes:
            return
        sel = self.class_list.curselection()
        if sel:
            idx = max(0, min(len(self._filtered_classes) - 1, sel[0] + delta))
        else:
            idx = 0 if delta >= 0 else len(self._filtered_classes) - 1
        self.class_list.selection_clear(0, "end")
        self.class_list.selection_set(idx)
        self.class_list.see(idx)
        self.class_list.activate(idx)

    def _assign_by_number(self, n: int):
        """Assign the N-th filtered class (1-based) using stored class relevance."""
        if self._is_typing_focus():
            return
        if 1 <= n <= len(self._filtered_classes):
            label = self._filtered_classes[n - 1]
            rel = self.class_relevance.get(label, False)
            self._assign(label, rel)

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

    def _apply_current_label_prefill(self):
        """When resuming in edit mode on an already-labeled tile (sparse label mode),
        pre-select the previous choice in the list and set the relevant checkbox.

        This is crucial for the "resume" workflow the user requested:
        - User can see what was labeled before.
        - Can re-assign a corrected label.
        - Or click "Skip / Move On" to leave the existing label untouched.
        """
        if not getattr(self, 'current_label', None):
            return
        try:
            if self.current_label in self._filtered_classes:
                idx = self._filtered_classes.index(self.current_label)
                self.class_list.selection_clear(0, "end")
                self.class_list.selection_set(idx)
                self.class_list.see(idx)
            else:
                self.new_var.set(self.current_label)

            if hasattr(self, 'relevant_var'):
                self.relevant_var.set(self.current_relevant)

            self.dialog_status_var.set(
                f"EDIT: previous '{self.current_label}' (relevant={self.current_relevant}). "
                "Re-assign or Skip to keep as-is."
            )
        except Exception as e:
            print(f"[GUI Dialog] Could not pre-fill current label: {e}")

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

    def _skip_current(self):
        """User explicitly chooses not to label this tile in the current pass.

        - Does not write (or overwrite) anything in the TileAnnotationDB for this tile.
        - Satisfies the LabelRequest so the worker can continue to the next tile.
        - On resume the tile will be offered again (unless it gets labeled via A/RED query
          or a future labeling pass).
        - In edit mode this means "keep whatever was there before and move on".
        """
        print("[GUI Dialog] User chose SKIP / Move On for current tile.")
        if getattr(self, 'current_req', None):
            try:
                self.current_req.set_skip()
            except Exception:
                pass
            self.current_req = None
        if self.on_assign:
            try:
                self.on_assign("__SKIPPED__", False)
            except Exception:
                pass
        # Keep window open and ready for next tile
        self._prepare_for_next(last_label="(skipped)", last_relevant=False)

    # ------------------------------------------------------------------
    # Navigation handlers (only active in Label Only mode)
    # Forwarded through MainWindow to the controller.
    # ------------------------------------------------------------------
    def _nav_prev(self):
        """Satisfy current tile req (so worker unblocks), then ask main to signal controller.
        This only happens for Label Only (allow_skip) requests; A/RED dialogs have no nav UI.
        """
        if getattr(self, 'current_req', None):
            try:
                self.current_req.set_skip()
            except Exception:
                pass
            self.current_req = None
        if self.on_assign:
            try:
                self.on_assign("__SKIPPED__", False)
            except Exception:
                pass
        self._prepare_for_next(last_label="(prev)", last_relevant=False)
        try:
            mw = getattr(self, 'main_window', None)
            if mw:
                mw._nav_prev_from_dialog()
            else:
                # Fallback (should not be needed)
                self.master._nav_prev_from_dialog()
        except Exception:
            pass

    def _nav_next(self):
        if getattr(self, 'current_req', None):
            try:
                self.current_req.set_skip()
            except Exception:
                pass
            self.current_req = None
        if self.on_assign:
            try:
                self.on_assign("__SKIPPED__", False)
            except Exception:
                pass
        self._prepare_for_next(last_label="(next)", last_relevant=False)
        try:
            mw = getattr(self, 'main_window', None)
            if mw:
                mw._nav_next_from_dialog()
            else:
                self.master._nav_next_from_dialog()
        except Exception:
            pass

    def _nav_jump(self):
        try:
            val = int(self.jump_var.get() or 0)
        except Exception:
            return
        if getattr(self, 'current_req', None):
            try:
                self.current_req.set_skip()
            except Exception:
                pass
            self.current_req = None
        if self.on_assign:
            try:
                self.on_assign("__SKIPPED__", False)
            except Exception:
                pass
        self._prepare_for_next(last_label="(jump)", last_relevant=False)
        try:
            mw = getattr(self, 'main_window', None)
            if mw:
                mw._nav_jump_from_dialog(val)
            else:
                self.master._nav_jump_from_dialog(val)
        except Exception:
            pass

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

    def set_current_request(self, req, known_classes=None, class_counts=None, class_relevance=None,
                            allow_skip=None, current_label=None, current_relevant=None):
        """Update this persistent window for a new tile (A/RED query or Label Only).

        Extended for the alternative efficient labeling mode:
        - allow_skip, current_label, current_relevant are forwarded so the Skip button
          and pre-fill logic work when the window is reused.
        """
        self.current_req = req
        if allow_skip is not None:
            self.allow_skip = bool(allow_skip)
        if current_label is not None:
            self.current_label = current_label
        if current_relevant is not None:
            self.current_relevant = bool(current_relevant)

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

        if getattr(self, 'allow_skip', False):
            self.dialog_status_var.set(
                "Label Only / sparse mode (resume). "
                "Assign relevant class or press Escape / click 'Skip / Move On' to skip unlabeled tiles. "
                "Already-labeled tiles are auto-skipped unless Edit Mode is on."
            )
        else:
            self.dialog_status_var.set("New A/RED query. Label this tile (select or create), then Assign.")

        # Re-apply prefill if we are editing a previously labeled tile
        if getattr(self, 'current_label', None):
            self._apply_current_label_prefill()

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
        self.annotation_manager: Optional[AnnotationManager] = None  # service layer (refactor in progress)
        self.edit_mode: bool = False
        self._stats_job = None
        self._pending_label_request: Optional[LabelRequest] = None
        self.discovered_classes: set = set()  # labels we have assigned in this run (for immediate UI feedback)
        self.run_class_counts: dict[str, int] = {}  # per-run counts for display in class boxes (not full DB history)
        self.class_relevance: dict[str, bool] = {}  # class name -> is_relevant (set at creation time)
        self._last_queried_global = -1

        self._build_ui()
        self._start_stat_poller()

        # Register for status from worker
        self.controller.on_stats = self._on_worker_stats

        # Ensure clean shutdown (prevents "terminate called without an active exception" and VSCode crash reports)
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        except Exception:
            pass

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
        file_menu.add_command(label="Exit", command=self._on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # ---- Scrollable main area (control bar + params can exceed viewport height) ----
        # Outer: canvas + vertical scrollbar so buttons/params at the bottom stay reachable.
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self._main_canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        self._main_vscroll = ttk.Scrollbar(outer, orient="vertical", command=self._main_canvas.yview)
        self._main_canvas.configure(yscrollcommand=self._main_vscroll.set)
        self._main_vscroll.pack(side="right", fill="y")
        self._main_canvas.pack(side="left", fill="both", expand=True)

        # Interior frame that holds the entire UI content
        scroll_inner = ttk.Frame(self._main_canvas)
        self._main_canvas_window = self._main_canvas.create_window((0, 0), window=scroll_inner, anchor="nw")

        def _on_inner_configure(_event=None):
            # Keep scrollregion tight to content
            self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Stretch interior to canvas width (vertical scroll only)
            try:
                self._main_canvas.itemconfigure(self._main_canvas_window, width=event.width)
            except Exception:
                pass

        scroll_inner.bind("<Configure>", _on_inner_configure)
        self._main_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            # Only scroll main window when pointer is over it (not over a child Toplevel)
            try:
                w = self.root.winfo_containing(event.x_root, event.y_root)
                if w is None:
                    return
                # Skip if over a different toplevel (labeling dialog, review, etc.)
                top = w.winfo_toplevel()
                if top is not self.root:
                    return
                if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                    self._main_canvas.yview_scroll(-3, "units")
                elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
                    self._main_canvas.yview_scroll(3, "units")
            except Exception:
                pass

        self._main_mousewheel_handler = _on_mousewheel
        # bind_all so wheel works over nested ttk widgets; handler filters other toplevels
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

        # Control bar
        ctrl = ttk.Frame(scroll_inner)
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
        ttk.Label(scroll_inner, textvariable=self.status_var, relief="sunken").pack(fill="x", padx=int(6*self.ui_scale), pady=int(2*self.ui_scale))

        # Main content: left params, center stats + classes, right preview stub
        body = ttk.Frame(scroll_inner)
        body.pack(fill="both", expand=True, padx=int(6*self.ui_scale), pady=int(4*self.ui_scale))

        # Parameters (many are live for next run)
        param_frame = ttk.LabelFrame(body, text="Parameters (applied on next Start)")
        param_frame.pack(side="left", fill="y", padx=(0, int(6*self.ui_scale)))

        self._add_param_row(param_frame, "Tile W (px, uniform)", "tile_w", self.config.tiling.tile_width)
        self._add_param_row(param_frame, "Tile H (px, uniform)", "tile_h", self.config.tiling.tile_height)
        self._add_param_row(param_frame, "Frame stride (every Nth)", "frame_stride", self.config.tiling.frame_stride)

        # Overlapping tiles controls (new). Checkbox enables; entries are overlap in pixels.
        # stride = tile_size - overlap (clamped >= 1). When disabled or 0, stride = tile size (non-overlapping).
        self.overlap_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="Enable overlapping tiles (stride = tile - overlap)",
                        variable=self.overlap_enabled_var).pack(anchor="w", pady=int(2*s))

        self._add_param_row(param_frame, "Overlap X (px)", "overlap_x", getattr(self.config.tiling, "overlap_x", 0))
        self._add_param_row(param_frame, "Overlap Y (px)", "overlap_y", getattr(self.config.tiling, "overlap_y", 0))

        self._add_param_row(param_frame, "Kappa (higher = MORE queries)", "kappa", self.config.ared.kappa, is_float=True)
        self._add_param_row(param_frame, "Buffer size", "buf_size", self.config.ared.l_buf_size)
        self._add_param_row(param_frame, "Cache threshold (L2)", "cache_thresh", self.config.label_cache.auto_label_threshold, is_float=True)
        self._add_param_row(param_frame, "DINO model name", "dino_model", self.config.features.model_name, is_str=True)
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

        # Label Only mode — now with built-in resume/sparse support (the requested alternative)
        self.label_only_var = tk.BooleanVar(value=getattr(self.config.tile_annotations, "label_only_default", False))
        ttk.Checkbutton(param_frame, text="Label Only Mode (no A/RED, no DINO — pure labeling + Skip/Resume)",
                        variable=self.label_only_var).pack(anchor="w", pady=int(3*s))

        # Terminal logging: gate high-volume repeating prints (per-tile / per-frame / cache hits)
        self.terminal_logging_var = tk.BooleanVar(
            value=bool(getattr(self.config.gui, "terminal_logging", True))
        )
        set_terminal_logging(self.terminal_logging_var.get())

        def _sync_terminal_logging(*_):
            enabled = bool(self.terminal_logging_var.get())
            set_terminal_logging(enabled)
            try:
                self.config.gui.terminal_logging = enabled
                # Also quiet/unquiet original A_REDIN internal VERBOSE_FLAGS
                self.config.ared.verbose_flags = [1, 5, 6] if enabled else []
                adapter = getattr(self.controller, "ared_adapter", None)
                if adapter is not None and getattr(adapter, "ared", None) is not None:
                    adapter.ared.verbose_flags = list(self.config.ared.verbose_flags)
            except Exception:
                pass

        ttk.Checkbutton(
            param_frame,
            text="Terminal logging (repeating per-tile / progress prints)",
            variable=self.terminal_logging_var,
            command=_sync_terminal_logging,
        ).pack(anchor="w", pady=int(3*s))
        self.terminal_logging_var.trace_add("write", _sync_terminal_logging)

        # Running metrics logging (every N tiles → runs/<run_id>/)
        ttk.Separator(param_frame, orient="horizontal").pack(fill="x", pady=int(4*s))
        metrics_log_frame = ttk.LabelFrame(param_frame, text="Running Metrics Log (paper QP/RR/F1)")
        metrics_log_frame.pack(fill="x", pady=int(2*s))
        ml0 = getattr(self.config, "metrics_logging", None)
        self.metrics_log_enabled_var = tk.BooleanVar(value=bool(getattr(ml0, "enabled", True)))
        ttk.Checkbutton(
            metrics_log_frame,
            text="Save running metrics every N tiles (and on stop/finish)",
            variable=self.metrics_log_enabled_var,
        ).pack(anchor="w", pady=int(2*s))
        self._add_param_row(
            metrics_log_frame,
            "Checkpoint every N tiles",
            "metrics_ckpt_every",
            int(getattr(ml0, "checkpoint_every", 5000) or 5000),
        )
        self._add_param_row(
            metrics_log_frame,
            "Runs output dir",
            "metrics_out_dir",
            str(getattr(ml0, "output_dir", "runs") or "runs"),
            is_str=True,
        )
        self.metrics_ckpt_on_video_var = tk.BooleanVar(
            value=bool(getattr(ml0, "checkpoint_on_video_end", True))
        )
        ttk.Checkbutton(
            metrics_log_frame,
            text="Also checkpoint when each video ends",
            variable=self.metrics_ckpt_on_video_var,
        ).pack(anchor="w", pady=int(1*s))
        self._metrics_run_line_var = tk.StringVar(value="No metrics run yet.")
        ttk.Label(
            metrics_log_frame,
            textvariable=self._metrics_run_line_var,
            font=("TkDefaultFont", int(9 * self.ui_scale)),
            relief="sunken",
            wraplength=int(260 * self.ui_scale),
        ).pack(fill="x", pady=int(2*s))

        # Initialize overlap checkbox from config (if stride < tile size, or explicit overlap > 0)
        tcfg0 = self.config.tiling
        initial_overlap = bool((tcfg0.stride_x is not None and tcfg0.stride_x < tcfg0.tile_width) or
                               (tcfg0.stride_y is not None and tcfg0.stride_y < tcfg0.tile_height) or
                               getattr(tcfg0, "overlap_x", 0) > 0 or
                               getattr(tcfg0, "overlap_y", 0) > 0)
        self.overlap_enabled_var.set(initial_overlap)
        ttk.Button(param_frame, text="Review / Edit Past Labels...", command=self._open_review_window).pack(fill="x", pady=int(3*s))
        ttk.Button(param_frame, text="Multi-Frame Browser (scroll many frames + select to label)", 
                   command=self._open_multi_frame_browser).pack(fill="x", pady=int(3*s))

        # Expanded DB management (see plan)
        db_mgmt = ttk.LabelFrame(param_frame, text="DB Management")
        db_mgmt.pack(fill="x", pady=int(4*s))
        ttk.Button(db_mgmt, text="Save / Flush DB", command=self._save_tile_annotations).pack(fill="x", pady=1)
        ttk.Button(db_mgmt, text="Load / Switch DB...", command=self._load_tile_annotations).pack(fill="x", pady=1)
        ttk.Button(db_mgmt, text="New DB...", command=self._new_tile_annotations).pack(fill="x", pady=1)
        ttk.Button(db_mgmt, text="Clone Current DB...", command=self._clone_tile_annotations).pack(fill="x", pady=1)
        ttk.Button(db_mgmt, text="Vacuum (compact)", command=self._vacuum_tile_annotations).pack(fill="x", pady=1)
        ttk.Button(db_mgmt, text="Bulk Edit / Remove Labels...", command=self._open_bulk_label_ops).pack(fill="x", pady=1)

        # Live DB info
        self._db_info_var = tk.StringVar(value="No annotation DB")
        ttk.Label(db_mgmt, textvariable=self._db_info_var, font=("TkDefaultFont", int(9*self.ui_scale)), 
                  relief="sunken", wraplength=220).pack(fill="x", pady=2)

        # Right side: stats + discovered classes
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=int(4*s))

        stats_frame = ttk.LabelFrame(right, text="Live Stats")
        stats_frame.pack(fill="x")

        self.stats_text = tk.Text(stats_frame, height=8, width=int(60 * min(s, 1.5)), state="disabled", font=("TkDefaultFont", int(11*s)))
        self.stats_text.pack(fill="x", padx=4, pady=4)

        # --- Metrics box (Query Precision + Relevant Recall as defined in the A/RED papers) ---
        # See IJSC_2026-1.pdf and SPIE_IVSP_2026.pdf for exact definitions.
        # Scrollable so the full expanded audit is readable.
        metrics_frame = ttk.LabelFrame(right, text="Metrics (Query Precision / Relevant Recall - RR includes first appearances)")
        metrics_frame.pack(fill="x", pady=(int(6*s), 0))

        # Container + vertical scrollbar for the (potentially very long) detailed audit
        text_container = ttk.Frame(metrics_frame)
        text_container.pack(fill="x", padx=4, pady=4)

        self.metrics_text = tk.Text(text_container, height=12, width=int(75 * min(s, 1.5)), state="disabled",
                                    wrap="word", font=("TkDefaultFont", int(10*s)), bg="#f8f8f8")
        self.metrics_text.pack(side="left", fill="both", expand=True)

        ysb = ttk.Scrollbar(text_container, orient="vertical", command=self.metrics_text.yview)
        ysb.pack(side="right", fill="y")
        self.metrics_text.configure(yscrollcommand=ysb.set)

        self.metrics_text.insert("1.0", "Click 'Compute from DB (last video)' after a run.\n"
                                        "EXPANDED + SCROLLABLE: every datapoint - total relevant tiles, total relevant queried, total A/RED queries (caches count as user queries), full work for QP/RR (RR includes first appearances of classes per paper positives def).")
        self.metrics_text.config(state="disabled")

        btn_row = ttk.Frame(metrics_frame)
        btn_row.pack(fill="x", padx=4, pady=2)
        ttk.Button(btn_row, text="Compute from DB (last video)", command=self._compute_metrics_from_db).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self._clear_metrics_display).pack(side="left", padx=4)

        self._last_metrics: Dict[str, Any] = {}
        self._metrics_auto_updated = False

        classes_frame = ttk.LabelFrame(right, text="Discovered Classes (active annotation DB + this run)")
        classes_frame.pack(fill="both", expand=True, pady=(int(6*s), 0))

        class_list_row = ttk.Frame(classes_frame)
        class_list_row.pack(fill="both", expand=True)
        self.class_listbox = tk.Listbox(class_list_row, height=10, font=("TkDefaultFont", int(13 * s)))
        self.class_listbox.pack(fill="both", expand=True, side="left")
        ysb = ttk.Scrollbar(class_list_row, orient="vertical", command=self.class_listbox.yview)
        ysb.pack(side="right", fill="y")
        self.class_listbox.configure(yscrollcommand=ysb.set)

        class_btn_row = ttk.Frame(classes_frame)
        class_btn_row.pack(fill="x", padx=4, pady=2)
        ttk.Button(
            class_btn_row,
            text="Remove selected class…",
            command=self._remove_selected_discovered_class,
        ).pack(side="left")
        ttk.Button(
            class_btn_row,
            text="Prune unused from list",
            command=self._prune_discovered_classes_and_refresh,
        ).pack(side="left", padx=4)

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

            # Overlap → stride computation (new)
            # stride = tile - overlap when enabled and overlap > 0. Otherwise stride = tile (non-overlap).
            try:
                tw = self.config.tiling.tile_width
                th = self.config.tiling.tile_height
                ox = max(0, int(getattr(self, "_overlap_x_var").get() or 0))
                oy = max(0, int(getattr(self, "_overlap_y_var").get() or 0))
                enabled = bool(self.overlap_enabled_var.get())
                if enabled and (ox > 0 or oy > 0):
                    self.config.tiling.stride_x = max(1, tw - ox)
                    self.config.tiling.stride_y = max(1, th - oy)
                    self.config.tiling.overlap_x = ox
                    self.config.tiling.overlap_y = oy
                else:
                    # Force clean non-overlapping
                    self.config.tiling.stride_x = tw
                    self.config.tiling.stride_y = th
                    self.config.tiling.overlap_x = 0
                    self.config.tiling.overlap_y = 0
            except Exception:
                # Fall back to safe non-overlap if anything is malformed
                self.config.tiling.stride_x = self.config.tiling.tile_width
                self.config.tiling.stride_y = self.config.tiling.tile_height
                self.config.tiling.overlap_x = 0
                self.config.tiling.overlap_y = 0

            self.config.ared.kappa = float(getattr(self, "_kappa_var").get())
            self.config.ared.l_buf_size = int(getattr(self, "_buf_size_var").get())
            self.config.label_cache.auto_label_threshold = float(getattr(self, "_cache_thresh_var").get())
            self.config.features.model_name = getattr(self, "_dino_model_var").get()

            # NEW exact annotation DB
            self.config.tile_annotations.db_path = getattr(self, "_tile_ann_db_var").get()
            self.config.tile_annotations.edit_mode_default = self.edit_mode_var.get()
            self.config.ared.data_augmentation_enabled = self.aug_var.get()
            self.config.tile_annotations.label_only_default = self.label_only_var.get()

            # Running metrics log (every N tiles → runs/)
            if not hasattr(self.config, "metrics_logging") or self.config.metrics_logging is None:
                from .config import MetricsLoggingConfig
                self.config.metrics_logging = MetricsLoggingConfig()
            self.config.metrics_logging.enabled = bool(self.metrics_log_enabled_var.get())
            self.config.metrics_logging.checkpoint_every = max(
                1, int(getattr(self, "_metrics_ckpt_every_var").get() or 5000)
            )
            self.config.metrics_logging.output_dir = str(
                getattr(self, "_metrics_out_dir_var").get() or "runs"
            ).strip() or "runs"
            self.config.metrics_logging.checkpoint_on_video_end = bool(
                self.metrics_ckpt_on_video_var.get()
            )

            # Terminal logging (repeating prints + optional A_REDIN VERBOSE_FLAGS)
            if hasattr(self, "terminal_logging_var"):
                self.config.gui.terminal_logging = bool(self.terminal_logging_var.get())
                set_terminal_logging(self.config.gui.terminal_logging)
                self.config.ared.verbose_flags = (
                    [1, 5, 6] if self.config.gui.terminal_logging else []
                )
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

        # Fresh per-run class tracking. Class list shows: selected annotation DB + this run only
        # (not embedding-cache history and not leftover A/RED labels from a previous model load).
        self.discovered_classes = set()
        self.run_class_counts = {}

        # Make sure A/RED query counts for this run start at zero.
        if self.controller and getattr(self.controller, 'ared_adapter', None):
            try:
                self.controller.ared_adapter.query_counts = {}
            except Exception:
                pass

        # Prepare label store (embedding similarity) — used for auto-labeling, NOT for class list names
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
            db = TileAnnotationDB(db_path=ann_path)
            self._set_active_tile_db(db, ann_path)
        else:
            self.tile_db = None
            self.controller.set_tile_database(None)
            self.annotation_manager = None
            self.controller.set_annotation_manager(None)

        # Refresh after DB is wired so list reflects only this DB
        try:
            self._refresh_class_list()
        except Exception:
            pass

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

        self._metrics_auto_updated = False

    def _stop(self):
        """Stop the worker and re-enable Start so the user can restart without restarting the whole program."""
        self.controller.stop()
        self.start_btn.config(state="normal")
        self.status_var.set("Stopped. You can change parameters/videos and press Start again.")
        if self.controller.stats:
            self._update_stats_display(self.controller.stats)

    def _on_closing(self):
        """Clean shutdown handler to avoid 'terminate called without an active exception' and VSCode crash reports.

        Common causes on Linux + OpenCV + threads + sqlite + Tk:
        - Worker thread still alive when Tk exits
        - Unreleased cv2.VideoCapture or sqlite connections
        - Pending after() jobs firing during destruction
        """
        print("[MainWindow] Shutdown requested (WM_DELETE or Exit).")
        try:
            # Unbind global mousewheel handlers installed for main-window scrolling
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    self.root.unbind_all(seq)
                except Exception:
                    pass

            # 1. Stop worker (signals events + drains label queue + joins)
            if hasattr(self, "controller") and self.controller:
                try:
                    self.controller.stop(join_timeout=2.0)
                except Exception:
                    pass

            # Cancel any pending stat-poller / after jobs
            if getattr(self, "_stats_job", None):
                try:
                    self.root.after_cancel(self._stats_job)
                except Exception:
                    pass
                self._stats_job = None

            # 2. Close DBs (this is critical for the sqlite WAL + "terminate" symptom)
            if getattr(self, "tile_db", None):
                try:
                    self.tile_db.close()
                except Exception:
                    pass
            if getattr(self, "annotation_manager", None):
                try:
                    self.annotation_manager.close()
                except Exception:
                    pass

            # 3. Flush label cache
            if getattr(self, "label_store", None):
                try:
                    self.label_store.save()
                except Exception:
                    pass

            # 4. Destroy child windows first (they may hold video caps or threads)
            for attr in ("_labeling_win", "_review_win", "_multi_frame_browser"):
                win = getattr(self, attr, None)
                if win is not None:
                    try:
                        if hasattr(win, "destroy"):
                            win.destroy()
                    except Exception:
                        pass

            # 5. Give Tk a tiny moment to process destructions
            try:
                self.root.update_idletasks()
            except Exception:
                pass

        except Exception as e:
            print("[MainWindow] Error during clean shutdown:", e)
        finally:
            try:
                self.root.quit()
            except Exception:
                pass
            try:
                self.root.destroy()
            except Exception:
                pass

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
                self._refresh_db_info()
                messagebox.showinfo("Tile Annotations", f"Annotation DB saved. Total entries: {count}")
            except Exception as e:
                messagebox.showerror("Tile Annotations", f"Save failed: {e}")
        else:
            messagebox.showwarning("Tile Annotations", "No annotation DB active.")

    # ------------------------------------------------------------------
    # Metrics display (Query Precision + Relevant Recall)
    # References the exact definitions in IJSC_2026-1.pdf and SPIE_IVSP_2026.pdf
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Navigation proxies called by the LabelingDialog when in Label Only mode.
    # These forward to the controller's navigation API.
    # ------------------------------------------------------------------
    def _nav_prev_from_dialog(self):
        """Forward nav request. The dialog side is responsible for satisfying
        the current LabelRequest (via set_skip) so the worker unblocks.
        We only clear our pending ref defensively and signal the controller.
        Navigation is ONLY used in label-only mode; A/RED path is unaffected.
        """
        if getattr(self, '_pending_label_request', None):
            try:
                self._pending_label_request.set_skip()
            except Exception:
                pass
            self._pending_label_request = None
        # Also clear dialog's view if present
        if hasattr(self, '_labeling_win') and getattr(self._labeling_win, 'current_req', None):
            try:
                self._labeling_win.current_req = None
            except Exception:
                pass
        if self.controller:
            self.controller.label_only_prev()

    def _nav_next_from_dialog(self):
        if getattr(self, '_pending_label_request', None):
            try:
                self._pending_label_request.set_skip()
            except Exception:
                pass
            self._pending_label_request = None
        if hasattr(self, '_labeling_win') and getattr(self._labeling_win, 'current_req', None):
            try:
                self._labeling_win.current_req = None
            except Exception:
                pass
        if self.controller:
            self.controller.label_only_next()

    def _nav_jump_from_dialog(self, frame: int):
        if getattr(self, '_pending_label_request', None):
            try:
                self._pending_label_request.set_skip()
            except Exception:
                pass
            self._pending_label_request = None
        if hasattr(self, '_labeling_win') and getattr(self._labeling_win, 'current_req', None):
            try:
                self._labeling_win.current_req = None
            except Exception:
                pass
        if self.controller:
            self.controller.label_only_jump_to_frame(frame)

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
            self._set_metrics_content("No metrics computed yet.\nRun A/RED (with DB labels) or use the button.")

    def _display_metrics_error(self, msg: str):
        if hasattr(self, 'metrics_text'):
            self._set_metrics_content(f"Error: {msg}")

    def _set_metrics_content(self, content: str):
        """Update metrics Text while preserving current scroll position as best as possible.
        This stops the view from jumping back to the top on every refresh."""
        if not hasattr(self, 'metrics_text'):
            return
        try:
            yview = self.metrics_text.yview()
            top_frac = yview[0] if yview else 0.0
        except Exception:
            top_frac = 0.0

        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", content)
        self.metrics_text.config(state="disabled")

        try:
            self.metrics_text.yview_moveto(top_frac)
        except Exception:
            pass

    def _refresh_metrics_display(self, result: Dict[str, Any]):
        if not hasattr(self, 'metrics_text'):
            return
        # Run parameters (kappa, tile size, frame stride, DB, model, etc.) if present
        rp = result.get("run_params") or {}
        rp_lines = []
        if rp:
            ts = rp.get("tile_size")
            tile_str = f"{ts[0]}x{ts[1]}" if isinstance(ts, (list, tuple)) and len(ts) == 2 else str(ts or "?")
            rp_lines = [
                "0. RUN PARAMETERS (reproducibility):",
                f"   kappa={rp.get('kappa', '?')}   tile_size={tile_str}   frame_stride={rp.get('frame_stride', '?')}",
                f"   stride=({rp.get('stride_x', '?')},{rp.get('stride_y', '?')})",
                f"   annotation_db={rp.get('annotation_db') or rp.get('db_path', '?')}",
                f"   dino_model={rp.get('dino_model', '?')}   l_buf={rp.get('l_buf_size', '?')} k={rp.get('k_comp_pts', '?')}",
                f"   label_cache={rp.get('label_cache_enabled', '?')} (thresh={rp.get('label_cache_threshold', '?')})",
                f"   data_aug={rp.get('data_augmentation_enabled', '?')}   edit_mode={rp.get('edit_mode', '?')} label_only={rp.get('label_only_mode', '?')}",
                "---------------------------------------------------------------------",
            ]

        lines = [
            "========== FULL A/RED METRICS AUDIT (EVERY SINGLE DATAPOINT) ==========",
            f"Video: {result.get('video', '?')}",
            f"QP: {result.get('query_precision', 0):.4f}    RR: {result.get('relevant_recall', 0):.4f}    F1: {result.get('f1_score', 0):.4f}",
            f"Classes discovered (A/RED queried this run / unique in run): {result.get('classes_discovered_x_of_y', '?')}",
            f"Total queries A/RED made (CACHE QUERIES COUNT AS USER QUERIES): {result.get('ared_queries_made', result.get('n_actual_queries', 0))}",
            f"Total stream tiles seen: {result.get('total_stream_tiles', result.get('total_points', 0))}",
            f"Total labeled (people-tagged) tiles in DB: {result.get('n_labeled', 0)}",
        ]
        lines.extend(rp_lines)
        lines.append("---------------------------------------------------------------------")

        audit = result.get("detailed_breakdown") or result.get("audit", {})
        if audit:
            lines.append("1. CORE COUNTS (what you asked for):")
            lines.append(f"   TOTAL_TILES_ACTUALLY_SENT_TO_ARED          = {audit.get('TOTAL_TILES_ACTUALLY_SENT_TO_ARED_THIS_RUN', '?')}")
            lines.append(f"   TOTAL_RELEVANT_TILES                       = {audit.get('TOTAL_RELEVANT_TILES', audit.get('total_relevant_tiles', '?'))}")
            lines.append(f"   TOTAL_RELEVANT_TILES_QUERIED (Total relevant queried) = {audit.get('TOTAL_RELEVANT_TILES_QUERIED', audit.get('total_relevant_tiles_queried', '?'))}")
            lines.append(f"   TOTAL_QUERIES_ARED_MADE (incl. all cache)  = {audit.get('TOTAL_QUERIES_ARED_MADE', result.get('ared_queries_made', '?'))}")
            lines.append(f"   TOTAL_LABELED_TILES_IN_DB                  = {audit.get('TOTAL_LABELED_TILES_IN_DB', result.get('n_labeled', '?'))}")
            lines.append(f"   CLASSES_DISCOVERED (A/RED queried / unique in run) = {audit.get('CLASSES_DISCOVERED_X_Y', result.get('classes_discovered_x_of_y', '?'))}")
            lines.append("")
            lines.append("2. PER-CLASS HUMAN TAGGED (from DB annotations):")
            for lab, cnt in sorted((audit.get("CLASS_COUNTS") or {}).items(), key=lambda x: -x[1]):
                r = (audit.get("RELEVANT_CLASS_COUNTS") or {}).get(lab, 0)
                lines.append(f"   {lab}: {cnt} total labeled ({r} relevant)")
            lines.append("")
            lines.append("3. FIRST OCCURRENCES (new-class positives part of should):")
            for lab, fr in sorted((audit.get("FIRST_OCCURRENCE_BY_CLASS") or {}).items(), key=lambda x:x[1]):
                lines.append(f"   {lab} first seen at frame {fr}")
            lines.append("")
            lines.append("4. SHOULD_QUERY GT POSITIVES (paper definition):")
            lines.append(f"   N_SHOULD_QUERY_TOTAL = {audit.get('N_SHOULD_QUERY_TOTAL', audit.get('should_query_total', '?'))}")
            lines.append(f"   N_FIRST_OF_CLASS     = {audit.get('N_FIRST_OF_CLASS', '?')}")
            lines.append(f"   N_RELEVANT_CLASS_SAMPLES = {audit.get('RELEVANT_CLASS_SAMPLES', audit.get('RELEVANT_POSITIVES_FOR_RR', '?'))}")
            lines.append("   (Note: RR uses ALL positives per paper: first appearances of any class + samples from relevant classes. See section 6.)")
            lines.append("")
            lines.append("5. A/RED QUERY OUTCOMES (broad positives for QP and RR):")
            lines.append(f"   TP (queried a positive) = {result.get('tp', audit.get('TP', '?'))}")
            lines.append(f"   FP (queried but not a positive) = {result.get('fp', audit.get('FP', '?'))}")
            lines.append(f"   FN (positive but not queried) = {result.get('fn', audit.get('FN', '?'))}")
            lines.append("   Note: positives = (first sample of any class) OR (any sample of a relevant-designated class)")
            lines.append("")
            lines.append("   (For reference: relevant-class samples only)")
            lines.append(f"   relevant class samples = {audit.get('RELEVANT_CLASS_SAMPLES', result.get('n_relevant_positives', '?'))}")
            lines.append(f"   relevant TP (queried among them) = {result.get('relevant_tp', audit.get('RELEVANT_TP', '?'))}")
            lines.append(f"   relevant FN = {result.get('relevant_fn', audit.get('RELEVANT_FN', '?'))}")
            lines.append("")
            lines.append("6. EXACT CALCULATIONS (formulas + numbers from papers):")
            lines.append(f"   QP = TP / (TP + FP)   --> {result.get('query_precision', 0)}")
            lines.append(f"   RR = TP / (TP + FN)   --> {result.get('relevant_recall', 0)}")
            lines.append(f"   F1 = 2 * QP * RR / (QP + RR)   --> {result.get('f1_score', 0)}")
            lines.append(f"   Classes discovered (queried by A/RED this run / unique classes in run): {result.get('classes_discovered_x_of_y', '?')}")
            lines.append(f"   (RR includes first appearances of classes + relevant samples)")
            lines.append(f"   (Cache-satisfied A/RED decisions are included in the query count above.)")
            lines.append("")
            bl = audit.get("RANDOM_BASELINE", {})
            lines.append("7. RANDOM BASELINE (from papers at same query count):")
            lines.append(f"   approx QP = relevant_rate = {bl.get('relevant_rate', '?')}")

        if "summary" in result:
            lines.append("")
            lines.append(result["summary"])
        lines.append("======================================================================")
        lines.append("Data sources: DB annotations (human labels) + A/RED decision log (pipeline.queried_identities + stats['ared_queries']). See metrics.py + SPIE_IVSP_2026 / IJSC_2026-1.")

        content = "\n".join(lines)
        self._set_metrics_content(content)

    def _update_metrics_on_finish(self):
        """Called when a run finishes. Tries to auto-compute using the last video + DB."""
        if getattr(self.controller, 'label_only_mode', False):
            # In pure label-only we have no A/RED queries, so just show stats
            n_l = self.controller.stats.get("tiles_processed", 0)
            # Still try to snapshot run params for reproducibility even in label-only
            run_p = getattr(self.controller, "_collect_run_params", lambda: {})()
            disp = {
                "video": self.controller.stats.get("current_video", "?"),
                "query_precision": 0.0,
                "relevant_recall": 0.0,
                "f1_score": 0.0,
                "classes_discovered_x_of_y": "0/0",
                "n_actual_queries": 0,
                "total_points": n_l,
                "n_labeled": n_l,
                "summary": "Label Only run — every tile was presented for human labeling (no A/RED query decisions). All labels are direct people-tagged.",
                "audit": {"total_annotations_in_db": n_l, "first_of_class_count": "N/A (full labeling)", "relevant_tiles_count": "see DB", "should_query_total": "N/A"},
            }
            if run_p:
                disp["run_params"] = run_p
            self._refresh_metrics_display(disp)
            return

        # Normal A/RED run — try to compute real metrics
        self._compute_metrics_from_db()

    def _load_tile_annotations(self):
        path = filedialog.askopenfilename(title="Load / Switch tile annotation DB", filetypes=[("SQLite DB", "*.db"), ("All", "*.*")])
        if not path:
            return
        try:
            if self.tile_db:
                try:
                    self.tile_db.close()
                except Exception:
                    pass
            db = TileAnnotationDB(db_path=path)
            self._set_active_tile_db(db, path)
            # Warn if current tile size has no data here
            self._warn_on_tile_size_mismatch(path)
            messagebox.showinfo("Tile Annotations", f"Switched to DB with {len(self.tile_db)} entries.")
        except Exception as e:
            messagebox.showerror("Load/Switch", str(e))

    def _refresh_db_info(self):
        if not hasattr(self, "_db_info_var"):
            return
        if self.tile_db:
            try:
                summ = self.tile_db.get_db_summary()
                sizes = ", ".join(f"{w}x{h}" for w, h in summ.get("tile_sizes", [])) or "—"
                name = Path(summ["path"]).name
                self._db_info_var.set(f"{name} | {summ['total_entries']} rows | {summ['num_videos']} vids | sizes: {sizes}")
            except Exception as e:
                self._db_info_var.set(f"DB info error: {e}")
        else:
            self._db_info_var.set("No annotation DB active")

    def _set_active_tile_db(self, db: "TileAnnotationDB", path: str):
        """Central helper: wires DB + manager + controller + config + UI state.
        Simplifies duplication across load/new/clone/start paths.
        Now also propagates current stride for overlap-aware scoping.
        """
        self.tile_db = db
        self.controller.set_tile_database(db)
        self.annotation_manager = AnnotationManager(db)
        tw = self.config.tiling.tile_width
        th = self.config.tiling.tile_height
        sx = getattr(self.config.tiling, 'stride_x', None)
        sy = getattr(self.config.tiling, 'stride_y', None)
        self.annotation_manager.set_scope(tile_size=(tw, th), stride=(sx, sy) if (sx or sy) else None)
        self.controller.set_annotation_manager(self.annotation_manager)
        self.config.tile_annotations.db_path = path
        if hasattr(self, "_tile_ann_db_var"):
            self._tile_ann_db_var.set(path)
        self._refresh_db_info()
        # Class list must track the newly selected DB only (drop other-DB names from view).
        try:
            self._refresh_class_list()
        except Exception:
            pass

    def _warn_on_tile_size_mismatch(self, db_path: str):
        """If we have a current tiling size, check if the loaded DB has any annotations at that size."""
        try:
            if not self.tile_db:
                return
            tw = self.config.tiling.tile_width
            th = self.config.tiling.tile_height
            sizes = self.tile_db.get_tile_sizes_for_video("")  # across all videos
            # get all sizes
            cur = self.tile_db.conn.cursor()
            cur.execute("SELECT DISTINCT tile_width, tile_height FROM annotations")
            all_sizes = [(int(r[0]), int(r[1])) for r in cur.fetchall()]
            if all_sizes and (tw, th) not in all_sizes:
                sizes_str = ", ".join(f"{w}x{h}" for w,h in all_sizes[:5])
                messagebox.showwarning(
                    "Tile Size Mismatch",
                    f"Current GUI tile size is {tw}x{th}.\n"
                    f"This DB contains annotations at other sizes: {sizes_str}{'...' if len(all_sizes)>5 else ''}\n\n"
                    "Browsers and lookups will be scoped to matching sizes only.\n"
                    "You can still label at the current size (new entries will be added)."
                )
        except Exception:
            pass

    def _new_tile_annotations(self):
        """Create a fresh empty DB and switch to it."""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Create New Annotation DB",
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db")],
            initialfile="drone_tile_annotations_new.db"
        )
        if not path:
            return
        try:
            if self.tile_db:
                try:
                    self.tile_db.close()
                except Exception:
                    pass
            # Creating by opening will make the tables
            db = TileAnnotationDB(db_path=path)
            self._set_active_tile_db(db, path)
            messagebox.showinfo("New DB", f"Created and switched to new DB: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("New DB", str(e))

    def _clone_tile_annotations(self):
        """Clone the current DB to a new file and switch to the clone."""
        from tkinter import filedialog
        if not self.tile_db:
            messagebox.showwarning("Clone", "No active DB to clone.")
            return
        src_path = self.tile_db.db_path
        default_name = f"{Path(src_path).stem}_clone.db"
        dest = filedialog.asksaveasfilename(
            title="Clone DB to...",
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db")],
            initialfile=default_name
        )
        if not dest:
            return
        try:
            if self.tile_db:
                try:
                    self.tile_db.close()
                except Exception:
                    pass
            # Use the class helper
            new_db = TileAnnotationDB.clone(src_path, dest, open_after=True)
            self._set_active_tile_db(new_db, dest)
            messagebox.showinfo("Clone", f"Cloned to {Path(dest).name} and switched.\nOriginal is untouched.")
        except Exception as e:
            messagebox.showerror("Clone DB", str(e))
            # Try to reopen original
            try:
                self.tile_db = TileAnnotationDB(db_path=src_path)
                self.controller.set_tile_database(self.tile_db)
            except Exception:
                pass

    def _vacuum_tile_annotations(self):
        if not self.tile_db:
            messagebox.showwarning("Vacuum", "No active DB.")
            return
        if not messagebox.askyesno("Vacuum", "Compact the DB (removes deleted space)? Safe but may take a moment."):
            return
        try:
            self.tile_db.vacuum()
            self._refresh_db_info()
            messagebox.showinfo("Vacuum", "DB compacted.")
        except Exception as e:
            messagebox.showerror("Vacuum", str(e))

    def _open_bulk_label_ops(self):
        """Open dialog for mass editing/removing/changing labels with specificity.
        Uses AnnotationManager + AnnotationFilter for precise control (by video, size, labels, frames, relevance).
        """
        if not self.annotation_manager:
            try:
                path = getattr(self.config.tile_annotations, 'db_path', 'drone_tile_annotations.db')
                db = TileAnnotationDB(db_path=path)
                self._set_active_tile_db(db, path)
            except Exception as e:
                messagebox.showerror("Bulk Ops", f"No DB: {e}")
                return

        BulkLabelOpsDialog(
            self.root,
            self.annotation_manager,
            self.config,
            ui_scale=self.ui_scale,
            on_done=self._on_bulk_label_ops_done,
        )

    # ------------------------------------------------------------------
    # NEW: Review / Edit past exact labels (works across runs and stride changes)
    # ------------------------------------------------------------------
    def _open_review_window(self):
        if not self.tile_db:
            # Try to open/create one from current config
            try:
                path = getattr(self.config.tile_annotations, 'db_path', 'drone_tile_annotations.db')
                db = TileAnnotationDB(db_path=path)
                self._set_active_tile_db(db, path)
            except Exception as e:
                messagebox.showerror("Review", f"Could not open annotation DB: {e}")
                return

        videos = self.tile_db.list_videos()
        if not videos:
            messagebox.showinfo("Review", "No annotations saved yet. Label some tiles while running A/RED (or load a previous DB).")
            return

        LabelReviewWindow(self.root, self.tile_db, ui_scale=self.ui_scale,
                          annotation_manager=getattr(self, 'annotation_manager', None))

    def _open_multi_frame_browser(self):
        """Open the new multi-frame visual browser.

        Shows an adjustable number of frames (via columns + scrolling) as thumbnails.
        Click to select a frame, then use the actions to explore its tiles and label them.
        This is intended as a fast visual way to scroll/review labeled content and
        jump into per-frame labeling, complementary to the sequential Label Only mode.
        """
        if not self.tile_db:
            try:
                path = getattr(self.config.tile_annotations, 'db_path', 'drone_tile_annotations.db')
                db = TileAnnotationDB(db_path=path)
                self._set_active_tile_db(db, path)
            except Exception as e:
                messagebox.showerror("Browser", f"Could not open annotation DB: {e}")
                return

        # Always allow opening the browser (even with zero annotations) so the user can
        # browse any video + start labeling frames using the stride. The internal "Browse any video"
        # button and the video combo (if annotations exist) will work.
        videos = self.tile_db.list_videos()
        if not videos:
            # Informational only; do not block opening.
            print("[MultiFrameBrowser] Opening with no prior annotations in DB.")

        MultiFrameLabelBrowser(
            self.root,
            self.tile_db,
            ui_scale=self.ui_scale,
            controller=self.controller,
            main_window=self,
            annotation_manager=getattr(self, 'annotation_manager', None)
        )

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

        vprint(f"[GUI] Received label REQUEST from ARED for tile global={getattr(getattr(req,'tile',None),'global_idx','?')} meta={getattr(req,'meta',{})}")
        self._pending_label_request = req

        # Guard against duplicate req for the exact same tile (shouldn't happen, but prevents "same tile twice")
        tile = getattr(req, 'tile', None)
        gidx = getattr(tile, 'global_idx', -1) if tile else -1
        if gidx != -1 and getattr(self, '_last_queried_global', -1) == gidx:
            # Already handled this tile's query; ignore stale/duplicate
            # CRITICAL: still satisfy the req or the worker thread will block forever on wait().
            # Use cancel/skip — never invent a fake class label like __DUPLICATE__.
            vprint("[GUI]   -> Duplicate request for same global_idx, ignoring (cancelling req to unblock worker).")
            try:
                if hasattr(req, "set_cancelled"):
                    req.set_cancelled(reason="duplicate")
                else:
                    req.set_skip()
            except Exception:
                pass
            return
        self._last_queried_global = gidx

        # Extract sparse labeling hints from the request (used by the alternative efficient label mode)
        meta = getattr(req, 'meta', {}) or {}
        allow_skip = bool(meta.get("allow_skip", False))
        cur_label = meta.get("current_label")
        cur_rel = bool(meta.get("current_relevant", False))

        # Class names: selected annotation DB + classes assigned this GUI run only.
        # Do NOT pull from embedding label_store or A/RED known labels (those mix other runs/models).
        classes = self._collect_scoped_class_names()

        # Numbers = times A/RED queried for the class during *this run*.
        counts = {}
        if self.controller.ared_adapter:
            counts = self.controller.ared_adapter.get_query_counts() or {}
        for lbl in classes:
            counts.setdefault(lbl, 0)

        # Build relevance map (prefer active annotation DB, then embedding cache)
        class_relevance = dict(getattr(self, 'class_relevance', {}))
        for lbl in classes:
            if lbl not in class_relevance:
                seeded = False
                if hasattr(self.controller, 'tile_db') and self.controller.tile_db:
                    try:
                        rel = self.controller.tile_db.get_class_relevance(lbl)
                        if rel is not None:
                            class_relevance[lbl] = rel
                            seeded = True
                    except Exception:
                        pass
                if not seeded and self.label_store:
                    rel = self.label_store.get_class_relevance(lbl)
                    if rel is not None:
                        class_relevance[lbl] = rel
                        seeded = True
                if not seeded:
                    class_relevance[lbl] = False

        def _assign_cb(label: str, relevant: bool):
            # Notification / UI update path.
            # For normal A/RED queries this records the class.
            # For label-only sparse mode, if the label is the skip sentinel we just ignore it
            # (the actual skip decision was already handled via req.skipped in the processor).
            if label == "__SKIPPED__":
                vprint("[GUI] Skip notification received (no class recorded).")
                self._pending_label_request = None
                return
            # Never record control-plane sentinels as discovered classes
            try:
                from .label_sentinels import is_control_label
                if is_control_label(label):
                    vprint(f"[GUI] Ignoring control-sentinel notification '{label}' (not a real class).")
                    self._pending_label_request = None
                    return
            except Exception:
                pass
            vprint(f"[GUI] Label SUBMITTED from dialog: '{label}' (relevant={relevant})")
            self._pending_label_request = None
            self.discovered_classes.add(label)
            # Track count for *this run only* so the class boxes start near zero instead of full DB history
            # Note: the A/RED query count is tracked inside the adapter when the decision was made.
            # We only need to remember the name for immediate clickability.
            self.class_relevance[label] = relevant
            self._refresh_class_list()
            self.status_var.set(f"Last label assigned: {label} (relevant={relevant})")
            vprint("[GUI] Dialog back to WAITING state for next A/RED query (worker continues processing non-queried tiles in background).")

        # Persistent window: create once, then update for new requests
        if not hasattr(self, '_labeling_win') or not self._labeling_win.winfo_exists():
            vprint("[GUI] Creating new persistent LabelingDialog for this query.")
            self._labeling_win = LabelingDialog(
                self.root,
                req,
                known_classes=classes,
                on_assign=_assign_cb,
                class_counts=counts,
                ui_scale=getattr(self, 'ui_scale', 1.6),
                class_relevance=class_relevance,
                allow_skip=allow_skip,
                current_label=cur_label,
                current_relevant=cur_rel,
            )
        else:
            vprint("[GUI] Updating existing persistent LabelingDialog with new query tile.")
            self._labeling_win.set_current_request(
                req,
                known_classes=classes,
                class_counts=counts,
                class_relevance=class_relevance,
                allow_skip=allow_skip,
                current_label=cur_label,
                current_relevant=cur_rel,
            )

        # Attach reference so dialog can call back for Label-Only navigation (Prev/Next/Jump).
        # We do NOT use .master because the dialog is parented to the raw tk.Tk root.
        # This keeps label-only nav fully separate from A/RED query flow (which never
        # enables allow_skip or creates the nav buttons/binds).
        if hasattr(self, '_labeling_win') and self._labeling_win:
            self._labeling_win.main_window = self

    def _collect_scoped_class_names(self) -> List[str]:
        """Class names for dialogs / main list: active annotation DB + this-run discoveries only.

        Explicitly excludes:
          - embedding label_store history (often from other sessions/DBs)
          - A/RED adapter known labels (loaded model / prior run state)
        so the UI does not show classes from unrelated runs.
        """
        names: set = set()
        # Prefer the GUI-owned active DB; fall back to controller helper if needed.
        try:
            if getattr(self, "tile_db", None) is not None:
                for lbl in self.tile_db.get_all_labels() or []:
                    if lbl:
                        names.add(lbl)
            elif hasattr(self.controller, "get_labels_from_annotation_db"):
                for lbl in self.controller.get_labels_from_annotation_db() or []:
                    if lbl:
                        names.add(lbl)
        except Exception:
            pass
        for lbl in getattr(self, "discovered_classes", set()) or set():
            if lbl:
                names.add(lbl)
        # Filter control sentinels if any slipped through
        try:
            from .label_sentinels import is_control_label, is_persistable_label
            names = {n for n in names if is_persistable_label(n) and not is_control_label(n)}
        except Exception:
            pass
        return sorted(names)

    def _db_label_set(self) -> set:
        """Unique labels currently present in the active annotation DB."""
        names: set = set()
        try:
            if getattr(self, "tile_db", None) is not None:
                for lbl in self.tile_db.get_all_labels() or []:
                    if lbl:
                        names.add(lbl)
        except Exception:
            pass
        return names

    def _count_annotations_for_label(self, label: str) -> int:
        if not label or not getattr(self, "tile_db", None):
            return 0
        try:
            if hasattr(self.tile_db, "count_by_label"):
                return int(self.tile_db.count_by_label(label))
            # Fallback: scan
            n = 0
            for v in self.tile_db.list_videos() or []:
                for ann in self.tile_db.get_annotations_for_video(v) or []:
                    if ann.get("label") == label:
                        n += 1
            return n
        except Exception:
            return 0

    def _prune_discovered_classes(self) -> int:
        """Drop session-only class names that no longer exist in the annotation DB.

        Returns how many names were removed from discovered_classes.
        Does not delete A/RED in-memory cluster state (out of scope for UI prune).
        """
        if not hasattr(self, "discovered_classes") or not self.discovered_classes:
            return 0
        db_labels = self._db_label_set()
        before = set(self.discovered_classes)
        # Keep only names still backed by at least one annotation row
        self.discovered_classes = {n for n in before if n in db_labels}
        removed = before - self.discovered_classes
        for n in removed:
            self.class_relevance.pop(n, None)
        return len(removed)

    def _prune_discovered_classes_and_refresh(self):
        n = self._prune_discovered_classes()
        self._refresh_class_list()
        self._refresh_db_info()
        if n:
            self.status_var.set(f"Pruned {n} unused class name(s) from the list (no DB annotations).")
        else:
            self.status_var.set("No unused class names to prune (all listed names still exist in the DB).")

    def _on_bulk_label_ops_done(self):
        """After bulk reassign/delete: prune ghost class names and refresh lists."""
        self._prune_discovered_classes()
        self._refresh_class_list()
        self._refresh_db_info()

    def _remove_selected_discovered_class(self):
        """Remove a discovered class from the UI list; optionally delete its DB annotations.

        - 0 annotations: drop from session discovered_classes immediately.
        - N>0 annotations: confirm bulk-delete of those rows, then prune.
        Also best-effort cleans the embedding label_store cache for that name.
        """
        sel = self.class_listbox.curselection()
        if not sel:
            messagebox.showinfo("Remove class", "Select a class in the Discovered Classes list first.")
            return
        idx = int(sel[0])
        labels_map = getattr(self, "_class_listbox_labels", None) or []
        if 0 <= idx < len(labels_map):
            label = labels_map[idx]
        else:
            # Fallback parse of display text: "name", "name (n)", "name [R]", "name (n) [R]"
            display = self.class_listbox.get(idx)
            label = display.split(" (")[0].split(" [")[0].strip()
        if not label:
            return
        try:
            from .label_sentinels import is_control_label
            if is_control_label(label):
                messagebox.showwarning("Remove class", f"'{label}' is a control sentinel and cannot be managed here.")
                return
        except Exception:
            pass

        n_db = self._count_annotations_for_label(label)
        if n_db > 0:
            ok = messagebox.askyesno(
                "Remove class",
                f"Class '{label}' has {n_db} annotation(s) in the active DB.\n\n"
                f"Delete all of those annotations and remove the class from the list?\n"
                f"(This does not rewrite a live A/RED model buffer.)",
            )
            if not ok:
                return
            try:
                if self.annotation_manager:
                    deleted = self.annotation_manager.bulk_delete(label=label, use_scope=False)
                elif self.tile_db:
                    deleted = self.tile_db.delete_by_filter(label=label)
                else:
                    deleted = 0
                vprint(f"[GUI] Removed class '{label}': deleted {deleted} annotation row(s).")
            except Exception as e:
                messagebox.showerror("Remove class", f"Failed to delete annotations: {e}")
                return
        else:
            ok = messagebox.askyesno(
                "Remove class",
                f"Class '{label}' has no annotations in the active DB "
                f"(session-only / never used for durable labels).\n\n"
                f"Remove it from the Discovered Classes list?",
            )
            if not ok:
                return

        self.discovered_classes.discard(label)
        self.class_relevance.pop(label, None)
        # Best-effort: stop embedding cache from auto-labeling this name
        try:
            if self.label_store and hasattr(self.label_store, "remove_by_label"):
                self.label_store.remove_by_label(label)
        except Exception as e:
            vprint(f"[GUI] label_store cleanup for '{label}' failed: {e}")

        self._prune_discovered_classes()
        self._refresh_class_list()
        self._refresh_db_info()
        self.status_var.set(f"Removed class '{label}' from list" + (f" (and {n_db} DB row(s))" if n_db else ""))

    def _refresh_class_list(self):
        self.class_listbox.delete(0, "end")
        # Scoped to selected annotation DB + classes created/used this run.
        all_labels = self._collect_scoped_class_names()
        # Parallel list of bare class names (index-aligned with listbox rows)
        self._class_listbox_labels: List[str] = list(all_labels)

        # Displayed numbers = how many times A/RED queried for the class during *this run*.
        counts = {}
        if self.controller.ared_adapter:
            counts = self.controller.ared_adapter.get_query_counts() or {}
        for lbl in all_labels:
            counts.setdefault(lbl, 0)

        # Ensure we have a relevance entry (seed from active DB first, then label_store as fallback)
        for lbl in all_labels:
            if lbl not in self.class_relevance:
                seeded = False
                if hasattr(self.controller, 'tile_db') and self.controller.tile_db:
                    try:
                        rel = self.controller.tile_db.get_class_relevance(lbl)
                        if rel is not None:
                            self.class_relevance[lbl] = rel
                            seeded = True
                    except Exception:
                        pass
                if not seeded and self.label_store:
                    rel = self.label_store.get_class_relevance(lbl)
                    if rel is not None:
                        self.class_relevance[lbl] = rel
                        seeded = True
                if not seeded:
                    self.class_relevance[lbl] = False

        for lbl in all_labels:
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
        run_dir = stats.get("metrics_run_dir") or ""
        metrics_line = stats.get("metrics_last_line") or ""
        text = (
            f"Status: {stats.get('status', '?')}   Video: {stats.get('current_video', '')}\n"
            f"Frames: {stats.get('frames_read', 0)}   Tiles: {stats.get('tiles_processed', 0)}\n"
            f"User labels needed (ARED queries): {stats.get('ared_queries', stats.get('user_queries', 0))}   "
            f"Cache hits: {stats.get('cache_hits', 0)}   Actual human dialogs this run: {stats.get('user_queries', 0)}\n"
            f"ARED clusters: {stats.get('ared_clusters', '?')}   Known labels: {stats.get('ared_known_labels', '?')}"
        )
        if metrics_line:
            text += f"\n{metrics_line}"
        if run_dir:
            text += f"\nRun log: {run_dir}"
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

        if hasattr(self, "_metrics_run_line_var"):
            if metrics_line:
                self._metrics_run_line_var.set(metrics_line)
            elif run_dir:
                self._metrics_run_line_var.set(f"Logging → {run_dir}")

        if stats.get("tiles_processed", 0) % 5 == 0 or stats.get("status") in ("finished", "stopped"):
            self._refresh_class_list()

        # Re-enable Start after a run ends so user can restart / load new videos without restarting the program
        status = stats.get("status", "")
        if hasattr(self, "start_btn"):
            if status in ("stopped", "finished", "error", "idle"):
                self.start_btn.config(state="normal")
            elif status in ("running", "paused"):
                self.start_btn.config(state="disabled")

        # Auto-show metrics box when processing finishes (at end of video).
        # Guard so it only happens once; otherwise the repeated poller would
        # keep calling _refresh which deletes+inserts and jumps the scroll view to top.
        if status in ("finished", "stopped") and not getattr(self, '_metrics_auto_updated', False):
            self._update_metrics_on_finish()
            self._metrics_auto_updated = True


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

    def __init__(self, master, tile_db: "TileAnnotationDB", ui_scale: float = 1.6,
                 annotation_manager: Optional["AnnotationManager"] = None):
        super().__init__(master)
        self.tile_db = tile_db
        self.annotation_manager = annotation_manager or (AnnotationManager(tile_db) if tile_db else None)
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
        tw = th = sx = sy = None
        try:
            parent = getattr(self, 'master', None)
            if parent and hasattr(parent, 'config'):
                tcfg = parent.config.tiling
                tw, th = tcfg.tile_width, tcfg.tile_height
                sx, sy = getattr(tcfg, 'stride_x', None), getattr(tcfg, 'stride_y', None)
        except Exception:
            pass
        if self.annotation_manager:
            self.annotation_manager.set_scope(video_path=v, tile_size=(tw, th) if tw else None, stride=(sx, sy) if (sx or sy) else None)
            self.annotations = self.annotation_manager.get_annotations(video=v, use_scope=True)
        else:
            self.annotations = self.tile_db.get_annotations_for_video(v, tile_width=tw, tile_height=th,
                                                                         stride_x=sx, stride_y=sy)
        self.ann_list.delete(0, "end")
        for i, a in enumerate(self.annotations):
            rel_mark = " [R]" if a["relevant"] else ""
            tw = a.get("tile_width", "?")
            th = a.get("tile_height", "?")
            txt = f"f{a['abs_frame']:06d} r{a['tile_row']}c{a['tile_col']} [{tw}x{th}]  {a['label']}{rel_mark}"
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

        sx = ann.get("stride_x")
        sy = ann.get("stride_y")
        try:
            if self.annotation_manager:
                key = TileKey(ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                              ann["tile_width"], ann["tile_height"], stride_x=sx, stride_y=sy)
                self.annotation_manager.db.set_annotation_for_key(key, new_label, new_rel,
                                                                  embedding=None,
                                                                  crop_x=ann.get("crop_x"), crop_y=ann.get("crop_y"))
            else:
                self.tile_db.set_annotation(
                    ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                    ann["tile_width"], ann["tile_height"],
                    new_label, new_rel,
                    embedding=None,
                    crop_x=ann.get("crop_x"), crop_y=ann.get("crop_y")
                )
            self.status_var.set(f"Saved: {new_label} (rel={new_rel})")
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
        sx = ann.get("stride_x")
        sy = ann.get("stride_y")
        try:
            if self.annotation_manager:
                key = TileKey(ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                              ann["tile_width"], ann["tile_height"], stride_x=sx, stride_y=sy)
                self.annotation_manager.db.delete_key(key)
            else:
                self.tile_db.delete_annotation(
                    ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                    ann["tile_width"], ann["tile_height"], stride_x=sx, stride_y=sy
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


# Bulk ops dialog extracted to gui_bulk_dialog.py (imported at top)

# =============================================================================
# MultiFrameLabelBrowser - NEW visual scrollable multi-frame overview + per-frame labeling
# =============================================================================

class MultiFrameLabelBrowser(tk.Toplevel):
    """
    A new GUI for quickly viewing and scrolling through many frames' labels at once.

    Features requested:
    - Adjustable number of frames visible "at once" (via columns slider + scrolling).
    - Scrollable strip/grid of frame thumbnails (whole-frame previews + label counts).
    - Click to select a frame.
    - Then jump into "label creation section for that frame":
        * "Explore Tiles on Frame" opens a grid of ALL tiles on the selected frame
          (generated live with GridTiler).
        * Labeled tiles show their current label + relevant marker.
        * Click any tile → Quick assign dialog (existing classes + new + relevant).
        * Saves directly to the exact TileAnnotationDB.
    - Also supports jumping the main Label Only sequential cursor to the frame.
    - Focused on visual review/scrolling of labels (complements the single-tile
      LabelingDialog and the list-based LabelReviewWindow).

    This is completely independent of the A/RED processing path.
    Only uses the TileAnnotationDB + video re-extraction + optional controller for jumps.
    """

    def __init__(self, master, tile_db: "TileAnnotationDB", ui_scale: float = 1.6,
                 controller: Optional["DroneAREDController"] = None,
                 main_window=None, annotation_manager: Optional["AnnotationManager"] = None):
        super().__init__(master)
        self.tile_db = tile_db
        self.annotation_manager = annotation_manager or (AnnotationManager(tile_db) if tile_db else None)
        self.controller = controller
        self.main_window = main_window
        self.ui_scale = float(ui_scale) if ui_scale else 1.6

        self.current_video: Optional[str] = None
        self.frame_to_anns: Dict[int, List[Dict]] = {}
        self.sorted_frames: List[int] = []
        self.frame_thumbs: Dict[int, ImageTk.PhotoImage] = {}
        self._last_thumb_w: int = 180
        self.selected_frame: Optional[int] = None
        self.frame_cards: Dict[int, tk.Widget] = {}
        self._card_img_labels: Dict[int, tk.Widget] = {}  # frame -> the Label widget holding the image (for async updates)

        # Virtual card management for smooth loading/unloading + bounded resources.
        # We use canvas.create_window with explicit positions instead of a huge growing grid.
        # Only a viewport + small buffer of cards are materialized as live widgets at any time.
        self._card_windows: Dict[int, int] = {}  # fidx -> canvas window item id
        self._materialized_frames: set = set()
        self._virtual_row_height: int = 140  # updated when we know thumb size + text
        self._virtual_cell_width: int = 200
        self._viewport_update_pending: Optional[str] = None  # after() id for debounced viewport refresh

        # For the per-frame tile explorer
        self.current_frame_tiles: List["Tile"] = []
        self.current_frame_tile_labels: Dict[int, Tuple[str, bool]] = {}  # tile_global_in_frame -> (label, rel)
        self._tile_photo_refs: List[ImageTk.PhotoImage] = []  # keep alive
        # Preview size for tiles in "Explore & Label" (upscales small tiles for readability).
        # Configurable via the explorer slider; default 300×300 as requested.
        self.tile_preview_size: int = 300

        # Track known class relevance so we can auto-apply it to tiles (prevents forgetting in long sessions)
        self.class_relevance: dict[str, bool] = {}

        # Virtual display window size (addressable frames). Actual live Tk widgets + PhotoImages
        # are limited to the current viewport + small buffer by the virtualization logic.
        self._displayed_count = 0
        self._last_materialized_thumb = 180
        self._last_thumb_w = 180
        self._last_cols = 4

        # Window setup (this must always run)
        self.title("Multi-Frame Label Browser - Adjustable Scroll + Per-Frame Labeling")
        base_w = int(1280 * min(self.ui_scale, 1.8))
        base_h = int(820 * min(self.ui_scale, 1.8))
        self.geometry(f"{base_w}x{base_h}")
        self.minsize(900, 600)
        self.resizable(True, True)

        self._build_ui()
        self._load_videos()

    def _get_live_tile_size(self):
        """Return (tw, th) preferring live GUI entry boxes (the typed values in MainWindow),
        then .config on main_window, then controller/tiler/config. This is what makes
        the multi-frame browser respect what the user typed without pressing Start.
        """
        tw = th = None
        source = "default"
        try:
            # 1) Live typed boxes in the main control window (highest priority)
            if self.main_window is not None:
                if hasattr(self.main_window, '_tile_w_var'):
                    try:
                        val = str(self.main_window._tile_w_var.get()).strip()
                        if val:
                            tw = int(val)
                            source = "live_var_w"
                    except Exception:
                        pass
                if hasattr(self.main_window, '_tile_h_var'):
                    try:
                        val = str(self.main_window._tile_h_var.get()).strip()
                        if val:
                            th = int(val)
                            source = "live_var_h" if source == "live_var_w" else "live_var"
                    except Exception:
                        pass

            # 2) Main window's config (what was last applied on Start, or initial)
            if (tw is None or th is None) and self.main_window is not None and hasattr(self.main_window, 'config'):
                try:
                    tcfg = self.main_window.config.tiling
                    tw = tw or getattr(tcfg, 'tile_width', None)
                    th = th or getattr(tcfg, 'tile_height', None)
                    if tw and th and source == "default":
                        source = "main_config"
                except Exception:
                    pass

            # 3) Live controller tiler (if one is active and sizes match intent)
            if (tw is None or th is None) and self.controller and getattr(self.controller, 'tiler', None):
                try:
                    tw = tw or getattr(self.controller.tiler, 'tile_w', None)
                    th = th or getattr(self.controller.tiler, 'tile_h', None)
                    if tw and th and source == "default":
                        source = "controller_tiler"
                except Exception:
                    pass

            # 4) Controller config
            if (tw is None or th is None) and self.controller and hasattr(self.controller, 'config'):
                try:
                    tcfg = self.controller.config.tiling
                    tw = tw or getattr(tcfg, 'tile_width', None)
                    th = th or getattr(tcfg, 'tile_height', None)
                    if tw and th and source == "default":
                        source = "controller_config"
                except Exception:
                    pass
        except Exception:
            pass

        if not tw or not th:
            tw = th = 256
            source = "hard_default_256"
        try:
            print(f"[MultiFrameBrowser] live tile size -> {tw}x{th} (source={source})")
        except Exception:
            pass
        return int(tw), int(th)

    def _get_live_stride(self, tw: Optional[int] = None, th: Optional[int] = None) -> Tuple[int, int]:
        """Return (stride_x, stride_y) from live GUI controls / config, falling back to non-overlap (stride == tile).

        If the user has typed overlap values + enabled the checkbox, we compute stride here too
        so browsers and explorers see the correct step without requiring Start.
        """
        sx = sy = None
        source = "default"

        try:
            # Highest priority: live overlap controls on the main window
            if self.main_window is not None:
                enabled = False
                if hasattr(self.main_window, "overlap_enabled_var"):
                    try:
                        enabled = bool(self.main_window.overlap_enabled_var.get())
                    except Exception:
                        pass

                ox = oy = 0
                if hasattr(self.main_window, "_overlap_x_var"):
                    try:
                        ox = max(0, int(str(self.main_window._overlap_x_var.get() or "0").strip()))
                    except Exception:
                        pass
                if hasattr(self.main_window, "_overlap_y_var"):
                    try:
                        oy = max(0, int(str(self.main_window._overlap_y_var.get() or "0").strip()))
                    except Exception:
                        pass

                if enabled and (ox > 0 or oy > 0):
                    # Derive from live overlap values (preferred when checkbox is on)
                    tw0 = tw or self._get_live_tile_size()[0]
                    th0 = th or self._get_live_tile_size()[1]
                    sx = max(1, int(tw0) - ox)
                    sy = max(1, int(th0) - oy)
                    source = "live_overlap"
                    return sx, sy

                # Checkbox OFF (or zero overlap): force non-overlapping stride = tile size.
                # Do NOT fall through to stale config.stride_* left over from a previous
                # overlapped Start — that made "overlap disabled" still use a smaller step.
                if self.main_window is not None and hasattr(self.main_window, "overlap_enabled_var"):
                    try:
                        if not enabled:
                            tw0 = tw or self._get_live_tile_size()[0]
                            th0 = th or self._get_live_tile_size()[1]
                            source = "live_nonoverlap"
                            return int(tw0), int(th0)
                    except Exception:
                        pass

            # Next: explicit stride fields if user ever types them directly (future-proof)
            # Currently we expose overlap; stride is derived. But if present on config, honor it.
            if self.main_window is not None and hasattr(self.main_window, "config"):
                try:
                    tcfg = self.main_window.config.tiling
                    if getattr(tcfg, "stride_x", None) is not None:
                        sx = int(tcfg.stride_x)
                    if getattr(tcfg, "stride_y", None) is not None:
                        sy = int(tcfg.stride_y)
                    if sx is not None and sy is not None:
                        source = "main_config_stride"
                except Exception:
                    pass

            # Controller tiler (live after Start)
            if (sx is None or sy is None) and self.controller and getattr(self.controller, "tiler", None):
                try:
                    sx = sx or getattr(self.controller.tiler, "stride_x", None)
                    sy = sy or getattr(self.controller.tiler, "stride_y", None)
                    if sx is not None and sy is not None:
                        source = "controller_tiler"
                except Exception:
                    pass

            # Controller config
            if (sx is None or sy is None) and self.controller and hasattr(self.controller, "config"):
                try:
                    tcfg = self.controller.config.tiling
                    sx = sx or getattr(tcfg, "stride_x", None)
                    sy = sy or getattr(tcfg, "stride_y", None)
                    if sx is not None and sy is not None:
                        source = "controller_config"
                except Exception:
                    pass
        except Exception:
            pass

        # Final fallback: non-overlapping using provided or live tile size
        if sx is None or sy is None:
            tw0, th0 = (tw, th) if (tw and th) else self._get_live_tile_size()
            sx = int(tw0)
            sy = int(th0)
            source = source or "nonoverlap_fallback"

        try:
            print(f"[MultiFrameBrowser] live stride -> {sx}x{sy} (source={source})")
        except Exception:
            pass
        return int(sx), int(sy)

    def _build_ui(self):
        s = self.ui_scale
        fs = int(11 * s)
        fsb = int(12 * s)

        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=int(8*s), pady=int(4*s))

        ttk.Label(top, text="Video:").pack(side="left")
        self.video_var = tk.StringVar()
        self.video_combo = ttk.Combobox(top, textvariable=self.video_var, width=55, state="readonly")
        self.video_combo.pack(side="left", padx=4)
        self.video_combo.bind("<<ComboboxSelected>>", lambda e: self._load_annotations_for_video())

        ttk.Button(top, text="Reload", command=self._load_videos).pack(side="left", padx=4)
        ttk.Button(top, text="Browse any video file (unlabeled OK)", command=self._browse_video_file).pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        # Adjustable number of frames visible "at once"
        ttk.Label(top, text="Columns (frames visible):").pack(side="left")
        self.cols_var = tk.IntVar(value=4)
        self.cols_scale = ttk.Scale(top, from_=1, to=8, variable=self.cols_var,
                                    orient="horizontal", length=int(140*s),
                                    command=self._on_cols_changed)
        self.cols_scale.pack(side="left", padx=4)
        ttk.Label(top, textvariable=self.cols_var, width=2).pack(side="left")

        ttk.Label(top, text="  Thumb size:").pack(side="left")
        self.thumb_w_var = tk.IntVar(value=180)
        # Debounce thumb size changes — live rebuild during drag causes scroll jumps + flashing
        self.thumb_scale = ttk.Scale(top, from_=80, to=320, variable=self.thumb_w_var,
                                      orient="horizontal", length=int(100*s),
                                      command=self._on_thumb_size_changed)
        self.thumb_scale.pack(side="left", padx=4)

        ttk.Button(top, text="Refresh Strip", command=self._refresh_frame_strip).pack(side="left", padx=8)

        self.only_relevant_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Only frames with relevant labels", variable=self.only_relevant_var,
                        command=self._apply_relevance_filter).pack(side="left")

        # Main split: left = scrollable frames strip, right = selected frame details + actions
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=int(6*s), pady=int(4*s))

        # LEFT: Scrollable frame strip (the key new visual browser)
        left = ttk.LabelFrame(main, text="Frames (click to select) - scroll to see more")
        left.pack(side="left", fill="both", expand=True, padx=(0, int(4*s)))

        # Canvas + scrollbar for the strip/grid - VIRTUALIZED for smooth load/unload + no reflow jumps.
        # Cards are placed with explicit create_window(x, y) at stable coordinates.
        # Only viewport + buffer cards are kept as live widgets (unloaded when far off-screen).
        self.strip_canvas = tk.Canvas(left, bg="#1a1a1a", highlightthickness=0)
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.strip_canvas.yview)
        self.strip_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.strip_canvas.pack(side="left", fill="both", expand=True)

        # Virtualized: no giant child frame. Cards are individually placed via create_window(x, y)
        # at stable precomputed coordinates. scrollregion is managed explicitly in _update_* and refresh.
        # This eliminates grid reflows, large jumps on fast scroll, and unbounded widget accumulation.
        self.strip_canvas.bind("<MouseWheel>", self._on_strip_mousewheel)
        self.strip_canvas.bind("<Button-4>", self._on_strip_mousewheel)
        self.strip_canvas.bind("<Button-5>", self._on_strip_mousewheel)
        # Debounced viewport update on scroll/configure (the heart of smooth virtual loading)
        self.strip_canvas.bind("<Configure>", self._on_canvas_configure)
        # We drive updates from wheel + scheduled viewport; motion not needed.

        # RIGHT: Selected frame + actions + tile explorer launcher
        right = ttk.LabelFrame(main, text="Selected Frame & Label Actions", width=int(480 * min(s, 1.5)))
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.sel_info_var = tk.StringVar(value="No frame selected. Click a thumbnail on the left.")
        ttk.Label(right, textvariable=self.sel_info_var, wraplength=int(380*s), justify="left").pack(anchor="w", padx=6, pady=4)

        # Quick stats
        self.sel_stats_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.sel_stats_var, font=("TkDefaultFont", fsb)).pack(anchor="w", padx=6)

        # Action buttons - "go to the label creation section for that frame"
        btns = ttk.Frame(right)
        btns.pack(fill="x", padx=6, pady=6)

        ttk.Button(btns, text="Explore & Label Tiles on this Frame",
                   command=self._open_frame_tile_explorer).pack(fill="x", pady=2)
        ttk.Button(btns, text="Jump to this frame in Label Only mode (sequential)",
                   command=self._jump_to_frame_in_main).pack(fill="x", pady=2)
        ttk.Button(btns, text="Open in Review Window (list view)",
                   command=self._open_in_review).pack(fill="x", pady=2)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)

        # Mini list of labels on the selected frame (for quick view)
        ttk.Label(right, text="Labels on selected frame:").pack(anchor="w", padx=4)
        self.frame_labels_list = tk.Listbox(right, height=12, font=("TkDefaultFont", fs))
        self.frame_labels_list.pack(fill="both", expand=False, padx=4, pady=2)
        # Double-click list entry to edit immediately (streamline)
        self.frame_labels_list.bind("<Double-Button-1>", lambda e: self._edit_selected_tile_from_list())

        ttk.Button(right, text="Edit selected tile label (from list above)",
                   command=self._edit_selected_tile_from_list).pack(fill="x", padx=4, pady=2)

        # Status bar
        self.browser_status_var = tk.StringVar(value="Load a video. Frames use virtual loading (smooth scroll + fixed-size image slots). Only visible cards consume heavy resources.")
        ttk.Label(self, textvariable=self.browser_status_var, relief="sunken").pack(fill="x", padx=int(6*s), pady=2)

    def _on_strip_configure(self, event=None):
        # Kept for compatibility; virtual path uses explicit scrollregion.
        try:
            self.strip_canvas.configure(scrollregion=self.strip_canvas.bbox("all"))
        except Exception:
            pass

    def _on_canvas_configure(self, event=None):
        # On canvas resize we may need to adjust positions if cols changed, but mainly schedule viewport.
        self._schedule_viewport_update(80)

    def _on_strip_mousewheel(self, event):
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", 0)
        if num == 5 or delta < 0:
            self.strip_canvas.yview_scroll(1, "units")
        elif num == 4 or delta > 0:
            self.strip_canvas.yview_scroll(-1, "units")
        # Debounced: prioritize visible + load/unload cards for the new viewport.
        # Using longer debounce + single scheduled func prevents flicker/jump storms on fast wheel.
        self._schedule_viewport_update(110)
        self.after(180, self._start_async_thumb_loading)

    def _schedule_viewport_update(self, delay_ms: int = 90):
        """Debounce viewport refresh (materialize visible cards + unload far ones).
        This is the key to flicker-free fast scrolling and bounded resources.
        """
        if self._viewport_update_pending is not None:
            try:
                self.after_cancel(self._viewport_update_pending)
            except Exception:
                pass
        self._viewport_update_pending = self.after(delay_ms, self._update_visible_cards)

    def _load_videos(self):
        vids = self.tile_db.list_videos()
        self.video_combo['values'] = vids
        if vids:
            self.video_var.set(vids[0])
            self._load_annotations_for_video()
        else:
            self.browser_status_var.set("No videos with annotations found in DB. Use 'Browse any video file'.")

    def _browse_video_file(self):
        """Allow the user to pick any video (even one with no labels yet) so they can see the
        full stride sequence and start labeling from the browser.
        """
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select video to browse/label",
            filetypes=[("Video files", "*.mp4 *.MP4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.current_video = path
        self.video_var.set(path)  # show the full path temporarily
        # Treat as having no annotations initially
        self.frame_to_anns = {}
        # Force load which will compute strided frames using video metadata + stride
        self._load_annotations_for_video()

    def _load_annotations_for_video(self):
        v = self.video_var.get()
        if not v:
            return
        self.current_video = v

        # Always go through the helper so we read the live entry boxes the user just typed.
        tw, th = self._get_live_tile_size()

        # Debug so user can see what size the browser actually decided to use
        try:
            print(f"[MultiFrameBrowser] Using tile size for annotations: {tw}x{th} (from live boxes if present)")
        except Exception:
            pass

        # Live stride from main window controls (critical for overlap)
        sx, sy = self._get_live_stride(tw, th)
        if self.annotation_manager:
            self.annotation_manager.set_scope(video_path=v, tile_size=(tw, th) if tw else None, stride=(sx, sy) if (sx or sy) else None)
            anns = self.annotation_manager.get_annotations(video=v, use_scope=True)
        else:
            anns = self.tile_db.get_annotations_for_video(v, tile_width=tw, tile_height=th,
                                                             stride_x=sx, stride_y=sy)
        self.frame_to_anns = {}
        for a in anns:
            f = a["abs_frame"]
            if f not in self.frame_to_anns:
                self.frame_to_anns[f] = []
            self.frame_to_anns[f].append(a)

        # Determine stride from the *live GUI variable* if possible (so it respects the
        # "Frame stride (every Nth)" entry even before you press Start), falling back to config.
        # This fixes the issue where it always used the default of 3.
        stride = 3
        try:
            if self.main_window and hasattr(self.main_window, '_frame_stride_var'):
                val = self.main_window._frame_stride_var.get()
                stride = max(1, int(val)) if val.strip() else 3
            elif self.controller and hasattr(self.controller, 'config') and self.controller.config:
                stride = max(1, int(self.controller.config.tiling.frame_stride))
            elif self.main_window and hasattr(self.main_window, 'config') and self.main_window.config:
                stride = max(1, int(self.main_window.config.tiling.frame_stride))
        except Exception:
            stride = 3

        # Get total frames from video so we can show EVERY frame according to stride (not just annotated ones)
        total_frames = 0
        try:
            import cv2
            cap = cv2.VideoCapture(v)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            total_frames = 0

        if total_frames > 0:
            candidate_frames = list(range(0, total_frames, stride))
        else:
            # Fallback to annotated frames only if we can't read the video
            candidate_frames = sorted(self.frame_to_anns.keys())

        # Apply only-relevant filter on top of the strided list if requested
        if self.only_relevant_var.get():
            self.sorted_frames = [f for f in candidate_frames
                                  if any(a.get("relevant") for a in self.frame_to_anns.get(f, []))]
        else:
            self.sorted_frames = candidate_frames

        # Keep the full strided list (pre-filter) so we can toggle the relevance filter live
        self._base_sorted_frames = list(self.sorted_frames)

        self.frame_thumbs.clear()
        self.selected_frame = None
        self.frame_cards.clear()
        self._clear_all_materialized_cards()  # ensure no stale canvas windows from previous video

        # The _displayed_count is the addressable range (affects scroll height + what the worker will consider).
        # Because of virtualization, we can afford a larger addressable window from the start;
        # only a small viewport+buffer of *actual widgets* are ever created (see _update_visible_cards).
        self._displayed_count = min(240, len(self.sorted_frames))

        # Seed class relevance from DB so we can auto-apply "relevant" flag for known classes
        self.class_relevance = {}
        try:
            for lbl in self.tile_db.get_all_labels():
                rel = self.tile_db.get_class_relevance(lbl)
                if rel is not None:
                    self.class_relevance[lbl] = rel
        except Exception:
            pass

        # Also inherit from main window if available (for consistency with main labeling session)
        if self.main_window and hasattr(self.main_window, 'class_relevance'):
            try:
                self.class_relevance.update(getattr(self.main_window, 'class_relevance', {}))
            except Exception:
                pass

        shown = min(self._displayed_count, len(self.sorted_frames))
        self.browser_status_var.set(
            f"Loaded {len(self.sorted_frames)} frames (stride={stride}, total~{total_frames}) from {Path(v).name}. "
            f"Virtualized: scroll freely. Only ~viewport cards materialized to avoid flicker/crashes."
        )
        self._refresh_frame_strip()

        if self.sorted_frames:
            self._select_frame(self.sorted_frames[0])
            # Ensure at least the top of the list gets materialized promptly
            self.after(40, self._update_visible_cards)

    def _apply_relevance_filter(self):
        """Recompute sorted_frames from the base strided list according to the checkbox.
        Then reset displayed and do a (full) refresh. Keeps thumbs where possible.
        """
        base = getattr(self, '_base_sorted_frames', None) or self.sorted_frames
        if self.only_relevant_var.get():
            self.sorted_frames = [f for f in base
                                  if any(a.get("relevant") for a in self.frame_to_anns.get(f, []))]
        else:
            self.sorted_frames = list(base)

        # On filter change we can keep the same displayed prefix length (or cap)
        self._displayed_count = min(getattr(self, '_displayed_count', 80) or 80, len(self.sorted_frames))
        self._clear_all_materialized_cards()
        self.frame_cards.clear()
        self._card_img_labels.clear()
        # Thumbs are kept (they are independent of relevance)
        self._refresh_frame_strip()
        # If current selected is filtered out, pick first
        if self.selected_frame not in self.sorted_frames and self.sorted_frames:
            self._select_frame(self.sorted_frames[0])

    def _refresh_frame_strip(self):
        """Full layout reset (used for columns, thumb size, filter toggle, initial load).
        Clears virtual materialized cards and recomputes geometry.
        Normal scrolling uses the much lighter _update_visible_cards instead.
        """
        # Save frac for size/col changes
        try:
            scroll_frac = self.strip_canvas.yview()[0]
        except Exception:
            scroll_frac = 0.0

        current_thumb = self._get_effective_thumb_w()
        if current_thumb != getattr(self, '_last_thumb_w', -1):
            self.frame_thumbs.clear()
        self._last_thumb_w = current_thumb  # always track latest effective for next comparison

        # Full clear of any previous virtual cards + widgets
        self._clear_all_materialized_cards()

        if not self.sorted_frames:
            # Minimal message directly on canvas
            try:
                self.strip_canvas.delete("all_msg")
            except Exception:
                pass
            self.strip_canvas.create_text(20, 20, text="No frames to display.", fill="#888", tags="all_msg", anchor="nw")
            return

        cols = max(1, int(self.cols_var.get()))
        thumb_w = current_thumb
        thumb_h = int(thumb_w * 0.65)

        # Compute stable layout metrics for virtual positioning
        self._virtual_cell_width = thumb_w + 8
        text_block_h = int(28 * min(self.ui_scale, 1.6))
        self._virtual_row_height = thumb_h + 8 + text_block_h   # ph + small pads + info labels

        frames_to_display = self._get_displayed_frames()
        total_items = len(frames_to_display)
        num_rows = (total_items + cols - 1) // cols if total_items else 1
        total_height = max(10, num_rows * self._virtual_row_height + 10)
        total_width = cols * self._virtual_cell_width + 10

        # Set the canvas scrollregion to the full virtual size (even if we only materialize a slice)
        self.strip_canvas.configure(scrollregion=(0, 0, total_width, total_height))

        # Clear any stray message
        try:
            self.strip_canvas.delete("all_msg")
        except Exception:
            pass

        self._card_img_labels.clear()
        # Note: frame_cards and _card_windows will be populated lazily by _update_visible_cards

        # Restore scroll (for col/thumb changes). Force Tk to commit the new scrollregion
        # and sizes so the subsequent viewport update sees correct yview() + winfo.
        try:
            self.strip_canvas.update_idletasks()
            self.strip_canvas.yview_moveto(scroll_frac)
            self.strip_canvas.update_idletasks()
        except Exception:
            pass

        # Schedule the virtual viewport population (creates only near-visible cards with reserved blanks)
        self._schedule_viewport_update(25)

        # Kick async for whatever ends up visible
        self.after(70, self._start_async_thumb_loading)

        # Re-highlight if needed (will apply when its card is (re)materialized)
        if self.selected_frame is not None:
            self.after(120, lambda: self._highlight_card(self.selected_frame))

    def _clear_all_materialized_cards(self):
        """Destroy all live card widgets and canvas windows. Used on full resets."""
        for fidx in list(self._materialized_frames):
            self._destroy_card(fidx)
        self.frame_cards.clear()
        self._card_windows.clear()
        self._materialized_frames.clear()
        self._card_img_labels.clear()

    def _get_displayed_frames(self):
        return self.sorted_frames[:getattr(self, '_displayed_count', len(self.sorted_frames))]

    def _get_effective_thumb_w(self):
        """Thumb width that incorporates current container width so frames scale up
        when the browser window or frame strip section is resized larger.
        The 'Thumb size' slider provides the base; extra horizontal space is
        distributed to grow the actual thumbnail frames.
        """
        slider = max(80, int(self.thumb_w_var.get()))
        try:
            cw = self.strip_canvas.winfo_width()
        except Exception:
            cw = 0
        cols = max(1, int(self.cols_var.get()))
        if cw > 120:
            needed = slider * cols + 16
            extra = cw - needed
            if extra > 20:
                add = extra // cols
                eff = slider + add
                max_fit = (cw - 16) // cols - 4
                eff = min(eff, max(80, max_fit))
                return max(80, min(520, eff))
            # Canvas wide enough to naturally fit larger than slider
            max_fit = (cw - 16) // cols - 4
            if max_fit > slider:
                return max(80, min(520, max_fit))
        return slider

    def _compute_layout_params(self):
        """Return (cols, thumb_w, thumb_h, cell_w, row_h) for current slider settings.
        Now uses effective thumb that grows with container width.
        """
        cols = max(1, int(self.cols_var.get()))
        thumb_w = self._get_effective_thumb_w()
        thumb_h = int(thumb_w * 0.65)
        cell_w = thumb_w + 8
        text_block_h = int(28 * min(getattr(self, 'ui_scale', 1.6), 1.6))
        row_h = thumb_h + 8 + text_block_h
        return cols, thumb_w, thumb_h, cell_w, row_h

    def _create_card_widget(self, fidx: int, thumb_w: int, thumb_h: int):
        """Create the card ttk.Frame (never gridded). Includes fixed-size blank image slot.
        The ph reserves the exact pixel size at all times so layout does not shift when the
        image loads or async updates happen.
        """
        anns = self.frame_to_anns.get(fidx, [])
        n_labels = len(anns)
        n_rel = sum(1 for a in anns if a.get("relevant"))
        has_rel = n_rel > 0

        card = ttk.Frame(self.strip_canvas, relief="ridge", borderwidth=1)

        # FIXED SIZE IMAGE SLOT: this is the "equally sized blank spot" requested.
        # pack_propagate(False) + explicit width/height means it claims its space immediately
        # and never resizes when we later put a real PhotoImage inside.
        # While the thumb is not here, this area stays exactly thumb_w x thumb_h, preventing
        # any layout shift or flicker as images pop in.
        ph = ttk.Frame(card, width=thumb_w, height=thumb_h)
        ph.pack_propagate(False)
        # Start completely blank (no text) so it truly looks like a reserved empty slot.
        img_lbl = ttk.Label(ph, text="", anchor="center", background="#222")
        img_lbl.pack(expand=True, fill="both")
        self._card_img_labels[fidx] = img_lbl
        ph.pack(padx=2, pady=2)

        # Info label (below the reserved image area)
        rel_mark = " [R]" if has_rel else ""
        info = f"Frame {fidx}\n{n_labels} labels{rel_mark}"
        info_lbl = ttk.Label(card, text=info, justify="center", font=("TkDefaultFont", int(9*self.ui_scale)))
        info_lbl.pack(pady=1)

        # Store ref so we can update text without full rebuild
        card._info_label = info_lbl

        # Click handlers
        def make_sel(ff=fidx, c=card):
            return lambda e: self._select_frame(ff, c)

        def make_dbl(ff=fidx):
            return lambda e: (self._select_frame(ff), self._open_frame_tile_explorer())

        # Bind on card and direct children (ph, labels)
        for child in (card, ph, img_lbl, info_lbl):
            child.bind("<Button-1>", make_sel(ff=fidx, c=card))
            child.bind("<Double-Button-1>", make_dbl(ff=fidx))

        self.frame_cards[fidx] = card
        return card

    def _place_or_update_card(self, fidx: int, x: int, y: int, thumb_w: int, thumb_h: int):
        """Ensure a card widget exists for fidx and is placed via create_window at (x,y).
        If already placed, try to move if its coords are stale (e.g. after cols change
        without full clear). Also, never apply a cached PhotoImage if its pixel size
        does not match the current thumb target — that would leave the "frame" small
        inside a resized box.
        """
        if fidx in self._card_windows:
            try:
                win_id = self._card_windows[fidx]
                coords = self.strip_canvas.coords(win_id)
                if coords and (abs(coords[0] - x) > 2 or abs(coords[1] - y) > 2):
                    self.strip_canvas.coords(win_id, x, y)
            except Exception:
                pass
            return self.frame_cards.get(fidx)

        card = self._create_card_widget(fidx, thumb_w, thumb_h)

        # Reserve the space in the layout from the moment the window is created
        win_id = self.strip_canvas.create_window(x, y, window=card, anchor="nw")
        self._card_windows[fidx] = win_id
        self._materialized_frames.add(fidx)

        # If we have a cached thumb, ONLY apply if the image resolution roughly matches
        # the slot we just created. Mismatched size (old cached after cols/thumb change)
        # is the main cause of "boxes resize but the actual frame image stays small/old".
        # Pop it so the async loader (which always uses live _get_effective) will regen.
        applied = False
        if fidx in self.frame_thumbs:
            try:
                tkimg = self.frame_thumbs[fidx]
                if abs(getattr(tkimg, 'width', lambda: 0)() - thumb_w) <= 6:
                    img_lbl = self._card_img_labels.get(fidx)
                    if img_lbl and img_lbl.winfo_exists():
                        img_lbl.configure(image=tkimg, text="")
                        img_lbl.image = tkimg
                        applied = True
                else:
                    # wrong size for this card's ph — drop so it gets regenerated at correct res
                    self.frame_thumbs.pop(fidx, None)
            except Exception:
                pass

        if not applied:
            # Ensure the img_lbl is clean blank for the exact new size (in case previous content)
            try:
                img_lbl = self._card_img_labels.get(fidx)
                if img_lbl and img_lbl.winfo_exists():
                    img_lbl.configure(image="", text="")
                    if hasattr(img_lbl, 'image'):
                        del img_lbl.image
            except Exception:
                pass

        return card

    def _destroy_card(self, fidx: int):
        """Unload a single card: delete its canvas window, destroy widget, clean refs.
        This releases X11 pixmap resources for that frame's image.
        """
        win_id = self._card_windows.pop(fidx, None)
        if win_id is not None:
            try:
                self.strip_canvas.delete(win_id)
            except Exception:
                pass
        card = self.frame_cards.pop(fidx, None)
        if card is not None:
            try:
                card.destroy()
            except Exception:
                pass
        self._card_img_labels.pop(fidx, None)
        self._materialized_frames.discard(fidx)

    def _update_visible_cards(self):
        """Core of the smooth loading/unloading system.
        Computes the current viewport (from yview), materializes only a window of cards
        around it (plus small buffer), and unloads cards that have scrolled far away.
        Uses fixed (x, y) placement so there is zero reflow or scroll jumping.
        The reserved ph inside each card guarantees a blank spot of correct size.
        """
        self._viewport_update_pending = None
        if not self.sorted_frames:
            return

        cols, thumb_w, thumb_h, cell_w, row_h = self._compute_layout_params()
        displayed = self._get_displayed_frames()
        n = len(displayed)
        if n == 0:
            return

        # Detect layout changes (cols or effective thumb/row size). If so, drop ALL
        # currently materialized so they are recreated with correct ph dimensions,
        # correct (x,y) positions for the *new* cols, and we can force correct-res thumbs.
        # This prevents stale cards (old sizes or old column positions) from lingering
        # and also ensures desired cards are always freshly placed after a cols change.
        old_tw = getattr(self, '_last_materialized_thumb', 0)
        old_c = getattr(self, '_last_cols', cols)
        layout_changed = (abs(thumb_w - old_tw) > 5) or (old_c != cols)
        if layout_changed and self._materialized_frames:
            for fidx in list(self._materialized_frames):
                self.frame_thumbs.pop(fidx, None)  # force fresh gen at the size used for new cards
                self._destroy_card(fidx)
            self.after(50, self._start_async_thumb_loading)
        self._last_materialized_thumb = thumb_w
        self._last_cols = cols

        # Determine visible row range from scroll fraction using accurate total_rows
        # (pixel y / row_h would also work since our coords are r * row_h).
        # This fixes cases where previous n/cols float math + changing row_h left
        # viewport with no desired cards → black empty canvas.
        try:
            top_f, bot_f = self.strip_canvas.yview()
        except Exception:
            top_f, bot_f = 0.0, 1.0

        total_rows = (n + cols - 1) // cols if n else 1
        first_visible_row = max(0, int(top_f * total_rows) - 2)
        last_visible_row = int(bot_f * total_rows) + 2

        # Larger buffer helps ensure cards appear promptly on scroll / after layout change
        buffer_rows = 5
        start_row = max(0, first_visible_row - buffer_rows)
        end_row = min(total_rows, last_visible_row + buffer_rows)

        # The logical frame indices we want materialized now
        desired = set()
        for r in range(start_row, end_row + 1):
            for c in range(cols):
                idx = r * cols + c
                if idx < n:
                    desired.add(displayed[idx])

        # Safety: if desired came out empty for the current view (can happen transiently
        # after layout/scroll frac changes), force at least the first couple rows of the
        # current view so the user never sees pure black.
        if not desired and n > 0:
            top_row = max(0, int(top_f * total_rows))
            for rr in range(top_row, min(total_rows, top_row + 3)):
                for c in range(cols):
                    idx = rr * cols + c
                    if idx < n:
                        desired.add(displayed[idx])

        # Unload anything materialized that is no longer desired
        for fidx in list(self._materialized_frames):
            if fidx not in desired:
                self._destroy_card(fidx)

        # Create/place the desired ones at their stable coordinates
        for idx, fidx in enumerate(displayed):
            if fidx not in desired:
                continue
            r = idx // cols
            c = idx % cols
            x = c * cell_w
            y = r * row_h
            self._place_or_update_card(fidx, x, y, thumb_w, thumb_h)

        # Make sure scrollregion still covers the full virtual area (in case resize etc.)
        total_rows = (n + cols - 1) // cols if n else 1
        total_h = total_rows * row_h + 4
        total_w = cols * cell_w + 4
        try:
            self.strip_canvas.configure(scrollregion=(0, 0, total_w, total_h))
        except Exception:
            pass

        # If selected frame is now materialized, make sure it's highlighted
        if self.selected_frame in self.frame_cards:
            self._highlight_card(self.selected_frame)

        # Opportunistic: if the user has scrolled near the end of current displayed,
        # trigger the extender (non-destructive).
        try:
            _, bot = self.strip_canvas.yview()
            if bot > 0.60:
                self.after(50, self._check_load_more)
        except Exception:
            pass

    def _generate_frame_pil(self, frame_idx: int, max_w: int = 180, max_h: int = 110) -> Optional["Image.Image"]:
        """Heavy work: decode + resize. Safe to call from background thread. Returns PIL or None."""
        if not self.current_video:
            return None
        try:
            import cv2
            cap = cv2.VideoCapture(self.current_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            h, w = frame.shape[:2]
            scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb).convert("RGB")
        except Exception as e:
            print(f"[MultiFrameBrowser] PIL generate error for frame {frame_idx}: {e}")
            return None

    def _get_frame_thumbnail(self, frame_idx: int, max_w: int = 180, max_h: int = 110) -> Optional[ImageTk.PhotoImage]:
        """Sync path (kept for compatibility). Uses cache or generates on main thread."""
        if frame_idx in self.frame_thumbs:
            return self.frame_thumbs[frame_idx]
        pil = self._generate_frame_pil(frame_idx, max_w, max_h)
        if pil is None:
            return None
        tkimg = ImageTk.PhotoImage(pil)
        self.frame_thumbs[frame_idx] = tkimg
        return tkimg

    def _refresh_annotations_light(self):
        """Re-fetch annotations after an edit. Updates only the count text on existing cards.
        Does NOT clear thumbs or destroy/rebuild the strip. This avoids the full reload annoyance.
        """
        if not self.current_video:
            return
        try:
            # Scope using the shared live-value helper (reads the boxes the user typed) + stride
            tw, th = self._get_live_tile_size()
            sx, sy = self._get_live_stride(tw, th)
            anns = self.tile_db.get_annotations_for_video(self.current_video, tile_width=tw, tile_height=th,
                                                           stride_x=sx, stride_y=sy)
            self.frame_to_anns = {}
            for a in anns:
                f = a["abs_frame"]
                self.frame_to_anns.setdefault(f, []).append(a)
            # Update texts on current cards only (thumbs stay loaded)
            for fidx in list(self.frame_cards.keys()):
                self._update_card_text_only(fidx)
            # If the right panel is showing the edited frame, refresh its label list too
            if getattr(self, 'selected_frame', None) is not None:
                try:
                    self._select_frame(self.selected_frame)
                except Exception:
                    pass
        except Exception as e:
            print(f"[MultiFrameBrowser] light refresh error: {e}")

    def _update_card_text_only(self, fidx):
        card = self.frame_cards.get(fidx)
        if not card or not card.winfo_exists():
            return
        anns = self.frame_to_anns.get(fidx, [])
        n_labels = len(anns)
        n_rel = sum(1 for a in anns if a.get("relevant"))
        has_rel = n_rel > 0
        rel_mark = " [R]" if has_rel else ""
        info = f"Frame {fidx}\n{n_labels} labels{rel_mark}"
        # Preferred: the stored direct ref from creation
        if hasattr(card, "_info_label") and card._info_label and card._info_label.winfo_exists():
            try:
                card._info_label.configure(text=info)
                return
            except Exception:
                pass
        # Fallback scan (older cards or edge cases)
        for child in card.winfo_children():
            try:
                if isinstance(child, ttk.Label):
                    txt = child.cget("text") or ""
                    if "\n" in str(txt):
                        child.configure(text=info)
                        break
            except Exception:
                pass

    def _get_visible_frame_indices(self):
        """Return list of frame indices whose cards are currently visible (or near) in the viewport.
        Used to prioritize async thumb loading for what the user is actually looking at.
        Works with the virtual canvas positioning (no reliance on strip_inner grid).
        """
        visible = []
        displayed = self._get_displayed_frames()
        if not displayed:
            return visible
        try:
            top_frac, bot_frac = self.strip_canvas.yview()
            # Approximate using logical rows
            cols = max(1, int(self.cols_var.get()))
            n = len(displayed)
            total_rows = (n + cols - 1) // cols
            first_row = max(0, int(top_frac * total_rows) - 1)
            last_row = min(total_rows, int(bot_frac * total_rows) + 2)
            for r in range(first_row, last_row):
                for c in range(cols):
                    idx = r * cols + c
                    if idx < n:
                        visible.append(displayed[idx])
            if not visible:
                visible = displayed[: max(8, cols * 2)]
        except Exception:
            visible = displayed[:8]
        return visible

    def _get_prioritized_load_list(self):
        """Visible frames (in order) first, then the remaining frames.
        Called frequently by the worker so scrolling immediately affects what gets loaded next.
        Only considers currently displayed frames (incremental loading to prevent resource exhaustion).
        """
        if not self.sorted_frames:
            return []
        frames = self._get_displayed_frames()
        visible = self._get_visible_frame_indices()
        vset = set(visible)
        vis_in_order = [f for f in frames if f in vset]
        rest = [f for f in frames if f not in vset]
        return vis_in_order + rest

    def _start_async_thumb_loading(self):
        """Load frame previews in a background thread so the browser window stays responsive
        for scrolling and editing while thumbs appear progressively.

        Improvements:
        - Single worker thread (no more parallel "chunk" loading from 0 and from 30 at the same time).
        - Strictly sequential within the worker.
        - Dynamically prioritizes frames that are currently scrolled into view.
        - Re-checks priority after every single thumbnail so panning/scrolling takes effect quickly.
        """
        if not self.sorted_frames or not self.current_video:
            return

        def worker():
            # Keep loading until everything for the current list is done.
            # Every iteration we ask for the current best order (visible first).
            while True:
                prio = self._get_prioritized_load_list()
                next_f = None
                for f in prio:
                    if f not in self.frame_thumbs:
                        next_f = f
                        break
                if next_f is None:
                    break  # nothing left to load for now

                # Snapshot current effective size on every item (layout/cols can change
                # while a long worker is running; closed-over value would be stale).
                cur_tw = self._get_effective_thumb_w()
                cur_mh = int(cur_tw * 0.65)
                pil = self._generate_frame_pil(next_f, cur_tw, cur_mh)
                if pil is not None:
                    def schedule(f=next_f, p=pil):
                        try:
                            tkimg = ImageTk.PhotoImage(p)
                            self.frame_thumbs[f] = tkimg
                            self.after(0, lambda ff=f, img=tkimg: self._update_card_thumb(ff, img))
                        except Exception as ex:
                            print(f"[MultiFrameBrowser] async thumb schedule error: {ex}")
                    self.after(0, schedule)

                # tiny sleep: keeps UI fluid + gives scroll events time to update visible set
                time.sleep(0.008)

            self.after(0, lambda: self._clear_remaining_loading_placeholders())
            self.after(0, lambda: self.browser_status_var.set(
                (self.browser_status_var.get() or "") + "  (thumbnails loaded)"))

        # Only ever one loader thread. If one is running it will see the new priority list
        # on its very next iteration (thanks to _get_prioritized_load_list).
        if hasattr(self, '_thumb_load_thread') and getattr(self._thumb_load_thread, 'is_alive', lambda: False)():
            return
        t = threading.Thread(target=worker, daemon=True, name="frame-thumb-loader")
        self._thumb_load_thread = t
        t.start()

    def _update_card_thumb(self, frame_idx: int, tkimg: ImageTk.PhotoImage):
        """Called on main thread to update a specific card's image label (async safe)."""
        img_lbl = self._card_img_labels.get(frame_idx)
        if img_lbl and img_lbl.winfo_exists():
            try:
                img_lbl.configure(image=tkimg, text="")  # replace placeholder text
                img_lbl.image = tkimg  # keep ref
            except Exception:
                pass
        # also keep in the card dict for rebuilds
        if frame_idx in self.frame_cards:
            self.frame_cards[frame_idx].img_ref = tkimg  # optional extra ref

    def _clear_remaining_loading_placeholders(self):
        """After the worker finishes a pass, leave permanent blanks for any that failed.
        (We keep the slot reserved; a subtle indicator is optional.)
        """
        for fidx, img_lbl in list(self._card_img_labels.items()):
            if img_lbl and img_lbl.winfo_exists():
                try:
                    # Only touch ones that never received an image
                    if not getattr(img_lbl, 'image', None):
                        # keep it as empty reserved area, or set a very faint marker
                        if not img_lbl.cget("image"):
                            img_lbl.configure(text="")  # stay as clean blank spot
                except Exception:
                    pass

    def _check_load_more(self, event=None):
        """When user scrolls near the bottom, incrementally extend the *virtual* total.
        Only extends the conceptual list + scrollregion. Actual widgets are created on-demand
        by _update_visible_cards when they enter the viewport. This is what prevents
        flicker, large jumps, and eventual resource exhaustion.
        """
        if not hasattr(self, '_displayed_count') or not self.sorted_frames:
            return
        try:
            _, bot = self.strip_canvas.yview()
            if bot > 0.55 and self._displayed_count < len(self.sorted_frames):
                old = self._displayed_count
                # Smaller steps feel smoother; still plenty fast
                self._displayed_count = min(len(self.sorted_frames), self._displayed_count + 80)
                if self._displayed_count > old:
                    # Recompute scrollregion for the new total without touching existing cards
                    cols, thumb_w, thumb_h, cell_w, row_h = self._compute_layout_params()
                    n = self._displayed_count
                    total_rows = (n + cols - 1) // cols if n else 1
                    total_h = total_rows * row_h + 4
                    total_w = cols * cell_w + 4
                    try:
                        cur_region = self.strip_canvas.cget("scrollregion")
                    except Exception:
                        cur_region = ""
                    self.strip_canvas.configure(scrollregion=(0, 0, total_w, total_h))
                    # Do NOT yview_moveto here — that was a major source of jumps during fast scroll.
                    # Let the user's scroll position be stable; new content simply appears further down.
                    # Schedule the viewport manager so any newly visible tail gets cards created.
                    self._schedule_viewport_update(60)
                    self.after(140, self._start_async_thumb_loading)
        except Exception:
            pass

    def _on_thumb_size_changed(self, val):
        """Debounced handler for the thumb size slider.
        Live rebuilds during drag cause scroll snapping, flashing, and layout thrashing.
        """
        if hasattr(self, '_thumb_debounce'):
            try:
                self.after_cancel(self._thumb_debounce)
            except Exception:
                pass
        self._thumb_debounce = self.after(280, self._refresh_frame_strip)

    def _on_cols_changed(self, val):
        """Debounced handler for the columns slider.
        Live full refreshes on every tick during drag destroy cards constantly and
        can leave the viewport without materialized frames (black areas) until settle.
        """
        if hasattr(self, '_cols_debounce'):
            try:
                self.after_cancel(self._cols_debounce)
            except Exception:
                pass
        self._cols_debounce = self.after(220, self._refresh_frame_strip)

    def _highlight_card(self, frame_idx: Optional[int]):
        # If the target is in the addressable set but not live yet, try to bring it in.
        if frame_idx is not None and frame_idx not in self.frame_cards:
            displayed = self._get_displayed_frames()
            if frame_idx in displayed:
                try:
                    cols, thumb_w, thumb_h, cell_w, row_h = self._compute_layout_params()
                    idx = displayed.index(frame_idx)
                    r = idx // cols
                    c = idx % cols
                    self._place_or_update_card(frame_idx, c * cell_w, r * row_h, thumb_w, thumb_h)
                except Exception:
                    pass
        for f, card in list(self.frame_cards.items()):
            try:
                if f == frame_idx:
                    card.configure(relief="solid", borderwidth=3)
                else:
                    card.configure(relief="ridge", borderwidth=1)
            except Exception:
                pass

    def _select_frame(self, frame_idx: int, card_widget=None):
        self.selected_frame = frame_idx
        # If the frame is within our virtual displayed set but not currently materialized,
        # materialize it immediately (and a small neighborhood) so highlight + visual selection works.
        displayed = self._get_displayed_frames()
        if frame_idx in displayed and frame_idx not in self.frame_cards:
            try:
                cols, thumb_w, thumb_h, cell_w, row_h = self._compute_layout_params()
                idx = displayed.index(frame_idx)
                r = idx // cols
                c = idx % cols
                x = c * cell_w
                y = r * row_h
                self._place_or_update_card(frame_idx, x, y, thumb_w, thumb_h)
            except Exception:
                pass
        self._highlight_card(frame_idx)

        anns = self.frame_to_anns.get(frame_idx, [])
        n = len(anns)
        n_rel = sum(1 for a in anns if a.get("relevant", False))

        self.sel_info_var.set(f"Frame {frame_idx}  |  {n} labeled tiles  |  {n_rel} relevant")
        self.sel_stats_var.set(f"Video: {Path(self.current_video).name if self.current_video else '?'}")

        # Populate quick labels list
        self.frame_labels_list.delete(0, "end")
        self._current_frame_anns_list = anns  # for edit action
        for a in anns:
            rel = " [R]" if a.get("relevant") else ""
            tw = a.get("tile_width", "?")
            th = a.get("tile_height", "?")
            txt = f"r{a['tile_row']}c{a['tile_col']} [{tw}x{th}]  {a['label']}{rel}"
            self.frame_labels_list.insert("end", txt)

        self.browser_status_var.set(f"Selected frame {frame_idx}. Use 'Explore & Label Tiles on this Frame' for the per-frame label creation grid.")

    def _open_frame_tile_explorer(self):
        """Open a scrollable grid of ALL tiles for the selected frame so each can be labeled/edited.

        Critical UX: the canvas must keep a correct scrollregion and support mouse-wheel
        scrolling. Without that, dense non-overlapping grids (small tile sizes → many rows)
        appear "cut off" after ~3–4 visible rows.
        """
        if self.selected_frame is None or not self.current_video:
            messagebox.showinfo("Explorer", "Select a frame first.")
            return

        explorer = tk.Toplevel(self)
        explorer.title(f"Tile Label Editor - Frame {self.selected_frame}")
        explorer.geometry(f"{int(1450 * min(self.ui_scale, 1.8))}x{int(920 * min(self.ui_scale, 1.8))}")
        explorer.minsize(900, 600)
        explorer.resizable(True, True)

        top = ttk.Frame(explorer)
        top.pack(fill="x", padx=8, pady=6)
        info_var = tk.StringVar(value=f"Frame {self.selected_frame} — click any tile to label/edit  |  scroll for more rows")
        ttk.Label(top, textvariable=info_var).pack(side="left")
        # Keep a handle so populate can update the status line with tile counts
        explorer._tile_explorer_info_var = info_var

        # Body: canvas + vertical (and horizontal) scrollbars so tall/wide grids are fully reachable
        body = ttk.Frame(explorer)
        body.pack(fill="both", expand=True, padx=4, pady=4)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        canvas = tk.Canvas(body, bg="#1f1f1f", highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(body, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tile_container = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=tile_container, anchor="nw")
        # Store on explorer so refresh can re-bind / re-use the same ids
        explorer._tile_canvas = canvas
        explorer._tile_container = tile_container
        explorer._tile_win_id = win_id

        # Preview size control (default 300×300; upscales small native tiles for easier viewing)
        ttk.Label(top, text="  Preview:").pack(side="left", padx=(12, 0))
        preview_var = tk.IntVar(value=int(getattr(self, "tile_preview_size", 300) or 300))
        explorer._tile_preview_var = preview_var
        preview_scale = ttk.Scale(
            top, from_=120, to=480, variable=preview_var, orient="horizontal", length=120,
        )
        preview_scale.pack(side="left", padx=4)
        preview_lbl = ttk.Label(top, text=f"{preview_var.get()}px", width=5)
        preview_lbl.pack(side="left")

        def _sync_preview_lbl(*_):
            try:
                preview_lbl.config(text=f"{int(preview_var.get())}px")
            except Exception:
                pass
        preview_var.trace_add("write", _sync_preview_lbl)

        def _apply_preview_size(*_):
            try:
                self.tile_preview_size = max(80, min(640, int(preview_var.get())))
            except Exception:
                self.tile_preview_size = 300
            self._populate_frame_tile_grid(explorer, tile_container, canvas)

        preview_scale.bind("<ButtonRelease-1>", _apply_preview_size)

        def _on_container_configure(_event=None):
            # Keep scrollregion in sync with the full content size (all rows/cols).
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(event=None):
            # Stretch the embedded frame to the canvas width so columns reflow on resize,
            # then refresh scrollregion so vertical height is never clipped.
            try:
                cw = event.width if event is not None else canvas.winfo_width()
                if cw > 50:
                    canvas.itemconfig(win_id, width=cw)
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        tile_container.bind("<Configure>", _on_container_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            """Scroll the tile grid with the mouse wheel (Windows/macOS/Linux)."""
            try:
                if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                    canvas.yview_scroll(-3, "units")
                elif getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
                    canvas.yview_scroll(3, "units")
            except Exception:
                pass
            return "break"

        # Bind wheel on canvas + bubble from children via bind_all scoped while explorer is open
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)
        tile_container.bind("<MouseWheel>", _on_mousewheel)
        tile_container.bind("<Button-4>", _on_mousewheel)
        tile_container.bind("<Button-5>", _on_mousewheel)

        def _bind_wheel_recursive(widget):
            try:
                widget.bind("<MouseWheel>", _on_mousewheel)
                widget.bind("<Button-4>", _on_mousewheel)
                widget.bind("<Button-5>", _on_mousewheel)
            except Exception:
                pass
            for child in widget.winfo_children():
                _bind_wheel_recursive(child)

        explorer._bind_tile_wheel = _bind_wheel_recursive
        explorer._on_tile_mousewheel = _on_mousewheel

        ttk.Button(
            top, text="Refresh / Re-tile",
            command=lambda: self._populate_frame_tile_grid(explorer, tile_container, canvas),
        ).pack(side="right", padx=4)

        explorer.update_idletasks()
        self.after(30, lambda: self._populate_frame_tile_grid(explorer, tile_container, canvas))
        self.after(100, _on_canvas_configure)

    def _populate_frame_tile_grid(self, explorer_win, container, canvas_ref=None):
        """Build a fully scrollable grid of every tile on the selected frame.

        Uses live tile size + stride (overlap if enabled). Thumb size scales down when
        there are many tiles (typical for small non-overlapping sizes) so more of the
        frame is visible without scrolling, while the canvas still scrolls to the rest.
        """
        for w in container.winfo_children():
            w.destroy()

        self.current_frame_tiles = []
        self.current_frame_tile_labels = {}
        self._tile_photo_refs.clear()

        if not self.current_video or self.selected_frame is None:
            ttk.Label(container, text="Could not load frame.").pack()
            return

        try:
            import cv2
            cap = cv2.VideoCapture(self.current_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.selected_frame))
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                ttk.Label(container, text="Could not decode frame.").pack()
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame_rgb.shape[:2]

            try:
                from .tiling import GridTiler
            except Exception:
                from drone_ared.tiling import GridTiler

            tw, th = self._get_live_tile_size()
            sx, sy = self._get_live_stride(tw, th)
            tiler = GridTiler(tile_width=tw, tile_height=th, stride_x=sx, stride_y=sy)
            tiles = tiler.tile_frame(frame_rgb, self.selected_frame, 0, video_path=self.current_video)
            self.current_frame_tiles = tiles

            if not tiles:
                ttk.Label(
                    container,
                    text=f"No tiles generated for {tw}x{th} stride=({sx},{sy}) on {frame_w}x{frame_h} frame.",
                ).pack()
                return

            # Available width for column math (prefer live canvas width)
            try:
                if canvas_ref is not None:
                    avail_w = max(400, canvas_ref.winfo_width() or 0)
                else:
                    avail_w = 0
                if avail_w < 200:
                    avail_w = max(800, container.winfo_width() or explorer_win.winfo_width() or 1200)
            except Exception:
                avail_w = 1200

            n_tiles = len(tiles)
            # Preview size: default 300×300 (configurable via explorer slider).
            # Small native tiles are *upscaled* so they stay easy to view; large tiles are
            # downscaled to fit the same box. Dense grids still scroll fully.
            try:
                pvar = getattr(explorer_win, "_tile_preview_var", None)
                if pvar is not None:
                    thumb_size = max(80, min(640, int(pvar.get())))
                else:
                    thumb_size = int(getattr(self, "tile_preview_size", 300) or 300)
            except Exception:
                thumb_size = 300
            self.tile_preview_size = thumb_size

            padding = 14
            cols = max(1, (avail_w - 24) // (thumb_size + padding))
            # Cap columns reasonably; more columns when previews are large keeps the grid usable.
            cols = max(1, min(cols, 12))

            # How many grid rows the tiler produced (for status)
            max_r = max((t.tile_row for t in tiles), default=0) + 1
            max_c = max((t.tile_col for t in tiles), default=0) + 1

            info_var = getattr(explorer_win, "_tile_explorer_info_var", None)
            if info_var is not None:
                info_var.set(
                    f"Frame {self.selected_frame}  |  {n_tiles} tiles  "
                    f"({max_c}×{max_r} grid, tile {tw}×{th}, stride {sx}×{sy})  "
                    f"|  scroll for all rows  |  click a tile to label"
                )

            for idx, tile in enumerate(tiles):
                label = None
                rel = False
                try:
                    if self.tile_db:
                        sx_k, sy_k = self._get_live_stride(tile.width, tile.height)
                        key = TileKey(
                            self.current_video, tile.frame_idx, tile.tile_row, tile.tile_col,
                            tile.width, tile.height, stride_x=sx_k, stride_y=sy_k,
                        )
                        hit = self.tile_db.lookup_key(key)
                        if hit:
                            label, rel = hit
                except Exception:
                    pass

                self.current_frame_tile_labels[idx] = (label, rel)

                card = ttk.Frame(container, relief="groove", borderwidth=2, padding=3)
                r, c = divmod(idx, cols)
                card.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

                # Scale to a fixed preview box (upscale small tiles, downscale large ones).
                # thumbnail() never enlarges; resize() does, which is what we want for
                # small non-overlap tiles (e.g. 128→300).
                small = tile.image.copy()
                ow, oh = small.size
                if ow > 0 and oh > 0:
                    scale = min(thumb_size / ow, thumb_size / oh)
                    nw = max(1, int(round(ow * scale)))
                    nh = max(1, int(round(oh * scale)))
                    small = small.resize((nw, nh), Image.Resampling.LANCZOS)
                tkimg = ImageTk.PhotoImage(small)
                self._tile_photo_refs.append(tkimg)

                img_lbl = ttk.Label(card, image=tkimg)
                img_lbl.pack()

                status = f"r{tile.tile_row}c{tile.tile_col}  {label or 'unlabeled'}{' [R]' if rel else ''}"
                ttk.Label(card, text=status, width=max(18, thumb_size // 10), anchor="center").pack(pady=2)

                def make_edit(t=tile, i=idx):
                    return lambda e: self._quick_label_tile(explorer_win, t, i, container, canvas_ref)

                for child in (card, img_lbl):
                    child.bind("<Button-1>", make_edit())
                    # Wheel over cards must still scroll the outer canvas
                    mw = getattr(explorer_win, "_on_tile_mousewheel", None)
                    if mw is not None:
                        try:
                            child.bind("<MouseWheel>", mw)
                            child.bind("<Button-4>", mw)
                            child.bind("<Button-5>", mw)
                        except Exception:
                            pass

            for i in range(cols):
                container.columnconfigure(i, weight=1)

            # Layout + scrollregion so every row is reachable
            container.update_idletasks()
            if canvas_ref is not None:
                try:
                    cw = canvas_ref.winfo_width()
                    win_id = getattr(explorer_win, "_tile_win_id", None)
                    if win_id is not None and cw > 50:
                        canvas_ref.itemconfig(win_id, width=cw)
                    canvas_ref.configure(scrollregion=canvas_ref.bbox("all"))
                    # Start scrolled to top so the first tiles are visible
                    canvas_ref.yview_moveto(0.0)
                except Exception:
                    pass

            # Bind wheel on all nested widgets created above
            binder = getattr(explorer_win, "_bind_tile_wheel", None)
            if binder is not None:
                try:
                    binder(container)
                except Exception:
                    pass

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Tile Explorer", str(e))


    def _quick_label_tile(self, parent, tile: "Tile", tile_local_idx: int, grid_container, canvas_ref):
        """Quick label dialog for a specific tile. Saves to DB and refreshes grid."""
        q = tk.Toplevel(parent)
        q.title(f"Label Tile r{tile.tile_row}c{tile.tile_col} (f{tile.frame_idx})")
        q.geometry("680x980")

        s = self.ui_scale
        # Image: upscale small tiles so they are easy to see in the edit dialog
        big = tile.image.copy()
        target = max(300, int(getattr(self, "tile_preview_size", 300) or 300))
        target = min(520, max(target, 420))  # roomy default for the dialog
        ow, oh = big.size
        if ow > 0 and oh > 0:
            scale = min(target / ow, target / oh)
            nw = max(1, int(round(ow * scale)))
            nh = max(1, int(round(oh * scale)))
            big = big.resize((nw, nh), Image.Resampling.LANCZOS)

        tkbig = ImageTk.PhotoImage(big)
        ttk.Label(q, image=tkbig).pack(pady=4)
        # keep ref on q
        q._img_ref = tkbig

        # Current
        cur = self.current_frame_tile_labels.get(tile_local_idx, (None, False))
        cur_label, cur_rel = cur

        ttk.Label(q, text=f"Current: {cur_label or 'unlabeled'}  (relevant={cur_rel})").pack()

        # Known classes from DB (good for this browser)
        known = []
        try:
            known = self.tile_db.get_all_labels()
        except Exception:
            pass
        if not known and self.main_window:
            try:
                known = list(getattr(self.main_window, 'discovered_classes', [])) or []
            except:
                pass

        # Determine suggested relevant flag from known class relevance (auto-apply to prevent errors)
        suggested_rel = cur_rel
        if cur_label and cur_label in self.class_relevance:
            suggested_rel = self.class_relevance[cur_label]

        large_font = ("TkDefaultFont", int(10*self.ui_scale))

        ttk.Label(
            q,
            text="Existing classes (↑↓ select, double-click or Assign; 1-9 quick-assign):",
        ).pack(anchor="w", padx=8)
        lb = tk.Listbox(q, height=10, exportselection=False, font=large_font)
        lb.pack(fill="x", padx=8)
        known_sorted = sorted(set(known))
        for k in known_sorted:
            lb.insert("end", k)
        if cur_label and cur_label in known_sorted:
            try:
                idx = known_sorted.index(cur_label)
                lb.selection_set(idx)
            except Exception:
                pass

        # New class entry + relevance (defined before key handlers so closures resolve cleanly)
        newf = ttk.Frame(q)
        newf.pack(fill="x", padx=8, pady=4)
        ttk.Label(newf, text="New / Edit class:").pack(side="left")
        new_var = tk.StringVar(value=cur_label or "")
        new_entry = ttk.Entry(newf, textvariable=new_var)
        new_entry.pack(side="left", fill="x", expand=True, padx=4)

        rel_var = tk.BooleanVar(value=suggested_rel)
        ttk.Checkbutton(q, text="Relevant (interesting / anomaly)", variable=rel_var).pack(anchor="w", padx=8)

        def do_assign():
            name = new_var.get().strip() or (lb.get(lb.curselection()[0]) if lb.curselection() else None)
            if not name:
                messagebox.showwarning("Label", "Enter or select a class name.")
                return
            rel = rel_var.get()
            # Enforce class-level relevance: if this class is known relevant, the tile must be too.
            # This prevents the common mistake of forgetting to tick "relevant" on every instance.
            if name in self.class_relevance and self.class_relevance[name]:
                rel = True
            # If the user marked this (new) class as relevant, remember it for future tiles
            if rel:
                self.class_relevance[name] = True
            try:
                cx, cy = tile.bbox[0], tile.bbox[1]
                sx, sy = self._get_live_stride(tile.width, tile.height)
                if self.annotation_manager:
                    key = TileKey(self.current_video, tile.frame_idx, tile.tile_row, tile.tile_col,
                                  tile.width, tile.height, stride_x=sx, stride_y=sy)
                    self.annotation_manager.db.set_annotation_for_key(key, name, rel, embedding=None, crop_x=cx, crop_y=cy)
                else:
                    self.tile_db.set_annotation(
                        self.current_video, tile.frame_idx,
                        tile.tile_row, tile.tile_col, tile.width, tile.height,
                        name, rel, embedding=None, crop_x=cx, crop_y=cy
                    )
                # Update local
                self.current_frame_tile_labels[tile_local_idx] = (name, rel)
                q.destroy()
                # Refresh the grid
                self._populate_frame_tile_grid(parent, grid_container, canvas_ref)
                # Light refresh: update counts on strip cards, keep thumbs loaded
                self._refresh_annotations_light()

                # Make the name known immediately for clickability in the main lists.
                # Do NOT bump any "query count" here — this is manual labeling in the browser,
                # not an A/RED query decision. The A/RED query counts come only from the adapter.
                try:
                    if self.main_window is not None:
                        self.main_window.discovered_classes.add(name)
                        if hasattr(self.main_window, '_refresh_class_list'):
                            self.main_window._refresh_class_list()
                except Exception:
                    pass
            except Exception as e:
                messagebox.showerror("Save", str(e))

        def _on_class_selected(event=None):
            try:
                sel = lb.get(lb.curselection())
                if sel in self.class_relevance:
                    rel_var.set(self.class_relevance[sel])
                new_var.set(sel)
            except Exception:
                pass
        lb.bind("<<ListboxSelect>>", _on_class_selected)
        lb.bind("<Double-Button-1>", lambda e: do_assign())

        def _nav_lb(delta: int):
            if not known_sorted:
                return "break"
            sel = lb.curselection()
            if sel:
                idx = max(0, min(len(known_sorted) - 1, sel[0] + delta))
            else:
                idx = 0 if delta >= 0 else len(known_sorted) - 1
            lb.selection_clear(0, "end")
            lb.selection_set(idx)
            lb.see(idx)
            lb.activate(idx)
            _on_class_selected()
            return "break"

        def _assign_lb_number(n: int):
            focus = q.focus_get()
            if focus is new_entry:
                return
            if 1 <= n <= len(known_sorted):
                name = known_sorted[n - 1]
                new_var.set(name)
                if name in self.class_relevance:
                    rel_var.set(self.class_relevance[name])
                do_assign()

        q.bind("<Up>", lambda e: _nav_lb(-1))
        q.bind("<Down>", lambda e: _nav_lb(1))
        for i in range(1, 10):
            q.bind(f"<Key-{i}>", lambda e, n=i: _assign_lb_number(n))

        ttk.Button(q, text="Assign / Update Label", command=do_assign).pack(fill="x", padx=8, pady=6)
        ttk.Button(q, text="Mark Background", command=lambda: (new_var.set("__BACKGROUND__"), rel_var.set(False), do_assign())).pack(fill="x", padx=8)
        ttk.Button(q, text="Cancel", command=q.destroy).pack(fill="x", padx=8, pady=2)
        q.after(100, lambda: lb.focus_set())

    def _edit_selected_tile_from_list(self):
        """Edit from the right-side list of labels on the selected frame."""
        sel = self.frame_labels_list.curselection()
        if not sel or self.selected_frame is None:
            return
        idx = sel[0]
        if not hasattr(self, '_current_frame_anns_list') or idx >= len(self._current_frame_anns_list):
            return
        ann = self._current_frame_anns_list[idx]

        # Re-extract the tile image for nice preview
        bbox = (ann.get("crop_x", ann["tile_col"]*ann["tile_width"]),
                ann.get("crop_y", ann["tile_row"]*ann["tile_height"]),
                ann.get("crop_x", ann["tile_col"]*ann["tile_width"]) + ann["tile_width"],
                ann.get("crop_y", ann["tile_row"]*ann["tile_height"]) + ann["tile_height"])

        img = extract_tile_from_video(ann["video_path"], ann["abs_frame"], bbox)

        # Simple editor (reuse spirit of review, but quick)
        ed = tk.Toplevel(self)
        ed.title(f"Edit label f{ann['abs_frame']} r{ann['tile_row']}c{ann['tile_col']}")
        ed.geometry("520x620")
        if img:
            disp = img.copy()
            disp.thumbnail((320, 320))
            tkd = ImageTk.PhotoImage(disp)
            ttk.Label(ed, image=tkd).pack()
            ed._ref = tkd

        lvar = tk.StringVar(value=ann["label"])
        ttk.Entry(ed, textvariable=lvar).pack(fill="x", padx=8, pady=4)

        rvar = tk.BooleanVar(value=ann["relevant"])
        ttk.Checkbutton(ed, text="Relevant", variable=rvar).pack()

        def save():
            try:
                new_label = lvar.get().strip() or "__UNLABELED__"
                new_rel = rvar.get()
                # Enforce known class relevance for this tile
                if new_label in self.class_relevance and self.class_relevance[new_label]:
                    new_rel = True
                if new_rel:
                    self.class_relevance[new_label] = True
                self.tile_db.set_annotation(
                    ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                    ann["tile_width"], ann["tile_height"],
                    new_label, new_rel,
                    crop_x=ann.get("crop_x"), crop_y=ann.get("crop_y")
                )
                ed.destroy()
                self._refresh_annotations_light()
            except Exception as e:
                messagebox.showerror("Edit", str(e))

        ttk.Button(ed, text="Save Change to DB", command=save).pack(fill="x", padx=8, pady=6)
        def _do_delete():
            sx = ann.get("stride_x")
            sy = ann.get("stride_y")
            sx = ann.get("stride_x")
            sy = ann.get("stride_y")
            if self.annotation_manager:
                key = TileKey(ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                              ann["tile_width"], ann["tile_height"], stride_x=sx, stride_y=sy)
                self.annotation_manager.db.delete_key(key)
            else:
                self.tile_db.delete_annotation(
                    ann["video_path"], ann["abs_frame"], ann["tile_row"], ann["tile_col"],
                    ann["tile_width"], ann["tile_height"], stride_x=sx, stride_y=sy)
            ed.destroy()
            self._refresh_annotations_light()
        ttk.Button(ed, text="Delete this annotation", command=_do_delete).pack(fill="x", padx=8)

    def _jump_to_frame_in_main(self):
        """Use the existing controller navigation so the main LabelingDialog will show tiles from this frame."""
        if self.selected_frame is None:
            return
        if not self.controller:
            messagebox.showinfo("Jump", "No controller available (start Label Only mode from main window).")
            return
        try:
            self.controller.label_only_jump_to_frame(self.selected_frame)
            self.browser_status_var.set(f"Jumped controller to frame {self.selected_frame}. Use the main labeling window.")
        except Exception as e:
            messagebox.showerror("Jump", str(e))

    def _open_in_review(self):
        """Open the classic list review window for convenience."""
        if self.current_video and self.tile_db:
            # The LabelReviewWindow loads its own list; just open it
            LabelReviewWindow(self, self.tile_db, ui_scale=self.ui_scale,
                              annotation_manager=getattr(self, 'annotation_manager', None))
