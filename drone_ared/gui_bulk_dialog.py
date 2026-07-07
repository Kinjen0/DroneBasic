"""
gui_bulk_dialog.py

Reusable GUI component for mass / bulk label editing and removal.

Provides a dialog with precise filters (via AnnotationFilter) so users can
safely do things like:
- Delete all "dirt" labels on a specific video at 128px tile size
- Change "person" to "pedestrian" for frames 100-500 where relevant=True
- Preview counts first

This keeps the bulk logic out of the main window, improving modularity.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Any

from .annotation_manager import AnnotationManager
from .annotation_domain import AnnotationFilter


class BulkLabelOpsDialog(tk.Toplevel):
    """Dialog for targeted bulk label operations.

    See _open_bulk_label_ops in main GUI for usage.
    """

    def __init__(self, master, manager: AnnotationManager, config: Any, ui_scale: float = 1.6,
                 on_done: Optional[Callable] = None):
        super().__init__(master)
        self.manager = manager
        self.config = config
        self.ui_scale = float(ui_scale) if ui_scale else 1.6
        self.on_done = on_done

        self.title("Bulk Label Operations - Precise Mass Edit/Remove")
        self.geometry(f"{int(700*min(self.ui_scale,1.8))}x{int(520*min(self.ui_scale,1.8))}")
        self.resizable(True, True)

        self._build_ui()
        # Prefill from current manager scope if available
        v = getattr(self.manager, '_current_video', None)
        if v:
            self.video_var.set(v)
        size = getattr(self.manager, '_current_tile_size', None)
        if size:
            tw, th = size
            self.tw_var.set(str(tw))
            self.th_var.set(str(th))
            self.use_current_size_var.set(True)
        self._refresh_preview()

    def _build_ui(self):
        s = self.ui_scale
        fs = int(10 * s)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=4)
        ttk.Label(top, text="Current scope from DB manager (can override below):").pack(anchor="w")
        self.scope_var = tk.StringVar(value="Using manager scope")
        ttk.Label(top, textvariable=self.scope_var, font=("TkDefaultFont", fs)).pack(anchor="w")

        filt_frame = ttk.LabelFrame(self, text="Filter criteria (more specific = safer mass changes)")
        filt_frame.pack(fill="x", padx=8, pady=4)

        lf = ttk.Frame(filt_frame)
        lf.pack(fill="x", pady=2)
        ttk.Label(lf, text="Labels to affect (comma sep or select):").pack(side="left")
        self.labels_var = tk.StringVar()
        ttk.Entry(lf, textvariable=self.labels_var, width=40).pack(side="left", padx=4)
        self.label_list = tk.Listbox(filt_frame, height=4, exportselection=False)
        self.label_list.pack(fill="x", padx=4, pady=2)
        self.label_list.bind("<<ListboxSelect>>", self._on_label_select)

        vf = ttk.Frame(filt_frame)
        vf.pack(fill="x", pady=2)
        ttk.Label(vf, text="Video (blank = all):").pack(side="left")
        self.video_var = tk.StringVar()
        ttk.Entry(vf, textvariable=self.video_var, width=50).pack(side="left", padx=4)
        ttk.Button(vf, text="Use Current", command=self._use_current_video).pack(side="left")

        sf = ttk.Frame(filt_frame)
        sf.pack(fill="x", pady=2)
        self.use_current_size_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Use current tile size from config", variable=self.use_current_size_var).pack(side="left")
        ttk.Label(sf, text="Override W:").pack(side="left")
        default_tw = getattr(getattr(self.config, 'tiling', None), 'tile_width', 256)
        default_th = getattr(getattr(self.config, 'tiling', None), 'tile_height', 256)
        self.tw_var = tk.StringVar(value=str(default_tw))
        ttk.Entry(sf, textvariable=self.tw_var, width=6).pack(side="left")
        ttk.Label(sf, text="H:").pack(side="left")
        self.th_var = tk.StringVar(value=str(default_th))
        ttk.Entry(sf, textvariable=self.th_var, width=6).pack(side="left")

        rf = ttk.Frame(filt_frame)
        rf.pack(fill="x", pady=2)
        self.relevant_var = tk.StringVar(value="")
        ttk.Label(rf, text="Relevant (blank=any):").pack(side="left")
        ttk.Combobox(rf, textvariable=self.relevant_var, values=["", "yes", "no"], width=6).pack(side="left")
        ttk.Label(rf, text="Frames from:").pack(side="left", padx=(8,0))
        self.fmin_var = tk.StringVar()
        ttk.Entry(rf, textvariable=self.fmin_var, width=8).pack(side="left")
        ttk.Label(rf, text="to:").pack(side="left")
        self.fmax_var = tk.StringVar()
        ttk.Entry(rf, textvariable=self.fmax_var, width=8).pack(side="left")

        act_frame = ttk.LabelFrame(self, text="Action")
        act_frame.pack(fill="x", padx=8, pady=4)
        self.action_var = tk.StringVar(value="reassign")
        ttk.Radiobutton(act_frame, text="Reassign to:", variable=self.action_var, value="reassign").pack(side="left")
        self.new_label_var = tk.StringVar()
        ttk.Entry(act_frame, textvariable=self.new_label_var, width=25).pack(side="left", padx=4)
        self.set_rel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(act_frame, text="Set relevant on new", variable=self.set_rel_var).pack(side="left")

        ttk.Radiobutton(act_frame, text="DELETE matching", variable=self.action_var, value="delete").pack(side="left", padx=10)

        prev_frame = ttk.LabelFrame(self, text="Preview (counts before change)")
        prev_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self.preview_text = tk.Text(prev_frame, height=8, width=70, state="disabled", font=("TkDefaultFont", fs))
        self.preview_text.pack(fill="both", expand=True, padx=4, pady=2)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8, pady=6)
        ttk.Button(btns, text="Refresh Preview", command=self._refresh_preview).pack(side="left")
        ttk.Button(btns, text="Execute (irreversible!)", command=self._execute_bulk).pack(side="left", padx=10)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")

        self._populate_known_labels()

    def _populate_known_labels(self):
        try:
            labels = self.manager.db.get_all_labels()
            self.label_list.delete(0, "end")
            for lbl in labels:
                self.label_list.insert("end", lbl)
        except Exception:
            pass

    def _on_label_select(self, event=None):
        try:
            sel = self.label_list.get(self.label_list.curselection())
            current = self.labels_var.get().strip()
            if current:
                if sel not in current.split(","):
                    self.labels_var.set(current + "," + sel)
            else:
                self.labels_var.set(sel)
        except Exception:
            pass

    def _use_current_video(self):
        v = getattr(self.manager, '_current_video', None)
        if v:
            self.video_var.set(v)

    def _build_filter(self) -> AnnotationFilter:
        labels = [l.strip() for l in self.labels_var.get().split(",") if l.strip()]
        video = self.video_var.get().strip() or None
        tw = th = None
        tw = 256
        th = 256
        if hasattr(self.config, 'tiling'):
            tw = getattr(self.config.tiling, 'tile_width', 256)
            th = getattr(self.config.tiling, 'tile_height', 256)
        if self.use_current_size_var.get():
            pass  # already set above
        else:
            try:
                tw = int(self.tw_var.get()) if self.tw_var.get() else tw
                th = int(self.th_var.get()) if self.th_var.get() else th
            except Exception:
                pass
        rel = None
        if self.relevant_var.get() == "yes":
            rel = True
        elif self.relevant_var.get() == "no":
            rel = False
        fmin = int(self.fmin_var.get()) if self.fmin_var.get().strip() else None
        fmax = int(self.fmax_var.get()) if self.fmax_var.get().strip() else None
        return AnnotationFilter(
            video_path=video, labels=labels or None,
            tile_width=tw, tile_height=th, relevant=rel,
            frame_min=fmin, frame_max=fmax
        )

    def _refresh_preview(self):
        filt = self._build_filter()
        try:
            # Route through manager (use_scope=False so dialog's precise filt wins; manager safely forwards)
            counts = self.manager.get_label_counts(use_scope=False, filt=filt)
            total = sum(counts.values())
            lines = [f"{lbl}: {cnt}" for lbl, cnt in counts.items()]
            text = f"Would affect {total} annotations:\n" + "\n".join(lines) if lines else "No matches for current filter."
            self.preview_text.config(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", text)
            self.preview_text.config(state="disabled")
            self.scope_var.set(f"Video scope: {filt.video_path or 'all'} | Size: {filt.tile_width}x{filt.tile_height or 'any'}")
        except Exception as e:
            self.preview_text.config(state="normal")
            self.preview_text.delete("1.0", "end")
            self.preview_text.insert("1.0", f"Preview error: {e}")
            self.preview_text.config(state="disabled")

    def _execute_bulk(self):
        action = self.action_var.get()
        filt = self._build_filter()
        if not filt.labels:
            messagebox.showwarning("Bulk", "Specify at least one label to affect (in filter).")
            return

        # Use the full user-specified filt for preview counts (ignore outer manager scope for this dialog's choices)
        # Go through manager.get_label_counts (with use_scope=False) for consistent filtering + safety.
        counts = self.manager.get_label_counts(use_scope=False, filt=filt)
        total = sum(counts.values())
        if total == 0:
            messagebox.showinfo("Bulk", "Nothing to do.")
            return

        if action == "delete":
            action_str = "DELETE"
            new_lbl = None
        else:
            new_lbl = self.new_label_var.get().strip()
            if not new_lbl:
                messagebox.showwarning("Bulk", "Enter a new label name.")
                return
            action_str = f"REASSIGN to '{new_lbl}'"

        if not messagebox.askyesno("Confirm Bulk", f"{action_str} {total} tiles?\n{counts}\n\nThis cannot be undone."):
            return

        try:
            if action == "delete":
                n = self.manager.bulk_delete(labels=filt.labels, use_scope=False, filt=filt)
                msg = f"Deleted {n} annotations."
            else:
                n = self.manager.bulk_reassign(old_label=None, new_label=new_lbl, old_labels=filt.labels, use_scope=False, filt=filt)
                msg = f"Reassigned {n} annotations to '{new_lbl}'."
            self.manager.db.conn.commit()
            messagebox.showinfo("Bulk Done", msg)
            self._refresh_preview()
            if self.on_done:
                self.on_done()
        except Exception as e:
            messagebox.showerror("Bulk Error", str(e))
