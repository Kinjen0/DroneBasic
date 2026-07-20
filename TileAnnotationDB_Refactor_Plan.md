# Tile Annotation Database & UI System Refactor Plan

**Date:** 2026-07-07  
**Context:** DroneBasic / drone_ared project (A/RED anomaly detection + human labeling of drone video tiles via GridTiler + DINO).  
**Primary pain points raised:**
1. Annotations are **not isolated by video + tile size**. Setting tile size 128 still shows/leaks 256px tiles/frames in browsers and review UIs.
2. The per-frame **tile viewer** (after selecting a frame in MultiFrameLabelBrowser) does not fill the window — content occupies roughly half, rest is black/empty canvas space.
3. No support for **significant / bulk DB operations**: mass delete labels (e.g. all "dirt"), mass reassign ("dirt" → "debris"), etc.
4. No convenient **DB lifecycle management**: rename/switch DB files to experiment with different labeling methodologies, clone a DB for editing experiments.
5. Overall code has grown monolithic. Need better **readability, expandability, OOP + Single Responsibility Principle (SRP)**.

This plan addresses the concrete bugs **and** proposes a cleaner architecture so future features (multi-scale, per-video projects, audit logs, export, ML-assisted bulk suggestions, etc.) are easy to add.

---

## 1. Current State Analysis (Problems & SRP Violations)

### 1.1 Storage (tile_database.py)
- `TileAnnotationDB` is a **god class**:
  - Manages sqlite connection, WAL, schema creation.
  - Does all CRUD (`lookup`, `set_annotation`, `get_annotations_for_video`, `delete_annotation`).
  - Provides convenience queries (`list_videos`, `get_all_labels`, `get_class_relevance`).
  - Holds a standalone helper `extract_tile_from_video`.
- Schema is already good for isolation:
  ```sql
  PRIMARY KEY (video_path, abs_frame, tile_row, tile_col, tile_width, tile_height)
  ```
  Different tile sizes are **separate rows**. The problem is **lack of filtering** at query + call sites.
- No `AnnotationFilter` object. No bulk update/delete.
- No notion of "current session scope" (video + active tile size).
- `get_annotations_for_video(v)` returns **everything** for the video, any resolution ever used.

### 1.2 Usage Sites (gui.py — biggest monolith)
- `MainWindow` (~1000+ lines): mixes param UI, starting pipelines, DB open/load, stats, metrics, launching sub-windows.
- `LabelReviewWindow`: calls `get_annotations_for_video(v)` with no size filter. List shows mixed resolutions.
- `MultiFrameLabelBrowser` (the multi-frame + per-frame explorer, ~1400 lines):
  - `_load_annotations_for_video`: `anns = self.tile_db.get_annotations_for_video(v)` — no filter.
  - Builds `frame_to_anns` and candidate frames from **all** historical sizes.
  - Only the per-frame `_populate_frame_tile_grid` does a current-size `lookup`.
  - Result: frames that only had 256 labels appear when using 128; counts are wrong.
- Per-frame tile grid (`_open_frame_tile_explorer` + `_populate...`):
  - Hard-coded `cols = 4`, `thumbnail((180,180))`.
  - `canvas.pack(fill="both", expand=True)`
  - `tile_container = ttk.Frame(canvas); canvas.create_window((0,0), window=..., anchor="nw")`
  - No `<Configure>` handler to recompute columns or stretch the window item.
  - Grid children determine a fixed natural width → large right-side black area on wide windows.
- Direct DB calls scattered everywhere (`set_annotation`, `delete`, `lookup` from many methods).
- Ad-hoc DB switching in `_load_tile_annotations` (closes old, opens new, mutates config). No clone, no "new project DB", no safety around "current tile size may not match DB contents".
- There is already a "Load different Annotation DB..." button and path field — good, but incomplete.

### 1.3 Pipeline / Controller (pipeline.py)
- Good: `lookup` **does** pass the current `tw, th` from the active `GridTiler` / meta. New labels are saved with correct size.
- `set_tile_database` injection exists.
- Label provider logic mixes exact DB + embedding store + GUI fallback.
- No bulk awareness.

### 1.4 Other Stores
- `PersistentLabelStore` (label_store.py) is separate (embedding similarity). It is **not** keyed by tile size/video identity. It is intentionally "looks like" and should probably stay loosely coupled.
- Two stores serve different purposes; any refactor should keep a clear boundary.

### 1.5 SRP / Modularity Violations Summary
- UI classes own too much (layout + event handling + data loading + business rules + async workers).
- No domain model for "what uniquely identifies a tile for labeling".
- Data access, filtering, and mutation are not separated.
- No service layer for cross-cutting ops (bulk edits, session-scoped views).
- Layout widgets are not reusable/components.
- DB file management is sprinkled in GUI + raw `TileAnnotationDB(db_path=...)`.

---

## 2. Goals for the New System

- **Correct isolation**: All "current work" views (MultiFrameBrowser, Review, labeling stats, per-frame grids) are **always scoped to (video, current_tile_size)** unless the user explicitly chooses "show all historical sizes".
- **Bulk power**: First-class support for mass label changes/deletes with safe preview + undo-friendly logging (at minimum audit in console + updated_ts).
- **DB as first-class workspace**: Easy "New labeling DB", "Clone current as...", "Switch DB", recent list. Changing DB should be a deliberate act with warnings about tile size mismatch.
- **Readable & expandable (OOP + SRP)**:
  - Small classes with single responsibilities.
  - Clear layers: Domain → Repository → Service → UI (view models / controllers).
  - Dependency injection (pass repositories/services instead of raw DB objects).
  - Reusable UI components (`ResponsiveTileGrid`, `VirtualFrameStrip`, `BulkLabelEditor`).
- **Backward compat**: Old DBs keep working. Mixed-size data remains queryable when user wants it.
- **Performance**: Filtering at DB level (SQL WHERE) when possible. Keep virtualized UIs.

---

## 3. Proposed Architecture (Layered, SRP)

### 3.1 Domain Layer (`drone_ared/domain/` or inside existing, small files)
```python
@dataclass(frozen=True)
class TileKey:
    video_path: str
    abs_frame: int
    tile_row: int
    tile_col: int
    tile_width: int
    tile_height: int

    def to_tuple(self) -> tuple: ...
    @classmethod
    def from_annotation_dict(cls, d): ...
    def size(self) -> Tuple[int,int]: ...

@dataclass
class TileAnnotation:
    key: TileKey
    label: str
    relevant: bool
    crop: Optional[Tuple[int,int,int,int]] = None
    updated_ts: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None   # or bytes
```

`TileKey` becomes the primary currency passed around instead of 6 separate ints. This alone improves readability dramatically.

### 3.2 Repository Layer (Pure data access, SRP: "talk to sqlite")
- `TileAnnotationRepository` (ABC or protocol)
  - `get(key: TileKey) -> Optional[TileAnnotation]`
  - `save(ann: TileAnnotation)`
  - `delete(key: TileKey)`
  - `query(filter: AnnotationFilter) -> List[TileAnnotation]`
  - `bulk_update_label(filter, new_label)`
  - `bulk_delete(filter)`
  - `distinct_labels(filter=None)`, `distinct_tile_sizes(video)`, `list_videos()`
  - `get_stats(filter=None)`
- `SqliteTileAnnotationRepository(TileAnnotationRepository)`
  - Holds the connection.
  - Implements the above with proper parameterized queries + the existing composite PK.
- (Future: `InMemoryRepository` for tests, or `ReadOnly` wrapper.)

Add `AnnotationFilter` dataclass (all fields Optional):
```python
@dataclass
class AnnotationFilter:
    video_path: Optional[str] = None
    labels: Optional[List[str]] = None
    tile_width: Optional[int] = None
    tile_height: Optional[int] = None
    relevant: Optional[bool] = None
    frame_range: Optional[Tuple[int,int]] = None
    # etc.
```

### 3.3 Service / Application Layer
- `AnnotationManager` (or `TileLabelingService`)
  - Wraps a repository + current "session context" (optional active video + tiling).
  - `get_annotations_for_current_scope()` (applies active tile size filter unless "include_all_sizes").
  - `save_for_tile(tile_or_key, label, relevant, ...)`
  - `bulk_reassign_labels(old_label: str, new_label: str, filter: AnnotationFilter = None)`
  - `purge_label(label: str, filter=None)`
  - `clone_to(new_path: Path)` (delegates to manager below)
  - Provides "current" view for UI without UI knowing SQL.
- `AnnotationDatabaseManager`
  - Knows about the filesystem side of DBs (in a directory, or explicit paths).
  - `open(path) -> AnnotationManager`
  - `create_new(base_name="project_labels") -> (path, manager)`
  - `clone(source_manager, dest_path) -> manager`
  - `close_current()`
  - Tracks "recent DBs", default location.
  - Handles safe close (commit + handle WAL files for copy).

The GUI and Pipeline talk to `AnnotationManager` (or the repo via the manager), never raw `TileAnnotationDB`.

### 3.4 UI / Presentation Layer (much thinner)
- Keep big windows but delegate:
  - `MultiFrameLabelBrowser` receives an `AnnotationManager` + active `TilingConfig` (or size tuple) + video.
  - It asks the manager for "frames with annotations in current size".
  - Per-frame explorer uses a new `ResponsiveTileGrid(explorer, tile_list, on_tile_click=...)`.
- `ResponsiveTileGrid`:
  - Owns the canvas + container.
  - On `<Configure>` (debounced): `cols = max(1, (width - margin) // (desired_thumb + spacing))`; destroy children; re-grid with current thumb size (or make thumb scale too).
  - Or: stretch the create_window width and compute cols to fill.
  - Supports "fill available" vs "fixed cols + slider".
- New dialog/panel: `BulkAnnotationEditor(manager, initial_filter=None)`
  - List or summary of matching annotations ("127 tiles labeled 'dirt' in 128px on DJI_0017").
  - "Reassign to: ____ + Relevant checkbox"
  - "Delete all matching"
  - Preview table (limited rows) + "Confirm (X affected)".
  - Optional: "Also affect embedding store?" (advanced).
- DB management menu or section:
  - Current DB path + entry count + "sizes used in this DB".
  - Buttons: New DB..., Open/ Switch..., Clone current to..., Compact/Backup.
  - On switch/clone: warn if current GUI tile size has no (or few) matching annotations in the target DB.

- Inject the manager into sub-windows instead of raw `tile_db`.

### 3.5 Other SRP Splits
- `VirtualScrollingStrip` (extract the card virtualization logic from MultiFrameLabelBrowser so it can be reused).
- `LabelRequest` / dialog coordination stays in controller but becomes clearer.
- Keep `PersistentLabelStore` mostly as-is (different responsibility: similarity, not identity).
- Config remains the source of truth for "active" tile size; pass `config.tiling` or a `(w,h)` down when constructing views.

---

## 4. Concrete Fixes (Prioritized)

### Phase 0 / Immediate (small changes, low risk)
1. Add optional filtering to `TileAnnotationDB` (or new repo methods) right away:
   - `get_annotations_for_video(video, tile_width=None, tile_height=None, ...)`
   - Update internal SELECT to add `AND tile_width=? AND tile_height=?` when provided.
2. In **all** call sites that mean "current session":
   - `MultiFrameLabelBrowser._load...`, `LabelReviewWindow`, review lists, stats, etc.
   - Pass the active size from `main_window.config.tiling` or controller.tiler.
   - Add a checkbox "Include other tile sizes for this video" (advanced) that relaxes the filter.
3. Fix the per-frame tile viewer **immediately**:
   - Bind `<Configure>` on the explorer canvas (debounced).
   - Compute dynamic `cols = max(2, (canvas_width - 20) // (thumb_w + 6))`
   - Or keep user "preferred thumb" but always compute cols to fill.
   - After computing cols, `container.grid_columnconfigure` or just re-`grid` the cards (destroy + recreate is acceptable for one frame's tiles — usually < 50-100).
   - Make the create_window item width track canvas width: `canvas.itemconfig(win, width=canvas_w)`.
   - Add a horizontal scrollbar for when user wants very large thumbs.
   - Consider a "Fill width" toggle vs fixed-cols mode.
4. Expose a couple bulk methods on the existing `TileAnnotationDB` (for quick wins):
   ```python
   def delete_by_label(self, label: str, video_path: Optional[str] = None,
                       tile_width: Optional[int] = None, ...) -> int
   def reassign_label(self, old_label: str, new_label: str, **filter_kwargs) -> int
   def get_label_counts(self, video_path=None, tile_size=None) -> Dict[str, int]
   ```
   Wire simple buttons or a mini dialog from the review window / multi-frame right panel.

### Phase 1 — Domain + Repository (clean foundation)
- Introduce `TileKey` + `TileAnnotation` + `AnnotationFilter`.
- Refactor `TileAnnotationDB` internals or create `SqliteTileAnnotationRepository` that uses the domain objects.
- Keep the old class as a thin adapter for one release if needed, or do a mechanical rename + update call sites.
- Add the bulk methods to the repository using safe UPDATE/DELETE.

### Phase 2 — Service + DB Manager
- `AnnotationDatabaseManager` + `AnnotationManager`.
- Update `PipelineController.set_tile_database(...)` → `set_annotation_manager(...)`.
- Update GUI startup and the existing load button to go through the manager.
- Implement clone (close repo, shutil.copy the .db + handle sidecars, reopen).

### Phase 3 — UI & Polish
- Responsive tile grid component.
- Bulk edit dialog (nice table of affected items, confirmation).
- DB switcher that shows "tile sizes present in this DB" and "match with current GUI size?".
- Propagate size filter into the frame strip virtualization (so only frames that have annotations **at the current size** contribute to the virtual list).
- Metrics / stats should also be able to scope to a size (they already try to be careful via processed tiles).
- Add "Compact DB" or VACUUM helper.
- Documentation in README + the plan file.

### Phase 4 — Expandability
- Make the two stores implement a common `LabelProvider` protocol where it makes sense.
- Support "annotation projects" (sub-tables or separate DBs per experiment).
- Export / import (CSV, COCO-style for tiles, etc.).
- Undo stack for bulk ops (record the old values before UPDATE).
- Versioned schema + migration on open.

---

## 5. Migration & Compatibility Notes

- Schema does **not** need change (tile_w/h already in PK and columns).
- Old mixed DBs remain usable; the filter simply selects the matching subset.
- When user opens an old DB with a new tile size that has zero annotations, the browser should show "No annotations at current 128px size for these frames. (X annotations exist at other sizes.)" + easy "Show all" button.
- `extract_tile_from_video` can move to a utility module or become a method on `TileKey`.
- Update all tests (if any) and the various places that construct `TileAnnotationDB(...)` directly (GUI, controller tests, run scripts).

---

## 6. Risks & Mitigations

- **Large refactor risk**: Do it in phases. Phase 0 (filtering + bulk on existing class + viewer fix) can ship quickly without breaking callers.
- **Tkinter layout is fiddly**: The responsive grid will need iteration + user testing on different window sizes and tile counts. Provide both "auto-fill" and "user cols + thumb size" modes (the strip already has good sliders — reuse the pattern).
- **Concurrent access / WAL**: Sqlite is fine; keep the existing WAL pragma. Manager should serialize opens.
- **Performance on huge DBs**: Add LIMIT + pagination to query when used for lists. The virtual strip already protects the frame browser.
- **User confusion on "which size am I labeling?"**: Surface the active tile size prominently (already in params). In the multi-frame browser title or status, show "Current tile size: 128x128 (filtering annotations)".

---

## 7. Suggested File / Module Structure After Refactor

```
drone_ared/
  domain/
    __init__.py
    tile_key.py          # TileKey, maybe TileAnnotation
    filters.py           # AnnotationFilter
  repositories/
    __init__.py
    base.py              # ABC TileAnnotationRepository
    sqlite.py            # SqliteTileAnnotationRepository
  services/
    annotation_manager.py
    db_manager.py        # AnnotationDatabaseManager (file lifecycle + cloning)
  gui/
    components/
      responsive_tile_grid.py
      virtual_frame_strip.py   # extracted
      bulk_label_dialog.py
    windows/             # thinner versions of the big classes
      multi_frame_browser.py
      label_review.py
    ...
  tile_database.py       # can become thin wrapper / legacy or deleted
  pipeline.py
  ...
docs/
  AnnotationSystem_Refactor_Plan.md   # this file
```

Or keep flatter if preferred (domain objects + `annotation_repository.py`).

---

## 8. Implementation Order Recommendation (for incremental work)

1. **Today/small changes**: 
   - Add size filter parameters to `get_annotations_for_video` (and update `list_videos`? or add `list_videos_with_sizes`).
   - Patch the 3-4 call sites in gui.py (browser load, review load, etc.) to pass current config tiling size.
   - Fix the explorer grid to dynamically fill (this was the "half window" complaint).
   - Add 2-3 bulk methods + a very simple UI entry point (e.g. from the right panel of browser: "Bulk ops on this video's current-size labels").
2. Introduce `TileKey` and start threading it (low blast radius).
3. Extract repository interface + implementation.
4. Build `AnnotationManager` + `DatabaseManager`.
5. Wire DB switch/clone UI + full bulk dialog.
6. Extract reusable components and clean the big window classes.
7. Update docs, add a couple unit tests for the repository bulk ops.

---

## 9. Open Questions for Discussion

- Should bulk reassign also try to update the embedding `PersistentLabelStore` (risky — different keys)?
- Preferred location for new DBs (next to videos? in a `label_dbs/` subdir?)?
- Do we want "label sets" or "sessions" inside one DB file later (tags or extra column)?
- How important is undo for bulk? (simple version: just show count and require confirmation.)
- Keep separate "exact" vs "similarity" stores, or eventually unify the query path?

---

This plan gives a clear path from the current working-but-fragile system to one that is correct w.r.t. tile sizes, powerful for label curation, and architecturally sound for long-term maintenance and new features.

Next step: decide on scope for the next coding session (quick filter + viewer fix, or start the domain model extraction). I can implement any phase or the whole thing incrementally.

## Progress (as of latest session)
- **DB management expanded**:
  - New buttons: New DB..., Clone Current DB..., Vacuum (compact).
  - Enhanced Load/Switch with path var sync, live info label, tile size mismatch warning.
  - DB summary display (name, row count, videos, sizes present).
  - All use safe close-before-copy for clone, update controller + config.
- **Rework started (OOP/SRP)**:
  - Added domain models: `TileKey` (frozen dataclass), `AnnotationFilter`, `TileAnnotation`.
  - Updated core DB APIs: `lookup_key`, `set_annotation_for_key`, `delete_key`; legacy 6-arg kept for compat.
  - Bulk methods now accept `AnnotationFilter` too.
  - Introduced `AnnotationManager` service (scope by video+size, bulk wrappers, summary).
  - Wired manager into MainWindow (startup, load, new, clone, fallbacks for review/browser). GUI now creates manager alongside raw db.
  - Example migration: per-frame explorer now uses `TileKey` + `lookup_key`.
  - `get_annotations_for_video` accepts filter.
  - Management UI grouped in LabelFrame, info auto-refreshes on ops.
- Phase 0 management + initial domain/service layer in place. Ready for deeper extraction (separate repo/service files, full caller migration, etc).