# drone_ared

A/RED (Anomalous / Relevant Event Detection) pipeline for drone footage.

## Quick Start

```bash
cd /home/wes/Desktop/PHXResearch/DroneBasic
python -m pip install -r requirements.txt   # or install the packages below
python run_drone_ared.py
```

### Required packages (beyond what A_RED already needs)
- opencv-python
- torch + transformers (already used by A_RED DINO code)
- pillow
- scikit-learn (already used)
- tkinter (usually comes with Python)

## High-Level Flow

1. Videos are read frame-by-frame (every Nth frame).
2. Each selected frame is split into tiles (default 224x224 grid).
3. Tiles are passed through a DINOv2 (or v3) model → embeddings.
4. Embeddings are streamed into A_REDIN (the original implementation, imported without modification).
5. When A_RED decides a tile is anomalous or near a relevant cluster, the GUI asks the user for a label + relevance flag.
6. A persistent label store (embedding NN cache) can auto-answer repeated/similar tiles on this or future runs.
7. Optional: save the current A_RED cluster state ("model") and reload it later for warm-start on similar data.

## GUI Features (as requested)

- Resizable main window + resizable labeling dialog (shrink/grow to fit any screen).
- Double-click (or Enter) on existing class in list to assign label instantly.
- Text entry + "Relevant" checkbox for creating brand new classes.
- Keyboard friendly (Return, Escape, filter box).
- "Save ARED Model" / "Load ARED Model" buttons + menu items (optional, off by default).
- **Merge ARED Models** (File menu + toolbar): combine two saved models with one of three strategies.
- Live stats, class list, pause/resume, etc.

## Model merging

Saved A_RED models are pickles of labeled buffer points + hyperparams
(`AREDState`). Merging never edits `A_REDimplementation/`; it rebuilds via
`AREDAdapter.process` (same idea as Load).

| Strategy | Behavior |
|----------|----------|
| **Squish** | Enlarge buffer (≈ 2×), force-replay **all** points from A then B. True buffer union. |
| **Ingest** | Rebuild A, then stream B through live A_RED. B’s label is used **only if** A_RED queries. Smart forgetting applies when the buffer is full. |
| **Interleave** | Fresh model; alternate points from A and B (query-aware, like ingest). |

Notes:

- **Order matters** (A-then-B ≠ B-then-A). A is always the “base” for hyperparams.
- Models must share the **same embedding dimension** (same DINO / tile setup).
- GUI: stop any running stream first → **File → Merge ARED Models…** (or **Merge Models** button).
- Programmatic:

```python
from drone_ared.model_merge import AREDModelMerger

result = AREDModelMerger().merge_files("model_a.pkl", "model_b.pkl", strategy="squish")
print(result.pretty())
result.adapter.save_state("merged.pkl")
```

Headless tests: `python -m unittest drone_ared.tests.test_model_merge -v`

## Running metrics (cumulative + batch)

With **Running Metrics Log** enabled, each Start writes `runs/<run_id>/`:

| File | Contents |
|------|----------|
| `run.json` | Params + checkpoints + final metrics |
| `checkpoints.csv` | Cumulative QP/RR/F1 **and** `batch_*` window scores |
| `batches.csv` | Batch-window extract for alternate reporting |
| `final_metrics.json` / `final_audit.txt` | Full-run (cumulative) package |

- **Cumulative** fields (`query_precision`, …): scores over all tiles from run start → checkpoint.
- **Batch** fields (`batch_query_precision`, …): scores for tiles since the previous checkpoint only. First-of-class still uses full-stream context (no double-counting across batches).

```bash
# Cumulative curves / report (unchanged defaults)
python run_report.py curves --runs-dir runs
python run_report.py report --runs-dir runs --out reports/latest

# Batch-window curves / report
python run_report.py batch-curves --runs-dir runs
python run_report.py batch-report --runs-dir runs --out reports/batch_latest
python run_report.py compare --metric batch_relevant_recall --runs-dir runs
```

Tests: `python -m unittest drone_ared.tests.test_batch_metrics -v`

## Expandability

- All major pieces are in their own files with ABCs or clear hooks:
  - `tiling.py` — add new tiler strategies

## Tiling and Overlap

The default is a non-overlapping grid. For 240×240 tiles (or any size) you can enable **overlapping tiles** in the GUI:

- Check **"Enable overlapping tiles (stride = tile - overlap)"**
- Enter **Overlap X (px)** and/or **Overlap Y (px)** (e.g. 32 or 48 for 240×240)
- The effective stride becomes `tile_size - overlap` (clamped ≥ 1).

Overlapping tiles help avoid cutting objects that cross tile boundaries. This increases the number of tiles (roughly by `(tile / (tile-overlap))²`).

All labels are stored with exact (video, frame, tile position, size). The system correctly recalls labels even when you change stride later (provided the (row, col) + size matches what was labeled). Different stride values for the same nominal tile size produce different grids; treat (tile size + stride) as a unit when building a corpus.
  - `feature_extractor.py` — swap DINOv3, add projector, etc.
  - `label_store.py` — replace with FAISS / sqlite-vss
  - `ared_adapter.py` — the only place that knows about the original A_RED
  - `model_merge.py` — Strategy-pattern merge modes (squish / ingest / interleave)
  - `gui.py` — add panels, export buttons, etc.
- `PipelineConfig` is a tree of dataclasses → easy to persist or expose more controls.

See the plan.md in the session folder for full design rationale and verification steps.
