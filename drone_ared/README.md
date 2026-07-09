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
- Live stats, class list, pause/resume, etc.

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
  - `gui.py` — add panels, export buttons, etc.
- `PipelineConfig` is a tree of dataclasses → easy to persist or expose more controls.

See the plan.md in the session folder for full design rationale and verification steps.
