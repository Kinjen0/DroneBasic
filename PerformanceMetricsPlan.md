# Performance Metrics Plan for Drone A/RED Evaluation

**Date**: 2026-07-02

**Goal**: Implement rigorous performance evaluation for the A/RED system on drone footage, modeled directly on the published A/RED papers in this repository. This is essential for turning the current working prototype into a credible paper on "A/RED effectiveness for SAR / anomaly detection in drone video."

## 1. Metrics from the Existing A/RED Papers

We must use (or closely adapt) the same primary metrics as the authors:

### From SPIE_IVSP_2026.pdf and IJSC_2026-1.pdf (and AIxDKE)

**Core Definitions** (Section 5 in SPIE, similar in IJSC):

A/RED is treated as a **binary classifier on the query/no-query decision**.

**Positives** (things that *should* trigger a query):
- The first sample of any previously unseen class (new class discovery).
- Any sample belonging to a class that the user has previously marked as **relevant**.

**Query Precision (QP)**:
```
Query Precision ≜ TP / (TP + FP)
```
- TP = correctly queried points (new class or relevant class instance).
- FP = points that were queried but were neither new classes nor from relevant classes.

**Relevant Recall (RR)**:
```
Relevant Recall ≜ TP / (TP + FN)
```
- Only computed over points that belong to *relevant* classes (per the user's designations).
- FN = relevant points that A/RED failed to query.

"Precision" and "Recall" in the papers always refer to the above (not standard multi-class classification metrics).

### Random Baseline (very important for claims)
From the papers:
- At a matched query budget, random querying achieves:
  - `QueryPrecision_RDM ≈ Relevant Rate` (fraction of points that are relevant in the dataset).
- They show A/RED significantly outperforms random on real data.

### Additional quantities they report
- Query rate (queries / total points or per unit time).
- Number of classes discovered (especially relevant ones).
- Performance across different values of **κ** (the paranoia parameter).
- Behavior on highly imbalanced data (relevant classes 1% or 5% of total).
- Relevant recall near 1.0 while keeping query burden low.
- Cumulative plots over the stream (how quickly relevant classes are found).

They use fully labeled datasets (parking lot tiles, MNIST variants, MVTec, etc.) so they can compute the above retrospectively after the streaming run.

## 2. Challenges for Our Drone Setup

- No exhaustive ground-truth labels on the DJI field videos.
- Open-world + streaming + human-in-the-loop nature.
- Tiles are generated on the fly; we cannot easily store raw images (per previous storage constraint).
- We are building labels interactively via the GUI + TileAnnotationDB.

**Solution Direction**:
Use the **exact TileAnnotationDB** (video + abs_frame + row/col + resolution + label + relevant) as the source of "ground truth" for evaluation.

The new **"Label Only" mode** (see below) is the key tool to efficiently build high-quality reference labeled sets.

## 3. Proposed Evaluation Protocol

### Phase 1: Build Reference Labels (Label-Only Mode)
- Run videos in "Label Only" mode.
- Label as many tiles as practical (use frame stride to control density).
- Designate relevant classes (e.g., "person", "vehicle", "anomaly", "target") vs. background.
- Use the existing persistent LabelingDialog + TileAnnotationDB.
- Can do multiple passes, correct via the Review window.
- Goal: create one or more "reference labeled videos" (e.g., 10k–50k labeled tiles across a few flights).

### Phase 2: Retrospective Metric Computation
For a reference labeled video/segment:

1. Load the full set of annotations for that video from the DB.
2. Determine the "ground truth" query decisions:
   - A tile is a "positive" if:
     - It is the first occurrence of its class in the stream order, **or**
     - Its final label was marked relevant.
3. Replay the exact same tile stream (same stride, same tile size).
4. Run A/RED (with different κ values).
5. Record which tiles A/RED actually queried.
6. Compute QP and RR against the GT positives.
7. Repeat many times with a random querier at the exact same number of queries → compare.

### Phase 3: Reporting (paper-ready)
- Tables/plots of QP and RR vs. κ.
- QP/RR vs. number of queries (or vs. % of tiles queried).
- Comparison to random baseline at matched budgets.
- Class discovery curves (cumulative relevant classes found vs. queries).
- Query burden numbers (queries per minute of video, or per 1000 tiles).
- Ablations: with/without data augmentation, different tile sizes, DINO variants, cache on/off.
- Qualitative: examples of what got queried (and whether they were actually useful).

This mirrors exactly how the authors evaluate in the PDFs.

## 4. Additional / Supporting Metrics

- **Query Efficiency**: queries needed to discover all relevant classes (or a target recall).
- **Cache Effectiveness**: % of potential queries satisfied by exact DB or embedding cache (reduces human burden).
- **Label Stability**: how often users correct labels in review (measures consistency).
- **Throughput**: tiles processed per second (with/without A/RED).
- **Human Time Proxy**: number of actual GUI interactions required.

## 5. Implementation Plan

### 5.1 Label-Only Mode (Highest Priority – Enables Everything)

See separate section below. This must be implemented first.

### 5.2 Offline Evaluation Harness

Create `drone_ared/eval/` or scripts:
- `compute_metrics.py`
- `run_evaluation.py --video DJI_0017.MP4 --kappa 1.0,2.0,3.0 --random-trials 20`

Core class: `AREDEvaluator`
- Takes a list of tiles (or re-generates them from video + tiler).
- Has access to the TileAnnotationDB (as "oracle" of final labels).
- Simulates the stream.
- Can run either:
  - Real `DroneAREDController` (with A/RED) in a special "eval mode" that logs every decision.
  - Or a pure simulation using the adapter + forced labels from DB.
- Records for every tile: `queried_by_ared`, `should_query` (from GT), `is_relevant`, `class_first_occurrence`.
- Computes QP, RR, random baseline, etc.

**Important**: For faithful replay, we must use the *same* order of tiles as during live labeling (use global tile order or abs_frame + row/col).

### 5.3 Integration Points
- Make the existing `TileAnnotationDB` queryable for "all annotations in video order".
- Add a method to get "GT should_query" decisions given the collected labels.
- During A/RED runs (even normal ones), log the sequence of `queried` decisions + tile identities. This log + the final DB labels can be used for metrics.
- Support "forced label mode" in the adapter/provider so we can replay using saved labels without GUI.

### 5.4 GUI / UX for Metrics
- After a run, button "Compute Metrics (using current DB)".
- Simple dialog to select reference video/segment + κ values.
- Output table + save CSV/JSON.
- Later: matplotlib plots (QP vs κ, discovery curves).

### 5.5 Ground Truth Collection Strategy
- Start with one or two shorter video segments.
- Label exhaustively or at high density in "Label Only" mode.
- Mark a small number of classes as relevant.
- Use planted targets in future flights for cleaner validation.

## 6. "Label Only" Mode Specification

**Purpose**:
- Fast, low-overhead way to label large numbers of tiles.
- No DINO loading, no A/RED, no feature extraction overhead.
- Builds the reference dataset needed for the metrics above.
- Re-uses the excellent existing high-volume labeling UI.

**Behavior**:
- User selects video(s) and parameters (tile size, frame stride) as usual.
- Clicks "Start Label Only" (separate from normal A/RED Start).
- System reads video, generates tiles (same GridTiler).
- For **every** tile (respecting stride):
  - Shows the persistent LabelingDialog with the tile image.
  - User can:
    - Double-click / Enter existing class.
    - Create new class + set relevant checkbox.
    - Mark as Background.
  - On assign: immediately saves to `TileAnnotationDB` with full identity (video, abs_frame, row, col, w, h, label, relevant).
  - Advances to next tile automatically.
- Same keyboard shortcuts, filter, etc. as normal mode.
- Can pause / resume.
- Progress: "Labeled 1247 / 18500 tiles".
- When done (or stopped), all labels are in the DB and ready for metrics or future A/RED runs.
- The Review / Edit Past Labels window works immediately for corrections.

**Implementation Notes**:
- Reuse `LabelingDialog`, `TileAnnotationDB`, video loop, and tiling code.
- New flag/mode: `label_only_mode`.
- In `DroneAREDController`:
  - If `label_only_mode`:
    - Skip creating `feature_extractor` and `ared_adapter`.
    - In the processing loop, for each tile: always create a `LabelRequest`, put it in the queue, wait for result, save directly to `tile_db`.
- In GUI:
  - Add a second button: "Start Label Only".
  - Or a radio / checkbox "Mode: A/RED | Label Only".
  - When starting label only, set `controller.label_only_mode = True` and call start.
  - Disable A/RED-specific controls or show different status.
- In `_gui_label_provider` or a new `_label_only_provider`: always request human label (no cache/ ARED logic).
- Still support the exact DB for "I already labeled this exact tile" (even in label-only, if re-running with different stride, auto-skip or show for confirmation).

**Advantages**:
- Very fast (no heavy model loads).
- Directly populates the DB that metrics will use.
- Users get comfortable with the labeling UI before running full A/RED.

## 7. Risks & Mitigations

- **Incomplete labeling**: Not every tile will be labeled in practice. Mitigate by:
  - Sampling strategies.
  - Reporting metrics on the subset that *was* labeled.
  - Using planted events for validation.
- **Labeler bias / inconsistency**: Use the review tool + multiple passes. Track corrections.
- **Seeking accuracy** when re-extracting tiles for review: acceptable for this use case (user can verify visually).
- **Reproducibility**: Always save the exact config (stride, tile size, video path) alongside the annotations.

## 8. Implementation Order (Recommended)

1. Implement **Label Only mode** (core enabler).
2. Add logging of query decisions with tile identities during normal A/RED runs.
3. Build basic `eval/metrics.py` that can load a video + its DB annotations and compute QP/RR for a simulated run.
4. Hook a "Compute Metrics" button in GUI (initially simple print/CSV).
5. Add random baseline comparator.
6. Support κ sweeps.
7. Add plots (matplotlib) and integrate with paper figures.
8. Document protocol in README or a small `docs/evaluation.md`.

## 9. Files to Touch

- New: `drone_ared/eval/metrics.py` (or `scripts/`)
- `drone_ared/pipeline.py` + `gui.py` (label only mode + mode flag)
- `drone_ared/tile_database.py` (maybe add helper methods for "get_ordered_annotations(video)" and "get_gt_query_decisions")
- Possibly small additions to `ared_adapter.py` for "eval replay with forced labels"
- `drone_ared/config.py` (small mode flag if desired)

## 10. Relation to Paper

This directly supports the paper goal stated in `PaperIdeas.md`:
> "Build an offline evaluation harness... Perform at least one full manual annotation pass... Run systematic κ sweeps... produce the first result tables/figures."

Once we have labeled reference data + these metrics, we can make strong, paper-quality claims about A/RED effectiveness on real drone SAR-style footage.

---

This plan is intentionally faithful to the definitions and evaluation style in `IJSC_2026-1.pdf` and `SPIE_IVSP_2026.pdf`.

Next step after review: implement the Label Only mode + basic metric computation skeleton.
