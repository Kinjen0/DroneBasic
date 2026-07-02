# PotentialFeatures — Areas for Improvement in the Drone A/RED System

**Purpose**: Concrete, prioritized suggestions for evolving the program.  
**Guiding constraint**: All changes must preserve faithful use of A/RED. The algorithm (kappa-driven anomalous decision + relevant-cluster query rule, circular buffer, cluster management, forgetting, etc.) must continue to make the query decisions. We can only improve the *inputs* it receives (better features/tiling), the *efficiency and usability* of the human loop, the *evaluation tooling*, and operational features. Never short-circuit or replace A/RED's internal logic.

This file is referenced from `PaperIdeas.md`.

---

## 1. High-Value / Low-Risk Improvements (Do These First)

### 1.1 Stronger Evaluation & Reproducibility Tooling
- **Offline evaluation harness** (highest priority for the paper):
  - Accept a directory of tiles (or a video + deterministic tiler) + a ground-truth JSON/CSV of `(global_idx, gold_label, gold_relevant)`.
  - Replay through the exact same AREDAdapter path.
  - Automatically compute Query Precision, Relevant Recall, #queries, cache behavior.
  - Support "injected anomaly" mode: programmatically mark certain tiles as gold-relevant.
- Add a `scripts/` or `eval/` folder with:
  - `compute_metrics.py`
  - `kappa_sweep.py`
  - `visualize_clusters.py` (can reuse/adapt code from `A_REDimplementation/A_RED/cluster_visualization.py` and `data_visualization.py`)
- Save full run artifacts: embeddings (with global_idx), query log, final label store snapshot, ARED replay state.

### 1.2 Better Logging & Post-Run Analysis
- Structured JSON logs (in addition to pretty terminal prints) for easy parsing.
- Per-run summary at end: tiles, queries, cache_hits, discovered classes, query reasons (anom_only / rel_only / both).
- Simple live plot (or matplotlib after run) of cumulative queries vs. tiles.
- Store the sequence of "was_queried" decisions so we can align them with any later gold labels.

### 1.3 Labeling UX Polish (Still High-Volume Friendly)
- Keyboard-only power mode (number keys for top-N classes, Tab to cycle, etc.).
- "This looks like the previous one" quick button (uses last assigned label+relevance).
- History panel: recently labeled tiles (thumbnails) with ability to correct a previous decision (and propagate back to ARED via replay or label store correction).
- Better class management: rename, merge two classes (with care — must update ARED clusters consistently or document that merge happens only in post-processing).
- Show a small "why was this queried?" hint in the dialog: "Anomalous (dist=0.42 > comp=0.31 / κ=1.0)" or "Near relevant cluster 'Person'".
  - This can be exposed by the adapter (it already has distance info from some paths; we can surface more from ARED state read-only).

### 1.4 Cache & Persistence Robustness
- Make label store threshold adaptive or per-class (some classes are tighter than others).
- Version the cache file; warn on mismatch of DINO model / pooling / tile size.
- Optional export/import of label store as human-readable (embedding hash + label) for auditing.
- Support "forget class" or "lower confidence on old labels" when environment changes (lighting season, new field).

### 1.5 Model Save/Load Polish
- GUI button to "Warm-start from previous flight" that also loads the label store.
- Store a small manifest with the exact config (kappa, DINO name, tile size) that produced the saved state.
- Auto-suggest replay when user loads videos that look similar to a saved state.

---

## 2. Feature & Representation Improvements (Must Keep Streaming + A/RED in Control)

### 2.1 Tiling Strategies (Beyond Uniform Grid)
- Add abstract `Tiler` implementations:
  - `MotionTiler` or `SaliencyTiler` (use simple frame differencing or cheap saliency to propose candidate regions, then still produce uniform-size tiles for DINO).
  - `MultiScaleTiler` (a few different tile sizes centered on the same point).
  - Overlap-aware or "jittered" grid for robustness.
- **Important**: Even adaptive tilers should emit a stream of fixed-size image crops + deterministic global indices. A/RED still sees one embedding at a time.

### 2.2 Richer Features While Staying Faithful
- Concatenate or fuse:
  - DINO embedding + cheap hand-crafted stats (mean brightness, edge density, color histogram) or optical-flow magnitude in the tile.
  - Small temporal context: embedding of current tile + embedding of same spatial tile from N frames ago (or delta).
- Test whether adding these dimensions helps A/RED's distance-based decisions (compare QP/RR).
- Keep the DINO backbone frozen (or document any fine-tuning carefully — the papers treat it as a fixed extractor in most experiments).

### 2.3 Optional Lightweight Projector / Autoencoder
- The SPIE paper used a small autoencoder on top of DINO to reduce to 16-D.
- We can add an optional trainable (but still offline-pretrained) projector head that is applied before ARED.
- Must be deterministic and part of the saved "feature config" so replay works.

---

## 3. Operational / SAR-Specific Features

- Geo-tagging: if DJI metadata or RTK is available, attach approximate lat/lon + altitude to each tile (or at least each queried tile). Huge for real SAR.
- "Mission mode" vs. "Analysis mode": lower κ or different cache policy when you want to be extremely paranoid during a critical search grid.
- Export of queried tiles with metadata (KML, CSV, image chips + labels) for after-action reports.
- Integration hooks: REST endpoint or MQTT so a ground station or another process can receive the "relevant" alerts in real time.
- Support for multiple simultaneous video streams (different drones or gimbal + nadir) with a shared ARED model or per-drone models.
- Night / thermal handling (different DINO or fine-tune; or fall back to different feature extractor).

---

## 4. Performance & Scale

- Make feature extraction and ARED processing faster:
  - Torch compile / TensorRT / ONNX for DINO on edge or ground station.
  - Larger batching with smarter flushing.
  - Profile where time is spent (currently DINO forward + occasional GUI).
- Better memory behavior for extremely long flights (the buffer already bounds the *labeled* points; the cache can grow — add an LRU or importance-based eviction for the label store).
- Asynchronous DINO extraction so the main video read loop never stalls.

---

## 5. Human-in-the-Loop Research Angles (Good for a Paper)

- Measure and reduce "effective query burden": cache hit rate + "time to label" stats.
- Class discovery curves: how many relevant classes are found after X queries / Y minutes of video.
- Label correction study: how often does the cache or previous decision need fixing?
- Multi-analyst: two users labeling the same stream (or different streams) — agreement on relevance, merging of discovered classes.
- "Background fatigue": long periods with no queries — does the operator stay attentive? (UI can occasionally surface very low-confidence normal examples for calibration.)

---

## 6. Things That Would Be Nice but Carry Risk of Subverting A/RED (Handle With Care)

- Any "pre-filter" that drops tiles before they reach `ared.process()`:
  - Must be documented as an *acceleration* or *operational filter*.
  - The paper should also report numbers for the *pure* A/RED path (all tiles fed).
  - Example safe version: drop obviously black or blurred frames using cheap heuristics, but still count them in denominator for burden metrics.
- Online updating of DINO or a projector during the stream:
  - Concept drift is real, but this moves us away from the static feature assumption in the original papers.
  - If done, treat it as a major experimental axis and compare against frozen-DINO A/RED.
- Replacing A/RED's query logic with a different active learning strategy:
  - Do this only in a "comparison system" branch, never as the main path when claiming "A/RED effectiveness".
- Making relevance a live per-assignment checkbox again:
  - We already fixed this. Don't regress — relevance must be a stable property of the discovered class.

**Rule of thumb**: If a change would let us get high "relevant recall" numbers without the actual A/RED anomalous/relevant-near test ever firing, it is dangerous for the scientific claim.

---

## 7. Code Organization & Maintainability Suggestions

- Move evaluation scripts out of root.
- Add a small `docs/` or keep expanding the README with "How to run an evaluation".
- Type hints + a few more unit tests around the adapter (peek vs. real, replay fidelity).
- Make the adapter expose a few more read-only views (current comparison cluster for the last point, last distance, etc.) so the GUI can show "why queried" without any mutation.
- Consider a thin `ared_viz.py` that can load a saved state + embeddings and produce the nice cluster evolution plots from the original repo.

---

## 8. Stretch / Future Work Ideas

- On-drone or Jetson-class deployment (very low query rate + edge cache, transmit only queried tiles).
- Integration with traditional SAR tools (flight planning software, map overlays).
- Handling of gimbal motion / orthorectification so "same location" tiles can be compared more meaningfully.
- Weak supervision: use occasional GPS-tagged ground observations to auto-seed or validate labels.
- Active forgetting tuned to scene dynamics (e.g., wind-blown grass should be forgotten faster than a static road).
- Uncertainty-aware querying: when A/RED is borderline, optionally ask the human even if the hard kappa rule didn't fire (but again, measure the pure version separately).

---

## Prioritization Suggestion (for the Paper + Usability)

1. Evaluation harness + first solid QP/RR numbers on annotated data (enables the paper).
2. Logging + visualization improvements (helps debugging + figures).
3. Tiling/feature experiments that stay inside the A/RED contract.
4. UX speed/comfort for long labeling sessions (cache + history).
5. SAR-specific exports and metadata.
6. Everything else.

---

**Remember**: The scientific contribution is "we took the A/RED algorithm as published, fed it real drone tiles via DINO, let it decide when to query a human, and measured how effective that was for surfacing relevant events." Any feature that makes the human's life easier or the system faster is valuable engineering, but the core measurement must be of A/RED's query behavior.

Add items here as we discover them during paper experiments.
