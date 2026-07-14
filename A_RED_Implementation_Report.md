# A/RED Implementation Report — Drone A/RED Project

**Date**: 2026-07-14  
**Scope**: How the original A/RED algorithm is used in this project, whether that usage matches the papers’ intent, what metrics we compute (and still omit), and how the interactive drone pipeline wraps an offline research codebase without rewriting the core.

**Primary sources**:
- Papers: `SPIE_IVSP_2026.pdf`, `IJSC_2026-1.pdf`, `AIxDKE_2026.pdf`
- Untouched library: `A_REDimplementation/A_RED/A_REDIN.py` (+ `Oracle.py`, buffers, etc.)
- Project bridge: `drone_ared/ared_adapter.py`, `drone_ared/pipeline.py`, `drone_ared/metrics.py`

---

## 1. Executive summary

**We are using A/RED as intended for the query decision.**

In the papers and in `A_REDIN.ARED`, A/RED is a **streaming, human-in-the-loop query controller**: for each new point it decides *whether* to ask for a (class label, relevance) pair. That decision is driven by:

1. **Anomaly relative to a comparison cluster** (κ-scaled distance vs cluster compactness), and  
2. **Proximity to already-relevant structure** (comparison neighborhood involving relevant clusters).

**We do not reimplement that decision.** Every normal (non–Label-Only) tile embedding is passed into the **unmodified** `ARED.process_first_point` / `ARED.process_point`. Labels enter A/RED **only** when A/RED itself calls `oracle.answer_query` on its real query branch.

What we *did* build around that core is the open-world, interactive, drone-video front-end the original repo never had:

| Layer | Role | Decides query? |
|-------|------|----------------|
| Video tiling + frame stride | Stream construction | No |
| DINOv2/v3 feature extractor | Point = embedding vector | No |
| **`A_REDIN.ARED`** | Cluster memory, κ rule, query/no-query | **Yes** |
| Exact TileAnnotationDB + embedding cache | Answer the query faster | No (answers only) |
| Human GUI | SME oracle for novel / uncached queries | No (answers only) |
| Metrics (QP / RR / F1 / baselines) | Evaluation after the fact | No |

**Label-Only mode is intentionally not A/RED** — it is a separate tool to build ground-truth annotations so that QP/RR can be computed later, exactly as offline paper datasets were pre-labeled.

---

## 2. What A/RED is (paper intent)

Across SPIE / IJSC / AIxDKE, A/RED is framed as:

- A **stream processor** over high-dimensional points (pixels or features).
- A system that **discovers classes** and **focuses labeling budget** on:
  - first samples of new classes, and  
  - samples of classes the user has marked **relevant**.
- Controlled by a **paranoia parameter κ** that trades **query rate** vs **query precision / relevant recall**.
- Evaluated as a **binary classifier on the query / no-query decision**, not as a classical multi-class accuracy leaderboard alone.

### 2.1 Paper positives (should query)

From SPIE §5 / IJSC §6 (and mirrored in our `metrics.py`):

**Positives** =
1. first sample of a given class in the stream, **or**  
2. samples from classes designated as **relevant**.

### 2.2 Paper metrics

| Metric | Definition (papers) | In this project |
|--------|---------------------|-----------------|
| **Query Precision (QP)** | TP / (TP + FP) over query decisions | Yes — primary |
| **Relevant Recall (RR)** | TP / (TP + FN) over positives | Yes — primary (full positives set) |
| **F1** | Harmonic mean of QP and RR | Yes — used in tables/plots in SPIE |
| **Query Rate (QR)** | (# queries) / (# streamed points) | Yes |
| **Relevant Rate** | fraction of stream that is relevant-class | Yes (for random QP baseline) |
| **Random QP** | ≈ Relevant Rate at matched budget | Yes (`baseline_random_query_precision*`) |
| **Random RR** | = Query Rate | **Yes (now explicit)** `baseline_random_relevant_recall` |
| **QP / RR improvement vs random** | SPIE Fig. 3 style ratios | **Yes (now)** |
| **Classes discovered** | count / annotation on QP–RR plots | Yes (`classes_discovered_x_of_y`) |
| **QP & RR vs time / stream index** | SPIE Fig. 3; IJSC cumulative curves | Partial — via **running checkpoints** every N tiles |
| **QP vs RR curves across κ** | SPIE Fig. 2; IJSC Fig. 5 | Via multi-run packages + `run_report.py compare` |
| **Random RR / QP over time** | plotted against A/RED | Partial — final + checkpoint baselines; not a full synthetic random stream plot yet |
| **Confusion matrix (class assignment)** | original ARED bookkeeping | Exists inside ARED; **not** our primary paper metric surface |
| **Per-query reason counts** (anomaly-only / relevant-near / both) | internal ARED counters | Available on ARED object; **not yet exported** into run.json |

---

## 3. Which code we run (zero edits to the library)

- Canonical class: **`A_REDIN.ARED`** imported as `_OriginalARED` in `ared_adapter.py`.
- **No modifications** under `A_REDimplementation/A_RED/` (import shim only: a fake `main.QS_VAR` so the library imports cleanly).
- Older `A_RED.py` is **not** used.

Constructor parameters we pass through from `AREDConfig` match the library:

- `kappa`, `l_buf_size`, `K_COMP_PTS`, `QS_VAR`, `DATA_AUG_VAR`,  
  `NGHBHOOD_MERGE`, `SINGLETON_MERGE`, `SMART_FORGETTING_VAR`, `VERBOSE_FLAGS`.

---

## 4. End-to-end data path (how a drone tile becomes an A/RED point)

```
Video frame (stride N)
    → GridTiler → fixed-size tile crop
    → DINOFeatureExtractor → embedding x ∈ R^d
    → AREDAdapter.process(x, tile, meta)
         → ARED.process_point(x)          # ORIGINAL
              → determine_comparison_cluster
              → anomalous?  OR  near-relevant?
              → if yes: oracle.answer_query(...)   # REAL QUERY
              → else: assign without query
    → if real query: provider answers:
         1) exact TileAnnotationDB hit (same video/frame/row/col/size/stride)
         2) embedding similarity cache
         3) human LabelingDialog
    → label+relevance returned only because A/RED asked
```

### 4.1 First point

Matches original: **always queries** (`process_first_point`).

### 4.2 Subsequent points — decision lives only in A_REDIN

Core branch (library, paraphrased):

```text
is_anomalous = (distance * kappa > cluster.comp_distance)
comp_cl_is_relevant = (relevant neighbor data is not None)

if is_anomalous or comp_cl_is_relevant:
    num_queries += 1
    new_label, new_rel = query(...)   # → our answer_query hook
    add to existing cluster OR split / neighborhood-merge
else:
    # no query — confusion-matrix peek only
```

That is the paper algorithm: **κ-controlled anomaly + relevance-aware neighborhood**.

---

## 5. How we answer queries without corrupting A/RED’s intent

### 5.1 The original Oracle assumption

Paper/code experiments assume a **fully labeled offline dataset**.  
`Oracle.answer_query(abs_idx)` always returns ground truth. There is **no interactive GUI** in the original repo (only matplotlib analysis plots).

### 5.2 Our open-world adaptation (necessary, localized)

We temporarily replace `oracle.answer_query` for the duration of **one** `process_*` call:

1. **Peek vs real query**  
   ARED often calls `answer_query` / peeks `oracle.y` for bookkeeping even when it does **not** take the query branch.  
   We treat the first call on a non-first point as a **peek** (provisional, never GUI) and only the **second / query-branch** call as a **real query** (cache / DB / human).  
   This is required so we do not ask the human on every tile — which would **not** be A/RED.

2. **`was_queried` is set only on the real query path**  
   Metrics and GUI “A/RED queries” count **A/RED’s decision**, even if the answer is filled by cache or exact DB.  
   That matches the project rule: *cache is an answer accelerator, not a decision bypass.*

3. **Growing label map + sparse `oracle.y`**  
   Papers use a fixed class set. We discover string class names live.  
   Adaptation is confined to the adapter’s oracle wrappers so `A_REDIN` keeps its bidict / conf_matrix expectations.

4. **Control sentinels** (`__STOPPED__`, skip, etc.)  
   Never learned as real classes; never stored as durable labels.  
   Stop/timeout cancel without teaching A/RED a junk class.

### 5.3 What we deliberately do *not* do

- We do **not** force labels into clusters for tiles A/RED did not query (except faithful **model replay** of previously labeled buffer points).
- We do **not** let DINO, the GUI, or the DB decide “this tile is anomalous.”
- We do **not** edit `A_REDIN` to add drone-specific logic.

---

## 6. Feature front-end: DINO is not A/RED

Papers evaluate A/RED on different feature spaces (shallow, DAGMM, DINOv2).  
In this project:

- **DINOv2/v3** produces the vector A/RED clusters.
- Optional **rotation augmentation** runs **after** a real A/RED query: new embeddings are inserted as labeled variants (`add_labeled_variant` path) to densify the cluster — analogous in spirit to paper data-augmentation variants, still **downstream of** the query decision.

---

## 7. Evaluation protocol vs papers

### 7.1 What papers do

- Fully labeled streams → after a run, compute QP/RR vs κ, vs random, over time, and plot QP–RR curves with class-discovery annotations.

### 7.2 What we do on drone video

1. **Label-Only mode** (or multi-frame browser) builds a **TileAnnotationDB** of human labels + relevance (reference set).  
2. **Normal A/RED runs** stream tiles; A/RED decides queries; answers may come from DB/cache/human.  
3. **Metrics** join:
   - `processed_identities` (tiles actually sent to A/RED this run),  
   - `queried_identities` / `ared_queries` (query decisions),  
   - DB annotations for those tiles (ground truth for “should query”).  
4. **Running metrics** every N tiles + **finalize** package under `runs/<run_id>/`.

This is the same *scientific* loop as the papers (stream → A/RED queries → evaluate against labels), adapted to **incomplete, interactive** labeling instead of a closed offline y-vector.

### 7.3 Metrics we currently save

**Primary (paper):** QP, RR, F1, QR, relevant rate, TP/FP/FN, random QP baseline, random RR (= QR), improvement ratios, classes discovered x/y, relevant tiles queried.

**Operational:** tiles processed, frames, cache hits, human dialogs, clusters, known labels, κ / tile / stride / buffer / DINO / DB path, elapsed time, per-checkpoint CSV.

**Files per run:**
- `run.json` — params + checkpoints + compact `final_metrics`
- `checkpoints.csv` — running table
- `final_metrics.json` — full final package (including detailed audit)
- `final_audit.txt` — human-readable work

### 7.4 Gaps vs papers (honest)

| Paper artifact | Status |
|----------------|--------|
| Full random **strategy** curve over time (not just closed-form baseline) | Not simulated as a second stream yet |
| Export of ARED’s `anom_only_queries` / `rel_only_queries` / `both_a_and_r_queries` | Counters exist in library; not in run logs yet |
| Multi-κ sweep automation (batch runner) | Manual via GUI + `run_report.py` compare |
| Strict RR (relevant-class-only, ignoring firsts) | Computed as `relevant_recall_strict` for audit; primary RR is full positives |
| Classification accuracy / conf_matrix dashboards | Internal to ARED; not GUI-primary |
| Query rate windowed bar charts (library `Stats.graph_query_rate_over_time`) | Approximated by our checkpoints + reporting plots |

None of these gaps change the fact that **query decisions are A/RED’s**.

---

## 8. Fidelity checklist

| Claim | Evidence |
|-------|----------|
| Query decisions only from A/RED | Only `process_first_point` / `process_point` in adapter; provider never called on peek path |
| Library unmodified | All edits outside `A_REDimplementation/A_RED/` |
| κ, buffer, k-NN comparison, neighborhood merge, smart forgetting | Passed into `ARED.__init__` from config / GUI |
| First point always queries | Original `process_first_point` |
| Non-query points not labeled by us for A/RED learning | Assigned internally; no GUI |
| Cache/DB after decision only | Provider stack in pipeline + real-query branch in adapter |
| Metrics match paper QP/RR positives | `compute_should_query_from_annotations` + `compute_query_metrics` |
| Interactive open-world | Growing labels + sparse y + human/cache oracle |

---

## 9. How this differs from “using A/RED in name only”

A non-faithful integration might:

- Threshold DINO distances and call that “A/RED”,
- Label every Nth tile,
- Or inject DB labels into clusters without a query.

**We do none of that.** The algorithm that decides *when* the SME is bothered is the same `A_REDIN` routine used in the paper experiments. Our contribution is the **drone streaming stack + interactive oracle + evaluation harness** required to run that algorithm on real SAR-style footage.

---

## 10. Practical operator notes

1. **Terminal logging** (GUI checkbox) now gates high-volume prints in:
   - `pipeline` (per-tile / per-frame),
   - `ared_adapter` (`[ARED] …` process path),
   - GUI dialog chatter,  
   and clears `VERBOSE_FLAGS` inside A_REDIN when off (empty list = quiet).  
   Start/stop/errors still print.

2. **Final metrics** on stop/finish always attempt a full evaluation and write:
   - `final_metrics.json`, `final_audit.txt`, and `final_metrics` inside `run.json`.  
   If the DB has no overlapping labels yet, the package still records the error + stream counters (so the run is never “silent”).

3. **Class list** shows labels from the **active annotation DB + this run**, not embedding-cache history from other sessions.

4. **Reporting**: `python run_report.py report --runs-dir runs` builds paper-style curves from saved packages.

---

## 11. Conclusion

**Yes — we are truly using A/RED as intended:** as the sole streaming controller of the label budget, driven by κ, cluster compactness, and relevance-aware comparison, with the human (or a faithful cache of prior human answers) acting as the oracle *only when A/RED queries*.

DINO, tiling, caches, and the GUI are **infrastructure**.  
QP/RR/F1 and the new random RR / improvement ratios are **evaluation**.  
`A_REDIN.ARED` remains the **brain**.

---

## 12. Key file map

| File | Role |
|------|------|
| `A_REDimplementation/A_RED/A_REDIN.py` | Canonical A/RED (untouched) |
| `drone_ared/ared_adapter.py` | Import shim, interactive oracle, process API, save/load replay |
| `drone_ared/pipeline.py` | Video loop, DINO, provider priority (DB → cache → GUI) |
| `drone_ared/metrics.py` | Paper QP/RR/F1 + baselines + audit |
| `drone_ared/run_metrics_logger.py` | Per-run packages + running checkpoints |
| `drone_ared/reporting/` + `run_report.py` | Paper-style plots/tables |
| `ARED_Integration_Summary.md` | Earlier short fidelity note (superseded in detail by this report) |
