# A/RED Integration Summary (Drone A/RED)

**Date**: 2026-07-08  
**Purpose**: Confirm that the drone implementation uses the original A/RED **strictly for all query decisions**, matching the algorithm in the papers (SPIE_IVSP_2026, IJSC_2026-1, and the code in `A_REDimplementation/A_RED/`). DINOv3 is only a feature front-end.

---

## 1. Which A/RED Implementation We Use

- We use **`A_REDIN.ARED`** (imported as `_OriginalARED` from `A_REDIN`).
- This is the version present in the paper-related experiments:
  - `FiniteBuffer`, `Subspace_Partition`, `determine_comparison_cluster` (k-NN via BallTree)
  - `kappa * distance > cluster.comp_distance` (diameter or average NN distance)
  - Neighborhood merge (when `K_COMP_PTS >= 2`)
  - Smart forgetting, singleton merge options, etc.
- The older `A_RED.py` variant is **not** used.
- **Zero modifications** were made to any file under `A_REDimplementation/A_RED/`. All adaptation lives in `drone_ared/ared_adapter.py`.

---

## 2. How Query Decisions Are Made — Strictly by A/RED

The only entry point for a new embedding is:

```python
# In ared_adapter.py
if first_point:
    self.ared.process_first_point(emb)
else:
    self.ared.process_point(emb)
```

Inside the **original untouched** `ARED` (A_REDIN.py):

- `process_first_point`:
  - Always performs a query via `self.query(...)` → `oracle.answer_query(...)`.

- `process_point(data_point)`:
  1. `determine_comparison_cluster(data_point)` — finds k closest labeled points in the buffer.
  2. `is_anomalous = distance * self.kappa > cluster.comp_distance`
  3. `comp_cl_is_relevant =` any of the k-nearest is from a relevant-labeled cluster.
  4. **Only if** `comp_cl_is_relevant or is_anomalous`:
     - Increment `num_queries`
     - `new_pt_label, new_pt_relevant = self.query(self.num_pts_streamed - 1)` ← **the single place labels are requested**
     - Then either add to existing cluster or `split` (new cluster).
  5. Otherwise the point is treated as ordinary (assigned to the comparison cluster **without** a query). ARED only peeks at `oracle.y[...]` afterward for confusion-matrix bookkeeping.

This is exactly the decision logic described in the papers (anomaly relative to cluster compactness + proximity to relevant clusters, controlled by κ).

**Labels are never injected into A/RED except through the query path that A/RED itself initiates.**

---

## 3. How We Supply Answers (Without Changing A/RED)

We temporarily replace `oracle.answer_query` with a hook **only for the duration of one `process_*` call**:

- We distinguish "peek" calls (the bookkeeping `answer_query` that original code does on every point) from "real query" calls (the second call that only happens on the anomalous/relevant branch).
- Peek calls receive a provisional label (no GUI). This is required for open-world interactive use; the original code had ground-truth for everything.
- Real query calls invoke the registered provider (GUI / cache / exact DB).
- The `was_queried` flag is set **only** on the real query path, inside the hook that ARED called because its internal logic decided a label was needed.
- After the call we restore the previous `answer_query`.

The provider (in `pipeline.py`) does:
1. Exact TileAnnotationDB lookup (by video+frame+row+col+size) — identity cache.
2. Embedding similarity cache (`PersistentLabelStore`).
3. Human GUI dialog.

**All three only run because A/RED reached the query branch.** They are answer accelerators, not decision bypasses. Cache hits are still counted as A/RED queries for metrics (the decision came from A/RED).

---

## 4. Role of DINOv3 (Strictly an Assistor)

- `DINOFeatureExtractor` (or DINOv2) produces a single vector per tile.
- That vector is the `data_point` passed to `ARED.process_point`.
- DINO has **zero** logic for:
  - Cluster formation
  - `anomalous(...)`
  - Relevance-near detection
  - `kappa`
  - Query / no-query decision
  - Buffer management or forgetting
- Optional rotation augmentation (`data_augmentation_enabled`) happens **after** a real A/RED query for a tile. We extract fresh DINO embeddings of the rotated crops and call the (non-decision) `ared.add_labeled_variant(...)` extension so the variants enrich the same cluster. This mirrors the spirit of the original `DATA_AUG_VAR` but is driven by A/RED's query decisions.

DINO = feature front-end only. A/RED = the algorithm.

---

## 5. Label-Only Mode (Intentionally Bypasses A/RED)

There is a separate "Label Only" checkbox / mode.
- It completely skips DINO and ARED.
- Every tile (per stride) is shown to the human for labeling.
- This is for building high-quality reference annotations needed to compute QP/RR later.
- When a normal A/RED run is started, `label_only_mode=False` and the ARED path is always taken.

---

## 6. Replay / "Load ARED Model"

- `save_state` exports the labeled points currently in ARED's buffer (embeddings + labels + relevance).
- `load_state` creates a fresh ARED with the same hyperparameters and replays the points using a special `meta={"replay": True}` short-circuit inside the answer hook.
- This reconstructs the prior clustering state. It does not "tell A/RED labels on points it would not have queried" — we are replaying the history of points that were queried (or assigned via prior decisions) in the saved session.

---

## 7. Summary of Strictness

| Aspect                        | How it is enforced                                                                 | Matches papers? |
|-------------------------------|------------------------------------------------------------------------------------|-----------------|
| Query decision source         | Only inside `ARED.process_point` / `process_first_point` via `anomalous` + relevant-near test | Yes |
| Labels supplied to ARED       | Only via monkey-patched `answer_query` that ARED calls on its query branch        | Yes |
| Peeks vs real queries         | Explicit `is_peek` guard; GUI/provider never called on peeks                      | Necessary adaptation for interactive use |
| DINO role                     | Embedding generator only; no decision code                                        | Yes (feature front-end) |
| Caches / exact DB             | Answer providers, invoked only after ARED query decision                          | Yes (performance, not bypass) |
| Non-queried points            | Assigned by ARED internally to comparison cluster; no label requested             | Yes |
| First point                   | Always queries (per original)                                                     | Yes |
| Label-Only mode               | Completely disables ARED creation and calls                                       | Explicit separate path |

No code path exists that walks `subspace_partition`, manually inserts labeled points, or forces a label into a cluster before ARED's query logic has run for that point (except faithful replay).

---

## 8. Files of Interest

- `drone_ared/ared_adapter.py` — the only file that knows the original ARED; all shims live here.
- `drone_ared/pipeline.py` — wires the provider and calls `ared_adapter.process`.
- `A_REDimplementation/A_RED/A_REDIN.py` — the canonical implementation we call (never edited).
- `A_REDimplementation/A_RED/Oracle.py` — reference (we use a dummy + patches).
- Papers: `SPIE_IVSP_2026.pdf`, `IJSC_2026-1.pdf`.

---

This integration was deliberately written so that A/RED remains the sole source of query / no-query decisions. DINOv3 (and the caches) are conveniences around it.