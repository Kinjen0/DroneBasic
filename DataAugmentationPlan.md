# Data Augmentation Plan for Drone A/RED (DINO + A/RED)

**Date**: 2026-07-02  
**Context**: We are building an interactive A/RED pipeline for anomaly/relevant event detection in drone video (uniform tiles → DINO embeddings → A_REDIN). The goal is a faithful evaluation of A/RED effectiveness.

**Requirement Summary** (verbatim from request):
- When A/RED makes a query and we obtain a label (from user GUI **or** database/cache), rotate the source tile image (3 rotations).
- Generate fresh DINO embeddings for the rotated images.
- Insert **all** (original + rotated) embeddings into the A/RED cluster with the **same** label + relevance.
- This must be **optional** (a flag).
- Make **as few changes as possible** to the core A_RED implementation (`A_REDimplementation/A_RED/` files).
- We **can** extend A_RED functionality if needed (small, clean additions are acceptable).
- Current built-in augmentation (`DATA_AUG_VAR`) is unsuitable.

## 1. Analysis of Existing Data Augmentation in A/RED

Current mechanism lives here:
- `A_REDIN.py`: passes `DATA_AUG_VAR` to `ARED(...)` and `FiniteBuffer`.
- `FiniteBuffer.py`: `insert_pt(...)`
  - If `DATA_AUG_VAR[0] == 1`:
    - Assumes the incoming `X` is a **flattened raw image** (e.g. 256×256 grayscale or similar).
    - Does `img = X.reshape(shape)`
    - For `k in 1,2,3`: `rot_img = np.rot90(img, k=k)`, flatten, and **append 3 extra copies** into the circular buffers (same label, relevance, cluster_key, true_abs_idx).
    - This multiplies the number of entries in `l_buf` (hence the `% 4` requirement on buffer size in some places).
- `main.py` and examples default to `DATA_AUG_VAR = (0, (256, 256))` or `(1, ...)` for pixel-based experiments.

**Why it does not work for us**:
- We feed **DINO embeddings** (e.g. 384-dim or 768-dim float vectors), not pixel arrays.
- You cannot meaningfully `rot90` an embedding vector.
- The augmentation happens *inside* the labeled buffer insert, after the embedding would have already been produced.
- Our tiles are square (good), but the data path is completely different (pre-DINO vs. post-pixel).
- Result: setting `DATA_AUG_VAR=(1, ...)` would either crash or produce garbage.

**Conclusion**: We will **not** rely on the internal pixel augmentation for DINO runs. We implement augmentation at the *image → DINO embedding* level in our wrapper code, then feed the resulting embeddings as additional labeled variants.

## 2. High-Level Design

### Core Idea
1. A/RED (via our adapter + provider) decides a tile needs a label → we obtain `(label, relevant)` from GUI or exact/embedding DB.
2. The **main** embedding for the original tile is processed normally by A/RED (this is what triggers the cluster assignment / query).
3. **If augmentation flag is on**:
   - Take the original `PIL.Image` tile (we already pass `tile_image` to `process()` and the provider).
   - Generate 3 rotated versions (90°, 180°, 270° — "rotate the image 3 times").
   - Run DINO on each rotated image → 3 new embeddings.
   - Insert those 3 embeddings into A/RED **with the identical label + relevance**, so they contribute to:
     - The same cluster(s)
     - `comp_distance` / diameter calculations
     - Nearest-neighbor search (ball trees)
     - Relevant-cluster logic on future points
4. The augmented points are **synthetic** — they should not:
   - Increment `num_queries` or "user queries" counters.
   - Be treated as new independent points from the video stream for most statistics.
   - Trigger new GUI queries.

### Augmentation Scope
- Only on points that caused a **real A/RED query decision** (i.e. the path where `was_queried=True` in the adapter).
- Applies whether the label came from the human or from cache/DB (as long as it satisfied a query).
- **Optional** via flag — default **off** (to keep behavior identical to papers for baseline comparisons).
- Performed at **insertion time** (live), not on every tile.

### Rotation Details
- Use `PIL.Image.rotate(angle, resample=Image.BICUBIC or LANCZOS, expand=False)`.
- `expand=False` keeps the output square (our tiles are configured square: 224/256/etc.).
- Rotate the raw tile **before** the DINO processor (so the full preprocessing pipeline — resize to model input, normalization — is applied to each view).
- Batch the 3 rotations + original (if convenient) for one DINO call when possible.
- Angles: `[90, 180, 270]` (3 rotations) → total 4 variants per queried tile (original is already inserted by the normal path).

## 3. Changes — Prioritized by "Fewest Edits to A_RED"

### 3.1 No (or Almost No) Changes to A_RED Core (Preferred Path)

We can get surprisingly far without touching `A_REDIN.py`, `FiniteBuffer.py`, etc.

**Strategy**:
- Perform the 3 extra DINO calls in our code.
- For each rotated embedding, call `ared_adapter.process(rot_emb, tile_image=None, meta={"augmented": True, "label": label, "relevant": relevant})`.
- Extend the existing **replay / forced-label** mechanism (already present for model load) to support "forced labeled variant" inserts.

In `ared_adapter.py` the `_interactive_answer` already has special cases:
- `if meta and meta.get("replay"):` → force the label.
- Peek vs. real query detection.

We can add:
```python
if meta.get("augmented"):
    # Force the label, and also tell downstream "do not count this as a query"
    ...
    was_queried = False   # or a separate flag
    return label, rel
```

Then, inside `process()`, when we see `meta.get("augmented")`, after calling `process_point(rot_emb)` we can optionally patch stats so `num_queries` etc. are not polluted.

**Problem with this approach**:
- `process_point` will still run the full logic: `determine_comparison_cluster`, `anomalous = distance * kappa > comp_distance`, `if ... or is_anomalous: query`.
- A rotation embedding may be far enough in DINO space that it triggers another "query".
- Even if the oracle returns the label silently, `self.num_queries += 1`, `anom_only_queries` etc. will be incremented, and the point may create a spurious new cluster or split.

**Mitigation (still zero core changes)**:
- Right after the main label, temporarily set a very high tolerance (monkey-patch `self.ared.kappa = 999` for the aug calls, then restore).
- Or temporarily monkey-patch `self.ared.anomalous` to always return `False`.
- Ugly but localized entirely in the adapter.

This path keeps `A_REDIN.py` 100% untouched.

### 3.2 Small Clean Extension to A_RED (Recommended)

Because the user explicitly allows "we can extend its functionality", we do a **minimal, well-contained addition**.

In `A_REDimplementation/A_RED/A_REDIN.py`, inside the `ARED` class, add one new public method (≈15-30 lines):

```python
def add_labeled_variant(self, data_point: np.ndarray, label: str, relevance: bool):
    """Add a data-augmented variant (e.g. DINO embedding of a rotation)
    for a point that has already been labeled.

    - Does NOT run query decision logic.
    - Does NOT increment num_queries / anom_only_queries etc.
    - Re-uses the same label/relevance (and therefore the same cluster if it exists).
    - Updates the labeled buffer and cluster statistics.
    """
    # 1. Find a cluster that already has this label (prefer most recently touched)
    cluster_key = None
    for ck, cl in reversed(list(self.subspace_partition.cluster_dict.items())):
        if cl.label == label:
            cluster_key = ck
            break

    if cluster_key is None:
        # Fallback: create a new one (rare, only if the main point hasn't finished yet)
        cluster_key = self.subspace_partition.create_new_cluster(
            label, relevance, [], [], self.QS_VAR
        )

    # 2. Allocate a synthetic / duplicated abs index for buffer purposes
    #    We can reuse the last real abs_index or create an internal one.
    #    For simplicity we duplicate the behavior of the original pixel aug.
    data_point_abs_idx = self.abs_index   # share with the "parent" queried point

    # 3. Insert into buffer (this will also handle ball trees + forgetting if full)
    forgotten = self.l_buf.insert_pt(
        data_point, cluster_key, label, relevance, data_point_abs_idx
    )

    # 4. Tell the cluster about the new point (updates comp_distance etc.)
    cl = self.subspace_partition.cluster_dict[cluster_key]
    cl.add_l_pt_no_comp_dist_update(data_point_abs_idx)   # or the normal add_l_pt if we want diameter update
    # (see existing add_l_pt_to_existing_cl for reference)

    # Optional: also back-fill oracle.y so later peeks are happy
    if hasattr(self.oracle, 'y'):
        try:
            self.oracle.y[data_point_abs_idx] = [label, relevance]
        except Exception:
            pass
```

Then, in our `AREDAdapter`, we can call:

```python
if augmentation_enabled:
    for rot_emb in rotated_embs:
        self.ared.add_labeled_variant(rot_emb, label, rel)
```

**Advantages**:
- Logic lives next to the existing `add_l_pt_to_existing_cl` / `update_structs_w_new_pt`.
- We can copy/adapt 10-15 lines from those methods.
- Zero impact on the streaming / query decision paths.
- Easy to make correct w.r.t. buffer indices and cluster membership.

This is the cleanest long-term solution.

### 3.3 Hybrid (Prototype in Adapter, Upstream Later)

Start with the monkey-patch hack in the adapter (zero edits). Once it works, add the `add_labeled_variant` helper as a small PR to the A_RED repo.

## 4. Detailed Implementation Steps

### Step 1: Configuration (minimal)
File: `drone_ared/config.py`

Add to `AREDConfig` (or a sibling `AugmentationConfig`):

```python
data_augmentation_enabled: bool = False
augmentation_rotations: list[int] = [90, 180, 270]   # degrees
# Future: also flips? light color jitter? (keep simple for now)
```

Expose `DATA_AUG_VAR` remains for backward compat but we will ignore/set to `(0,...)` when our flag is used.

Update `PipelineConfig` load/save.

### Step 2: Augmentation Helper (new or extend)
Create `drone_ared/augmentation.py` (or add to `feature_extractor.py`):

```python
from PIL import Image
import numpy as np
from typing import List

def rotate_image(pil_img: Image.Image, angle: int, resample=Image.BICUBIC) -> Image.Image:
    return pil_img.rotate(angle, resample=resample, expand=False)

class EmbeddingAugmenter:
    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def get_rotated_embeddings(self, pil_image: Image.Image, angles: List[int]) -> List[np.ndarray]:
        if not angles:
            return []
        rotated_imgs = [rotate_image(pil_image, a) for a in angles]
        # Batch for efficiency
        embs = self.fe.extract_images(rotated_imgs)
        return [embs[i] for i in range(len(rotated_imgs))]
```

Make `DINOFeatureExtractor` support being passed a list cleanly (it already batches).

### Step 3: Wire the Flag and Call Site
Primary files:
- `drone_ared/ared_adapter.py`
- `drone_ared/pipeline.py` (pass `tile_image` reliably, forward config)

In `AREDAdapter.__init__` / config:
```python
self.data_augmentation_enabled = config.data_augmentation_enabled
self.aug_rotations = getattr(config, 'augmentation_rotations', [90,180,270])
```

After a real label is obtained (in `process()`, after the `process_point` call, when `was_queried`):

```python
if was_queried and obtained_label and self.data_augmentation_enabled and tile_image is not None:
    try:
        from .augmentation import EmbeddingAugmenter
        augmenter = EmbeddingAugmenter(...)  # or cache one
        rot_embs = augmenter.get_rotated_embeddings(tile_image, self.aug_rotations)
        for remb in rot_embs:
            self._insert_augmented_variant(remb, obtained_label, obtained_rel)
    except Exception as e:
        print(f"[Adapter] Augmentation failed (non-fatal): {e}")
```

Implement `_insert_augmented_variant` using either the monkey approach or the new `add_labeled_variant`.

Also update the `info` dict returned so the pipeline/GUI can log " + 3 augmented".

### Step 4: Optional Small Extension in A_RED (if we choose the clean path)
File: `A_REDimplementation/A_RED/A_REDIN.py`

Add the method described in section 3.2.

Also consider updating `FiniteBuffer.insert_pt` docstring or adding a comment that our DINO aug is handled upstream.

### Step 5: GUI Exposure (usability)
In `drone_ared/gui.py`:
- Add a checkbox in the parameters frame: "Data Augmentation (rotate labeled tiles 3×)"
- On Start, read it into `config.ared.data_augmentation_enabled`
- Optional: show in stats "Augmented inserts: X"

### Step 6: Label DB / Persistence Interaction
- When we save to `TileAnnotationDB` or the embedding `PersistentLabelStore`, we save the **original** tile / emb.
- Augmentation is re-generated on the fly the next time that label is used in a live run (if flag is on).
- For ARED model save/replay (`save_state` / `load_state`): the replayed points are the ones that were in `l_buf` at save time. If aug was on during the original run, the rotated variants will already be in the saved buffer snapshot → they will be replayed automatically. Good.

If we want deterministic "augment even on replay", we can re-apply in `load_state` after each replayed point, but this is optional.

### Step 7: Stats & Logging Hygiene
- Do **not** count aug inserts toward `num_queries`, `user_queries`, `cache_hits`.
- Add counters: `self.num_augmented_inserts = 0`
- Log clearly: `[ARED] Added 3 rotated variants for label 'X'`

### Step 8: Testing & Validation
1. **Unit / smoke**:
   - Enable flag, feed a known tile, obtain label.
   - After the query, inspect `adapter.ared.l_buf` or `get_current_clusters_summary()`.
   - Verify cluster for the label has `n_points` increased by 3 (or 4 including original).
2. **Correctness**:
   - Rotated images must be different (visually) but same semantic label.
   - DINO on rotated tile must succeed (square input).
3. **No side effects when disabled** (default).
4. **Interaction with**:
   - Exact label DB (replay same video with different stride)
   - Embedding cache hits
   - ARED save/load
   - kappa (aug should not change query rate)
5. **Paper experiments**:
   - Run with/without aug as an ablation.
   - Measure effect on Relevant Recall, Query Precision, number of clusters, stability of relevant clusters.

### Step 9: Documentation
- Update `drone_ared/README.md`
- Mention in `PaperIdeas.md` under "Potential Features" or evaluation axes.
- Add comment in `ared_adapter.py`: "Data augmentation for DINO is implemented here, not via A_RED's DATA_AUG_VAR."

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Rotated DINO embs land in wrong cluster | The `add_labeled_variant` (or forced insert) explicitly uses the label we just got from the user. |
| Buffer bloat (4× on every query) | Document it. User can increase `l_buf_size`. Original A/RED already has this behavior when their aug is on. |
| Performance cost | Only on queried tiles (rare by design of A/RED + kappa). Batch the 3 rotations. Make optional. |
| Non-determinism of rotation quality | Fix resample filter + document. |
| Seeking changes to A_RED internals later | Keep the augmentation code in the adapter + one small helper method. Easy to maintain. |
| "Square image" requirement | Our GridTiler already guarantees square tiles. |

## 6. File Change Summary (Minimal Set)

- `drone_ared/config.py` — add flags
- `drone_ared/augmentation.py` — new (small)
- `drone_ared/ared_adapter.py` — main integration + `_insert_augmented_variant`
- `drone_ared/pipeline.py` — ensure `tile_image` and config are passed through
- `drone_ared/gui.py` — optional checkbox + maybe a stat line
- `A_REDimplementation/A_RED/A_REDIN.py` — **optional** 20-line helper method (only if we take the clean path)
- `drone_ared/README.md` — usage note
- (Later) tests / eval scripts

## 7. Open Decisions for the Team

1. Exact rotation angles? (90/180/270 is the classic for the original A/RED aug.)
2. Should we also do horizontal/vertical flips? (User request was rotations.)
3. Apply augmentation on cache hits that satisfy a query, or only fresh human labels?
4. When the label comes from the persistent embedding cache (similarity), do we still have the original `tile_image`? (In current code the provider receives `tile_img` only for the GUI path. We may need to stash the image for the current batch.)
5. Do we want to store the fact that a particular annotation was augmented (for reproducibility)?

## 8. Next Actions After Plan Approval

1. Implement the config + augmentation helper (pure, easy to test).
2. Wire the call site in the adapter using the "monkey kappa temporarily" approach first (zero A_RED edits).
3. Test on a short video with the flag on/off.
4. If behavior is good, add the small `add_labeled_variant` helper to A_REDIN for cleanliness.
5. Add GUI toggle.
6. Run a small with/without aug comparison and record cluster sizes + query behavior.

This design keeps the spirit of the original A/RED data augmentation (more views of the same labeled event inside the buffer/clusters) while being compatible with a modern DINO front-end and remaining optional and minimally invasive. 

---

**Status**: Plan written. Ready for review / implementation start.