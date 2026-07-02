# Paper Ideas: Evaluating the Effectiveness of A/RED for Anomaly and Relevant Event Detection in Drone Footage

**Focus:** Search & Rescue (SAR), field monitoring, and general anomaly detection in real-world drone video streams.

**Goal:** Produce a rigorous, publishable study (in the style of the existing A/RED papers in this repository) that examines *how well A/RED works* when applied to this domain, rather than just "we built a demo."

**Date of this plan:** 2026-07-02

---

## 1. Core Research Question

How effective is A/RED (Anomalous / Relevant Event Detection) as an active, human-in-the-loop algorithm for discovering relevant anomalies in streaming drone imagery under realistic field conditions?

Sub-questions:
- After an initial warm-up period on "normal" background, does A/RED reliably surface true outliers (new classes or instances near relevant clusters) while suppressing queries on routine variation?
- What are the query burden vs. relevant recall trade-offs when using DINO embeddings + uniform spatial tiling on real drone video?
- How sensitive is performance to κ (the paranoia parameter), buffer size, tile granularity, and feature choices?
- Can the system support operational SAR use cases (low false query load on long boring flights, high chance of catching the rare relevant event)?

---

## 2. Relationship to Existing A/RED Papers in This Repository

Reference and build directly on the papers present:

- **IJSC_2026-1.pdf** — "Real-Time Memory-Bounded A/RED: Discovering Rare Relevant Classes in Streaming Data" (Loveland, Clark, Gentile, Ritter).  
  Introduces the memory-bounded version with circular buffer + ball trees + forgetting. Core algorithm description (Alg. 1), Query Precision and Relevant Recall metrics, evaluation on NICE synthetic, skewed MNIST, EMNIST, MVTec. Emphasizes outperforming random querying while keeping memory bounded.

- **SPIE_IVSP_2026.pdf** — "Shallow vs. Deep Features for Anomalous/Relevant Event Detection in High-Dimensional Streaming Imagery" (Loveland, Clark, Ritter).  
  Directly relevant because it uses **overhead video of a parking lot**, tiles it, compares raw pixels vs. DINOv2 vs. DAGMM features fed to A/RED. Reports performance across κ values and different definitions of "relevant" (1% and 5% of data). DINOv2 performed best. Uses similar tiling + feature pipeline ideas.

- **AIxDKE_2026.pdf** — "Memory-Bounded A/RED: Scalable Active Detection of Rare Relevant Events in Indefinite Length Streams" (same authors).  
  Very similar to IJSC; focuses on scalability for long streams.

- **A_REDimplementation/A_RED/MNIST_2D.pdf** — Earlier 2D visualization / illustration paper.

**How our work differs / extends:**
- Real drone field footage (open fields, natural variation, lighting/vegetation changes, altitude/gimbal effects) instead of controlled parking-lot or synthetic/MNIST data.
- First *interactive human-in-the-loop* deployment with live GUI (the original papers use a static pre-labeled Oracle for both answering queries and computing offline metrics).
- Explicit focus on SAR / practical anomaly detection utility.
- Emphasis on "warm-up then selective querying" behavior observed in the current implementation.
- Goal is an *effectiveness study* (quantitative + qualitative) rather than primarily an algorithmic improvement paper.

We must cite these works heavily and position our contribution as an applied evaluation + systems paper that tests the claims of the algorithmic papers in a new, operationally relevant domain.

---

## 3. Current System Snapshot (What We Actually Have)

- **Videos/**: Several large DJI .MP4 field flights (hundreds of MB to GB). Some saved .pkl label stores from prior runs.
- **Pipeline**:
  - `GridTiler`: uniform spatial grid, **only full non-clipped tiles** (important for validity).
  - `DINOFeatureExtractor`: facebook DINOv2 / timm DINOv3 models, mean/cls/max pooling + optional L2 norm. Batched.
  - `AREDAdapter`: zero-edit bridge to `A_REDimplementation/A_RED/A_REDIN.ARED` (and Oracle).
  - `PersistentLabelStore`: embedding NN cache for auto-answering similar tiles (future runs or later in stream).
  - `DroneAREDController`: threaded streaming (frame stride, batching).
  - `gui.py`: high-volume persistent resizable LabelingDialog (only shown on real A/RED queries), per-class relevance, zoom, etc.
- Behavior: After initial points, normal background tiles are absorbed without queries. Outliers trigger A/RED queries (GUI appears only then). Clusters grow and relevance affects future decisions.
- Save/load: ARED "model" state via replay of labeled buffer points; label cache is persistent.

**Key constraint observed throughout development:** The implementation must (and does) let the original A/RED logic decide *when* and *why* to query.

---

## 4. A/RED Fidelity Review — Is It Properly Using the Algorithm? (Must Be Ironclad for a Paper)

**Conclusion: Yes — the current architecture respects the core ideas of A/RED and does not subvert them.** This section should appear (perhaps condensed) in the paper's Method or as a subsection "Faithful Implementation of A/RED".

### Core A/RED Ideas (from the papers + source)
From `A_REDIN.py` and the papers:
- `process_point` (and `process_first_point`) computes a **comparison cluster** using nearest labeled points (accelerated by ball trees when K_COMP_PTS > 1).
- A point is **anomalous** if `distance * kappa > cluster.comp_distance` (QS_VAR controls diameter vs. avg-NN).
- Query decision (inside ARED, untouched): `if comp_cl_is_relevant or is_anomalous`.
- Only on that branch does it call `query()` → `oracle.answer_query()` a **second time** to obtain the human (or cached) label + relevance.
- There is always a first "peek" `answer_query` (for internal accounting / correct_class_counter) even on non-queried points.
- Relevance is a property attached to the **cluster / label** and influences future comparison cluster choice and query decisions.
- Queried points go into the circular `l_buf` (FiniteBuffer). Forgetting, small-cluster merging, neighborhood merge, etc. all happen inside the original code.
- No ground-truth y array is required for the online streaming path; the Oracle only supplies answers on demand.

### How Our Code Preserves These (No Subversion)
1. **Unmodified core**: `from A_REDIN import ARED as _OriginalARED`. No edits to `A_REDIN.py`, `Oracle.py`, `FiniteBuffer.py`, etc.
2. **Peek vs. real query distinction** (in `ared_adapter.py`):
   - Monkey-patch of `oracle.answer_query` is installed only for the duration of a single `process()` call.
   - `hook_calls` counter + `is_peek = (call_num == 1 and self.num_points_processed > 0)` detects the accounting peek.
   - On peek: provisional label or cache hit; **never invokes the GUI provider**.
   - Only the second call (or first_point) takes the real path that may show the GUI or use cache miss → human label.
   - This exactly mirrors the "two calls on query path, one peek otherwise" behavior in the original `process_point`.
3. **Decision logic is inside ARED**: The adapter never looks at distance or kappa itself to decide whether to ask the human. It only reacts to how many times ARED called the answer hook during that point.
4. **Relevance handling**:
   - Stored per-class (not per-assignment) in `class_relevance` dict.
   - Checkbox affects only *new* class creation (correctly, because relevance defines the class for future decisions).
   - Seeded from `label_store.get_class_relevance()`.
   - Passed back through the provider → ARED uses it when creating/adding to clusters.
5. **Buffer / forgetting / clustering untouched**: `save_state` only exports the labeled points that are currently in `l_buf`. `load_state` replays them via `process(..., meta={"replay": True, ...})` which forces the exact labels so clusters and comp_distances are reconstructed faithfully. No direct mutation of `subspace_partition`, `l_buf`, etc.
6. **Cache is only a label *provider***: The PersistentLabelStore can short-circuit the human for both peeks and real queries, but it never changes *whether* ARED decides a point needs a query. Cache hits still let ARED update its internal structures with the (correct) label/relevance.
7. **Open-world labels**: `_GrowingLabelMap` + `_SparseY` only expand the machinery the original code already expects (bidict + y); they do not alter query conditions or cluster logic.
8. **Logging guards**: Explicit "ARED decided to QUERY" vs. "did NOT trigger a (real) query" messages. GUI only appears for queue items that came from real decisions.

### Potential Subversion Risks (and Why They Are Avoided)
- Forcing every tile through GUI → **avoided** (the whole point of the later fixes).
- Ignoring kappa by always/never querying → **impossible** because the if-branch is inside ARED.
- Treating relevance as a per-tile checkbox that changes after class creation → **fixed** (per-class at creation time).
- Bypassing the circular buffer or manually adding points → never done.
- Using ARED only for clustering after offline labeling → no; we stream exactly as intended.
- Replaying without proper labels → replay path forces the saved (label, relevant) pairs.

**Recommendation for paper**: Include a short "Implementation Fidelity" subsection + pseudo-code or call sequence diagram showing peek vs. real path. This makes the effectiveness claims credible.

If we ever add heuristics that short-circuit before calling `ared.process()`, we must document them as "early rejection for operational efficiency" and still measure the effect on the pure A/RED path.

---

## 5. Proposed Paper Structure (Modeled on the Existing Papers)

Typical structure observed:
1. Abstract (emphasize metrics + SAR motivation + "first interactive field evaluation")
2. Introduction (streaming drone imagery challenges, SAR needs, human-in-loop assumption, κ as control knob)
3. Related Work (cite the three A/RED papers + classic AD, rare category, active learning for anomaly, DINO in video)
4. Method
   - A/RED recap (Alg. 1 from papers)
   - Drone pipeline: video ingest → GridTiler (full tiles only) → DINO → AREDAdapter (interactive + cache + replay) → persistent GUI
   - Fidelity discussion (see §4)
   - Parameters exposed (κ, tile size/stride, frame stride, buffer size, DINO variant/pooling, cache threshold)
5. Dataset
   - Field collection description (altitude, sensor, environments, total frames/tiles)
   - Characterization of "normal" vs. potential relevant events
   - (Future) planted SAR targets or annotated anomalies
6. Evaluation Methodology (detailed in §6)
7. Results (quantitative tables/figures + qualitative examples)
8. Discussion (κ sensitivity, warm-up behavior, DINO value, human factors, operational implications for SAR)
9. Limitations & Future Work
10. Conclusion

**Figures/tables to plan for**:
- Query rate (queries / tiles) vs. κ (multiple videos)
- Relevant Recall and Query Precision vs. κ (once we have retrospective GT)
- Cluster growth over time / number of discovered classes
- Example queried tiles (with labels + relevance) — "what surfaced"
- Comparison to random baseline at matched query budget
- Timing / throughput (tiles/sec)
- Cache hit rate over time (value of persistence)

---

## 6. How to Rigorously Test Effectiveness

This is the most important planning section. The papers' credibility comes from clear metrics + baselines.

### Primary Metrics (directly from SPIE / IJSC papers)
- **Query Precision** ≜ TP / (TP + FP)  
  Where a "positive" = a tile that *should* be queried: (a) first example of a previously unseen class, or (b) example of a class previously marked relevant by the user.
- **Relevant Recall** ≜ TP / (TP + FN)  
  Fraction of all tiles that belong to relevant classes that were actually queried.

These treat the query/no-query decision as the classifier. Perfect for A/RED.

We also track:
- Total queries vs. total tiles (query burden)
- Number of classes discovered
- Fraction of queries that were "new class" vs. "near relevant"
- Cache hit rate (reduces human burden)

### The Ground-Truth Problem on Field Footage
Current videos have no exhaustive labels. Solutions (use a combination):

1. **Retrospective full labeling on a sampled test set**  
   After (or during) runs, have a human label a random or stratified sample of tiles (or all tiles from selected short segments) with (class, relevant). Use this to build an offline "should query" oracle.

2. **Planted / known events**  
   Conduct flights with known "targets" (people, vehicles, objects placed in field). Mark their approximate frame/time + rough location. Convert to expected query events.

3. **SME review of all queries + sampled non-queries**  
   For the queries the system *did* make, record whether the SME agrees it was worth querying. For long non-query stretches, spot-check that nothing important was missed.

4. **Simulated injection**  
   Take normal footage, digitally insert rare objects at known locations/times, run the system, measure whether they trigger queries (and at what κ).

### Baselines
- **Random querying** at the same average query rate as A/RED (the main comparison in the papers). Repeat many times.
- **Always query** (upper bound on recall, terrible burden).
- **Offline AD** (e.g. isolation forest or one-class SVM on all embeddings after the fact) for reference — note this is not streaming / active.
- (Stretch) A simple motion or background-subtraction heuristic.

### Experimental Protocol Ideas
- Fix DINO model + tiling. Sweep κ on the same video(s): 0.5, 1.0, 2.0, 3.0, 5.0, 8.0 ...
- Multiple independent runs or contiguous segments to get statistics.
- Warm-up analysis: plot cumulative queries starting from tile 0; identify when "settling" occurs.
- Cache ablation: with vs. without PersistentLabelStore (effect on effective human queries).
- Feature ablations: different DINO variants, pooling methods, with/without L2.
- Tile granularity: 224, 256, 320, different strides (overlap vs. non-overlap).
- Buffer size sensitivity.
- For each condition record: QP, RR (once GT exists), #queries, #discovered classes, wall time, human labels required.

### Qualitative / Operational Evaluation
- Case studies: "During a 12-minute boring field flight, A/RED made 17 queries. 3 were new background variations, 2 were relevant (person + ATV)."
- Video or storyboard of the actual queried tiles in temporal order.
- SME feedback: "Would you have noticed this without the system?" "Was the query rate acceptable?"

### Reproducibility
- Save embeddings + global tile indices + the sequence of (label, relevant) decisions.
- Provide a replay script that feeds the same stream into a fresh ARED (or our adapter) and recomputes metrics.
- The existing label .pkl + ARED model save already give a strong starting point.

### Logging Already Helps
Our terminal logs (`[Pipeline] Finished tile ... ARED queried?`, `[ARED] A_RED decided to QUERY`, etc.) plus the GUI state can be parsed to compute per-run statistics without extra instrumentation.

---

## 7. Data Strategy for a Strong Paper

Current data is a good start ("field" rather than parking lot), but for publication we need more:
- Document flight parameters (height AGL, speed, camera angle, time of day, weather).
- Multiple environments.
- Controlled anomaly injection flights.
- At least one fully annotated short sequence (all tiles labeled) for clean metric computation.
- Consider releasing a small "Drone-ARED-Field" tile dataset (with ethics / privacy review).

If GPS/metadata is present in the DJI files, extract it so queried tiles can be geo-referenced — very powerful for SAR.

---

## 8. Potential Threats to Validity (Address in Paper)

- Lack of exhaustive GT on current footage → mitigate with multiple annotation methods above.
- DINO embeddings were (in some original experiments) computed with knowledge of the whole set; here we do pure streaming inference.
- Human labeler fatigue / consistency for long sessions → persistent dialog + cache help; report inter-session agreement if possible.
- "Relevant" is subjective → operational definition + multiple raters if feasible.
- Warm-up length depends on scene complexity → report per-video statistics.

---

## 9. Suggested Next Steps & Milestones (to keep the work straight)

1. **Fidelity audit complete** (this document + code comments). Done.
2. Create `PotentialFeatures.md` (see separate file) — improvement ideas that do *not* break A/RED semantics.
3. Build an **offline evaluation harness**:
   - Script that can take a set of tiles + a "gold" label/relevance file.
   - Replays the stream, records every query decision vs. gold "should query".
   - Computes QP / RR + plots.
4. Perform at least one full manual annotation pass on a manageable video segment or sampled tiles.
5. Run systematic κ sweeps on 1–2 videos; produce the first result tables/figures.
6. Collect or simulate SAR-specific anomalies (people in open, camouflaged, vehicles, etc.).
7. Draft Method + Evaluation sections first (they will drive what data/metrics we actually need).
8. Consider reaching out to the original authors for feedback or co-authorship (they are at New College of Florida).
9. Plan a small user study (one or two SAR-experienced observers) for qualitative feedback.
10. Write the paper iteratively while improving tooling.

---

## 10. Deliverables That Would Support the Paper

- Reproducible run scripts + config files used for the reported experiments.
- The evaluation harness + any retrospective labeling tools.
- High-quality figures (queried tile montages, κ curves, cluster timelines).
- Video supplement showing live GUI + example flights.
- (Optional but strong) Public small dataset of field tiles with labels.

---

## Notes / Open Questions for the Team

- Should we also explore a "pure anomaly" (no relevance) baseline inside the same framework?
- Is there value in temporal modeling (track tiles across frames) or should we stay strictly per-tile like the original papers?
- How do we handle extreme class imbalance in reporting (most of a long flight may be one "normal" class)?
- Long-term: on-drone or edge deployment vs. ground-station analyst workflow?

This plan is intentionally modeled on the structure, metrics, and scientific tone of IJSC/SPIE/AIxDKE papers while being honest about the current data situation and the interactive nature of our implementation.

---

**Sub-file:** See `PotentialFeatures.md` (in the same directory) for concrete suggestions on where the *program* can be improved without compromising A/RED correctness. Those items can be framed in the paper as "systems contributions" or "practical deployment considerations" or future work.

Write the paper *after* we have at least one solid quantitative comparison (QP/RR vs. random) on annotated data. The fidelity review above is non-negotiable for credibility.
