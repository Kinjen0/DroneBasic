#!/usr/bin/env python3
"""
t-SNE visualization of labeled tile embeddings (Labeling-Ready).

Primary source: SQLite TileAnnotationDB (drone_tile_annotations.db) rows that
store a DINO embedding blob.

Fallback: pickle PersistentLabelStore (drone_ared_labels.pkl) in either
Labeling-Ready simple format or root "entries" format.

Usage (from Labeling-Ready root):
    python scripts/tsne_label_embeddings.py --db drone_tile_annotations.db
    python scripts/tsne_label_embeddings.py --db drone_tile_annotations.db --max-per-class 100
    python scripts/tsne_label_embeddings.py --labels drone_ared_labels.pkl
    python scripts/tsne_label_embeddings.py --db ... --features pixels --max-per-class 50
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_sqlite(db_path: Path, require_embedding: bool = True) -> List[Dict[str, Any]]:
    """Load entries from TileAnnotationDB (prefer package helper, else raw SQL)."""
    try:
        from drone_ared.tile_database import TileAnnotationDB
        db = TileAnnotationDB(db_path=str(db_path))
        try:
            return db.iter_embedding_entries(require_embedding=require_embedding)
        finally:
            try:
                db.close()
            except Exception:
                pass
    except Exception as e:
        print(f"[tsne] Package TileAnnotationDB load failed ({e}); using raw sqlite3")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        if require_embedding:
            cur.execute(
                """
                SELECT video_path, abs_frame, tile_row, tile_col, tile_width, tile_height,
                       stride_x, stride_y, crop_x, crop_y, label, relevant,
                       embedding, embedding_dim
                FROM annotations
                WHERE embedding IS NOT NULL
                """
            )
        else:
            cur.execute(
                """
                SELECT video_path, abs_frame, tile_row, tile_col, tile_width, tile_height,
                       stride_x, stride_y, crop_x, crop_y, label, relevant,
                       embedding, embedding_dim
                FROM annotations
                """
            )
        out: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            (
                vpath, frame, trow, tcol, tw, th,
                sx, sy, cx, cy, label, relevant, emb_blob, emb_dim,
            ) = row
            if not label or str(label).startswith("__"):
                continue
            emb = None
            if emb_blob is not None:
                arr = np.frombuffer(emb_blob, dtype=np.float32)
                if emb_dim is not None and int(emb_dim) > 0:
                    arr = arr.reshape(-1)[: int(emb_dim)]
                emb = np.asarray(arr, dtype=np.float32).reshape(-1)
            elif require_embedding:
                continue
            if cx is not None and cy is not None and tw is not None and th is not None:
                bbox = (int(cx), int(cy), int(cx) + int(tw), int(cy) + int(th))
            elif tcol is not None and trow is not None and tw and th:
                bbox = (
                    int(tcol) * int(tw), int(trow) * int(th),
                    int(tcol) * int(tw) + int(tw), int(trow) * int(th) + int(th),
                )
            else:
                bbox = None
            out.append({
                "label": str(label),
                "relevant": bool(relevant),
                "embedding": emb,
                "meta": {
                    "video_path": vpath,
                    "frame": int(frame) if frame is not None else -1,
                    "frame_idx": int(frame) if frame is not None else -1,
                    "row": int(trow) if trow is not None else -1,
                    "col": int(tcol) if tcol is not None else -1,
                    "tile_width": tw,
                    "tile_height": th,
                    "stride_x": sx,
                    "stride_y": sy,
                    "crop_x": cx,
                    "crop_y": cy,
                    "bbox": bbox,
                },
            })
        return out
    finally:
        conn.close()


def load_from_pkl(path: Path) -> List[Dict[str, Any]]:
    """Load root-style entries or simple Labeling-Ready pkl."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    entries: List[Dict[str, Any]] = []

    # Root PersistentLabelStore with structured entries
    if isinstance(data, dict) and "entries" in data:
        raw = data["entries"]
        for e in raw:
            if isinstance(e, dict):
                entries.append({
                    "embedding": np.asarray(e["embedding"], dtype=np.float32).reshape(-1),
                    "label": str(e["label"]),
                    "relevant": bool(e.get("relevant", False)),
                    "meta": dict(e.get("meta") or {}),
                })
            else:
                entries.append({
                    "embedding": np.asarray(e.embedding, dtype=np.float32).reshape(-1),
                    "label": str(e.label),
                    "relevant": bool(getattr(e, "relevant", False)),
                    "meta": dict(getattr(e, "meta", None) or {}),
                })
        return entries

    # Simple Labeling-Ready format: parallel lists
    if isinstance(data, dict) and "embeddings" in data and "labels" in data:
        embs = data["embeddings"]
        labels = data["labels"]
        rels = data.get("relevances") or [False] * len(labels)
        for emb, lab, rel in zip(embs, labels, rels):
            if not lab or str(lab).startswith("__"):
                continue
            entries.append({
                "embedding": np.asarray(emb, dtype=np.float32).reshape(-1),
                "label": str(lab),
                "relevant": bool(rel),
                "meta": {},
            })
        return entries

    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and "embedding" in e and "label" in e:
                entries.append({
                    "embedding": np.asarray(e["embedding"], dtype=np.float32).reshape(-1),
                    "label": str(e["label"]),
                    "relevant": bool(e.get("relevant", False)),
                    "meta": dict(e.get("meta") or {}),
                })
        return entries

    raise ValueError(f"Unrecognized label store format in {path}")


def balanced_subsample(
    entries: Sequence[Dict[str, Any]],
    max_per_class: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if max_per_class is None or max_per_class <= 0:
        return list(entries)
    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_label[e["label"]].append(e)
    out: List[Dict[str, Any]] = []
    for label in sorted(by_label.keys()):
        group = by_label[label]
        if len(group) <= max_per_class:
            out.extend(group)
        else:
            idx = rng.choice(len(group), size=max_per_class, replace=False)
            out.extend(group[i] for i in idx)
    return out


# ---------------------------------------------------------------------------
# t-SNE + plots
# ---------------------------------------------------------------------------

def run_tsne(X: np.ndarray, seed: int, perplexity: Optional[float] = None) -> np.ndarray:
    n = X.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    if n == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    perp = perplexity if perplexity is not None else min(30.0, max(5.0, (n - 1) / 3.0))
    perp = min(perp, n - 1)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def cohesion_stats(X: np.ndarray, labels: Sequence[str], max_pairs: int = 2000) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    labels_arr = np.asarray(list(labels))
    n = len(labels_arr)
    stats: Dict[str, Any] = {"n": n}

    if n >= 2:
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(X)
        _, idxs = nn.kneighbors(X)
        same = sum(1 for i in range(n) if labels_arr[idxs[i, 1]] == labels_arr[i])
        stats["nn_same_label_rate"] = same / n
    else:
        stats["nn_same_label_rate"] = float("nan")

    within: Dict[str, float] = {}
    rng = np.random.default_rng(0)
    for lbl in sorted(set(labels_arr.tolist())):
        pts = X[labels_arr == lbl]
        m = len(pts)
        if m < 2:
            within[lbl] = float("nan")
            continue
        n_pairs = min(max_pairs, m * (m - 1) // 2)
        if m * (m - 1) // 2 <= max_pairs:
            i, j = np.triu_indices(m, k=1)
        else:
            i = rng.integers(0, m, size=n_pairs)
            j = rng.integers(0, m, size=n_pairs)
            mask = i != j
            i, j = i[mask], j[mask]
        dists = np.linalg.norm(pts[i] - pts[j], axis=1)
        within[lbl] = float(dists.mean()) if len(dists) else float("nan")
    stats["mean_within_class_l2"] = within
    return stats


def print_cohesion(stats: Dict[str, Any]) -> None:
    print(f"[tsne] n={stats['n']}  1-NN same-label rate={stats.get('nn_same_label_rate', float('nan')):.3f}")
    print("[tsne] Mean within-class pairwise L2:")
    for lbl, v in (stats.get("mean_within_class_l2") or {}).items():
        print(f"  {lbl}: {v:.4f}" if v == v else f"  {lbl}: n/a")


def _label_color_map(labels: Sequence[str]) -> Dict[str, Any]:
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    return {lbl: cmap(i % cmap.N) for i, lbl in enumerate(unique)}


def plot_class_tsne(
    coords: np.ndarray,
    labels: Sequence[str],
    relevant: Sequence[bool],
    out_path: Path,
    title: str,
) -> None:
    colors = _label_color_map(labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in sorted(set(labels)):
        mask = np.array([l == lbl for l in labels])
        rel = np.array(relevant)[mask]
        pts = coords[mask]
        if (~rel).any():
            ax.scatter(
                pts[~rel, 0], pts[~rel, 1],
                c=[colors[lbl]], label=f"{lbl} (n={mask.sum()})",
                s=28, alpha=0.55, edgecolors="none",
            )
        if rel.any():
            ax.scatter(
                pts[rel, 0], pts[rel, 1],
                c=[colors[lbl]],
                s=55, alpha=0.9, edgecolors="k", linewidths=0.6,
                label=f"{lbl} [R] (n={int(rel.sum())})",
            )
    handles, leg_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(leg_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[tsne] Wrote {out_path}")


def write_coords_csv(
    path: Path,
    coords: np.ndarray,
    labels: Sequence[str],
    extra_cols: Optional[Dict[str, Sequence[Any]]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x", "y", "label"]
    if extra_cols:
        fieldnames.extend(extra_cols.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(len(labels)):
            row = {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": labels[i]}
            if extra_cols:
                for k, vals in extra_cols.items():
                    row[k] = vals[i]
            w.writerow(row)
    print(f"[tsne] Wrote {path}")


# ---------------------------------------------------------------------------
# Pixel features (optional baseline)
# ---------------------------------------------------------------------------

_frame_cache: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
_caps: Dict[str, Any] = {}


def _get_frame_rgb(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    if not HAS_CV2:
        return None
    key = (video_path, int(frame_idx))
    if key in _frame_cache:
        return _frame_cache[key]
    if video_path not in _caps:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Try resolving relative to ROOT / parent
            for base in (ROOT, ROOT.parent):
                alt = base / Path(video_path).name
                if alt.exists():
                    cap = cv2.VideoCapture(str(alt))
                    if cap.isOpened():
                        video_path = str(alt)
                        break
            if not cap.isOpened():
                _frame_cache[key] = None
                return None
        _caps[video_path] = cap
    cap = _caps[video_path]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        _frame_cache[key] = None
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if len(_frame_cache) > 64:
        _frame_cache.clear()
    _frame_cache[key] = rgb
    return rgb


def close_video_caps() -> None:
    for cap in _caps.values():
        try:
            cap.release()
        except Exception:
            pass
    _caps.clear()
    _frame_cache.clear()


def crop_tile_from_meta(meta: Dict[str, Any]) -> Optional[Any]:
    if Image is None:
        return None
    video_path = meta.get("video_path")
    if not video_path:
        return None
    try:
        frame_idx = int(meta.get("frame", meta.get("frame_idx", -1)))
    except (TypeError, ValueError):
        return None
    bbox = meta.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    frame = _get_frame_rgb(str(video_path), frame_idx)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    return Image.fromarray(crop)


def build_pixel_matrix(
    entries: Sequence[Dict[str, Any]],
    pixel_size: int = 0,
) -> Tuple[np.ndarray, List[str], List[bool], int]:
    vecs: List[np.ndarray] = []
    labels: List[str] = []
    relevant: List[bool] = []
    n_failed = 0
    for e in entries:
        img = crop_tile_from_meta(e.get("meta") or {})
        if img is None:
            n_failed += 1
            continue
        if pixel_size and pixel_size > 0:
            img = img.resize((pixel_size, pixel_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        vecs.append(arr)
        labels.append(e["label"])
        relevant.append(bool(e.get("relevant", False)))
    if not vecs:
        return np.zeros((0, 1), dtype=np.float32), [], [], n_failed
    # Pad/truncate to common length
    dim = max(v.shape[0] for v in vecs)
    X = np.zeros((len(vecs), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        X[i, : v.shape[0]] = v
    return X, labels, relevant, n_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE of labeled embeddings from TileAnnotationDB or label-store pickle",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite annotation DB (preferred). Default: drone_tile_annotations.db if present.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional pickle label store (used if --db missing/empty, or explicitly set)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "tsne",
        help="Output directory for plots and CSV",
    )
    p.add_argument(
        "--features",
        choices=("dino", "pixels"),
        default="dino",
        help="Feature source: stored DINO embeddings or re-cropped raw pixels",
    )
    p.add_argument(
        "--pixel-size",
        type=int,
        default=64,
        help="Resize crops to N×N before flatten (pixels mode; 0 = native)",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=200,
        help="Cap samples per class (0 = no cap)",
    )
    p.add_argument(
        "--no-balance",
        action="store_true",
        help="Use all entries (ignore --max-per-class)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    default_db = ROOT / "drone_tile_annotations.db"
    default_pkl = ROOT / "drone_ared_labels.pkl"

    entries: List[Dict[str, Any]] = []
    source_desc = ""

    db_path = args.db
    if db_path is None and default_db.exists() and args.labels is None:
        db_path = default_db

    if db_path is not None:
        if not db_path.exists():
            raise SystemExit(f"Annotation DB not found: {db_path}")
        require_emb = args.features == "dino"
        entries = load_from_sqlite(db_path, require_embedding=require_emb)
        source_desc = str(db_path)
        if args.features == "dino":
            entries = [e for e in entries if e.get("embedding") is not None]
        print(f"[tsne] Loaded {len(entries)} entries from SQLite {db_path}")

    if (not entries) and (args.labels is not None or default_pkl.exists()):
        pkl_path = args.labels if args.labels is not None else default_pkl
        if not pkl_path.exists():
            raise SystemExit(f"Label store not found: {pkl_path}")
        entries = load_from_pkl(pkl_path)
        source_desc = str(pkl_path)
        print(f"[tsne] Loaded {len(entries)} entries from pickle {pkl_path}")

    if not entries:
        raise SystemExit(
            "No labeled entries found. Provide --db path to an annotation DB with embeddings, "
            "or --labels path to a pickle store."
        )

    counts = Counter(e["label"] for e in entries)
    for lbl, c in counts.most_common():
        print(f"  {lbl}: {c}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    max_per = None if args.no_balance else args.max_per_class
    subset = balanced_subsample(entries, max_per, args.seed)
    print(f"[tsne] Class plot using {len(subset)} points (max_per_class={max_per})")

    if args.features == "pixels":
        if not HAS_CV2:
            raise SystemExit("--features pixels requires opencv-python")
        if Image is None:
            raise SystemExit("--features pixels requires Pillow")
        print(f"[tsne] Building pixel vectors (pixel_size={args.pixel_size or 'native'}) ...")
        try:
            X, labels, relevant, n_failed = build_pixel_matrix(subset, pixel_size=args.pixel_size)
        finally:
            close_video_caps()
        if n_failed:
            print(f"[tsne] Warning: failed to crop {n_failed} tiles (missing video/frame/bbox)")
        if len(labels) < 2:
            raise SystemExit("Not enough pixel vectors for t-SNE")
        dim = int(X.shape[1])
        print(f"[tsne] Pixel feature matrix: n={X.shape[0]} dim={dim}")
        out_png = args.out_dir / "tsne_classes_pixels.png"
        out_csv = args.out_dir / "tsne_classes_pixels.csv"
        size_note = f"{args.pixel_size}×{args.pixel_size}" if args.pixel_size > 0 else "native"
        title = (
            f"Raw pixel tiles (t-SNE, n={len(labels)}, dim={dim}, "
            f"size={size_note}, seed={args.seed})\n{source_desc}"
        )
    else:
        usable = [e for e in subset if e.get("embedding") is not None]
        if len(usable) < 2:
            raise SystemExit(
                "Not enough embeddings for t-SNE. Re-run A/RED with annotations that "
                "store embedding blobs, or convert with scripts that re-extract DINO features."
            )
        # Align embedding dims (take most common dim)
        dims = Counter(int(e["embedding"].shape[0]) for e in usable)
        best_dim = dims.most_common(1)[0][0]
        usable = [e for e in usable if int(e["embedding"].shape[0]) == best_dim]
        if len(usable) < 2:
            raise SystemExit(f"Not enough embeddings with common dim (saw dims={dict(dims)})")
        X = np.stack([e["embedding"] for e in usable]).astype(np.float32)
        labels = [e["label"] for e in usable]
        relevant = [bool(e.get("relevant", False)) for e in usable]
        out_png = args.out_dir / "tsne_classes.png"
        out_csv = args.out_dir / "tsne_classes.csv"
        title = (
            f"DINO label embeddings (t-SNE, n={len(labels)}, dim={best_dim}, "
            f"seed={args.seed})\n{source_desc}"
        )

    stats = cohesion_stats(X, labels)
    print_cohesion(stats)

    print("[tsne] Running t-SNE ...")
    coords = run_tsne(X, seed=args.seed)
    plot_class_tsne(coords, labels, relevant, out_path=out_png, title=title)
    write_coords_csv(out_csv, coords, labels, extra_cols={"relevant": relevant})
    print("[tsne] Done.")


if __name__ == "__main__":
    main()
