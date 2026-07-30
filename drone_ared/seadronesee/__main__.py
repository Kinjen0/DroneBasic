"""
Headless / CLI entry for SeaDronesSee auto A/RED.

Examples:
  python -m drone_ared.seadronesee --mode raw --tile 32 --split train --max-images 2
  python -m drone_ared.seadronesee --mode dino --tile 32 --max-tiles 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import SeaDronesSeeConfig
from .runner import SeaDronesSeeRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SeaDronesSee auto-labeled A/RED pipeline")
    p.add_argument("--root", default="SeaDroneSeeProcessedDataExport", help="Dataset export root")
    p.add_argument("--split", default="train", choices=["train", "val", "both"])
    p.add_argument("--mode", default="raw", choices=["raw", "dino"], help="Feature backend")
    p.add_argument("--tile", type=int, default=32, help="Tile width/height")
    p.add_argument("--stride", type=int, default=None, help="Stride (default = tile)")
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--db", default="seadronesee_tile_annotations.db")
    p.add_argument("--dino-model", default=None, help="HF model id when --mode dino")
    p.add_argument("--save-model", default=None, help="Optional path to save A_RED state at end")
    p.add_argument("--load-model", default=None, help="Optional path to load A_RED state before start")
    p.add_argument("--checkpoint-every", type=int, default=500)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SeaDronesSeeConfig.default()
    cfg.dataset_root = args.root
    cfg.split = args.split
    cfg.feature_mode = args.mode
    tw = max(1, int(args.tile))
    st = int(args.stride) if args.stride is not None else tw
    cfg.tiling.tile_width = tw
    cfg.tiling.tile_height = tw
    cfg.tiling.stride_x = st
    cfg.tiling.stride_y = st
    cfg.tiling.overlap_x = max(0, tw - st)
    cfg.tiling.overlap_y = max(0, tw - st)
    cfg.ared.kappa = float(args.kappa)
    cfg.max_images = args.max_images
    cfg.max_tiles = args.max_tiles
    cfg.tile_annotations_db = args.db
    cfg.metrics_logging.checkpoint_every = int(args.checkpoint_every)
    if args.dino_model:
        cfg.features.model_name = args.dino_model

    runner = SeaDronesSeeRunner(cfg)

    def _print_stats(s):
        print(
            f"[stats] status={s.get('status')} tiles={s.get('tiles_processed')} "
            f"queries={s.get('ared_queries')} img={s.get('current_video')} "
            f"pos={s.get('gt_positives')} {s.get('metrics_last_line') or ''}"
        )

    runner.on_stats = _print_stats

    if args.load_model:
        runner.load_ared_state(args.load_model)

    runner.start()
    try:
        while runner.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted — stopping…")
        runner.stop()

    # Wait until finished if still winding down
    t0 = time.time()
    while runner.is_running() and time.time() - t0 < 30:
        time.sleep(0.2)

    if args.save_model:
        try:
            runner.save_ared_state(args.save_model)
            print(f"Saved model → {args.save_model}")
        except Exception as e:
            print(f"Save failed: {e}")

    print("Final stats:", runner.stats)
    if runner.stats.get("metrics_run_dir"):
        print("Metrics run dir:", runner.stats["metrics_run_dir"])
    return 0 if runner.stats.get("status") in ("finished", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
