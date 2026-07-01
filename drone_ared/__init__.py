"""
drone_ared - A/RED (Anomalous/Relevant Event Detection) for drone video anomaly detection.

Uses tiling + DINOv* embeddings + A_REDIN for streaming discovery of rare/relevant events in drone footage.

Full OOP design with extensive comments for future expansion.

Key components:
- Tiling strategies
- Pluggable feature extractors (DINOv2 / DINOv3 via HF)
- Persistent label cache to minimize human labeling
- Interactive resizable Tkinter GUI for high-volume tile labeling
- ARED adapter that works with the original A_RED implementation (zero or minimal edits)
- Optional ARED model state save/restore for "warm start" on similar data
- GUI-driven control of the entire pipeline

See README.md and individual module docs.
"""

__version__ = "0.1.0"
