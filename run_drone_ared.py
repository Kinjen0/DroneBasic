#!/usr/bin/env python3
"""
Entry point for the Drone A/RED application.

Usage:
    python run_drone_ared.py

This launches the Tkinter GUI which is the primary way to control the system
(loading videos, tuning parameters, saving/loading caches and ARED models, etc.).

All heavy logic lives in the drone_ared package.
"""

import tkinter as tk
from drone_ared.config import PipelineConfig
from drone_ared.gui import MainWindow


def main():
    root = tk.Tk()
    # Optional: load a saved config if you want persistence between launches
    # try:
    #     cfg = PipelineConfig.load("drone_config.json")
    # except Exception:
    cfg = PipelineConfig.default()

    app = MainWindow(root, initial_config=cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
