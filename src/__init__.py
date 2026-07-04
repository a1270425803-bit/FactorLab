"""FactorLab — AI-driven A-share quantitative factor mining system."""

import sys
from pathlib import Path

_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

__version__ = "0.1.0"
__author__ = "YippeeXu"
__license__ = "MIT"

__all__ = [
    "main",
    "engine",
    "pipeline",
    "batch_pipeline",
    "score",
    "backtest",
    "robustness_checker",
    "diversity_gate",
    "checker",
    "sandbox",
    "database",
    "config",
]
