import os
from pathlib import Path

DEFAULT_PATH = Path(os.path.realpath(__file__)).parents[1]

__all__ = [
    "DEFAULT_PATH",
]
