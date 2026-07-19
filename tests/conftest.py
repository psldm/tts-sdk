"""Pytest configuration: make the flat src/ modules importable."""

import os
import sys

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
sys.path.insert(0, SRC_DIR)
