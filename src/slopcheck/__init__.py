"""slopcheck: lint prose against published AI-writing tells."""

from .checks import Hit, run_checks
from .metrics import Metrics, compute
from .report import Config, Result, analyze, render_json, render_text
from .rules import RULES, Rule
from .text import Document

__version__ = "0.1.0"
__all__ = [
    "RULES",
    "Config",
    "Document",
    "Hit",
    "Metrics",
    "Result",
    "Rule",
    "analyze",
    "compute",
    "render_json",
    "render_text",
    "run_checks",
]
