"""slopcheck: lint prose against published AI-writing tells."""

from . import scorer, stylometry, voice
from .checks import Hit, run_checks
from .metrics import Metrics, compute
from .report import Config, Result, analyze, render_json, render_text
from .rules import RULES, Rule
from .stylometry import Stylometry
from .text import Document
from .voice import Voiceprint

__version__ = "0.2.0"
__all__ = [
    "RULES", "Config", "Document", "Hit", "Metrics", "Result", "Rule",
    "Stylometry", "Voiceprint", "analyze", "compute", "render_json",
    "render_text", "run_checks", "scorer", "stylometry", "voice",
]
