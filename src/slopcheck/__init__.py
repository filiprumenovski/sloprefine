"""slopcheck: lint prose against published AI-writing tells."""

from . import agent, audit, reader, scorer, stylometry, voice
from .agent import Drift, Instruction, Review, drift, review, style_contract
from .checks import Hit, run_checks
from .metrics import Metrics, compute
from .report import Config, Result, analyze, render_json, render_text
from .rules import RULES, Rule
from .stylometry import Stylometry
from .text import Document
from .voice import Voiceprint

__version__ = "0.4.0"
__all__ = [
    "RULES",
    "Config",
    "Document",
    "Drift",
    "Hit",
    "Instruction",
    "Metrics",
    "Result",
    "Review",
    "Rule",
    "Stylometry",
    "Voiceprint",
    "agent",
    "analyze",
    "audit",
    "compute",
    "drift",
    "reader",
    "render_json",
    "render_text",
    "review",
    "run_checks",
    "scorer",
    "style_contract",
    "stylometry",
    "voice",
]
