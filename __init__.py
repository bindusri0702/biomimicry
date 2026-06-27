"""Biomimicry spiral assistant — fully automated, LLM-driven Define -> Biologize -> Discover -> Abstract."""
from .orchestrator import build_spiral
from .stages.abstract import build_abstract_subgraph
from .stages.biologize import build_biologize_subgraph
from .stages.define import build_define_subgraph
from .stages.discover import build_discover_subgraph
from .state import SpiralState

__all__ = ["build_spiral", "build_define_subgraph", "build_biologize_subgraph",
           "build_discover_subgraph", "build_abstract_subgraph", "SpiralState"]
