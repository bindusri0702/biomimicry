"""Spiral Controller — wires the four stages into one LangGraph state machine.

The pipeline is fully automated (no human gates / interrupts), so the controller is
a plain linear forward chain over the shared `SpiralState`:

    define -> biologize -> discover -> abstract -> END

A checkpointer is optional (only useful for online crash-resume); none is required
because nothing interrupts.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .stages.abstract import build_abstract_subgraph
from .stages.biologize import build_biologize_subgraph
from .stages.define import build_define_subgraph
from .stages.discover import build_discover_subgraph
from .state import SpiralState


def build_spiral(checkpointer=None):
    """Build and compile the full spiral controller graph (linear forward chain)."""
    g = StateGraph(SpiralState)
    g.add_node("define", build_define_subgraph())
    g.add_node("biologize", build_biologize_subgraph())
    g.add_node("discover", build_discover_subgraph())
    g.add_node("abstract", build_abstract_subgraph())

    g.add_edge(START, "define")
    g.add_edge("define", "biologize")
    g.add_edge("biologize", "discover")
    g.add_edge("discover", "abstract")
    g.add_edge("abstract", END)

    return g.compile(checkpointer=checkpointer)
