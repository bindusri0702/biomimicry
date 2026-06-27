"""Offline-first retrieval for the Discover stage."""
from .base import Retriever, RetrievalHit, get_retriever, tokenize
from .corpus import StrategyDoc, load_corpus

__all__ = ["Retriever", "RetrievalHit", "get_retriever", "tokenize",
           "StrategyDoc", "load_corpus"]
