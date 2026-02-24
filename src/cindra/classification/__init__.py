"""Provides classification algorithms for distinguishing cells from artifacts."""

from .classify import Classifier, classify

__all__ = [
    "Classifier",
    "classify",
]
