# -*- coding: utf-8 -*-
"""Neural Feature Alignment (NFA) Core Library"""

from .agop import (
    AGOPComputer, decompose_matrix, compute_projections
)
from .nnfpp import build_nnfpp_graph, find_max_strength_path
from .eeg_features import extract_eeg_features, get_feature_names
from .alignment import compute_alignment, compute_alignment_for_article

__version__ = "1.0.0"
__author__ = "Mochizuki-BUPT"
