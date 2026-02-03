# -*- coding: utf-8 -*-
"""NNFPP (Neural Network Feature Propagation Path) Graph Construction"""

import os
import math
import pickle
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation between two vectors."""
    if x is None or y is None:
        return 0.0
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if np.isnan(x).any() or np.isnan(y).any():
                return 0.0
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                return 0.0
            r, _ = spearmanr(x, y)
            return 0.0 if np.isnan(r) else r
        except:
            return 0.0


def build_nnfpp_graph(projections: Dict[int, np.ndarray], 
                      eigenvalues: Dict[int, np.ndarray],
                      top_k: int = 10) -> nx.DiGraph:
    """Build NNFPP graph with Spearman-weighted edges."""
    G = nx.DiGraph()
    layers = sorted(projections.keys())
    
    if len(layers) < 2:
        return G
    
    # Add nodes
    for layer in layers:
        proj = projections[layer]
        eigs = eigenvalues.get(layer, np.ones(top_k))
        k = min(top_k, proj.shape[1], len(eigs))
        for i in range(k):
            G.add_node((layer, i), layer=layer, feature_index=i,
                      eigenvalue=float(eigs[i]) if i < len(eigs) else 0.0)
    
    # Add edges between adjacent layers
    for i, layer in enumerate(layers[:-1]):
        next_layer = layers[i + 1]
        proj_curr = projections[layer]
        proj_next = projections[next_layer]
        k_curr = min(top_k, proj_curr.shape[1])
        k_next = min(top_k, proj_next.shape[1])
        
        for fi in range(k_curr):
            for fj in range(k_next):
                corr = spearman_corr(proj_curr[:, fi], proj_next[:, fj])
                if abs(corr) > 1e-9:
                    G.add_edge((layer, fi), (next_layer, fj), 
                              corr=corr, type='inter')
    return G


def find_max_strength_path(G: nx.DiGraph, epsilon: float = 1e-12
                           ) -> Tuple[Optional[List], float]:
    """Find Maximum Strength (MS) path: argmax Σ log(|ρ| + ε)"""
    if not G or G.number_of_nodes() == 0:
        return None, float('-inf')
    
    nodes_by_layer = defaultdict(list)
    min_layer, max_layer = float('inf'), float('-inf')
    
    for node, data in G.nodes(data=True):
        layer = data.get('layer')
        if layer is not None:
            nodes_by_layer[layer].append(node)
            min_layer = min(min_layer, layer)
            max_layer = max(max_layer, layer)
    
    if min_layer == float('inf'):
        return None, float('-inf')
    
    # Sort by eigenvalue
    for layer in nodes_by_layer:
        nodes_by_layer[layer].sort(
            key=lambda n: G.nodes[n].get('eigenvalue', 0), reverse=True)
    
    # Dynamic programming
    strength = defaultdict(lambda: float('-inf'))
    pred = {}
    
    for node in nodes_by_layer[min_layer]:
        strength[node] = 0.0
    
    for layer in sorted(l for l in nodes_by_layer if min_layer <= l < max_layer):
        next_layer = layer + 1
        if next_layer not in nodes_by_layer:
            continue
        for u in nodes_by_layer[layer]:
            if strength[u] <= float('-inf') + 1:
                continue
            for v in G.successors(u):
                if G.nodes[v].get('layer') != next_layer:
                    continue
                edge = G.edges[u, v]
                if edge.get('type') != 'inter':
                    continue
                corr = edge.get('corr', 0)
                if abs(corr) < 1e-9:
                    continue
                w = math.log(abs(corr) + epsilon)
                new_s = strength[u] + w
                if new_s > strength[v]:
                    strength[v] = new_s
                    pred[v] = u
    
    # Find best end node
    best_s, end = float('-inf'), None
    for node in nodes_by_layer.get(max_layer, []):
        if strength[node] > best_s:
            best_s = strength[node]
            end = node
    
    if end is None or best_s <= float('-inf') + 1:
        return None, float('-inf')
    
    # Backtrack
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        if G.nodes[curr].get('layer') == min_layer:
            break
        curr = pred.get(curr)
    
    if not path or G.nodes[path[-1]].get('layer') != min_layer:
        return None, float('-inf')
    
    return path[::-1], best_s


def save_graph(G: nx.DiGraph, path: str):
    """Save graph to pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str) -> Optional[nx.DiGraph]:
    """Load graph from pickle file."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_path(path: List, output_dir: str, method: str = "ms"):
    """Save path to pickle and JSON."""
    if not path:
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"path_{method}.pkl"), 'wb') as f:
        pickle.dump(path, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, f"path_{method}.json"), 'w') as f:
        json.dump([list(n) for n in path], f)
