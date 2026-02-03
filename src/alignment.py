# -*- coding: utf-8 -*-
"""Cross-Modal Feature Alignment (Pearson correlation)"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def load_eeg_features(path: str) -> Optional[pd.DataFrame]:
    """Load EEG features to wide format."""
    try:
        df = pd.read_csv(path)
        loc_col = 'region' if 'region' in df.columns else 'location'
        df['feature_id'] = df[loc_col] + '_' + df['feature_name']
        return df.pivot_table(index='token_id', columns='feature_id', values='value').sort_index()
    except Exception as e:
        logger.error(f"Load EEG failed: {e}")
        return None


def load_llm_path(path: str) -> Optional[List[Tuple[int, int]]]:
    """Load LLM path from pickle."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
        return None


def load_projections(path: str) -> Optional[np.ndarray]:
    """Load projection vectors."""
    try:
        return np.load(path)
    except:
        return None


DEFAULT_MIN_SAMPLES = 30


def pearson_corr(x: np.ndarray, y: np.ndarray,
                 min_samples: int = DEFAULT_MIN_SAMPLES) -> Tuple[float, float]:
    """Compute Pearson correlation with a minimum sample size check."""
    try:
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x, y = x[mask], y[mask]
        if len(x) < min_samples or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return np.nan, np.nan
        r = pearsonr(x, y)
        return (r.statistic, r.pvalue) if hasattr(r, 'statistic') else (r[0], r[1])
    except:
        return np.nan, np.nan


def compute_alignment(llm_proj: Dict[int, np.ndarray], eeg_df: pd.DataFrame,
                      path_nodes: List[Tuple[int, int]],
                      min_samples: int = DEFAULT_MIN_SAMPLES) -> List[Dict]:
    """Compute FA scores between LLM and EEG features."""
    results = []
    eeg_tokens = set(eeg_df.index)

    for layer, feat_idx in path_nodes:
        if layer not in llm_proj:
            continue
        proj = llm_proj[layer]
        if feat_idx >= proj.shape[1]:
            continue

        llm_vec = proj[:, feat_idx]
        common = sorted(set(range(len(llm_vec))) & eeg_tokens)
        if len(common) < min_samples:
            continue

        for eeg_id in eeg_df.columns:
            eeg_vec = eeg_df.loc[common, eeg_id].values
            llm_sub = llm_vec[common]
            r, p = pearson_corr(llm_sub, eeg_vec, min_samples=min_samples)
            if not np.isnan(r):
                parts = eeg_id.split('_', 1)
                results.append({
                    'llm_layer': layer, 'llm_feature_idx': feat_idx,
                    'eeg_region': parts[0], 'eeg_feature_name': parts[1] if len(parts) > 1 else eeg_id,
                    'pearson_corr': r, 'pearson_p': p, 'fa_strength': abs(r)
                })
    return results


def compute_alignment_for_article(article_id: int, model_name: str,
                                  llm_proj_dir: str, llm_path_file: str,
                                  eeg_file: str, top_k: int = 10,
                                  min_samples: int = DEFAULT_MIN_SAMPLES) -> List[Dict]:
    """Compute alignment for one article."""
    eeg_df = load_eeg_features(eeg_file)
    path_nodes = load_llm_path(llm_path_file)
    if eeg_df is None or not path_nodes:
        return []
    
    layers = sorted(set(l for l, _ in path_nodes))
    llm_proj = {}
    for layer in layers:
        proj = load_projections(os.path.join(llm_proj_dir, f"layer_{layer}_top{top_k}_projections.npy"))
        if proj is not None:
            llm_proj[layer] = proj
    
    if not llm_proj:
        return []
    
    results = compute_alignment(llm_proj, eeg_df, path_nodes, min_samples=min_samples)
    for r in results:
        r['article_id'] = article_id
        r['model_name'] = model_name
    return results


def save_results(results: List[Dict], path: str):
    """Save alignment results to CSV."""
    if not results:
        return
    df = pd.DataFrame(results)
    cols = ['article_id', 'model_name', 'llm_layer', 'llm_feature_idx',
            'eeg_region', 'eeg_feature_name', 'pearson_corr', 'pearson_p', 'fa_strength']
    df = df[[c for c in cols if c in df.columns]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
