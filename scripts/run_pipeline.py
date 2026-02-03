#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFA Pipeline Runner

Orchestrates the full AGOP → NNFPP → EEG → Alignment pipeline.

Usage:
    python scripts/run_pipeline.py --config config/default.yaml --steps all
    python scripts/run_pipeline.py --steps agop --model Qwen/Qwen2.5-7B --articles 0 1
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pickle

import yaml
import numpy as np
import torch
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agop import (AGOPComputer, decompose_matrix, compute_projections,
                  save_results as save_agop_results, load_results as load_agop_results)
from nnfpp import build_nnfpp_graph, find_max_strength_path, save_graph, save_path
from eeg_features import extract_eeg_features
from alignment import compute_alignment_for_article, save_results as save_alignment_results


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_agop(config, model_path, articles, data_root, results_dir, logger):
    """Step 1: Compute AGOP matrices, eigendecomposition, and projections."""
    top_k = config['feature_extraction']['top_k']
    computer = AGOPComputer(model_path)

    for article_id in articles:
        article_dir = os.path.join(data_root, "articles", f"article_{article_id}")
        words_path = os.path.join(article_dir, f"article_{article_id}.pkl")
        if not os.path.exists(words_path):
            logger.warning(f"Article {article_id} not found at {words_path}, skipping")
            continue

        with open(words_path, 'rb') as f:
            words = pickle.load(f)
        text = ' '.join(words)

        inputs = computer.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"]

        out_dir = os.path.join(results_dir, "agop", model_path.split("/")[-1],
                               f"article_{article_id}")

        for layer_idx in range(computer.num_layers):
            logger.info(f"Article {article_id}, Layer {layer_idx}/{computer.num_layers-1}")

            # Gradients and AGOP
            grads = computer.compute_gradients(input_ids, layer_idx)
            agop_matrix = computer.compute_agop(grads)
            if agop_matrix is None:
                continue

            # Eigendecomposition
            eigenvalues, eigenvectors = decompose_matrix(agop_matrix)

            # Word-level hidden states and projections
            word_states = computer.extract_word_states(input_ids, words, layer_idx)
            projections = compute_projections(word_states, eigenvectors, k=top_k)

            save_agop_results(out_dir, layer_idx,
                              agop=agop_matrix, eigenvalues=eigenvalues,
                              eigenvectors=eigenvectors, word_states=word_states,
                              projections=projections)

            del grads, agop_matrix
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f"AGOP complete for article {article_id}")


def run_nnfpp(config, model_path, articles, results_dir, logger):
    """Step 2: Build NNFPP graph and find maximum strength path."""
    top_k = config['nnfpp']['top_k_features']
    epsilon = config['nnfpp']['epsilon']
    model_name = model_path.split("/")[-1]

    for article_id in articles:
        agop_dir = os.path.join(results_dir, "agop", model_name, f"article_{article_id}")
        if not os.path.exists(agop_dir):
            logger.warning(f"No AGOP results for article {article_id}, skipping NNFPP")
            continue

        projections, eigenvalues = {}, {}
        for f in sorted(os.listdir(agop_dir)):
            if f.endswith("_projections.npy"):
                layer_idx = int(f.split("_")[1])
                projections[layer_idx] = np.load(os.path.join(agop_dir, f))
            elif f.endswith("_eigenvalues.joblib"):
                layer_idx = int(f.split("_")[1])
                eigenvalues[layer_idx] = joblib.load(os.path.join(agop_dir, f))

        if not projections:
            continue

        G = build_nnfpp_graph(projections, eigenvalues, top_k=top_k)
        path, strength = find_max_strength_path(G, epsilon=epsilon)

        out_dir = os.path.join(results_dir, "nnfpp", model_name, f"article_{article_id}")
        save_graph(G, os.path.join(out_dir, "graph.pkl"))
        if path:
            save_path(path, out_dir, method="ms")
            logger.info(f"Article {article_id}: path length={len(path)}, strength={strength:.4f}")


def run_eeg(config, articles, data_root, results_dir, logger):
    """Step 3: Extract EEG features."""
    tmin, tmax = config['eeg']['tmin'], config['eeg']['tmax']

    for article_id in articles:
        epochs_path = os.path.join(data_root, "eeg", f"article_{article_id}_epo.fif")
        if not os.path.exists(epochs_path):
            logger.warning(f"EEG epochs not found: {epochs_path}, skipping")
            continue
        out_dir = os.path.join(results_dir, "eeg", f"article_{article_id}")
        ok = extract_eeg_features(epochs_path, out_dir, tmin=tmin, tmax=tmax)
        logger.info(f"EEG extraction article {article_id}: {'ok' if ok else 'failed'}")


def run_alignment(config, model_path, articles, results_dir, logger):
    """Step 4: Compute cross-modal alignment."""
    top_k = config['feature_extraction']['top_k']
    min_samples = config['alignment'].get('min_samples', 30)
    model_name = model_path.split("/")[-1]
    all_results = []

    for article_id in articles:
        proj_dir = os.path.join(results_dir, "agop", model_name, f"article_{article_id}")
        path_file = os.path.join(results_dir, "nnfpp", model_name,
                                 f"article_{article_id}", "path_ms.pkl")
        eeg_file = os.path.join(results_dir, "eeg", f"article_{article_id}",
                                "region_features.csv")

        results = compute_alignment_for_article(
            article_id, model_name, proj_dir, path_file, eeg_file,
            top_k=top_k, min_samples=min_samples)
        all_results.extend(results)
        logger.info(f"Alignment article {article_id}: {len(results)} pairs")

    if all_results:
        out_path = os.path.join(results_dir, "alignment", f"{model_name}_alignment.csv")
        save_alignment_results(all_results, out_path)
        logger.info(f"Saved {len(all_results)} alignment results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="NFA Pipeline")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--steps", nargs='+', default=['all'],
                        choices=['agop', 'nnfpp', 'eeg', 'alignment', 'all'])
    parser.add_argument("--model", type=str)
    parser.add_argument("--articles", nargs='+', type=int)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path))

    logging.basicConfig(level=getattr(logging, config.get('logging', {}).get('level', 'INFO')))
    logger = logging.getLogger(__name__)

    steps = args.steps if 'all' not in args.steps else ['agop', 'nnfpp', 'eeg', 'alignment']
    model_path = args.model or config['model']['default_path']
    articles = args.articles or list(range(config['alignment']['num_articles']))
    data_root = config['paths']['data_root']
    results_dir = config['paths']['results_dir']

    logger.info(f"NFA pipeline: steps={steps}, model={model_path}, articles={articles}")

    if 'agop' in steps:
        run_agop(config, model_path, articles, data_root, results_dir, logger)
    if 'nnfpp' in steps:
        run_nnfpp(config, model_path, articles, results_dir, logger)
    if 'eeg' in steps:
        run_eeg(config, articles, data_root, results_dir, logger)
    if 'alignment' in steps:
        run_alignment(config, model_path, articles, results_dir, logger)


if __name__ == "__main__":
    main()
