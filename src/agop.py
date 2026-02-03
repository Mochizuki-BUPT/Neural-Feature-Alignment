# -*- coding: utf-8 -*-
"""LLM Feature Extraction via AGOP (Average Gradient Outer Product)"""

import os
import gc
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import joblib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class AGOPComputer:
    """AGOP feature extractor: A_l = (1/n) Σ ∇_{h_l} f(x) (∇_{h_l} f(x))^T"""

    def __init__(self, model_path: str, device: Optional[str] = None,
                 dtype: torch.dtype = torch.float16):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self._init_model()
        self.num_layers = self._get_num_layers()
        self.layer_inputs: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=self.dtype,
            device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        model.eval()
        return tokenizer, model

    def _get_num_layers(self) -> int:
        if hasattr(self.model, "config"):
            return self.model.config.num_hidden_layers
        return len(self.model.model.layers)

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise ValueError("Unsupported model architecture")

    def _get_final_norm(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        return None

    def _register_hooks(self):
        self.layer_inputs = {}
        self.hooks = []
        for i, layer in enumerate(self._get_layers()):
            def get_hook(idx):
                def hook(module, inp):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        self.layer_inputs[idx] = inp[0].detach()
                return hook
            self.hooks.append(layer.register_forward_pre_hook(get_hook(i)))

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.layer_inputs = {}

    def compute_gradients(self, input_ids: torch.Tensor,
                          layer_idx: int) -> Optional[torch.Tensor]:
        """Compute ∇_{h_l} L for each position. Returns [1, seq_len-1, hidden_dim]."""
        seq_len = input_ids.size(1)
        input_ids = input_ids.to(self.device)

        try:
            # Capture layer input via hook
            self._register_hooks()
            with torch.no_grad():
                pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
                self.model(input_ids, position_ids=pos_ids)
            if layer_idx not in self.layer_inputs:
                return None
            layer_input = self.layer_inputs[layer_idx]
            self._remove_hooks()

            # Prepare for gradient computation
            layer_input_grad = layer_input.clone().detach().requires_grad_(True)
            all_layers = self._get_layers()
            subsequent_layers = list(all_layers[layer_idx + 1:])
            final_norm = self._get_final_norm()

            gradients = []
            for pos in tqdm(range(seq_len - 1), desc=f"Layer {layer_idx} gradients"):
                self.model.zero_grad()
                if layer_input_grad.grad is not None:
                    layer_input_grad.grad.detach_()
                    layer_input_grad.grad.zero_()

                h = layer_input_grad
                for sub_layer in subsequent_layers:
                    h = sub_layer(h, position_ids=pos_ids)[0]
                if final_norm:
                    h = final_norm(h)
                logits = self.model.lm_head(h)

                loss = torch.nn.functional.cross_entropy(
                    logits[:, pos, :], input_ids[:, pos + 1])
                loss.backward(retain_graph=True)

                if layer_input_grad.grad is not None:
                    gradients.append(layer_input_grad.grad[:, pos:pos+1, :].detach().clone())
                else:
                    gradients.append(torch.zeros(1, 1, layer_input_grad.shape[-1],
                                                 device=self.device,
                                                 dtype=layer_input_grad.dtype))

                del h, logits, loss
                if (pos + 1) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return torch.cat(gradients, dim=1) if gradients else None

        finally:
            self._remove_hooks()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_word_states(self, input_ids: torch.Tensor, words: List[str],
                            layer_idx: int) -> Optional[np.ndarray]:
        """Extract word-level hidden states (first subword token per word)."""
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.size(1)
        text = ' '.join(words)

        with torch.no_grad():
            pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            outputs = self.model(input_ids, position_ids=pos_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx + 1].detach().cpu()

        # Build word-to-token mapping via offset mapping.
        # NOTE: offset_mapping behavior varies across tokenizers (e.g. BPE vs SentencePiece);
        # this alignment assumes single-space word boundaries and may need adjustment for others.
        enc = self.tokenizer(text, return_offsets_mapping=True,
                             add_special_tokens=False, truncation=False)
        offsets = enc["offset_mapping"]

        word_spans = []
        pos = 0
        for i, w in enumerate(words):
            start = pos + (1 if i > 0 else 0)
            word_spans.append((start, start + len(w), i))
            pos = start + len(w)

        word_first = [-1] * len(words)
        for tok_idx, (ts, te) in enumerate(offsets):
            if ts == te == 0:
                continue
            for ws, we, wi in word_spans:
                if ws <= ts < we or ts == ws:
                    if word_first[wi] == -1:
                        word_first[wi] = tok_idx
                    break

        states = []
        hdim = hidden.shape[-1]
        for wi in range(len(words)):
            fp = word_first[wi]
            if 0 <= fp < hidden.shape[1]:
                states.append(hidden[0, fp, :].float().numpy())
            else:
                states.append(np.zeros(hdim, dtype=np.float32))

        return np.array(states)

    def compute_agop(self, gradients: torch.Tensor) -> Optional[np.ndarray]:
        """Compute AGOP matrix: A = (1/n) Σ ∇h ∇h^T"""
        if gradients is None:
            return None
        grads = gradients.squeeze(0).float().to(self.device)
        agop = torch.einsum('ni,nj->ij', grads, grads) / grads.shape[0]
        return agop.cpu().numpy()


def decompose_matrix(matrix: np.ndarray, ensure_positive: bool = True):
    """Eigendecomposition with GPU acceleration if available."""
    matrix = 0.5 * (matrix + matrix.T)
    if torch.cuda.is_available():
        try:
            m = torch.tensor(matrix, dtype=torch.float32, device='cuda')
            eigenvalues, eigenvectors = torch.linalg.eigh(m)
            if ensure_positive:
                eigenvalues = torch.clamp(eigenvalues, min=0.0)
            idx = torch.argsort(eigenvalues, descending=True)
            return eigenvalues[idx].cpu().numpy(), eigenvectors[:, idx].cpu().numpy()
        except:
            pass
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if ensure_positive:
        eigenvalues = np.maximum(eigenvalues, 0)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


def compute_projections(hidden_states: np.ndarray, eigenvectors: np.ndarray,
                        k: int = 10) -> np.ndarray:
    """Project hidden states onto top-k eigenvectors: s = H @ v"""
    k = min(k, eigenvectors.shape[1])
    return hidden_states.astype(np.float32) @ eigenvectors[:, :k].astype(np.float32)


def save_results(output_dir: str, layer_idx: int, **kwargs):
    """Save computation results."""
    os.makedirs(output_dir, exist_ok=True)
    for name, data in kwargs.items():
        if data is not None:
            if isinstance(data, np.ndarray) and name == 'projections':
                np.save(os.path.join(output_dir, f"layer_{layer_idx}_{name}.npy"), data)
            else:
                joblib.dump(data, os.path.join(output_dir, f"layer_{layer_idx}_{name}.joblib"))


def load_results(input_dir: str, layer_idx: int) -> Dict:
    """Load computation results."""
    results = {}
    for name in ['agop', 'eigenvalues', 'eigenvectors', 'word_states']:
        path = os.path.join(input_dir, f"layer_{layer_idx}_{name}.joblib")
        if os.path.exists(path):
            results[name] = joblib.load(path)
    proj_path = os.path.join(input_dir, f"layer_{layer_idx}_projections.npy")
    if os.path.exists(proj_path):
        results['projections'] = np.load(proj_path)
    return results
