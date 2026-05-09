---
title: >-
  [Paper Note] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features
description: >-
  [NeurIPS 2025][Causal Inference][positional encoding] CAPE learns the causal DAG structure among features from tabular data, embeds it into hyperbolic space to generate causality-aware rotary positional encodings (RoPE), enabling Transformers to process non-sequential yet causally structured feature data, with significant performance gains on downstream multi-omics tasks.
tags:
  - NeurIPS 2025
  - Causal Inference
  - positional encoding
  - causal structure learning
  - hyperbolic embedding
  - rotary position encoding
  - multi-omics
date: 2026-05-08
content_hash: daad3f6930a6ae64
---

# Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features

**Conference**: NeurIPS 2025
**arXiv**: [2509.16629](https://arxiv.org/abs/2509.16629)
**Code**: [https://github.com/Catchxu/CAPE](https://github.com/Catchxu/CAPE)
**Area**: Causal Inference / Transformer
**Keywords**: positional encoding, causal structure learning, hyperbolic embedding, rotary position encoding, multi-omics

## TL;DR
CAPE learns the causal DAG structure among features from tabular data, embeds it into hyperbolic space to generate causality-aware rotary positional encodings (RoPE), enabling Transformers to process non-sequential yet causally structured feature data, with significant performance gains on downstream multi-omics tasks.

## Background & Motivation
**Background**: Positional encodings in Transformers (e.g., sinusoidal, RoPE) assume data with a natural sequential order (word order, spatial arrangement of image patches), achieving great success in NLP and CV.

**Limitations of Prior Work**: Many real-world datasets (e.g., gene expression, proteomics, economic indicators) have features without a predefined sequential order, yet exhibit complex causal relationships. Existing positional encoding methods cannot capture such non-sequential causal structures.

**Key Challenge**: Existing methods either ignore inter-feature causal relationships (e.g., sorting by expression level) or use static pre-trained embeddings as surrogate positional encodings, neither of which genuinely exploits causal structural information among features.

**Goal**: How to generate positional encodings for non-sequential yet causally related features, such that the self-attention mechanism of Transformers becomes causality-aware?

**Key Insight**: Inspired by special relativity—causal connections correspond to relative positions in hyperbolic spacetime—the causal graph is embedded into hyperbolic space, naturally preserving two key properties: causal strength and causal specificity.

**Core Idea**: Learn the causal DAG among features → embed into hyperbolic space → convert to rotary positional encodings, so that attention scores decay with causal distance.

## Method

### Overall Architecture
CAPE is a three-step framework: the input is an $N \times M$ tabular dataset $\bm{X}$ ($N$ observations, $M$ non-sequential features), and the output is a causality-aware rotary positional encoding $\bm{\varphi}_{v_j}$ for each feature $v_j$. The three steps are: (1) causal structure learning → weighted DAG; (2) hyperbolic space embedding → preserving causal properties; (3) conversion to rotary form → injection into Transformer.

### Key Designs

1. **Causal Structure Learning (Step I)**:

    - **Function**: Learn the causal structure among features from observed data $\bm{X}$, represented as a weighted adjacency matrix $\bm{A}$.
    - **Mechanism**: A nonlinear structural equation model (SEM) is formulated in a VAE framework—encoder $\bm{Z} = f(\bm{X})(\bm{I} - \bm{A})$, decoder $\bm{X} = f^{-1}(\bm{Z}(\bm{I}-\bm{A})^{-1})$. Joint optimization via ELBO + sparsity regularization $\|\bm{A}\|_1$ + acyclicity constraint $\text{tr}(e^{\bm{A} \odot \bm{A}}) - M = 0$.
    - **Design Motivation**: Nonlinear SEM captures complex causal relationships; continuous optimization of the acyclicity constraint avoids combinatorial search; threshold $\tau$ prunes noisy edges.

2. **Hyperbolic Space Embedding (Step II)**:

    - **Function**: Embed the causal DAG into hyperbolic space (hyperboloid model), generating a $(d+1)$-dimensional embedding $\bm{p}_{v_j}$ for each node.
    - **Mechanism**: Embeddings are optimized via regularized graph contrastive learning. The contrastive loss $\mathcal{L}_{\text{con}}$ pulls causally connected nodes closer (positive samples are $k$-hop neighbors) and pushes unrelated nodes apart, weighted by $|\bm{A}_{mn}|$ (causal strength). The regularization term $\Omega = \bm{\pi}_{v_m} d_l(\bm{p}_{v_m}, \bm{p_o})$ uses PageRank scores $\pi$ to penalize causally generic nodes (high out-degree root nodes), forcing them toward the origin.
    - **Design Motivation**: Hyperbolic space is naturally suited for modeling tree-like/DAG structures. Two key properties are preserved—**causal strength** (proximity ↔ strong causal relationship) and **causal specificity** (distance from origin ↔ specificity of leaf nodes). Riemannian SGD is used for optimization on the manifold.

3. **Rotary Positional Encoding Conversion (Step III)**:

    - **Function**: Map hyperbolic embeddings to the Poincaré ball and convert them to rotary form for injection into the Transformer.
    - **Mechanism**: The diffeomorphism $f_d: \mathcal{H}^d \to \mathcal{B}^d$ first maps embeddings to the Poincaré ball to obtain $\bm{e}_{v_j}$; then $\bm{\varphi}_v = c \cdot \bm{e}_v$ (with $c=\pi/4$) serves as rotation angles to construct a block-diagonal rotation matrix $\bm{R}(\bm{\varphi}_v)$. Attention is computed as $\mathcal{A} = (\bm{q}_{v_m}^i)^\top \bm{R}(\bm{\varphi}_{v_n} - \bm{\varphi}_{v_m}) \bm{k}_{v_n}^i$.
    - **Design Motivation**: The spherical geometry of the Poincaré ball is more suitable for rotary encoding; the rotary form is compatible with linear self-attention, and the relative positional encoding depends only on the causal distance difference $\bm{\varphi}_{v_n} - \bm{\varphi}_{v_m}$.

### Loss & Training
- Step I: $\mathcal{L}_{\text{DAG}} = -\mathcal{L}_{\text{ELBO}} + \lambda_s \|\bm{A}\|_1 + \frac{\rho}{2}|h(\bm{A})|^2 + \alpha h(\bm{A})$, solved via augmented Lagrangian method.
- Step II: $\mathcal{L}_{\mathcal{H}} = \frac{1}{M} \sum_j \mathcal{L}_{\text{con}}(\bm{p}_{v_j}) + \lambda_g \Omega(\bm{p}_{v_j})$, optimized with Riemannian SGD.
- Step III: No additional trainable parameters; encoding is obtained via direct mapping.

## Key Experimental Results

### Main Results
The gene perturbation prediction (GPP) task is evaluated on single-cell scRNA-seq datasets using two Transformer backbones, scBERT and scGPT:

| Model | Positional Encoding | Single-gene Perturbation MSE | Double-gene Perturbation MSE |
|-------|--------------------|-----------------------------|------------------------------|
| scBERT | Static absolute PE (default) | 0.224 | 0.230 |
| scBERT | Learnable relative PE | 0.219 (−0.005) | 0.215 (−0.015) |
| scBERT | **CAPE** | **0.193 (−0.031)** | **0.189 (−0.041)** |
| scGPT | Learnable absolute PE (default) | 0.202 | 0.201 |
| scGPT | Learnable relative PE | 0.195 (−0.007) | 0.204 (+0.003) |
| scGPT | **CAPE** | **0.182 (−0.020)** | **0.176 (−0.025)** |

CAPE reduces MSE by an average of 11.1%, whereas causality-agnostic relative PE reduces it by only 2.7%.

### Ablation Study
Using scGPT as the backbone:

| Configuration | Single-gene Perturbation MSE | Double-gene Perturbation MSE |
|--------------|------------------------------|------------------------------|
| CAPE (full) | 0.182 (±0.005) | 0.176 (±0.008) |
| CAPE-null (no PE) | 0.234 (±0.014) | 0.238 (±0.017) |
| CAPE-w/o-CSL (no causal learning) | 0.209 (±0.010) | 0.213 (±0.011) |
| CAPE-w/o-hyperbolic (Euclidean substitute) | 0.192 (±0.008) | 0.196 (±0.008) |
| CAPE-w/o-rotary (additive PE) | 0.201 (±0.009) | 0.208 (±0.010) |

### Key Findings
- Causal structure learning (CSL) is the most critical component; removing it increases double-gene MSE from 0.176 to 0.213 (+21%).
- The rotary form is also important, with a notable performance gap over additive PE.
- The contribution of hyperbolic space modeling is moderate but consistently positive, indicating that curvature-aware optimization does help reflect causal graph structure more faithfully.
- Theoretical properties are validated on synthetic data: attention decays with causal distance, decays with causal generality, and is robust to positional perturbations.

## Highlights & Insights
- The **unified framework of causal graph → hyperbolic space → RoPE** is remarkably elegant, organically integrating methods from three distinct fields: causal discovery, hyperbolic geometry, and rotary attention. The design of directly mapping causal distance to attention decay is highly natural.
- **Theoretical analysis is rigorous**: three key properties are proven (causal distance decay, causal generality decay, and robustness), providing mathematical guarantees for the method's effectiveness.
- **Transferable design paradigm**: any Transformer application involving non-sequential features (e.g., feature interactions in recommender systems, knowledge graph nodes, atoms in molecular graphs) can draw inspiration from this "structure → hyperbolic embedding → rotary encoding" paradigm.

## Limitations & Future Work
- The causal structure learning component assumes acyclicity (DAG), making it inapplicable to causal systems with feedback loops.
- Validation is currently limited to biological omics data; evaluation on other non-sequential causal domains such as economics and social sciences is absent.
- The accuracy of causal learning is sensitive to data volume and noise; DAG learning may be unreliable in small-sample, high-dimensional settings.
- Computational complexity: the three-stage training pipeline of causal learning + hyperbolic embedding + Transformer may be slow, though the appendix indicates the complexity analysis shows it remains acceptable.

## Related Work & Insights
- **vs. RoPE / absolute PE and other standard methods**: These methods assume a predefined order; CAPE extends positional encoding to unordered causally structured features.
- **vs. default PE in scGPT / scBERT**: These models use expression-level-sorted order or pre-trained representations as surrogate PE, fundamentally ignoring causal relationships.
- **vs. NOTEARS / DAG-GNN**: CAPE borrows from these causal discovery methods, but its innovation lies in converting the learned DAG into positional encodings rather than treating it as an end goal.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of causal graph → hyperbolic embedding → RoPE is highly novel; the cross-domain methodological fusion is particularly ingenious.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic and multi-omics experiments are comprehensive, though broader domain coverage would be beneficial.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, exposition is clear, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for Transformer modeling of non-sequential data, though the practical application scope is currently somewhat narrow.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)
- [\[NeurIPS 2025\] Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks](domain-adapted_granger_causality_for_real-time_cross-slice_attack_attribution_in.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] Demystifying Spectral Feature Learning for Instrumental Variable Regression](demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)

<!-- RELATED:END -->
