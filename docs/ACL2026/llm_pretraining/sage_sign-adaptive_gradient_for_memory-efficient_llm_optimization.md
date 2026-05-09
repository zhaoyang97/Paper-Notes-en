---
title: >-
  [Paper Note] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization
description: >-
  [ACL 2026][LLM Pretraining][Optimizer] This paper proposes the SAGE optimizer, which addresses the "embedding layer dilemma" of lightweight optimizers by combining Lion-style sign update directions with an $O(d)$-memory adaptive damping scaling factor $\mathbf{H}_t$. SAGE achieves new state-of-the-art perplexity on Llama models (up to 1.3B parameters) with significantly reduced optimizer memory overhead.
tags:
  - ACL 2026
  - LLM Pretraining
  - Optimizer
  - Memory Efficiency
  - Embedding Layer
  - Sign-based Optimization
  - Adaptive Scaling
date: 2026-05-08
content_hash: dfe04899200e84cf
---

# SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization

**Conference**: ACL 2026
**arXiv**: [2604.07663](https://arxiv.org/abs/2604.07663)
**Code**: [GitHub](https://github.com/naubull2/SAGE-optimizer)
**Area**: LLM Pre-training
**Keywords**: Optimizer, Memory Efficiency, Embedding Layer, Sign-based Optimization, Adaptive Scaling

## TL;DR

This paper proposes the SAGE optimizer, which addresses the "embedding layer dilemma" of lightweight optimizers by combining Lion-style sign update directions with an $O(d)$-memory adaptive damping scaling factor $\mathbf{H}_t$. SAGE achieves new state-of-the-art perplexity on Llama models (up to 1.3B parameters) with significantly reduced optimizer memory overhead.

## Background & Motivation

**State of the Field**: AdamW is the standard optimizer for LLM pre-training, but its two full-size momentum states ($O(Vd)$) consume memory equivalent to twice the model size, making it a critical memory bottleneck. Lightweight alternatives such as Lion (single momentum) and SinkGD (stateless normalization) have made notable progress.

**Limitations of Prior Work**: Lightweight optimizers perform well on dense layers but fail on embedding layers. Embedding layer gradients exhibit sparsity and high variance due to Zipfian token frequency distributions, which stateless methods cannot handle effectively. As a result, approaches such as SinkGD resort to hybrid designs that fall back to AdamW for embedding layers, partially negating their memory savings.

**Root Cause**: The embedding layer is the largest contributor to optimizer state memory (with $V > 100{,}000$), yet it is precisely where lightweight optimizers fail. Achieving true memory efficiency requires solving the embedding layer problem.

**Paper Goals**: Design a lightweight optimizer that can successfully replace AdamW on embedding layers.

**Starting Point**: Lion's update magnitude is a static 1.0 (uniform across all dimensions), offering no control over high-variance dimensions. A bounded adaptive scaling factor that selectively damps high-variance dimensions could provide stability while preserving memory efficiency.

**Core Idea**: SAGE = Lion's sign direction + a novel $O(d)$ adaptive damping scaling factor $\mathbf{H}_t$. This factor is based on an EMA of gradient absolute values ($L_1$ norm), is theoretically bounded with $\|\mathbf{H}_t\|_\infty \leq 1.0$, applies stronger damping to high-variance dimensions, and degenerates to Lion's scaling of 1.0 for quiet dimensions.

## Method

### Overall Architecture

A hybrid optimizer is employed: SAGE handles embedding layers and 1D parameters (bias/norm) with $O(Vd) + O(d)$ state, while SinkGD handles dense 2D weights with $O(1)$ state. Compared to the SinkGD+AdamW hybrid, this reduces optimizer state memory for the embedding layer by approximately 50%.

### Key Designs

1. **$O(d)$ Adaptive Damping Scaling Factor $\mathbf{H}_t$**:

    - **Function**: Selectively damp the update magnitude of high-variance dimensions.
    - **Mechanism**: For the embedding layer, the mean absolute gradient value along each embedding dimension $j$ is computed as $(\mathbf{s}_t)_j = \frac{1}{V} \sum_{i=1}^V |g_{t,ij}|$ (an $O(d)$ vector). An EMA of this quantity yields $\hat{\mathbf{S}}_t$, and a layer-level RMS serves as the reference $\sigma_{rms}$. The damping factor is then $(\mathbf{H}_t)_j = \min(\sigma_{rms} / (\hat{\mathbf{S}}_t)_j, 1)$. For "quiet" dimensions ($\hat{S}_j < \sigma_{rms}$), the ratio exceeds 1 and is clipped to 1 (degenerating to Lion); for "noisy" dimensions ($\hat{S}_j > \sigma_{rms}$), the factor is damped below 1.
    - **Design Motivation**: Dimension-wise adaptivity is achieved with $O(d)$ rather than $O(Vd)$ state, incurring negligible memory overhead. Boundedness guarantees that updates are never more aggressive than Lion's, enabling theoretical convergence guarantees.

2. **Instantaneous Stability Constraint**:

    - **Function**: Prevent instability caused by sudden gradient spikes that EMA lag cannot handle in time.
    - **Mechanism**: In addition to the EMA-based damping $\mathbf{D}_t^{ema}$, an instantaneous damping term $\mathbf{D}_t^{inst}$ is computed from current-batch statistics. The final scaling factor takes the element-wise minimum: $(\mathbf{H}_t)_j = \min(\mathbf{D}_t^{ema}, \mathbf{D}_t^{inst}, 1)$.
    - **Design Motivation**: Analogous to Adaptive Gradient Clipping (AGC), this provides immediate protection against catastrophic instability.

3. **Adaptive Generalization of Lion**:

    - **Function**: Extend Lion from static to adaptive scaling.
    - **Mechanism**: Lion's update is $\hat{\mathbf{U}}_t^{Lion} = \mathbf{C}_t \odot \mathbf{1}$, whereas SAGE's update is $\hat{\mathbf{U}}_t^{SAGE} = \mathbf{C}_t \odot \mathbf{H}_t$. Lion is a special case of SAGE with $\mathbf{H}_t$ fixed to $\mathbf{1}$. Since $\|\mathbf{H}_t\|_\infty \leq 1$, SAGE constitutes a "safe generalization" of Lion.
    - **Design Motivation**: Safer updates permit the use of higher learning rates, enabling better convergence.

### Loss & Training

Decoupled weight decay (AdamW-style) is applied. SAGE maintains one $O(Vd)$ momentum state and one $O(d)$ adaptive state, yielding a total memory footprint of approximately half that of AdamW.

## Key Experimental Results

### Main Results (Test Perplexity)

| Method | 270M PPL | Memory | 1.3B PPL | Memory |
|--------|----------|--------|----------|--------|
| AdamW | 37.35 | 2.1GB | 27.81 | 9.8GB |
| Lion | 30.24 | 1.0GB | 28.37 | 4.9GB |
| SinkGD-Hybrid | 34.30 | 0.9GB | 28.71 | 1.9GB |
| **SAGE-Hybrid** | **29.95** | **0.5GB** | **24.33** | **0.9GB** |

### Ablation Study

| Configuration | PPL (270M) | Note |
|---------------|-----------|------|
| SAGE-Hybrid | 29.95 | Full method |
| SinkGD-Pure | 192.7 | Stateless method fails on embedding layer |
| SAGE-Pure | 116.0 | SAGE alone on all layers is also insufficient |
| Lion-Hybrid | 32.10 | Replacing embedding-layer AdamW with Lion |

### Key Findings
- SAGE-Hybrid achieves the lowest perplexity across all model sizes, with optimizer memory of only ~10% of AdamW.
- SinkGD-Pure confirms the embedding layer dilemma — purely stateless optimizers fail catastrophically on embedding layers.
- SAGE's boundedness permits higher learning rates than Lion, which is a key driver of performance gains.
- The hybrid design (SAGE for embeddings + SinkGD for dense layers) is the optimal combination.

## Highlights & Insights
- **Diagnosis of the "Embedding Layer Dilemma"** is precise: gradient sparsity and high variance in embedding layers are identified as the root cause of lightweight optimizer failure, enabling a targeted solution.
- **The $O(d)$ adaptive scaling design** is extremely memory-efficient: it compresses $V \times d$ gradient information into a $d$-dimensional mean absolute value vector, tracked by a $d$-dimensional EMA with negligible additional memory.
- **The generalization perspective from Lion to SAGE** is elegant: SAGE is a provably safe generalization of Lion, offering both theoretical guarantees and intuitive justification.

## Limitations & Future Work
- Experiments are conducted only up to 1.3B parameters; effectiveness at larger scales (7B+) remains unverified.
- SAGE is evaluated solely on the Llama architecture; generalization to other architectures (e.g., Mixture of Experts) is unknown.
- Pre-training token counts and dataset sizes are relatively small (a RedPajama subset is used); large-scale pre-training performance remains to be validated.
- Systematic comparison with other low-rank methods such as GaLore and APOLLO is absent (though APOLLO results are reported as poor).

## Related Work & Insights
- **vs. AdamW**: Optimizer state memory is reduced by ~10×, while perplexity is simultaneously improved.
- **vs. Lion**: SAGE is an adaptive generalization of Lion that enables higher learning rates via bounded damping.
- **vs. SinkGD**: SinkGD must fall back to AdamW for embedding layers; SAGE eliminates this fallback.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Both the diagnosis of the embedding layer dilemma and the $O(d)$ adaptive scaling solution are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐ Model scales are limited; larger-scale validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly derived and theoretical analysis is rigorous.
- **Value**: ⭐⭐⭐⭐ Provides a practical optimizer solution for memory-constrained LLM training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](../../NeurIPS2025/llm_pretraining/breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[ACL 2026\] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)
- [\[NeurIPS 2025\] Memory Mosaics at Scale](../../NeurIPS2025/llm_pretraining/memory_mosaics_at_scale.md)
- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/llm_pretraining/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)

<!-- RELATED:END -->
