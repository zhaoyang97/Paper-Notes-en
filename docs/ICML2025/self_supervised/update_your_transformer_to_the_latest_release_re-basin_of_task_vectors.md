---
title: >-
  [Paper Note] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors
description: >-
  [ICML2025][Self-Supervised Learning][Model Re-Basining] Proposed TransFusion, a two-level weight permutation method (inter-head + intra-head) specifically designed for Transformers, enabling data-free and training-free migration of fine-tuned knowledge (task vectors) from old models to new foundation models.
tags:
  - "ICML2025"
  - "Self-Supervised Learning"
  - "Model Re-Basining"
  - "Task Vectors"
  - "Transformer Weight Alignment"
  - "Weight Permutation"
  - "Data-Free Migration"
date: 2026-05-08
content_hash: b61becb357042425
---

# Update Your Transformer to the Latest Release: Re-Basin of Task Vectors

**Conference**: ICML2025  
**arXiv**: [2505.22697](https://arxiv.org/abs/2505.22697)  
**Code**: [TransFusion](https://github.com/aimagelab/TransFusion)  
**Area**: Self-Supervised  
**Keywords**: Model Re-Basining, Task Vectors, Transformer Weight Alignment, Weight Permutation, Data-Free Migration  

## TL;DR

Proposed TransFusion, a two-level weight permutation method (inter-head + intra-head) specifically designed for Transformers, enabling data-free and training-free migration of fine-tuned knowledge (task vectors) from old models to new foundation models.

## Background & Motivation

### Background

**Background**: Model update issue: Pre-trained models are frequently updated, making older fine-tuned models obsolete and requiring retraining.

### Proposed Solution

**Proposed Solution**: Task vectors: $\tau=\theta_A^{ft}-\theta_A$, with the target being $\theta_B^{ft}=\theta_B+\pi(\tau)$.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing re-basin methods only apply to MLPs/CNNs, as Multi-Head Attention suffers from the "head contamination" issue.

### Key Challenge

**Key Challenge**: Permutation inconsistency caused by residual connections.

## Method

### Step 1: Inter-Head Alignment

- SVD singular values define a permutation-invariant spectral distance: $d_{ij}=\|\Sigma_i-\Sigma_j\|$
- Distance matrix $D_{ij}=d_{ij}^q+d_{ij}^k+d_{ij}^v$
- Hungarian algorithm is used to find the optimal head pairing.

### Step 2: Intra-Head Alignment

- Maximize the inner product through row permutations within the paired heads.
- Synthesize $P_{attn}=P_{inter}\circ\{P_{intra}^{(h)}\}$.
- Jointly optimize q, k, and v.

### Step 3: Handling Residual Connections

- The compensation mapping $\mathcal{I}_i=P_{W_0}P_{in}^\top$ allows both branches to share the permutation.
- Separately handle the first residual (attention → addition) and the second residual (FFN → addition).
- Ensure permutation consistency on both sides of the residual connection.

### Migration Formula

$$\tilde{\theta}_B^{ft}=\theta_B+\alpha\pi(\tau)$$

where $\alpha$ is a scaling factor (with $\alpha=1$ achieving the best performance in experiments).

**Theorem 3.1**: The structured two-level permutation preserves functional equivalence: $O'=OP_{attn}$.

### Computational Complexity

$O(Ld_m^3)$, equivalent to Git Re-Basin (Proposition 3.2). Key operations: SVD $O(d_k^2 d_m)$ + Hungarian $O(H^3)$ + LAP $O(d_k^3)$.

### One-Time Matching, Multi-Time Reuse

In scenarios with multiple task vectors, the permutation $\pi$ for $\theta_A \to \theta_B$ only needs to be computed once and can be reused for all task vectors fine-tuned based on $\theta_A$. Model merging can also be performed on the target $\theta_B$.

## Key Experimental Results

### Visual Classification (CLIP ViT-B/16 CommonPool→Datacomp)

| Method | EuroSAT Task↑ | Supp.↑ | DTD Task↑ | SVHN Task↑ |
|---|---|---|---|---|
| $\theta_B$ zero-shot | 49.02 | 68.73 | 47.50 | 45.97 |
| $\theta_B+\tau$ (naive) | -7.62 | -16.15 | -0.15 | -22.00 |
| Git Re-Basin | +0.95 | -0.48 | -0.91 | +0.79 |
| Optimal Transport | -14.05 | -5.28 | -0.53 | -12.30 |
| **TransFusion** | **+4.95** | **-0.06** | **+0.21** | **+3.64** |

### NLP Tasks (QQP/SST2, etc.)

- TransFusion also effectively improves migration performance.

### $\alpha$ Sensitivity Analysis

- Downstream task performance is maximized when $\alpha \approx 1$.
- Generalization capability (support set) is more stable when $\alpha \geq 0.5$.
- naive transport remains unstable across all values of $\alpha$.

### Ablation Study

| Component Removed | Impact |
|---|---|
| Inter-head alignment | Significant performance drop |
| Intra-head alignment | Moderate drop |
| Residual handling | Model collapse |
| Spectral distance → Cosine distance | Degraded head pairing quality |

## Highlights & Insights

1. Data-free and training-free: Requires only two sets of weights.
2. Elegant utilization of permutation invariance in spectral distance.
3. Mathematical proof provided for functional equivalence.
4. One-time matching can be reused for all task vectors.

## Limitations & Future Work

- LAP approximation offers no optimality guarantee.
- Migration is constrained to identical architectures.
- Discriminability decreases when the head count is large.

## Related Work & Insights

- Ainsworth et al. (2023) Git Re-Basin
- Ilharco et al. (2023) Task Vectors

## Rating

⭐⭐⭐⭐ — The two-level permutation strategy is elegant with theoretical guarantees, and the problem addressed is highly practical.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[ICML 2025\] MTL-UE: Learning to Learn Nothing for Multi-Task Learning](mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)
- [\[CVPR 2025\] Do Your Best and Get Enough Rest for Continual Learning](../../CVPR2025/self_supervised/do_your_best_and_get_enough_rest_for_continual_learning.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)
- [\[ICLR 2026\] NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering](../../ICLR2026/self_supervised/neo_no-optimization_test-time_adaptation_through_latent_re-centering.md)

</div>

<!-- RELATED:END -->
