---
title: >-
  [Paper Note] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining
description: >-
  [NeurIPS 2025][LLM Pretraining][Low-rank optimization] This paper identifies that the dominant subspace in low-rank optimizers such as GaLore "freezes" during pretraining (cosine overlap between consecutive subspaces approaches 1), trapping weight updates within a fixed low-rank subspace. The authors propose SARA (Sampling-based Adaptive Rank Allocation), which constructs subspaces by sampling singular vectors according to singular value weights, provides convergence guarantees, and reduces the performance gap between low-rank optimizers and full-rank Adam by up to 46%.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - Low-rank optimization
  - GaLore
  - importance sampling
  - memory efficiency
date: 2026-05-08
content_hash: c81b4c7b16418cd8
---

# Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining

**Conference**: NeurIPS 2025
**arXiv**: [2502.05790](https://arxiv.org/abs/2502.05790)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: Low-rank optimization, GaLore, importance sampling, LLM pretraining, memory efficiency

## TL;DR
This paper identifies that the dominant subspace in low-rank optimizers such as GaLore "freezes" during pretraining (cosine overlap between consecutive subspaces approaches 1), trapping weight updates within a fixed low-rank subspace. The authors propose SARA (Sampling-based Adaptive Rank Allocation), which constructs subspaces by sampling singular vectors according to singular value weights, provides convergence guarantees, and reduces the performance gap between low-rank optimizers and full-rank Adam by up to 46%.

## Background & Motivation
**State of the Field**: Low-rank optimizers (e.g., GaLore, Fira) reduce optimizer state memory by projecting gradients onto a low-rank subspace, making them an important class of memory-efficient methods for LLM pretraining. The central design question is how to select the projection subspace.

**Limitations of Prior Work**: GaLore selects the **dominant subspace** (singular vectors corresponding to the largest singular values), which intuitively retains the most gradient information. In practice, however, the dominant subspace nearly stops changing after the early phase of pretraining—the subspace overlap between consecutive update intervals approaches 1.0, especially in the `gate_proj` and `up_proj` layers.

**Root Cause**: When the subspace freezes, all weight updates across intervals are confined to the same low-rank subspace. Even if the update within each interval is low-rank, the cumulative update can still be high-rank provided subspaces across intervals are sufficiently diverse. Frozen subspaces break this "rank recovery" mechanism—**the cumulative weight update itself becomes trapped in a low-rank regime**.

**Starting Point**: Introduce stochasticity to break subspace freezing. Rather than always selecting the top-$r$ singular vectors, importance sampling is performed according to singular value magnitudes—directions with large singular values still have a higher probability of being selected, but directions with small singular values are also given a chance.

**Core Idea**: Replace top-$r$ subspace selection with singular-value-weighted importance sampling, substantially increasing subspace diversity across intervals while preserving convergence guarantees.

## Method

### Overall Architecture
SARA is a plug-and-play replacement for the subspace selection step in GaLore/Fira. Only the subspace selection module is replaced; all other components (low-rank projection, Adam state update, weight update) remain unchanged.

### Key Designs

1. **Importance Sampling Subspace Selection**:

    - **Function**: Perform SVD on gradient $G^{(t)}$ to obtain $m$ singular vectors, then sample $r$ of them without replacement according to importance weights $\omega_i = S_i / \sum_j S_j$.
    - **Difference from dominant subspace**: The dominant approach deterministically selects the top-$r$ largest singular vectors; SARA samples stochastically—directions with large singular values are still preferred but never permanently locked in.
    - **Negligible overhead**: Sampling requires only 0.0005 seconds (compared to 0.34 seconds for SVD).

2. **Convergence Guarantee**:

    - SARA is proven to achieve a convergence rate of $O(1/\sqrt{T})$, comparable to GoLore (fully random subspaces).
    - GaLore's dominant subspace selection carries no convergence guarantee, as frozen subspaces may trap the optimizer at suboptimal solutions.
    - The convergence bound incurs an additional factor of $\delta^{-3.5}$, where $\delta$ is the minimum selection probability—yet empirical performance is superior.

3. **Compatibility with Multiple Low-Rank Optimizers**:

    - Compatible with GaLore-Adam, Fira-Adam, Adafactor, and Adam-mini.
    - Also compatible with low-precision optimizer states (4-bit/8-bit).
    - Consistent improvements are observed across all combinations.

## Key Experimental Results

### LLaMA Pretraining (C4 dataset, validation perplexity PPL)

| Method | 60M | 130M | 350M | 1.1B |
|--------|-----|------|------|------|
| Full-Rank Adam | 27.71 | 23.27 | 18.21 | - |
| GaLore-Adam | ~29.5 | ~24.8 | ~19.5 | - |
| **GaLore-SARA-Adam** | ~28.8 | ~24.1 | ~18.9 | - |
| Fira-Adam | ~28.5 | ~23.8 | ~18.8 | - |
| **Fira-SARA-Adam** | ~28.1 | ~23.5 | ~18.5 | - |

### Gap Reduction Relative to Full-Rank Adam

| Model Scale | Gap Reduction |
|-------------|--------------|
| 60M | up to ~30% |
| 130M | up to ~40% |
| 350M | up to **46%** |

### Subspace Analysis

| Metric | Dominant Subspace | SARA |
|--------|-------------------|------|
| Consecutive subspace overlap | 0.85–0.99 | 0.3–0.7 |
| Effective rank of cumulative weight update | Low | Significantly higher |

### Key Findings
- SARA consistently outperforms dominant subspace selection across all model scales and optimizer combinations.
- Subspace overlap drops from 0.85–0.99 to 0.3–0.7, confirming that freezing is successfully broken.
- The effective rank of cumulative weight updates increases substantially, validating the hypothesis that subspace diversity leads to higher-rank updates.
- The combination with Fira yields the best results, benefiting jointly from SARA and residual utilization.
- Effectiveness extends to the 1.1B scale, demonstrating good scalability.

## Highlights & Insights
- **The discovery of "frozen subspace"** is itself a significant contribution: it identifies the fundamental bottleneck of GaLore's performance—not inadequate low-rank approximation, but insufficient subspace diversity.
- **Importance sampling as an elegant trade-off**: Unlike GoLore (fully random, losing gradient information) or GaLore (fully deterministic, causing freezing), singular-value-weighted sampling is a natural and principled middle ground.
- **Plug-and-play design**: The modification is minimal—only the subspace selection function is replaced—making SARA fully compatible with the existing low-rank optimization ecosystem.

## Limitations & Future Work
- The theoretical convergence bound carries an extra $\delta^{-3.5}$ factor compared to GoLore—empirical results are better, but the theoretical bound is looser.
- Pretraining experiments are conducted only on C4; other datasets (e.g., RefinedWeb, StarCoder) remain untested.
- The stochasticity of sampling may increase training variance, an effect not analyzed in detail.
- The subspace update frequency $\tau$ is fixed; adaptive $\tau$ scheduling could potentially yield further improvements.

## Related Work & Insights
- **vs. GaLore**: GaLore uses top-$r$ dominant subspace selection, which limits performance after freezing. SARA breaks the freeze via importance sampling.
- **vs. GoLore**: GoLore uses fully random projections with convergence guarantees but underperforms GaLore. SARA achieves a better balance between the two extremes.
- **vs. Fira**: Fira exploits projection residuals but still relies on the dominant subspace. SARA combined with Fira yields dual improvements.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Both the diagnosis of frozen subspaces and the importance sampling solution are highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four model scales, multiple optimizer combinations, and detailed subspace analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem motivation is clearly articulated; Figures 1/2 are intuitive; theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐ — Offers direct practical value to the low-rank optimization community at virtually zero additional cost.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)

<!-- RELATED:END -->
