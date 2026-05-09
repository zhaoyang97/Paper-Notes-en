---
title: >-
  [Paper Note] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers
description: >-
  [NeurIPS 2025][LLM/NLP][Parameterization] CompleteP parameterization (α=1) is the only scheme that simultaneously achieves hyperparameter transfer along the depth dimension and complete feature learning, saving 12–34% FLOPs over μP on deep models.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Parameterization
  - Depth Scaling
  - Hyperparameter Transfer
  - Compute Efficiency
  - CompleteP
  - μP
date: 2026-05-08
content_hash: 711ed59937ce1408
---

# Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2505.01618](https://arxiv.org/abs/2505.01618)
**Code**: [https://github.com/EleutherAI/nanoGPT-mup/tree/completep](https://github.com/EleutherAI/nanoGPT-mup/tree/completep)
**Area**: LLM Scaling Laws, Hyperparameter Optimization
**Keywords**: Parameterization, Depth Scaling, Hyperparameter Transfer, Compute Efficiency, CompleteP, μP

## TL;DR
CompleteP parameterization (α=1) is the only scheme that simultaneously achieves hyperparameter transfer along the depth dimension and complete feature learning, saving 12–34% FLOPs over μP on deep models.

## Background & Motivation
As LLMs grow in scale, hyperparameter search (learning rate, initialization variance, etc.) must be repeated at substantial cost. μP (maximal update parameterization) enables a "tune small, deploy large" strategy by guaranteeing hyperparameter transfer along the width dimension. However, μP **addresses only width scaling**: when model depth $L$ varies, optimal hyperparameters drift, resulting in two consequences: (1) hyperparameters must be re-tuned for each depth, which is time-consuming; (2) forgoing re-tuning leads to suboptimal training and wasted compute.

Prior work extends μP to the depth dimension by introducing a residual scaling factor α∈[0.5,1], yet disagreement persists over the optimal α: Yang et al. argue α=0.5 is best and that HP transfer is impossible, while Bordelon et al. find α=1 to be theoretically superior. This paper systematically compares these two extremes and conclusively establishes that α=1 is the correct choice.

## Method

### Depth Scaling of Residual Blocks
The residual connections of a Transformer are parameterized as:

$$h^{\ell+1} = h^{\ell} + L^{-\alpha} \mathcal{F}_\ell(h^{\ell})$$

where $\mathcal{F}_\ell$ denotes the MLP or attention block at layer $\ell$, and α controls the decay rate of each layer's residual contribution. α=0 corresponds to standard parameterization (SP), α=0.5 is the value recommended by Yang et al., and α=1 is CompleteP as proposed in this work.

### Complete Parameterization Rules (Table 1)
CompleteP encompasses not only residual scaling but also coordinated adjustments across multiple hyperparameters. Defining the width multiplier $m_N = N/N_\text{base}$ and the depth multiplier $m_L = L/L_\text{base}$:

- **Hidden layer initialization variance**: $\sigma^2_\text{base} \cdot m_N^{-1}$ (inherited from μP)
- **Hidden layer learning rate**: $\eta_\text{base} \cdot m_N^{-1} \cdot m_L^{\alpha-1}$ (decreases with depth when α=1)
- **Pre-LN learning rate**: $\eta_\text{base} \cdot m_L^{\alpha-1}$ (a key extension absent in SP/μP)
- **Bias learning rate**: $\eta_\text{base} \cdot m_L^{\alpha-1}$
- **Residual block multiplier**: $m_L^{-\alpha}$ (i.e., $L^{-1}$ when α=1)
- **AdamW ε (residual blocks)**: $\epsilon_\text{base} \cdot m_N^{-1} \cdot m_L^{-\alpha}$
- **Weight decay**: $\lambda_\text{base} \cdot m_N$ (inherited from μP)

These extensions—particularly the depth scaling of the LayerNorm learning rate and AdamW ε—constitute an important practical contribution of this work. Without these adjustments, α=0.5 exhibits training instability.

### Three Design Desiderata

**Desideratum 1: Stable Initialization.** Requires $\|h^\ell\|^2/N = \Theta(1)$ for all layers, constraining α≥0.5.

**Desideratum 2: Maximal Residual Stream Updates.** Each layer's parameter update should contribute $\Theta(1/L)$ to $h^{\ell+1}$, establishing the depth dependence of the learning rate as $\eta = \Theta(L^{1-\alpha})$, and constraining α≤1.

**Desideratum 3: Complete Feature Learning (the central novel contribution).** The representation $h^\ell$ at every layer must not be "lazy" with respect to any subset of preceding parameters—that is, it must not degenerate into its linearized approximation. Specifically, for α<1, the nonlinear terms in the Taylor expansion are of order $\Theta(L^{\alpha-2})$, which vanish relative to the linear terms as L→∞, causing deep networks to effectively degenerate into linear models. **Only when α=1 are the nonlinear and linear terms of the same order**, guaranteeing complete feature learning.

This theoretical analysis explains why α=0.5 appears adequate at small depths but underperforms α=1 at large depths: the optimal hyperparameters for shallow models depend on the balance between linear and nonlinear dynamics, a balance that α<1 disrupts.

## Experimental Validation

### Hyperparameter Transfer Along Depth
Under a fixed 300M token training budget, learning rate and initialization standard deviation transfer is evaluated for depths L=2 to L=128:
- **SP / μP / α=0.5**: Optimal learning rate drifts substantially with depth, precluding stable transfer.
- **CompleteP (α=1)**: Optimal learning rate and $\sigma_\text{init}$ remain stable across all depths, with loss contours forming concentric structures.

Under compute-optimal settings (20 tokens per parameter, batch size selected by FLOPs, weight decay tuned), sensitivity to learning rate decreases, yet α=1 still consistently achieves lower loss without additional hyperparameter search.

### Optimal Width-to-Depth Ratio N:L
Models are trained at three scales ($P_\text{non-emb} \in \{50\text{M}, 300\text{M}, 1.5\text{B}\}$) with 7–10 N:L configurations per scale:

| Model Scale | μP Optimal N:L | CompleteP Optimal N:L | CompleteP ≤1% Loss Range |
|-------------|---------------|----------------------|--------------------------|
| 50M | ~40 | ~40 | N:L≥~12 |
| 300M | ~50 | ~50 | N:L≥~15 |
| 1.5B | ~62 | ~62 | **N:L≥11.8** (μP requires ≥38.7) |

CompleteP substantially broadens the acceptable N:L range, allowing narrow-deep models to remain near-optimal—a finding of significant practical relevance for memory-constrained hardware (e.g., layer-by-layer streaming inference/training).

### FLOP Savings (vs. μP)

| Configuration | FLOP Savings |
|---------------|-------------|
| 1.5B optimal N:L (N=1984, L=32) | **11.8%** |
| 1.5B deepest (N=832, L=179) | **34.4%** |
| 300M deepest (N=448, L=125) | Substantial |

The advantage of CompleteP over μP grows with depth, as μP suffers increasingly severe hyperparameter misalignment at large depths.

### Downstream Tasks (Zero-Shot, 1.5B Model, 20 TPP)

| Task | μP (Optimal N:L) | CompleteP (Optimal) | μP (L=179) | CompleteP (L=179) |
|------|-----------------|--------------------|-----------|--------------------|
| HellaSwag | 53.3±0.5 | **54.2±0.5** | 49.1±0.5 | **52.7±0.5** |
| ARC-Easy | 54.4±1.0 | **55.6±1.0** | 50.0±1.0 | **54.6±1.0** |
| LAMBADA | 54.3±0.7 | **54.9±0.7** | 51.8±0.7 | **53.3±0.7** |
| PIQA | 70.7±1.1 | **71.5±1.1** | 69.6±1.1 | **70.6±1.1** |
| BoolQ | 58.4±0.9 | **60.7±0.9** | 57.8±0.9 | **59.0±0.9** |
| Downstream Avg. | 54.3±0.3 | **55.2±0.3** | 52.0±0.3 | **54.3±0.3** |

Upstream gains transfer consistently to downstream tasks, with CompleteP's advantage particularly pronounced on deep models (54.3 vs. 52.0).

### Long-Training Validation at 200 TPP
Further training at 200 TPP on 50M and 300M models shows that CompleteP achieves the lowest validation loss across all configurations, confirming that the conclusions hold under extended training.

## Highlights & Insights
1. **First demonstration of depth HP transfer**: Successfully transfers hyperparameters from L=2 to L=128, surpassing the depths of LLaMA-70B/405B.
2. **Theoretical contribution—complete feature learning**: Desideratum 3 is introduced, proving that α<1 causes the nonlinear dynamics of layers to vanish in the deep limit (lazy regime), and that only α=1 mathematically guarantees fully nonlinear learning at every layer.
3. **Practical contribution—complete parameterization table**: Table 1 provides closed-form scaling rules for all hyperparameters of Pre-LN Transformers with AdamW, ready for direct engineering use.
4. **Width-depth ratio flexibility**: Challenges the conventional wisdom that N:L≈100 is optimal (Kaplan et al.), demonstrating that this conclusion was an artifact of depth misalignment under SP.
5. **Minimal implementation overhead**: Changes are confined to the residual multiplier $L^{-1}$, per-group learning rates, and ε scaling, requiring no architectural modifications.

## Limitations & Future Work
- The largest experimental scale is 1.5B non-embedding parameters; validation at 7B+ scale remains absent.
- Experiments are conducted exclusively on the SlimPajama dataset; robustness across data distributions is untested.
- The theoretical analysis is derived under a fixed token count regime; direct generalization to the compute-optimal setting requires additional assumptions.
- Downstream evaluation signals are relatively weak at small model scales.
- Architectural variants such as MoE or parallel sub-networks are not addressed.

## Related Work & Insights
- **μP (Yang et al., 2022)**: Establishes the foundation for width-wise HP transfer; this work extends it to the depth dimension.
- **Yang et al., 2024**: Advocates α=0.5 and asserts that HP transfer is impossible; this work directly refutes that claim.
- **Bordelon et al., 2024**: Provides theoretical arguments for α=1 in the infinite-depth limit; this work supplies large-scale empirical confirmation.
- **Kaplan et al., 2020**: Proposes N:L≈100 as the optimal ratio; this work demonstrates that the conclusion is confounded by the deficiencies of SP.
- **Large et al., 2024**: Empirically adopts a scheme similar to α=1 but lacks systematic theoretical grounding.

## Rating
⭐⭐⭐⭐⭐ (5/5)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales](hyperparameter_transfer_enables_consistent_gains_of_matrix-preconditioned_optimi.md)
- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)
- [\[NeurIPS 2025\] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](wider_or_deeper_scaling_llm_inference-time_compute_with_adaptive_branching_tree_.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)

<!-- RELATED:END -->
