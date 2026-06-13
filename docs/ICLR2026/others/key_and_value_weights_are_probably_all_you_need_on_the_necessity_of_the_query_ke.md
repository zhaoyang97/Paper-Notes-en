---
title: >-
  [Paper Note] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention
description: >-
  [ICLR 2026][Self-Attention] This paper theoretically demonstrates that the Query/Key/Value weight triplet in Transformer self-attention is redundant — the Query weight matrix can be replaced by an identity matrix (reduci…
tags:
  - "ICLR 2026"
  - "Self-Attention"
  - "Query Weights"
  - "Parameter Redundancy"
  - "Implicit Regularization"
  - "Architecture Simplification"
date: 2026-05-08
content_hash: 0dd9fefe8f5e1762
---

# Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention

**Conference**: ICLR 2026
**arXiv**: [2510.23912](https://arxiv.org/abs/2510.23912)  
**Code**: [GitHub](https://github.com/MarkoKarbevski/Wqkv_necessity)  
**Area**: Transformer Architecture
**Keywords**: Self-Attention, Query Weights, Parameter Redundancy, Implicit Regularization, Architecture Simplification

## TL;DR
This paper theoretically demonstrates that the Query/Key/Value weight triplet in Transformer self-attention is redundant — the Query weight matrix can be replaced by an identity matrix (reducing attention parameters by 25%). GPT-style models trained from scratch confirm that performance is preserved under appropriate hyperparameter adjustment, and training remains stable at 3× lower weight decay, suggesting implicit regularization.

## Background & Motivation

### State of the Field

**Background**: Transformer training and deployment are resource-intensive, motivating various architectural optimizations (quantization, efficient attention, weight sharing, normalization simplification). Recent work shows that normalization layers and attention parameters can be rearranged or simplified.

**Limitations of Prior Work**: It remains unclear whether the attention mechanism is over-parameterized and whether all three of the Q/K/V weights are necessary. Graef (2024) proved that both Q and O are redundant in settings without skip connections or normalization, but this does not cover practical architectures.

**Key Challenge**: Attention output depends on the input only through the products $XW_Q$, $XW_K$, $XW_V$ — a scaling construction can propagate a basis transformation from one layer to the next, making $W_Q = I_d$ theoretically lossless.

**Key Insight**: Theory-first approach → progressive proofs (single-layer / multi-layer / skip connections / weight sharing) → empirical validation on GPT.

## Method

### Theoretical Results

1. **Theorem 4.1 (Single-Layer Free Lunch)**: In any Transformer without normalization, $W_Q$ in a single layer can be eliminated via weight reparameterization. This result can be applied to pretrained models after removing normalization.

2. **Theorems 4.2–4.3 (Multi-Layer Elimination)**: Without normalization, $W_Q = I_d$ holds across all layers under either of two conditions: (i) skip connections are applied only around the attention sub-layer, or (ii) weights are shared across layers.

3. **Theorem 8.4 (ReLU MLP and Residuals)**: Precisely characterizes when a residual connection can be absorbed by a ReLU MLP, revealing the structural expressivity boundary of skip connections.

### Empirical Validation
- GPT-2-style models (117M–124M parameters) trained from scratch on OpenWebText.
- The 117M model with $W_Q = I_d$ matches the performance of the full 124M baseline.
- Reallocating saved parameters to the MLP yields significant improvement over the 124M baseline.
- Training remains stable at 3× lower weight decay, suggesting implicit regularization.

### Intuition
- Attention depends on $XW_QW_K^TX^T$, where $W_Q$ acts as a basis transformation that can be "transferred" to the processing of the previous layer's output.
- Removing $W_Q$ reduces attention logits from a quadratic function of parameters to a linear one, simplifying optimization.

## Key Experimental Results

### GPT Model Validation

### Main Results

| Configuration | Parameters | Validation Loss | Notes |
|---|---|---|---|
| Baseline GPT-124M | 124M | Baseline | Full attention |
| $W_Q=I$ (117M) | 117M (−8%) | **Matches baseline** | 7M fewer params, same performance |
| $W_Q=I$ + MLP expansion (124M) | 124M | **Exceeds baseline** | Parameter reallocation is superior |

### Training Stability

### Ablation Study

| Configuration | Minimum Stable Weight Decay |
|---|---|
| Standard GPT | $\lambda$ |
| $W_Q=I$ | **$\lambda/3$** |

### Key Findings
- Removing $W_Q$ incurs no performance degradation, confirming its redundancy.
- Reallocating saved parameters to the MLP is more beneficial than to the attention module.
- Stable training at 3× lower weight decay suggests that eliminating $W_Q$ provides implicit regularization.
- Orthogonal to GQA/MQA — can be combined with these techniques.

## Highlights & Insights
- **Theory-Driven Architectural Simplification**: Rather than empirically testing whether removal works, the paper mathematically proves *why* it is valid, providing both confidence and precise applicability conditions.
- **Optimization Simplification**: Setting $W_Q = I$ transforms attention logits from $XW_QW_K^TX^T$ (quadratic in learned weights) to $XW_K^TX^T$ (linear), which may explain the observed improvement in training stability.
- **Structural Expressivity Boundary**: Theorem 8.4 precisely characterizes when a ReLU MLP can or cannot absorb a residual connection — an independent theoretical contribution.

## Limitations & Future Work
- Validation is limited to the 117M–124M scale; confirmation on larger models (7B+) is needed.
- Only $W_Q$ elimination is tested; removing $W_K$ or $W_V$ is left for future work.
- The presence of LayerNorm introduces additional approximation, weakening theoretical guarantees from exact to approximate.
- Only pretraining loss is evaluated; downstream task performance is not assessed.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Theoretical proof of attention weight redundancy is a fundamental contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Scale is limited; validation on larger models is needed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and progressively structured.
- **Value**: ⭐⭐⭐⭐ If validated at scale, this work could have significant implications for Transformer design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Coordination of Value-Maximizing Bidders](../../ICML2026/others/on_the_coordination_of_value-maximizing_bidders.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](../../NeurIPS2025/others/recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[ICLR 2026\] Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields](refine_now_query_fast_a_decoupled_refinement_paradigm_for_implicit_neural_fields.md)

</div>

<!-- RELATED:END -->
