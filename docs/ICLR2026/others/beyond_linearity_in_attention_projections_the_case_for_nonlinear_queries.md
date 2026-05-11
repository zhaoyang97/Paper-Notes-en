---
title: >-
  [Paper Note] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries
description: >-
  [ICLR 2026 Workshop (GRaM)][nonlinear query] Motivated by the theoretical finding of algebraic redundancy in $W_Q$, this work replaces the linear Query projection with a nonlinear residual form $Q(X)=(X+f_\theta(X))/2$…
tags:
  - "ICLR 2026 Workshop (GRaM)"
  - "nonlinear query"
  - "attention projection"
  - "identity prior"
  - "bottleneck MLP"
  - "transformer architecture"
date: 2026-05-08
content_hash: 31b1d3dc4a3fb452
---

# Beyond Linearity in Attention Projections: The Case for Nonlinear Queries

**Conference**: ICLR 2026 Workshop (GRaM)  
**arXiv**: [2603.13381](https://arxiv.org/abs/2603.13381)  
**Code**: [GitHub](https://github.com/MarkoKarbevski/beyond_query_linearity)  
**Area**: Others  
**Keywords**: nonlinear query, attention projection, identity prior, bottleneck MLP, transformer architecture

## TL;DR

Motivated by the theoretical finding of algebraic redundancy in $W_Q$, this work replaces the linear Query projection with a nonlinear residual form $Q(X)=(X+f_\theta(X))/2$, outperforming a baseline with +12.5% more parameters while keeping parameter count unchanged.

## Background & Motivation

**Algebraic Redundancy Finding**: Karbevski & Mijoski (2024) proved the existence of reparameterization invariance in Transformers — for any invertible matrix $\Theta$, mapping $(X, W_Q, W_K, W_V, W_O)$ to $(X\Theta, \Theta^{-1}W_Q, ...)$ leaves the MHA output unchanged. Setting $\Theta = W_Q$ reduces $W_Q \to I$, showing that the linear parameters of $W_Q$ can be entirely absorbed by adjacent layers — rendering them algebraically redundant.

**Empirical Validation**: A model with $W_Q = I$ performs comparably to the standard baseline and remains stable under 3× lower weight decay, confirming that the identity mapping is a sound prior for the query pathway.

**Core Reasoning**: Since linear parameters are redundant (absorbable), introducing **nonlinearity** is necessary for effective parameter allocation in the query path — nonlinear transformations cannot be absorbed.

**Information Bottleneck Perspective**: Generating four vectors (q, k, v, residual) from a single token $x$ as linear functions of $x$ forms an information bottleneck. A nonlinear query partially decouples this bottleneck.

**Why Query**: Under GQA, $W_K/W_V$ are shared; only $W_Q$ can be safely replaced without disrupting the sharing structure.

## Method

### Core Formula

$$Q(X) = (X + f_\theta(X)) / 2$$

### Structure of $f_\theta$

Bottleneck MLP: $f_\theta(X) = \text{LN}(\text{GELU}(\text{RMSNorm}(X) \cdot W_1) \cdot W_2)$
- $W_1 \in \mathbb{R}^{d \times r}$, $W_2 \in \mathbb{R}^{r \times d}$, $r = d/2$
- Total matrix parameters $2dr = d^2$, on the same order as the original $W_Q$
- Normalization layers add only $O(d)$ parameters (<0.1%)

### Design Highlights

1. **Identity Anchor**: The $X$ term anchors to a known good prior ($W_Q=I$), ensuring gradient flow and training stability.
2. **1/2 Scaling Factor**: Follows the recommendation of Karbevski & Mijoski to prevent magnitude inflation.
3. **K/V Unchanged**: Key and Value projections remain standard linear mappings.
4. **Compatibility**: Compatible with modern architectures including GQA/MQA, RoPE, and MoE.

## Key Experimental Results

### Main Results (GPT-3 Small, ~124M, OpenWebText, 60k steps ≈ 29B tokens)

| Model | Non-embedding Params | Val Loss (59k) | Relative Gain |
|-------|---------------------|----------------|---------------|
| Baseline | 85M | 2.956 | 0 |
| MLP 4.75 (wider MLP, +12.5% params) | 96M | 2.927 | 0.98% |
| MLP 4.75 (scaled LR) | 96M | 2.928 | 0.94% |
| **Res. GELU (Ours)** | **85M** | **2.919** | **1.24%** |
| **Res. GELU (best hyperparams)** | **85M** | **2.915** | **1.40%** |

### Training Stability

| Configuration | Outcome | Note |
|---------------|---------|------|
| Baseline, WD=0.05 | Diverges before 20k steps | Standard model unstable |
| Res. GELU, WD=0.03, LR=3e-3 | Stable to 60k | Tolerates 5× higher LR |

### Key Findings

- Training far exceeds Chinchilla optimum (29B tokens vs. 2.5B optimal), ensuring that improvements are not artifacts of token-budget deficiency.
- All models observe identical training and validation data under a fixed random seed, ensuring strict variable control.
- The nonlinear variant yields the largest gains during warmup, smaller gains in the mid-phase, and a recovery of gains at the end for the best variant.
- The authors explicitly note that 1.40% likely represents a lower bound rather than an upper bound, given the very limited hyperparameter search.

## Highlights & Insights

- **Theory-Driven Architecture Modification**: The logical chain starting from algebraic redundancy is complete and rigorous ($W_Q$ redundant → linear projection ineffective → nonlinearity necessary).
- **Parameter-Neutral Improvement**: Outperforms a model with +12.5% more parameters without adding any parameters — indicating that parameter efficiency, not capacity, is the bottleneck.
- **Dual Win in Training Stability**: The nonlinear variant not only achieves better performance but also remains stable under more aggressive hyperparameters (lower WD, higher LR).
- **Fully Open-Source**: Code and checkpoints are publicly released.

## Limitations & Future Work

- Validated only at a single scale (~124M); performance on larger models remains untested (does redundancy persist at scale?).
- No multi-seed experiments conducted (mitigated by fixed data ordering and extended training).
- Inference speed is not measured — the nonlinear formulation introduces sequential dependencies (the bottleneck MLP must complete before attention).
- Hyperparameter search is highly limited; various normalization variants and activation functions are not systematically explored.
- Downstream task performance is not evaluated; only pretraining validation loss is reported.

## Related Work & Insights

- **Kernel Attention (Performer, etc.)**: Applies nonlinear feature maps after $Q=XW_Q$; this work directly replaces $W_Q$.
- **MLP-Attention (Zhang'24)**: Replaces all Q/K/V projections with MLPs, but adds ~10% parameters and lacks theoretical motivation.
- **Nonlinear LoRA**: Targets fine-tuning scenarios; this work targets pretraining architecture design.
- **Always Skip Attention (Ji et al.)**: Reveals the unique dependence of self-attention on skip connections, echoing the identity anchor design in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ Theory-driven architectural modification with a novel direction, though the modification itself is modest.
- Experimental Thoroughness: ⭐⭐⭐ Single scale and single dataset, but variable control is exceptionally rigorous.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and concise mathematics within workshop page limits.
- Value: ⭐⭐⭐⭐ Could be highly significant if validated at larger model scales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hilbert-Guided Sparse Local Attention](hilbert-guided_sparse_local_attention.md)
- [\[NeurIPS 2025\] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](../../NeurIPS2025/others/obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)
- [\[CVPR 2026\] NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries](../../CVPR2026/others/nailia_multimodal_nail_design_retrieval_based_on_dense_intent_descriptions_and_p.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
