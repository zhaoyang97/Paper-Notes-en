---
title: >-
  [Paper Note] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging
description: >-
  [AAAI 2026][LLM/NLP][model merging theft] This paper proposes MergeBarrier, a plug-and-play defense method that disrupts linear mode connectivity (LMC) between a protected model and its homologous counterparts by applying orthogonal projection transformations to attention layers and activation-function-unfolding reparameterization to FFN layers, thereby actively preventing unauthorized model merging without degrading model performance.
tags:
  - AAAI 2026
  - LLM/NLP
  - model merging theft
  - MergeBarrier
  - orthogonal projection
  - weight reparameterization
  - linear mode connectivity
date: 2026-05-08
content_hash: bad2d7364226ddee
---

# Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging

**Conference**: AAAI 2026
**arXiv**: [2511.10712](https://arxiv.org/abs/2511.10712)
**Code**: None
**Area**: LLM/NLP
**Keywords**: model merging theft, MergeBarrier, orthogonal projection, weight reparameterization, linear mode connectivity

## TL;DR

This paper proposes MergeBarrier, a plug-and-play defense method that disrupts linear mode connectivity (LMC) between a protected model and its homologous counterparts by applying orthogonal projection transformations to attention layers and activation-function-unfolding reparameterization to FFN layers, thereby actively preventing unauthorized model merging without degrading model performance.

## Background & Motivation

### State of the Field

**State of the Field**: Model merging has become an increasingly popular technique for efficiently expanding LLM capabilities. By integrating multiple expert models, practitioners can obtain combined abilities without additional data collection or high-end GPUs.

**Limitations of Prior Work**: Model merging introduces a novel intellectual property threat — "model merging theft." Free-riders can unauthorized merge restricted open-source models into their own for commercial purposes, and existing passive methods such as watermarking and fingerprinting may not survive the merging process, making post-hoc attribution difficult.

**Root Cause**: Existing approaches fail to simultaneously satisfy three critical properties: (1) actively preventing unauthorized merging (rather than detecting it after the fact); (2) compatibility with general open-source environments (no additional authorization components required); and (3) high security with negligible performance degradation. Passive methods cannot prevent merging; authorization-based methods are incompatible with open-source release; and simulator-based methods sacrifice either security or utility.

**Paper Goals**: This work formally defines the model merging theft defense problem for the first time and designs an active defense scheme satisfying all three properties above.

**Starting Point**: The approach targets the underlying mechanism of model merging — linear mode connectivity (LMC). Homologous models share a low-loss basin and can thus be smoothly interpolated and merged. Disrupting this connectivity causes the merged model to suffer significant performance degradation.

**Core Idea**: Protected models are moved out of the low-loss basin via mathematically equivalent weight transformations, such that linear interpolation paths between the protected model and its homologous counterparts traverse high-loss regions, rendering the merged result non-functional. Since the transformations are mathematically equivalent, the protected model's functionality remains intact when used in isolation.

## Method

### Overall Architecture

MergeBarrier handles the attention layers and FFN layers of a Transformer separately: orthogonal projection transformations are applied to attention layers, while activation-function-unfolding reparameterization is applied to FFN layers. Both transformations guarantee functional equivalence during inference, yet weight displacement during merging causes performance collapse.

### Key Designs

1. **Attention Weight Projection**:

    - Mechanism: The paired linear structure of Q and K in the attention mechanism is exploited by applying a shared orthogonal projection matrix $P$ to both.
    - Mathematical guarantee: Since $PP^\top = I$, the projection cancels out in the attention computation, leaving the output unchanged.
    - Optimization objective: Maximize the Frobenius norm distance between the merged model weights and the original low-loss region.
    - Extension to MHA/GQA: Block-diagonal orthogonal matrices are used to accommodate multi-head and grouped-query attention.
    - Acceleration: Randomized SVD (RSVD) reduces the eigendecomposition complexity from $O(n^3)$ to $O(n^2k)$.

2. **FFN Weight Reparameterization**:

    - Mechanism: As FFN layers lack symmetric structure, the activation function is approximated via Taylor expansion, replacing original weights with polynomial coefficients.
    - Design Motivation: The reparameterized weights reside in a different space from the original weights, preventing attackers from performing interpolation-based merging in the original weight space.
    - Expansion point selection: The midpoint of the training set's feature representations is used to minimize approximation error.
    - Security guarantee: The noise introduced by the Taylor remainder degrades weight inversion into the NP-hard Learning With Errors (LWE) problem.
    - Additional benefit: The remainder noise satisfies differential privacy (DP), providing defense against membership inference attacks.

3. **Orthogonal Matrix Optimization**:

    - The ideal case $P = -I$ maximizes displacement but exposes the true weights.
    - Relaxation strategy: Only the directions corresponding to the largest eigenvalues are flipped (−1) while others are retained (+1), balancing security and concealment.

### Loss & Training

- Optimization objective: Maximize the Frobenius norm distance between merged weights and the original low-loss region.
- Equivalent to maximizing $\|W_q(P-I)\|_F^2 + \|W_k(P-I)\|_F^2$.
- The FFN component is approximated locally via Taylor expansion around the selected expansion point; expansion order $N$ controls the precision–security trade-off.
- The overall approach is a post-processing method requiring no additional training.

## Key Experimental Results

### Main Results

Experiments are conducted on LLaMA-2-13B, protecting three expert models:

| Merging Method | Accuracy (Unprotected) | Accuracy (Protected) | Protection Effect (Δ) |
|---------------|----------------------|---------------------|----------------------|
| Task Arithmetic | 68.5% | 42.3% | −26.2% |
| TIES | 71.2% | 45.1% | −26.1% |
| DARE | 69.8% | 43.7% | −26.1% |

### Original Model Performance Preservation

| Metric | Original Model | Protected Model | Retention Rate |
|--------|---------------|-----------------|----------------|
| WizardLM (MT-Bench) | 7.21 | 7.18 | 99.6% |
| WizardMath (GSM8K) | 63.4% | 63.1% | 99.5% |
| Code Alpaca (HumanEval) | 45.7% | 45.2% | 98.9% |

| Merging Method | Unprotected | MergeBarrier | PaRaMS | TaylorMLP |
|---------------|-------------|-------------|--------|-----------|
| Task Arithmetic | Normal | Significant post-merge performance drop | Partially effective | Protects only a single layer; weak security |
| TIES | Normal | Significant post-merge performance drop | Reversible via decoding | Reversible |
| DARE | Normal | Significant post-merge performance drop | Reversible via decoding | Single-layer protection only |

Models protected by MergeBarrier exhibit negligible accuracy loss when used in isolation.

### Ablation Study

- Orthogonal projection and FFN reparameterization are each individually effective, with the combined approach achieving the best results.
- The number of flipped eigenvalue directions $k$ governs the security–concealment trade-off.
- RSVD acceleration substantially reduces computational cost while preserving effectiveness.

### Key Findings

- MergeBarrier is the only method simultaneously satisfying the properties of proactivity, compatibility, security, and utility.
- PaRaMS's permutation-based protection can be reversed via weight alignment, and its scaling-based protection can be undone using base model parameters.
- TaylorMLP protects only a single linear layer per Transformer block, leaving the remaining weights exposed.
- The NP-hardness of inverting the Taylor remainder provides a theoretical security guarantee for FFN reparameterization.

## Highlights & Insights

- The defense is motivated by the LMC mechanism, yielding a theoretically clean and elegant design.
- The self-cancellation property of orthogonal projections guarantees lossless performance, which is mathematically elegant.
- The security analysis connecting Taylor remainders to the LWE problem introduces cryptographic hardness into model protection — a pioneering approach.
- The plug-and-play design requires no external authorization components and is fully compatible with open-source release scenarios.

## Limitations & Future Work

- Validation is limited to LLaMA-2-13B; experiments on larger models and diverse architectures are lacking.
- Analysis of adaptive attackers (i.e., adversaries aware of the MergeBarrier mechanism and attempting to reverse it) is insufficiently thorough.
- FFN reparameterization modifies the model structure, which may introduce inference latency despite claims of parallelizability.
- The expansion point depends on the training data distribution; out-of-distribution inputs may increase approximation error.

## Related Work & Insights

- PaRaMS defends via MLP permutation and attention scaling, but both protections are reversible.
- TaylorMLP protects single-layer weights via Taylor expansion, providing insufficient security.
- This paper introduces LWE cryptographic hardness into deep learning model protection — an approach generalizable to other model security scenarios.
- The concept of LMC is of significant value for understanding the mechanisms underlying model merging.

## Rating

⭐⭐⭐⭐ (4/5)

The theoretical motivation is clear, the method design is elegant, and the mathematical derivations are rigorous. The lossless property of orthogonal projections and the LWE security guarantee are notable highlights. However, the experimental scale is limited (13B only), and the analysis of adaptive attacks could be more thorough. The work holds significant practical value for intellectual property protection of open-source models.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](../../ICCV2025/llm_nlp/fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](an_invariant_latent_space_perspective_on_language_model_inve.md)
- [\[AAAI 2026\] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)
- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)

<!-- RELATED:END -->
