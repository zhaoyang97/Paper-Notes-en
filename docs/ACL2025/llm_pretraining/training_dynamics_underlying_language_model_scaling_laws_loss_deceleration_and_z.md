---
title: >-
  [Paper Note] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning
description: >-
  [ACL 2025][LLM Pretraining][Scaling Laws] This work discovers the "loss deceleration" phenomenon in language model training—where the loss curves undergo a piecewise linear transition in log-log space. The root cause is identified as "zero-sum learning" (ZSL), where systematic opposition in per-token gradients leading to destructive interference offsets improvements in some tokens with deterioration in others. Scaling up mitigates ZSL by lowering the deceleration-onset loss $…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Scaling Laws"
  - "Training Dynamics"
  - "Loss Deceleration"
  - "Zero-Sum Learning"
  - "Broken Scaling Laws"
date: 2026-05-08
content_hash: 9bad6e1531938d69
---

# Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning

**Conference**: ACL 2025  
**arXiv**: [2506.05447](https://arxiv.org/abs/2506.05447)  
**Code**: [https://github.com/mirandrom/zsl](https://github.com/mirandrom/zsl)  
**Area**: LLM Pretraining  
**Keywords**: Scaling Laws, Training Dynamics, Loss Deceleration, Zero-Sum Learning, Broken Scaling Laws

## TL;DR
This work discovers the "loss deceleration" phenomenon in language model training—where the loss curves undergo a piecewise linear transition in log-log space. The root cause is identified as "zero-sum learning" (ZSL), where systematic opposition in per-token gradients leading to destructive interference offsets improvements in some tokens with deterioration in others. Scaling up mitigates ZSL by lowering the deceleration-onset loss $L_d$ and increasing the post-deceleration slope $r_d$, providing a directly actionable mechanism to bypass scaling law bottlenecks.

## Background & Motivation

**Background**: While scaling laws proposed by Kaplan et al. (2020) can accurately predict loss after model expansion, they are fundamentally empirical fits and do not explain *how* scaling improves loss (i.e., the underlying training dynamics mechanisms).

**Limitations of Prior Work**: (a) Theoretical explanations mostly focus on data distribution attributes (Michaud et al., 2023) or intrinsic model capacity (Sharma & Kaplan, 2022), leaving what actually happens during the training process largely unexplored; (b) Known phenomena like loss plateauing and saturation exist, but there is no unified framework connecting them to scaling improvements; (c) There is a lack of actionable mechanisms—knowing only "larger is better" does not help improve models without increasing their scale.

**Key Challenge**: The power-law form of scaling laws implies smooth training dynamics, but the authors discover that actual loss curves exhibit an abrupt slope change (deceleration) in log-log space, indicating a qualitative transition point in training dynamics.

**Goal**: To identify and formalize the loss deceleration phenomenon, propose the underlying mechanism (zero-sum learning), and demonstrate how scaling mitigates this mechanism, establishing a foundation for future methods to "improve models without relying on scale."

**Key Insight**: To analyze the root cause of macroscopic loss deceleration from a microscopic perspective of per-example (per-token) gradients and loss dynamics.

**Core Idea**: The root cause of loss deceleration is per-token gradient opposition (ZSL), and scaling up improves final loss by mitigating ZSL.

## Method

### Overall Architecture
1. **Characterizing the Phenomenon**: Fit the piecewise linear behavior of the loss curve using the Broken Neural Scaling Law (BNSL) to extract three interpretable parameters: $L_d$ (deceleration-threshold loss), $t_d$ (deceleration-threshold steps), and $r_d$ (post-deceleration log-log slope).
2. **Explaining the Mechanism**: Propose the ZSL hypothesis—systematic opposition in per-token gradients causes destructive interference, acting as the root cause of loss deceleration.
3. **Connecting with Scaling**: Demonstrate how scaling up reduces $L_d$ and $t_d$ while increasing $r_d$.

### Key Designs

1. **BNSL Fitting and Interpretable Parameterization (Eqn. 2)**:

    - Loss Estimation: $\hat{L}_T = L_d \cdot (t_d / T)^{r_d}$
    - $L_d$: Loss value when deceleration occurs (lower is better)
    - $t_d$: Step count when deceleration occurs (smaller indicates earlier deceleration)
    - $r_d$: Slope in the log-log space after deceleration (larger means faster loss decay)
    - These three parameters fully describe the loss improvements brought by scaling.

2. **Formalizing Zero-Sum Learning (ZSL)**:

    - Destructive interference measure: $D(\Delta\ell) = 1 - \frac{|\sum_i \Delta\ell_i|}{\sum_i |\Delta\ell_i|}$, ranging from 0 to 1, where higher values indicate more cancellation of loss changes among tokens
    - Gradient destructive interference: $\vec{D}(\nabla_\theta \ell) = 1 - \frac{|\sum_i \nabla_\theta \ell_i|}{\sum_i |\nabla_\theta \ell_i|}$, averaged per-parameter
    - Key decomposition: $|\Delta L| = M(\Delta\ell) \cdot (1 - D(\Delta\ell))$, where $M$ is the average magnitude of token-level loss changes

3. **Quantifying ZSL Contribution to Deceleration**:

    - An increase in $D(\Delta\ell)$ from 0.5 to 0.95 leads to a 10$\times$ reduction in loss improvement
    - A decrease in $M(\Delta\ell)$ from 0.75 to 0.5 leads to only a 1.5$\times$ reduction in loss improvement
    - Conclusion: ZSL (the $D$ term) dominates deceleration rather than the reduction of token-level loss magnitude (the $M$ term)

4. **Gradient Opposition as the Root Cause of ZSL**:

    - Under the first-order training dynamics assumption, $D(\tilde{\Delta}\ell)$ originates from the opposition of per-token gradient projections in the update direction
    - Experimental validation: Gradient interference rises sharply close to 1.0 on the eve of deceleration

## Key Experimental Results

### Main Results (Table 1: Loss Deceleration Measurements)

| Model | $\downarrow L_d$ | $\downarrow t_d$ | $\uparrow r_d$ | $\hat{L}_T$ | $L_T$ |
|-------|--------|--------|--------|---------|-------|
| 14M   | 4.05   | 5900   | 0.013  | 3.86    | 3.88  |
| 37M   | 3.60   | 5900   | 0.016  | 3.39    | 3.40  |
| 78M   | 3.38   | 5900   | 0.020  | 3.14    | 3.15  |
| 144M  | 3.25   | 6000   | 0.023  | 2.98    | 2.99  |
| 285M  | 3.14   | 5300   | 0.025  | 2.85    | 2.87  |
| 472M  | 3.16   | 4600   | 0.035  | 2.77    | 2.80  |
| OLMo-1B | 2.86 | 3700  | 0.034  | 2.39    | 2.40  |
| OLMo-7B | 2.64 | 4600  | 0.053  | 2.04    | 2.03  |

- The error of $\hat{L}_T$ compared to $L_T$ is within 1%, validating the piecewise linear model's effectiveness
- $L_d$ and $r_d$ monotonically improve as model scale increases
- 14M $\to$ 7B: $r_d$ increases from 0.013 to 0.053 (4$\times$ increase), and $L_d$ decreases from 4.05 to 2.64

### Ablation Study

- **D(Δℓ) vs M(Δℓ)** (Fig. 5): $D$ going from 0.5 $\to$ 0.95 accounts for a 10$\times$ decrease in loss improvement, whereas $M$ accounts for only 1.5$\times$ $\to$ ZSL is the primary cause.
- **Temporal Dynamics of Gradient Interference** (Fig. 3-4): $D(\Delta\ell)$ rises sharply immediately before deceleration. Gradient interference $D(\nabla\ell)$ starts very high (>0.9) early in training and approaches 1.0 near deceleration.
- **Architecture/Data/Optimizer Ablations** (Appendix C): Deceleration and ZSL are consistently observed across different architectures (GPT, Llama), datasets (C4, Dolma), and optimizers (Adam, SGD), showing that it is a universal phenomenon.

### Key Findings
- Deceleration is a universal, qualitative phenomenon rather than noise or an artifact of specific settings.
- ZSL is the main driver of deceleration, rather than the decay of token-level loss magnitude.
- Scaling up improves loss by mitigating ZSL (either by lowering the peak of $D(\Delta\ell)$ or delaying its rise).

## Highlights & Insights
- **Interpretable Scaling Law Parameterization**: $\hat{L}_T = L_d (t_d/T)^{r_d}$ breaks down the opaque power law into three physically meaningful quantities, offering deeper insights than traditional Chinchilla-style fits.
- **Bridge from Micro to Macro**: Establishes a complete causal chain from per-token gradient opposition to macroscopic loss deceleration.
- **Actionability**: ZSL targets a specific concrete objective—reducing destructive interference among per-token gradients, which could potentially be addressed via curriculum learning, gradient surgery, or data mixing strategies.

## Limitations & Future Work
1. Concrete methods to mitigate ZSL (such as gradient surgery) have not been experimentally implemented in this study; they represent "potential future directions."
2. Analysis is based on full-batch gradients. In practice, training leverages mini-batch SGD, which requires proxy approximations for measuring ZSL.
3. The experimental scale reaches up to 472M parameters (self-trained) + OLMo 7B (pretrained checkpoints). Whether new phenomena emerge at larger scales (70B+) remains unknown.
4. The validity of the BNSL first-order approximation (piecewise linearity) over significantly longer training runs requires further verification.

## Related Work & Insights
- Complementary to Kaplan et al. (2020) scaling laws: While the latter describes the relationship between final loss and scale, this work characterizes the behavior of loss *during training* in relation to scale.
- ZSL shares conceptual similarities with gradient conflict in multi-task learning (Liu et al., 2021) but highlights token-level conflict within a *single task* (language modeling).
- Inspiration: Monitoring $D(\Delta\ell)$ dynamically during training could enable adaptive adjustments of learning rates, data-mixing ratios, or model capacity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to systematically identify and formalize loss deceleration + ZSL)
- Theoretical Depth: ⭐⭐⭐⭐⭐ (Complete formalization and causal verification chain)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Broad coverage of scales, but lacks intervention experiments)
- Value: ⭐⭐⭐⭐ (Points out promising directions but hasn't fully materialized practical solutions yet)
- Overall Recommendation: ⭐⭐⭐⭐⭐ (An important theoretical contribution to the field of scaling laws)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](../../ICLR2026/llm_pretraining/scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](../../ICLR2026/llm_pretraining/pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](../../ICML2026/llm_pretraining/scaling_depth_capacity_via_zeroone-layer_model_expansion.md)

</div>

<!-- RELATED:END -->
