---
title: >-
  [Paper Note] Guiding Diffusion Models with Semantically Degraded Conditions
description: >-
  [CVPR 2026][Image Generation][Classifier-Free Guidance] This paper proposes Condition-Degradation Guidance (CDG), which replaces the null prompt $\emptyset$ in CFG with a semantically degraded condition $\boldsymbol{c}_{\text{deg}}$, transforming the guidance paradigm from a coarse-grained "good vs. empty" contrast to a fine-grained "good vs. slightly worse" contrast. Through a stratified degradation strategy—first degrading content tokens, then context-aggregating tokens—CDG constructs adaptive negative samples and achieves plug-and-play improvements in compositional generation accuracy on models including SD3, FLUX, and Qwen-Image, with negligible additional overhead.
tags:
  - CVPR 2026
  - Image Generation
  - Classifier-Free Guidance
  - Condition Degradation Guidance
  - Text-to-Image
  - Diffusion Models
  - Compositional Generation
date: 2026-05-08
content_hash: 840aafda2bbe6220
---

# Guiding Diffusion Models with Semantically Degraded Conditions

**Conference**: CVPR 2026
**arXiv**: [2603.10780](https://arxiv.org/abs/2603.10780)
**Code**: [Ming-321/Classifier-Degradation-Guidance](https://github.com/Ming-321/Classifier-Degradation-Guidance)
**Area**: Image Generation
**Keywords**: Classifier-Free Guidance, Condition Degradation Guidance, Text-to-Image, Diffusion Models, Compositional Generation

## TL;DR

This paper proposes Condition-Degradation Guidance (CDG), which replaces the null prompt $\emptyset$ in CFG with a semantically degraded condition $\boldsymbol{c}_{\text{deg}}$, transforming the guidance paradigm from a coarse-grained "good vs. empty" contrast to a fine-grained "good vs. slightly worse" contrast. Through a stratified degradation strategy—first degrading content tokens, then context-aggregating tokens—CDG constructs adaptive negative samples and achieves plug-and-play improvements in compositional generation accuracy on models including SD3, FLUX, and Qwen-Image, with negligible additional overhead.

## Background & Motivation

**Central Role and Limitations of CFG**: Classifier-Free Guidance (CFG) is a cornerstone of modern text-to-image models, yet its reliance on semantically vacuous null prompts $\emptyset$ yields suboptimal performance on complex compositional tasks such as text rendering, attribute binding, and spatial reasoning.

**Geometric Entanglement of Guidance Signals**: The large semantic gap between $\boldsymbol{c}$ and $\emptyset$ introduces interference components along the principal denoising directions, entangling content generation with style and structural information.

**Limitations of Existing Approaches**: Process-correction methods (APG, TCFG) retain the $\boldsymbol{c}$ vs. $\emptyset$ formulation and apply post-hoc corrections, addressing symptoms rather than root causes. Negative-sample construction methods—such as weaker models, random perturbations, or VLM-generated negatives—are either semantically blind or require auxiliary models.

**Key Intuition**: The semantically proximate contrast $\boldsymbol{c}$ vs. $\boldsymbol{c}_{\text{deg}}$ enables common-mode rejection—canceling shared denoising components and retaining a pure semantic correction signal.

**Functional Dichotomy of Tokens**: Tokens in Transformer-based text encoders naturally divide into **content tokens** (encoding object semantics) and **context-aggregating tokens** (padding and special tokens that absorb global context via attention), a structural property that can guide the degradation strategy design.

**Demand for Lightweight Plug-and-Play Solutions**: Practical deployment requires guidance improvements that are training-free, model-agnostic, and computationally negligible.

## Method

### Overall Architecture

CDG replaces the null prompt $\emptyset$ in the CFG formulation with a semantically degraded condition $\boldsymbol{c}_{\text{deg}}$:

$$D_\theta^{\text{CDG}}(\boldsymbol{x}_\sigma;\sigma,\boldsymbol{c}) = D_\theta(\boldsymbol{x}_\sigma;\sigma,\boldsymbol{c}) + (w-1)\big(D_\theta(\boldsymbol{x}_\sigma;\sigma,\boldsymbol{c}) - D_\theta(\boldsymbol{x}_\sigma;\sigma,\boldsymbol{c}_{\text{deg}})\big)$$

The pipeline for constructing $\boldsymbol{c}_{\text{deg}}$ proceeds as follows: ① extract self-attention maps from a designated Transformer block $\lambda_{\text{block}}$ → ② construct a graph and compute token importance via Weighted PageRank (WPR) → ③ generate a binary mask according to the stratified degradation strategy → ④ perform masked interpolation between the original and null conditions.

### Key Designs: Stratified Degradation

- **Token Functional Dichotomy**: WPR analysis reveals that content tokens (e.g., "minecraft," "cooking") attain substantially higher importance scores than context-aggregating tokens (padding and special tokens), validating the hypothesis that the two types encode semantics at different granularities.
- **Weighted PageRank Analysis**: The self-attention map is modeled as a directed graph with tokens as nodes and attention weights as edge weights. Token importance is determined by iterating $\boldsymbol{s}^{(k+1)} = \frac{A^T\boldsymbol{s}^{(k)}}{\|A^T\boldsymbol{s}^{(k)}\|_1}$ until convergence.
- **Unified Degradation Ratio $R_{\text{deg}} \in [0,2]$**: $r_{\text{content}} = \min(R_{\text{deg}}, 1.0)$ and $r_{\text{CtxAgg}} = \max(R_{\text{deg}}-1.0, 0)$. When $R_{\text{deg}} \le 1$, only content tokens (fine-grained semantics) are degraded; when $R_{\text{deg}} > 1$, context-aggregating tokens (coarse-grained semantics) are further degraded.
- **Default $R_{\text{deg}}=1.0$**: At this setting, all content tokens are degraded, WPR computation is skipped, and overhead is effectively zero.
- **Mask Construction and Reuse**: $\boldsymbol{c}_{\text{deg}} = \boldsymbol{m} \odot \boldsymbol{c} + (1-\boldsymbol{m}) \odot \emptyset$. The mask is computed once at the first denoising step and reused across all subsequent steps, making the overhead negligible.

### Geometric Interpretation and Theoretical Analysis

Under the manifold hypothesis, the authors use SVD to approximate the principal denoising subspace $\mathcal{S}_{\boldsymbol{c}}(t)$ from conditional predictions over MS-COCO prompts and define two metrics:

- **Geometric Decoupling** (orthogonality of the guidance signal to the principal denoising subspace): $\text{Decoupling}(\mathcal{S}_g, \mathcal{S}_c) = \frac{1}{k}\sum_{i=1}^k \sin^2(\theta_i)$, where values approaching 1 indicate near-perfect orthogonality. CDG maintains near-perfect orthogonality throughout generation, whereas CFG exhibits severe entanglement in early denoising stages.
- **Interference Energy Ratio** (fraction of guidance energy residing in the denoising subspace): $\text{Interference}(\Delta\boldsymbol{\varepsilon}) = \frac{\|P_{\mathcal{S}_c(t)}\Delta\boldsymbol{\varepsilon}\|_F^2}{\|\Delta\boldsymbol{\varepsilon}\|_F^2}$, where lower values indicate less interference. CDG exhibits minimal interference energy, while CFG wastes significant energy along misaligned directions.
- **Common-Mode Rejection**: Since $\boldsymbol{c}$ and $\boldsymbol{c}_{\text{deg}}$ are semantic neighbors sharing similar normal-direction components, the difference $\Delta\boldsymbol{\varepsilon}_{\text{CDG}} \propto \nabla_{z_t}\log\frac{p_t(z_t|\boldsymbol{c})}{p_t(z_t|\boldsymbol{c}_{\text{deg}})}$ naturally cancels shared components and retains a pure semantic correction signal—an effect that CFG cannot achieve due to the excessive semantic distance between $\boldsymbol{c}$ and $\emptyset$.

## Key Experimental Results

### Main Results (MS-COCO 2017 Validation Set)

| Model | Method | FID ↓ | CLIP Score ↑ | Aesthetic ↑ | VQA Score ↑ |
|-------|--------|-------|-------------|------------|------------|
| SD3 | CFG | 35.69 | 31.73 | 5.66 | 91.44 |
| SD3 | CDG | **34.05** | **32.00** | **5.70** | **92.40** |
| SD3 | CADS | 36.16 | 31.72 | 5.65 | 91.44 |
| SD3 | PAG | 50.60 | 30.15 | 5.52 | 81.27 |
| SD3.5 | CFG | 34.56 | 31.85 | 6.21 | 91.94 |
| SD3.5 | **CDG** | **33.07** | **31.96** | **6.26** | **92.61** |
| FLUX.1 | CFG | 38.55 | 31.20 | 6.06 | 90.31 |
| FLUX.1 | **CDG** | **37.11** | **31.21** | **6.15** | **90.62** |
| Qwen | CFG | 42.45 | 32.11 | 2.57 | 93.66 |
| Qwen | **CDG** | **39.02** | **32.31** | 2.54 | **93.93** |

### GenAI-Bench Compositional Reasoning

| Model | Method | Spatial ↑ | Comp ↑ | Differ ↑ | Univ ↑ |
|-------|--------|----------|--------|---------|--------|
| SD3.5 | CFG | 79.66 | 73.70 | 75.10 | 72.21 |
| SD3.5 | **CDG** | **80.69** | **76.06** | **78.74** | **73.13** |

CDG yields the largest gains on Differentiation (+3.64) and Comparison (+2.36), demonstrating that the "good vs. slightly worse" paradigm is most advantageous for tasks requiring fine-grained semantic discrimination. More modest improvements on FLUX.1 are consistent with its use of Guidance Distillation.

### Ablation Study

- **Stratified degradation is the primary driver**: The stratified variant achieves VQA scores 5.9–12.2 points higher and FID values 0.9–16.8 lower than the non-stratified baseline.
- **WPR is not essential but provides theoretical grounding**: Within the stratified framework, WPR-based ranking performs comparably to random ranking (FID 33.89 vs. 34.17); WPR primarily offers determinism and a principled justification for the $R_{\text{deg}}=1.0$ boundary.
- **Asymmetric response to $R_{\text{deg}}$**: Metrics change sharply in the $[0,1]$ interval (content token degradation) and more gradually in the $[1,2]$ interval (context-aggregating token degradation), corroborating the functional dichotomy hypothesis.
- **Ablation design**: Systematic comparisons are conducted across WPR ranking, random ranking, reversed ranking, stratified, and non-stratified configurations.
- **Computational efficiency**: Step-wise WPR incurs +47.2% overhead; one-time computation incurs +3.6%; the default $R_{\text{deg}}=1.0$ configuration is effectively zero-overhead (WPR is skipped).

### Key Findings

- Improvements on FLUX.1 are modest because its Guidance Distillation reduces reliance on inference-time guidance, further suggesting that CDG gains correlate positively with a model's dependence on the guidance signal.
- Qwen-Image uses `<|im_end|>` rather than padding as a context aggregator; CDG remains effective in this setting, validating the generalizability of the stratified degradation strategy across different token-type architectures.
- CDG achieves the largest gains on tasks requiring fine-grained semantic discrimination—Differentiation and Comparison—consistent with the design intent of the "good vs. slightly worse" contrast paradigm.
- CDG is orthogonal to methods such as PAG and is compatible with downstream applications including image-to-image generation and ControlNet.

## Highlights & Insights

- Reveals the functional dichotomy between content tokens and context-aggregating tokens in Transformer-based text encoders, providing a theoretical foundation for guidance signal design.
- Offers intuitive and quantifiable explanations for CDG's superiority over CFG through geometric analysis (Decoupling and Interference Energy metrics).
- Plug-and-play, training-free, and model-agnostic; near-zero overhead in the default configuration, making deployment highly practical.
- Consistent improvements across four architectures (SD3, SD3.5, FLUX.1, Qwen-Image) validate the generality of the approach.
- $R_{\text{deg}}$ provides an interpretable continuous control space: $[0,1]$ governs fine-grained semantics, $[1,2]$ governs coarse-grained context.
- Ablation design is carefully constructed; the CFG* experiment directly visualizes residual semantics in the degraded condition, enhancing method interpretability.
- Orthogonally composable with methods such as PAG, and supports extensions including image-to-image generation and ControlNet.

## Limitations & Future Work

- Gains are limited on models that already employ Guidance Distillation (e.g., FLUX.1), indicating reduced effectiveness when inference-time guidance dependence is low.
- The optimal value of $R_{\text{deg}}$ may require per-model tuning, although the default of 1.0 performs well in most cases.
- The method assumes a clear content/context-aggregating dichotomy within the text encoder; applicability to non-standard encoder architectures remains to be verified.
- Systematic evaluation on very long or highly complex prompts is absent.
- The CFG* validation experiment is primarily qualitative; more rigorous theoretical proofs establishing sufficient conditions for common-mode rejection are lacking.
- Validation is limited to Transformer-based diffusion models; applicability to UNet-based architectures is not discussed.

## Related Work & Insights

- **CFG framework improvements**: APG (geometric correction by projecting guidance signals onto a subspace orthogonal to the denoising direction) and TCFG (SVD decomposition of denoising signals) both retain the null prompt and apply post-hoc corrections without addressing the underlying semantic impoverishment.
- **Model-level negative samples**: Autoguidance employs a weaker model for negative signals; Weak-to-Strong Diffusion leverages a reflection mechanism. Both require maintaining auxiliary models, incurring high deployment costs.
- **Internal mechanism-level methods**: PAG (perturbing self-attention matrices) and SEG (smoothing energy curvature) operate on the model's computation flow rather than its inputs, and are orthogonally composable with CDG.
- **Input-level degradation**: ICG (random prompt substitution), CADS (unstructured Gaussian noise), SFG (spatially varying negative samples), and DNP (VLM-generated negative prompts) are either semantically blind or require expensive external models, and none exploits the intrinsic semantic structure of token embeddings within the prompt itself.
- **CDG's unique positioning**: CDG is the first method to leverage the functional dichotomy between content tokens and context-aggregating tokens in text encoders to realize adaptive semantic degradation at the input level, combining theoretical interpretability with practical lightweight design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The semantic degradation strategy grounded in the token functional dichotomy offers a fresh perspective; the paradigm shift from "good vs. empty" to "good vs. slightly worse" guidance is thought-provoking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four models, multiple metrics, GenAI-Bench compositional reasoning, comprehensive ablations, geometric analysis, and a cleverly designed CFG* validation experiment.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-articulated geometric intuition, and good alignment between theoretical derivations and experimental results.
- **Value**: ⭐⭐⭐⭐ — A practical plug-and-play solution with principled improvements over CFG, offering immediate applicability to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions (CDG)](guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)

</div>

<!-- RELATED:END -->
