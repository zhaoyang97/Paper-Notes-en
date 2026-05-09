---
title: >-
  [Paper Note] Guiding Diffusion Models with Semantically Degraded Conditions (CDG)
description: >-
  [CVPR 2026][Image Generation][Diffusion Model Guidance] Condition-Degradation Guidance (CDG) replaces the null prompt $\emptyset$ in CFG with a semantically degraded condition $\boldsymbol{c}_{\text{deg}}$, transforming the guidance from a "good vs. empty" comparison to a refined "good vs. almost good" contrast. This significantly improves the compositional generation precision of diffusion models without requiring any training.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Model Guidance
  - Condition Degradation
  - Text-to-Image
  - Compositional Generation
  - Attention Analysis
date: 2026-05-08
content_hash: 5bccd069ef3091e8
---

# Guiding Diffusion Models with Semantically Degraded Conditions (CDG)

**Conference**: CVPR 2026  
**arXiv**: [2603.10780](https://arxiv.org/abs/2603.10780)  
**Code**: [GitHub](https://github.com/Ming-321/Classifier-Degradation-Guidance)  
**Area**: Image Generation  
**Keywords**: Diffusion Model Guidance, Condition Degradation, Text-to-Image, Compositional Generation, Attention Analysis  

## TL;DR

Condition-Degradation Guidance (CDG) replaces the null prompt $\emptyset$ in CFG with a semantically degraded condition $\boldsymbol{c}_{\text{deg}}$, transforming the guidance from a "good vs. empty" comparison to a refined "good vs. almost good" contrast. This significantly improves the compositional generation precision of diffusion models without requiring any training.

## Background & Motivation

Classifier-Free Guidance (CFG) is a cornerstone of modern text-to-image diffusion models, enhancing generation quality by extrapolating between unconditional and conditional predictions. However, the core issue of CFG lies in its reliance on the **semantically void null prompt** $\emptyset$:

1. **Geometric Entanglement**: The semantic distance between condition $\boldsymbol{c}$ and $\emptyset$ is too large, causing the guidance signal to mix content generation with style/structure, producing entangled gradient signals.
2. **Compositional Failure**: CFG often fails in complex tasks, including text rendering errors, spatial relationship confusion, and imprecise attribute binding.
3. **Limitations of Prior Work**: Process correction methods (e.g., APG, TCFG) still retain the $\boldsymbol{c}$ vs. $\emptyset$ contrast framework and only perform post-hoc corrections; negative sample reconstruction methods are either semantically blind (random noise) or rely on expensive external models (VLM-generated negative samples), failing to utilize the intrinsic semantic structure of the prompt's own token embeddings.

The authors' Core Idea is: if one replaces $\emptyset$ with a semantically close degraded condition $\boldsymbol{c}_{\text{deg}}$, a **common-mode rejection effect** can be achieved—the normal components shared by two semantic neighbors are automatically canceled out during subtraction, leaving only the pure semantic correction signal.

## Method

### Overall Architecture

The guidance formula for CDG is:

$$D_\theta^{\text{CDG}}(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}) = D_\theta(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}) + (w-1)(D_\theta(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}) - D_\theta(\boldsymbol{x}_\sigma; \sigma, \boldsymbol{c}_{\text{deg}}))$$

Mechanism:
1. Extract token importance (Weighted PageRank) from the self-attention maps of the text encoder.
2. Categorize tokens into **content tokens** (encoding object semantics) and **context aggregation tokens** (encoding global context) based on importance.
3. Construct the degraded condition $\boldsymbol{c}_{\text{deg}}$ via a stratified degradation strategy.
4. Use $\boldsymbol{c}_{\text{deg}}$ instead of the null prompt for guidance in CFG.

### Key Designs

1. **Token Functional Dichotomy & Weighted PageRank Analysis**: In transformer text encoders, tokens naturally split into two categories—content tokens (e.g., "minecraft", "cooking") carry fine-grained semantics, while context aggregation tokens (padding and special tokens) absorb global context via attention. The authors model the self-attention matrix as a graph and use the WPR algorithm to calculate importance scores for each token, validating this dichotomy where content tokens have significantly higher importance.

2. **Stratified Degradation**: A unified degradation ratio $R_{\text{deg}} \in [0,2]$ is introduced, mapped to degradation ratios for the two token types via $r_{\text{content}} = \min(R_{\text{deg}}, 1.0)$ and $r_{\text{CtxAgg}} = \max(R_{\text{deg}}-1.0, 0)$. This ensures content tokens are degraded before context aggregation tokens. $R_{\text{deg}}=1.0$ acts as a natural "semantic boundary": the $[0,1]$ interval removes fine-grained semantics, while $(1,2]$ removes coarse global semantics. Setting $R_{\text{deg}}=1.0$ by default degrades all content tokens with near-zero overhead as no WPR calculation is needed.

3. **Masked Interpolation for Degraded Conditions**: A binary mask $\boldsymbol{m}$ is generated based on importance ranking, and masked interpolation is performed between the original and null conditions via $\boldsymbol{c}_{\text{deg}} = \boldsymbol{m} \odot \boldsymbol{c} + (1-\boldsymbol{m}) \odot \emptyset$. The degraded condition retains the global semantic skeleton (context aggregation tokens) while losing fine-grained semantic details (content tokens), achieving a precise "good vs. almost good" contrast.

### Loss & Training

CDG is a **training-free**, plug-and-play module:
- The mask $\boldsymbol{m}$ is calculated only once during the first denoising step and reused thereafter.
- An intervention block index $\lambda_{\text{block}}$ specifies which transformer block to extract attention maps from.
- Mask construction is triggered at $\lambda_{\text{block}}$, and all subsequent blocks use $\boldsymbol{c}_{\text{deg}}$.
- No external models or additional training are required.

### Geometric Analysis

Starting from the manifold hypothesis, the authors propose two metrics to explain the superiority of CDG:
- **Geometric Decoupling**: Measures the orthogonality between the guidance signal and the main denoising subspace; CDG maintains near-perfect orthogonality throughout.
- **Interference Energy Ratio**: Measures the energy ratio of the guidance signal projected onto the denoising subspace; CDG shows minimal interference.

The common-mode rejection effect of CDG cancels out the normal components shared by $\boldsymbol{c}$ and $\boldsymbol{c}_{\text{deg}}$, leaving only pure semantic correction signals.

## Key Experimental Results

### Main Results

| Model | Method | FID↓ | CLIP Score↑ | Aesthetic↑ | VQA Score↑ |
|------|------|------|-------------|------------|------------|
| SD3 | CFG | 35.69 | 31.73 | 5.66 | 91.44 |
| SD3 | **Ours** | **34.05** | **32.00** | **5.70** | **92.40** |
| SD3.5 | CFG | 34.56 | 31.85 | 6.21 | 91.94 |
| SD3.5 | **Ours** | **33.07** | **31.96** | **6.26** | **92.61** |
| FLUX.1 | CFG | 38.55 | 31.20 | 6.06 | 90.31 |
| FLUX.1 | **Ours** | **37.11** | **31.21** | **6.15** | **90.62** |
| Qwen | CFG | 42.45 | 32.11 | 2.57 | 93.66 |
| Qwen | **Ours** | **39.02** | **32.31** | 2.54 | **93.93** |

GenAI-Bench compositional reasoning (SD3.5): CDG achieves a +3.64 gain in Differentiation and +2.36 in Comparison.

### Ablation Study

| Importance Ranking | Stratified Degradation | FID↓ | VQA Score↑ |
|-----------|---------|------|------------|
| WPR | ✓ | 33.89 | 92.21 |
| Random | ✓ | 34.17 | 92.27 |
| WPR | ✗ | 35.06 | 86.31 |
| Reverse WPR | ✗ | 50.73 | 80.10 |
| Random | ✗ | 47.02 | 83.55 |

**Stratified degradation is the primary driver of performance**: The two stratified variants (first two rows) significantly outperform all non-stratified variants (last three rows), with VQA scores improving by 5.9-12.2 points.

### Key Findings

1. **Stratified degradation is more important than WPR ranking**: Within the stratified framework, WPR performs similarly to random ranking, but WPR provides the theoretical foundation and the basis for the $R_{\text{deg}}=1.0$ boundary.
2. **Smaller gains on FLUX**: FLUX utilizes guidance distillation, reducing its reliance on inference-time guidance.
3. **CFG* validation experiments** confirm the content/context aggregation token dichotomy—removing content tokens leads to a sharp drop in CLIP Score, while removing context aggregation tokens has a milder impact.
4. **Extremely high computational efficiency**: The one-time calculation strategy adds only 3.6% overhead; with the default $R_{\text{deg}}=1.0$, the overhead is nearly zero.

## Highlights & Insights

- Reveals the functional dichotomy of content vs. context aggregation tokens in transformer text encoders, which is a fundamental property of transformer encoders rather than a specific architecture feature.
- The "good vs. almost good" guidance paradigm is geometrically superior to "good vs. empty"—the guidance signal is orthogonal to the denoising direction, preventing energy waste.
- Elegantly explains the working mechanism of CDG through an analogy to the common-mode rejection effect.
- Plug-and-play, zero training, and near-zero overhead provide extremely high practical value.

## Limitations & Future Work

- CDG shows smaller improvements on models already using guidance distillation (e.g., FLUX).
- While $R_{\text{deg}}=1.0$ works well by default, different tasks or styles might require fine-tuning.
- Currently only validated in text-to-image scenarios; other modalities like video generation remain to be explored.
- While WPR analysis provides theoretical insight, the necessity of analysis tools is debatable given that stratified degradation itself is the key in practice.

## Related Work & Insights

- **APG/TCFG**: Geometric corrections within the CFG framework that treat symptoms rather than root causes.
- **PAG/SEG**: Perturbing attention/energy curvature at the internal mechanism level, which is orthogonal and complementary to CDG.
- **Autoguidance**: Uses weak models to provide negative signals, requiring additional model tuning.
- **Insights**: The logic of CDG can be extended to other conditional generation frameworks—constructing adaptive, semantically-aware negative samples is a key principle for achieving precise semantic control.

## Rating

- **Novelty**: 8/10 — The approach of constructing degraded conditions based on token functional dichotomy is novel, supported by deep geometric analysis.
- **Experimental Thoroughness**: 9/10 — Tests across four models, multiple benchmarks, and comprehensive ablations and mechanism validations.
- **Writing Quality**: 9/10 — Clear logic, tightly integrating geometric analysis with experimental observations.
- **Value**: 8/10 — A practical, plug-and-play solution that provides a new principled framework for diffusion model guidance design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)
- [\[CVPR 2026\] Guiding a Diffusion Model by Swapping Its Tokens](guiding_a_diffusion_model_by_swapping_its_tokens.md)
- [\[CVPR 2026\] Guiding a Diffusion Transformer with the Internal Dynamics of Itself](guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)

</div>

<!-- RELATED:END -->
