---
title: >-
  [Paper Note] Activation Steering for Masked Diffusion Language Models
description: >-
  [ICLR 2026][Image Restoration][activation steering] This work is the first to apply activation steering to Masked Diffusion Language Models (MDLMs)…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "activation steering"
  - "masked diffusion LM"
  - "safety"
  - "refusal direction"
  - "LLaDA"
date: 2026-05-08
content_hash: 1fcfc34c7b6a1d6b
---

# Activation Steering for Masked Diffusion Language Models

**Conference**: ICLR 2026
**arXiv**: [2512.24143](https://arxiv.org/abs/2512.24143)  
**Code**: Available  
**Area**: Image Restoration
**Keywords**: activation steering, masked diffusion LM, safety, refusal direction, LLaDA

## TL;DR
This work is the first to apply activation steering to Masked Diffusion Language Models (MDLMs), demonstrating that refusal behavior in MDLMs is likewise governed by a single low-dimensional direction. Globally projecting out this direction at every denoising step completely bypasses safety alignment. Unlike autoregressive models, effective directions can be extracted from pre-instruction tokens—reflecting the non-causal, parallel processing nature of diffusion models.

## Background & Motivation
**Background**: Activation steering has been extensively studied in autoregressive LLMs (e.g., refusal direction elimination by Arditi et al.), yet MDLMs such as LLaDA operate under a fundamentally different generation mechanism—iterative unmasking rather than token-by-token generation. Whether analogous low-dimensional control directions exist in MDLMs has remained entirely unknown.

**Limitations of Prior Work**: (1) Inference-time control in MDLMs is limited to sampling-level guidance (e.g., DIJA); representation-level control is absent. (2) Autoregressive jailbreak attacks such as GCG and PAIR transfer poorly to diffusion models due to architectural and generative differences.

**Key Challenge**: MDLMs employ non-causal attention (all tokens attend to one another), in stark contrast to the causal attention of autoregressive models—where only the last token has access to the full input, whereas in MDLMs every token does. The choice of position and timestep for activation steering must therefore be reconsidered from first principles.

**Goal**: Do MDLMs possess a refusal direction? At which denoising stage, which layers, and which token positions is steering most effective?

**Key Insight**: The paper directly adapts the contrastive direction extraction methodology of Arditi et al. to MDLMs, while additionally exploring MDLM-specific token positions (including pre-instruction tokens) and denoising timesteps.

**Core Idea**: Transplant the activation steering primitives of autoregressive LLMs to MDLMs, thereby revealing representation properties unique to the diffusion paradigm.

## Method

### Overall Architecture
Three stages: (1) extract candidate steering directions from contrastive prompt pairs (harmful vs. benign) as normalized mean-difference vectors; (2) select the optimal (layer, token position) combination on a validation set; (3) at inference time, project out the selected direction at every denoising step, across all layers and all token positions.

### Key Designs

1. **Direction Extraction**:

    - Function: Extract steering directions from the activation differences of 128 harmful and 128 benign prompts.
    - Mechanism: $v_i^{(\ell)} = \text{normalize}(\mu_{+,i}^{(\ell)} - \mu_{-,i}^{(\ell)})$, computing one direction per (layer $\ell$, token position $i$) pair, with the best candidate selected by sweeping all combinations on the validation set.
    - Key Finding: Effective directions can be extracted not only from post-instruction tokens (as in autoregressive models) but also from **pre-instruction tokens**—because MDLM's non-causal attention ensures every token encodes the full input context.

2. **Direction Application (Projection)**:

    - Function: At every reverse diffusion step, project all activations onto the orthogonal complement of the steering direction.
    - Formula: $\tilde{h}_i^{(\ell)} = h_i^{(\ell)} - \langle h_i^{(\ell)}, v \rangle v$
    - Global application: applied across all layers × all token positions × all denoising steps.
    - The diffusion sampling procedure itself is not modified.

3. **Diffusion-Specific Ablation Findings**:

    - Early denoising steps contribute disproportionately; the first diffusion block has the largest individual impact.
    - Middle-to-late transformer layers are most effective.
    - Sensitivity heatmaps reveal highly consistent patterns between LLaDA-8B and LLaDA-1.5.
    - MMaDA exhibits a distinct pattern (broader degradation without clear localization).

### Loss & Training
No training is required (training-free). Direction extraction requires only a single forward pass; global projection is applied at inference time.

## Key Experimental Results

### Main Results (JailbreakBench, 100 harmful instructions)

| Method | LLaDA-8B Refusal↓ | LLaDA-8B Safety↓ | LLaDA-1.5 Refusal↓ |
|--------|-------------------|-------------------|---------------------|
| Direct | ~98% | ~100% | ~98% |
| GCG (suffix optimization) | ~95% | ~98% | - |
| PAIR (automated jailbreak) | ~70% | ~85% | - |
| Slice (prefix initialization) | ~50% | ~65% | - |
| **Activation Steering (post)** | **0–16%** | **16–25%** | **~low** |
| **Activation Steering (pre)** | **~similar** | **~similar** | **~similar** |

### Ablation Study

| Ablation Dimension | Finding |
|--------------------|---------|
| Token position | Pre-instruction and post-instruction tokens are equally effective |
| Denoising timestep | Early steps are most influential (first block contributes most) |
| Transformer layer | Middle-to-late layers are most sensitive |
| Cross-lingual transfer | Strong transfer between English and Chinese |
| Cross-architecture transfer | MDLM→AR transfer fails |

### Key Findings
- **Refusal behavior in MDLMs is governed by a single low-dimensional direction**—a phenomenon observed in autoregressive LLMs now reproduced in a fundamentally different architecture.
- Pre-instruction tokens yield effective steering directions—impossible in autoregressive models due to causal attention constraints—highlighting the non-causal nature of diffusion models.
- GCG is nearly ineffective against MDLMs, underscoring the need for attack methods designed specifically for the diffusion paradigm.
- Cross-architecture transfer failure indicates that safety representations are architecture-dependent: the same concept is encoded differently in AR and MDLM frameworks.
- Early denoising steps are most critical—consistent with findings in A2D and suggestive of a "shallow alignment" problem in diffusion models.

## Highlights & Insights
- **Cross-architecture comparison of safety representations**: The "refusal" concept is encoded differently in AR and MDLM models, yet is shared across English and Chinese within the same MDLM—indicating that architecture shapes representational structure more than language does.
- **Safety implications of non-causal attention**: Because every token in an MDLM attends to the full input, safety directions are extractable from any position—simultaneously expanding the attack surface.
- **Complementarity with A2D**: A2D addresses alignment defensively via token-level [EOS] steering; this work attacks from the representation level via activation steering. Together, they delineate a new frontier in MDLM safety research.

## Limitations & Future Work
- Safety refusal is the sole case study; other control objectives (e.g., toxicity, style) remain unvalidated.
- Steering patterns for MMaDA differ from those for LLaDA, suggesting the method may not generalize universally.
- Global projection may degrade MDLM performance on benign tasks; utility evaluation is insufficient.
- Direction extraction requires a dataset of harmful prompts, which may be a practical constraint in deployment settings.

## Related Work & Insights
- **vs. Arditi et al. (refusal directions in AR LLMs)**: This work ports their methodology to MDLMs, recovering analogous phenomena while revealing diffusion-specific structural differences.
- **vs. A2D / DIJA**: A2D targets defense and DIJA targets attack, both operating at the token or sampling level; this work operates at the representation level and is more lightweight.
- **vs. AlphaSteer**: AlphaSteer employs null-space constraints for precise steering in AR LLMs; this work exposes an analogous low-dimensional control structure within MDLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of activation steering to MDLMs, revealing diffusion-specific representational properties.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three MDLMs evaluated with cross-architecture transfer and detailed ablations (layer / token / timestep).
- Writing Quality: ⭐⭐⭐⭐ Clear methodology with rich ablation visualizations.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for MDLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](../../ICML2026/image_restoration/plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/image_restoration/consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](../../ICML2026/image_restoration/early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)

</div>

<!-- RELATED:END -->
