---
title: >-
  [Paper Note] Diffusion-CAM: Faithful Visual Explanations for dMLLMs
description: >-
  [ACL 2026][Image Restoration][Diffusion Multimodal Model] Diffusion-CAM is the first interpretability method for diffusion-based multimodal LLMs (dMLLMs), extracting structurally valid intermediate representations from denoising trajectories with four post-processing modules (adaptive kernel denoising, distribution-aware confidence gating, contextual background decay, single-instance causal debiasing), significantly outperforming autoregressive CAM baselines on COCO Caption and GranDf.
tags:
  - ACL 2026
  - Image Restoration
  - Diffusion Multimodal Model
  - Class Activation Mapping
  - Visual Explanation
  - Explainable AI
  - Parallel Generation
content_hash: a3200e0dbefb57fa
---

# Diffusion-CAM: Faithful Visual Explanations for dMLLMs

**Conference**: ACL 2026
**arXiv**: [2604.11005](https://arxiv.org/abs/2604.11005)
**Code**: [GitHub](https://github.com/ZzzzzZhhmm/Diffusion-CAM)
**Area**: Image Restoration
**Keywords**: Diffusion Multimodal Model, Class Activation Mapping, Visual Explanation, Explainable AI, Parallel Generation

## TL;DR
Diffusion-CAM is the first interpretability method for diffusion-based multimodal LLMs (dMLLMs), extracting structurally valid intermediate representations from denoising trajectories with four post-processing modules (adaptive kernel denoising, distribution-aware confidence gating, contextual background decay, single-instance causal debiasing), significantly outperforming autoregressive CAM baselines on COCO Caption and GranDf.

## Background & Motivation

**Background**: Multimodal LLMs are transitioning from autoregressive architectures (LLaVA, Qwen-VL) to diffusion-based architectures (LaViDa, LLaDA-V, MMaDA). Diffusion models generate entire sentences through parallel mask denoising, improving generation speed and global coherence.

**Limitations of Prior Work**: (1) Existing CAM methods (LLaVA-CAM, TAM) rely on autoregressive models' sequential, attention-rich properties — dMLLMs lack explicit token-level attention weights and left-to-right causal structure; (2) Directly applying traditional CAM to dMLLMs produces diffuse, non-specific heatmaps; (3) Parallel denoising produces smooth, distributed activation patterns fundamentally different from autoregressive local sequential dependencies.

**Key Challenge**: The architectural advantages of dMLLMs (parallel generation, global planning) are precisely the obstacles for traditional interpretability tools — the latter assume sequential dependency while the former operates in parallel.

**Goal**: Design the first visual explanation method adapted for diffusion-based multimodal models.

**Key Insight**: Identify "structurally valid" intermediate steps in the denoising trajectory — where image-conditioned spatial information is preserved and can be linked to final predictions via gradients.

**Core Idea**: Extract gradient CAM from structurally valid steps of the denoising process + four diffusion-specific post-processing modules to address spatial noise, background diffusion, and redundant token correlations.

## Method

### Overall Architecture
Adapt CAM to dMLLMs: (1) register hooks at intermediate transformer blocks to extract features and gradients; (2) dynamically locate image token position boundaries; (3) backpropagate from final response scores to effective-step image region features to generate base CAM; (4) refine with four post-processing modules.

### Key Designs

1. **Diffusion CAM Adaptation (Three-Step Modification)**: Selects denoising steps satisfying feasibility conditions, dynamically locates image spans, and applies Grad-CAM aggregation. Design assumes no fixed image token positions, using universal feasibility criteria.

2. **Adaptive Kernel Denoising Module**: Suppresses high-frequency architectural artifacts from Transformer self-attention via dynamically scaled filter kernels considering denoising steps, spatial variance, and resolution.

3. **Distribution-Aware Confidence Gating + Contextual Background Decay + Single-Instance Causal Debiasing**: Respectively addresses high-variance activation artifacts, background residual signals, and abnormally high activations from repeated tokens.

## Key Experimental Results

### Main Results (COCO Caption + GranDf)

| Method | Localization Accuracy | Background Suppression | Visual Fidelity |
|--------|----------------------|----------------------|-----------------|
| LLaVA-CAM | Baseline | Weak | Weak |
| Grad-CAM (direct) | Poor | Poor | Poor |
| **Diffusion-CAM** | **SOTA** | **SOTA** | **SOTA** |

### Key Findings
- Directly applying autoregressive CAM to dMLLMs completely fails
- All four post-processing modules are complementary and indispensable
- Denoising step selection is crucial: only structurally valid steps yield meaningful visual attribution

## Highlights & Insights
- First reveals the fundamental challenge of dMLLM interpretability: parallel generation vs sequential dependency conflict
- The "structurally valid step" concept provides a general principle for attribution in non-autoregressive models

## Limitations & Future Work
- Validated only on LaViDa series; adaptability to other dMLLMs remains to be confirmed
- Four modules' hyperparameters require model-specific adjustment
- Computational overhead exceeds autoregressive CAM

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](../../CVPR2026/image_restoration/blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[AAAI 2026\] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](../../AAAI2026/image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
