---
title: >-
  [Paper Note] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding
description: >-
  [CVPR 2026][Image Restoration][Visual token resolution] This paper proposes Blink, a framework that dynamically expands and discards visual tokens across different Transformer layers of an MLLM — simulating the human "ra…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Visual token resolution"
  - "dynamic attention"
  - "multimodal large language models"
  - "saliency guidance"
  - "token super-resolution"
date: 2026-05-08
content_hash: 7d46fd2374bc2bb1
---

# Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding

**Conference**: CVPR 2026
**arXiv**: [2512.10548](https://arxiv.org/abs/2512.10548)
**Code**: N/A
**Area**: Multimodal Large Language Models / Image Restoration (Visual Perception Enhancement)
**Keywords**: Visual token resolution, dynamic attention, multimodal large language models, saliency guidance, token super-resolution

## TL;DR
This paper proposes Blink, a framework that dynamically expands and discards visual tokens across different Transformer layers of an MLLM — simulating the human "rapid blinking" scanning process — to adaptively enhance visual perception within a single forward pass, improving LLaVA-1.5 performance across multiple multimodal benchmarks.

## Background & Motivation
**Background**: Multimodal large language models (MLLMs) have achieved remarkable progress on vision-language tasks (e.g., LLaVA, Qwen-VL), yet their visual perception capability remains insufficient and prone to hallucinations.

**Limitations of Prior Work**: Existing MLLMs process visual inputs using conventional LLM architectures, lacking explicit exploitation of salient visual regions. Post-processing methods (e.g., identifying salient regions followed by cropping and re-inference) are computationally inefficient and can only focus on a single region at a time.

**Key Challenge**: Humans perceive visual scenes through a dynamic "scan–focus–shift" process, whereas MLLMs treat all visual tokens uniformly and cannot simulate cross-layer attention shifts.

**Goal**: How can an MLLM's visual perception capability be dynamically enhanced within a single forward pass?

**Key Insight**: A pilot study first uncovers two key insights — (a) different layers attend to different visual regions, and (b) allocating additional computation to high-attention tokens improves perception — which then motivate the design of the dynamic framework.

**Core Idea**: Leveraging the non-uniform distribution of attention maps, the framework dynamically decides at each layer whether to expand (via super-resolution enhancement) or discard visual tokens, thereby simulating the human "scan–focus–shift" cognitive process.

## Method

### Overall Architecture
Standard MLLM forward pass → At selected layers: compute saliency map → determine whether expansion/discard thresholds are exceeded → if so, apply the TokenSR module to expand tokens in salient regions → discard expanded tokens in subsequent layers if attention shifts → final output.

### Key Designs

1. **Saliency-Guided Scanning (SGS)**:

    - At each participating layer $L$, the attention of the last text token over all visual tokens is computed as:
    $S_v^{(L)} = q_{t_n}^{(L)} (k_v^{(L)})^\top$
    - Visual tokens are reshaped into an $H \times W$ grid and divided into $p \times p$ patches; the aggregated saliency of each patch is then computed.
    - A saliency ratio is defined as $\rho^{(L)} = \frac{\mathcal{S}_{r_{\max}}^{(L)}}{\sum_i \mathcal{S}_{r_i}^{(L)}}$, reflecting the degree of attention concentration.
    - **Design Motivation**: The pilot study reveals substantial variation in attention distributions across layers; high concentration indicates that the model is "confident" in attending to a specific region, making that region a suitable candidate for enhancement.

2. **Dynamic Token Resolution (DTR)**:

    - **Token Expansion**: When $\rho^{(L)} > \tau_{\text{exp}}$, the TokenSR module performs super-resolution enhancement on the salient patch:
    $hs_{SR}^{(L)} = \text{TokenSR}^{(L)}(hs_{LR}^{(L)})$
      The enhanced tokens are then inserted into the sequence: $[hs_s; hs_v; hs_{SR}; hs_t]$
    - **Token Discarding**: When $\rho^{(L)} < \tau_{\text{drop}}$, previously expanded tokens are removed and the original sequence is restored.
    - **Design Motivation**: Expansion increases computational investment in salient regions, while discarding prevents low-information tokens from interfering with subsequent reasoning.

3. **Token Super-Resolution Module (TokenSR)**:

    - A lightweight module consisting of three 2D convolutional layers with ReLU activations.
    - During training, salient-region tokens from the full image are upscaled, with tokens from the corresponding cropped image serving as reference; the training objective minimizes KL divergence between the two.
    - The MLLM backbone is frozen; only TokenSR parameters are trained.
    - **Design Motivation**: Inspired by image super-resolution, a lightweight network recovers fine-grained details from low-resolution tokens while preserving semantic consistency.

### Loss & Training
- TokenSR training: minimizes KL divergence between enhanced tokens and cropped reference tokens.
- Training data: LLaVA-1.5 training set (COCO + GQA + OCR-VQA + TextVQA + VisualGenome).
- All operations are performed prior to layer normalization, ensuring the Transformer correctly handles expanded or pruned sequences.

## Key Experimental Results

### Main Results (LLaVA-1.5-7B)

| Benchmark | Vanilla | Blink-interp | Blink | Gain |
|-----------|---------|-------------|-------|------|
| MME Perception | 1505.72 | 1514.08 | **1519.74** | +14.02 |
| MME Cognition | 357.86 | 353.21 | **361.79** | +3.93 |
| GQA | 61.93 | 61.93 | **61.98** | +0.05 |
| MMBench | 64.60 | **64.69** | **64.69** | +0.09 |
| MMBench-CN | 58.08 | 58.51 | **58.59** | +0.51 |
| POPE | 85.17 | 85.17 | **85.23** | +0.06 |
| ScienceQA | 69.46 | 69.51 | **69.66** | +0.20 |
| MM-Vet | 32.20 | 31.70 | **33.40** | +1.20 |

### Ablation Study

| Configuration | MME Total | Change | Notes |
|---------------|-----------|--------|-------|
| Blink (full) | **1881.53** | — | Best |
| w/o SGS (random selection) | 1879.38 | -2.15 | Saliency guidance is necessary |
| w/o DTR (fixed period) | 1840.46 | -41.07 | Dynamic resolution adjustment is critical |
| w/o Drop | 1884.03 | +2.50 | Omitting discard yields a marginal gain under Blink |
| High $\tau_{\text{exp}}$ | 1865.54 | -15.99 | Excessively high threshold limits effective expansion |

### Key Findings
- Removing the DTR module causes the largest performance drop (−41.07), confirming it as the core component of the framework.
- Blink-interp (training-free interpolation) also improves MME Perception by 8.36 points, demonstrating the intrinsic value of the dynamic inference pipeline.
- The full Blink model consistently matches or outperforms the baseline across all benchmarks.
- The selected layer range (layers 12–18) corresponds to the "correct-attention intermediate layer" interval identified in the pilot study.

## Highlights & Insights
- The two findings from the pilot study (cross-layer attention shifts + efficacy of increased computation on salient tokens) provide a solid empirical foundation for the method design.
- The "dynamic scan–focus" mechanism elegantly simulates the human visual cognition process.
- The plug-and-play design requires training only the lightweight TokenSR module, with the backbone entirely frozen.
- The Blink-interp variant demonstrates that gains can be achieved through the inference pipeline alone, even without training.

## Limitations & Future Work
- The absolute performance gains are modest (MME total score +17.95), though the direction is promising.
- Validation is limited to LLaVA-1.5-7B; evaluation on larger models and more recent architectures remains to be conducted.
- The thresholds $\tau_{\text{exp}}$ and $\tau_{\text{drop}}$ require manual tuning; adaptive learning of these values warrants exploration.
- Currently, only one salient patch is selected per layer; scenarios with multiple salient regions may require an extended design.

## Related Work & Insights
- Post-processing zoom methods (e.g., LLaVA-HR) require multiple forward passes and are computationally inefficient.
- Visual token pruning approaches (e.g., FastV, LLaVA-PruMerge) represent a complementary direction — Blink focuses on "enhancing the important" rather than "removing the unimportant."
- Insight: The internal attention distributions of MLLMs contain rich visual perception signals that merit further investigation.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of dynamic token resolution adjustment is novel, and the pilot study provides strong motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven benchmarks, detailed ablation studies, and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from empirical findings to method design is clearly articulated.
- Value: ⭐⭐⭐⭐ Introduces a new perspective for enhancing MLLM visual perception in a plug-and-play manner.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[ACL 2026\] Diffusion-CAM: Faithful Visual Explanations for dMLLMs](../../ACL2026/image_restoration/diffusion-cam_faithful_visual_explanations_for_dmllms.md)
- [\[AAAI 2026\] TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis](../../AAAI2026/image_restoration/tmdc_a_two-stage_modality_denoising_and_complementation_framework_for_multimodal.md)

</div>

<!-- RELATED:END -->
