---
title: >-
  [Paper Note] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping
description: >-
  [ICLR 2026][Multimodal VLM][MLLM] This paper proposes AttWarp, a plug-and-play test-time image warping method that leverages the MLLM's own cross-modal attention maps to perform rectilinear grid resampling — expanding hi…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "MLLM"
  - "image warping"
  - "attention-guided"
  - "fine-grained perception"
  - "test-time intervention"
date: 2026-05-08
content_hash: 58baab97ad56cc2e
---

# Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping

**Conference**: ICLR 2026
**arXiv**: [2510.09741](https://arxiv.org/abs/2510.09741)  
**Code**: [Project Page](https://dwipddalal.github.io/Attwarp/)  
**Area**: Multimodal VLM
**Keywords**: MLLM, image warping, attention-guided, fine-grained perception, test-time intervention

## TL;DR
This paper proposes AttWarp, a plug-and-play test-time image warping method that leverages the MLLM's own cross-modal attention maps to perform rectilinear grid resampling — expanding high-attention regions and compressing low-attention regions — achieving consistent accuracy improvements, enhanced compositional reasoning, and reduced hallucinations across 5 benchmarks and 4 MLLMs.

## Background & Motivation
**Background**: Multimodal large language models (MLLMs) such as LLaVA and Qwen-VL have made notable progress in image-based dialogue and reasoning, yet still exhibit significant deficiencies in fine-grained perception — missing small objects, confusing visually similar entities, and misinterpreting spatial relationships.

**Limitations of Prior Work**: Existing improvement methods either require external detectors (bounding boxes/masks), rely on multi-step reasoning chains, or employ cropping/masking strategies that discard global context.

**Key Challenge**: Small objects lose spatial detail during feature extraction, and subsequent attention-level improvements cannot recover this information; yet naive cropping or magnification discards global layout.

**Goal**: To enhance the resolution of query-relevant regions through spatial transformations at the input level, without modifying model weights or architecture.

**Key Insight**: Analogous to human foveal vision — densely sampling attended regions while sparsely sampling the periphery, thereby preserving global information.

**Core Idea**: Use the model's own attention to guide a single rectilinear warping transformation, enabling the same model to "see more clearly."

## Method

### Overall Architecture
Input image + query → MLLM extracts cross-modal attention maps → aggregated into an attention score matrix → marginal attention distributions computed → CDF inverse transform produces warp mapping → bilinear resampling yields warped image → same MLLM processes warped image to generate answer.

### Key Designs
1. **Rectilinear Warping**: The 2D attention matrix is decomposed into horizontal and vertical marginal distributions $m_x(j), m_y(i)$. The CDF is computed and its inverse is used as the warp mapping:
    $f_X^{\text{Warp}}(j) = W \cdot M_x^{-1}(j/W), \quad f_Y^{\text{Warp}}(i) = H \cdot M_y^{-1}(i/H)$
   This preserves the regular grid structure for compatibility with standard visual encoders. All original image information is retained (no cropping or masking); only pixel density is redistributed.

2. **AttWarp-Chain (Iterative Warping)**: Warping improves attention → better attention yields better warping, forming a positive feedback loop. KL divergence serves as the termination criterion:
    $\mathcal{D}_{KL}(P^{(d)} | P^{(d-1)}) < \epsilon_{KL}$

3. **AttWarp-Distill (Distilled Variant)**: A lightweight network (CLIP ViT-L/14 + FiLM conditioning + Conv1D) is trained to directly predict marginal distributions $(\hat{m}_x, \hat{m}_y)$ from image-text pairs, bypassing the attention extraction step. Trained with L1 loss; inference requires a single forward pass, achieving 3× speedup over ViCrop.

4. **Attention Score Matrix**: Cross-modal attention is extracted from specified decoder layers of the MLLM and averaged across output tokens, attention heads, and layers, then upsampled to image resolution and smoothed:
    $\tilde{A}_{i,j} = \frac{1}{n_{\text{out}} \cdot n_{\text{heads}} \cdot |\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \sum_m \sum_h a^{(\ell,h)}_{m,t}$

### Loss & Training
- AttWarp / AttWarp-Chain: No training required; purely test-time methods.
- AttWarp-Distill: Trained on TextVQA/GQA/DocVQA training sets using teacher attention as supervision targets with L1 loss.

## Key Experimental Results

### Main Results
Results on LLaVA-v1.5-7B (accuracy %):

| Method | TextVQA | GQA | MMMU | POPE | DocVQA |
|--------|---------|-----|------|------|--------|
| Base MLLM | 49.3 | 60.5 | 36.9 | 85.3 | 18.1 |
| ViCrop | 56.3 | 60.9 | 37.2 | 87.0 | 22.5 |
| AttWarp | 58.1 | 63.7 | 40.4 | 87.5 | 25.5 |
| AttWarp-Chain | **60.3** | **64.4** | **41.6** | **88.2** | **27.6** |
| Δ vs. strongest baseline | +4.0 | +3.5 | +4.4 | +1.2 | +5.1 |

Consistent improvements are also observed on Qwen2.5-VL (+2.1–3.6%).

### Ablation Study
Attention distribution improvement validation (TextVQA):

| Metric | w/o Warping | w/ AttWarp |
|--------|-------------|------------|
| Pointing Game Accuracy | 37.4% | 42.4% (+5%) |
| Proportion (attention within bbox) | 0.117 | 0.155 (+3.8%) |

Distribution shift analysis: AttWarp achieves KID = 31.5 vs. Non-Rectilinear Warp KID = 174.9 (distance from training distribution), demonstrating that rectilinear warping introduces negligible distribution shift.

### Key Findings
- Warping demonstrably concentrates attention on correct regions, improving Pointing Game accuracy by 5%.
- The rectilinear design is critical — non-rectilinear warping causes severe distribution shift (KID increases from 31.5 to 174.9).
- AttWarp-Distill requires only 8.7 TFLOPs, close to the Base MLLM's 8.5 TFLOPs and far more efficient than ViCrop's 24.2 TFLOPs.
- Error analysis indicates that AttWarp primarily reduces errors in fine-grained detail perception and compositional reasoning.

## Highlights & Insights
- **Philosophy of "Constructive Distortion"**: Inspired by human foveal vision, actively warping the input is a principled and effective strategy for improving perception.
- **Plug-and-Play**: Requires no model modification; consistently effective across 4 diverse MLLM architectures (LLaVA, Qwen-VL, InternVL, InstructBLIP).
- **Information-Preserving**: Unlike cropping or masking, warping retains all pixel information and only redistributes density.
- **CDF Inverse Transform Framework**: The mathematical framework converting attention distributions into warp mappings is elegant and concise, requiring only a single CDF forward pass.
- **Positive Feedback in AttWarp-Chain**: Iterative reinforcement where warping improves attention and attention improves warping, with KL divergence providing automatic termination.
- **Distribution Preservation Analysis**: Rigorous validation that rectilinear warping does not introduce distribution shift (verified via KID/FID/Mahalanobis distance).

## Limitations & Future Work
- Requires two MLLM forward passes (one for attention extraction, one for inference), doubling latency.
- Warping may suppress peripheral context beneficial for global reasoning, particularly for questions requiring full-scene understanding.
- Absolute scale information is lost after warping, potentially affecting size-related questions.
- The number of AttWarp-Chain iterations depends on the KL threshold hyperparameter.
- Attention quality is a prerequisite — if the initial attention is severely misaligned, warping will be counterproductive.
- There is no theoretical upper bound on warp magnitude; extreme warping may cause severe compression of non-target regions.
- Application to video understanding models (temporally consistent warping) remains unexplored.

## Related Work & Insights
- Compared to test-time intervention methods such as FGVP/SoM/ViCrop: AttWarp is the only method that preserves complete image information.
- Compared to APIPrompting: the latter overlays attention heatmaps, introducing non-original information; AttWarp maintains pure image input.
- A modern revival of classical methods such as seam carving and saliency-aware warping — whereas traditional approaches rely on optimization (minutes per image), AttWarp operates via a single CDF forward pass.
- **Insight**: Input-level intervention (as opposed to intermediate representation manipulation) is an underexplored yet effective strategy for improving perceptual models.
- **Implication for Embodied AI / AR Devices**: AttWarp-Distill's single-pass inference is well-suited for low-latency deployment scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — Attention-guided warping is a novel idea; the CDF inverse transform framework is elegant, drawing inspiration from foveal vision.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 benchmarks, 4 models, distribution analysis, attention verification, and error analysis; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation → method → experiments flow logically; figures are clear and analysis is thorough.
- Value: ⭐⭐⭐⭐ — High practical value as a plug-and-play approach, though it is fundamentally a test-time trick with limited theoretical depth.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Token Warping Helps MLLMs Look from Nearby Viewpoints](../../CVPR2026/multimodal_vlm/token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [\[ICML 2026\] RESTORE: Improving Visual Token Reduction via Distortion Correction for Enhanced MLLM Inference Efficiency](../../ICML2026/multimodal_vlm/improving_visual_token_reduction_via_rectifying_distortions_for_efficient_multim.md)
- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](../../CVPR2026/multimodal_vlm/ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[ICLR 2026\] Grounding-IQA: Grounding Multimodal Language Models for Image Quality Assessment](grounding-iqa_grounding_multimodal_language_model_for_image_quality_assessment.md)
- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)

</div>

<!-- RELATED:END -->
