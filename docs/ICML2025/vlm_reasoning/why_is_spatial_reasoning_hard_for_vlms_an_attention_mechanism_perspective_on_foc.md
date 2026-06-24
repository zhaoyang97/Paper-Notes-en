---
title: >-
  [Paper Note] Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas
description: >-
  [ICML 2025][VLM Reasoning][Spatial Reasoning] This work investigates the causes of spatial reasoning failures in VLMs from a mechanistic interpretability perspective, finding that image tokens obtain only ~10% of attention despite making up 90% of the input, and that the geometric distribution of attention is the key factor. The authors propose AdaptVis, a training-free decoding method that adaptively adjusts image attention temperature based on runtime confidence…
tags:
  - "ICML 2025"
  - "VLM Reasoning"
  - "Spatial Reasoning"
  - "Attention Mechanism"
  - "VLM Interpretability"
  - "Confidence-aware Decoding"
  - "Attention Intervention"
date: 2026-05-08
content_hash: 43511094906284c2
---

# Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas

**Conference**: ICML 2025  
**arXiv**: [2503.01773](https://arxiv.org/abs/2503.01773)  
**Code**: [github.com/shiqichen17/AdaptVis](https://github.com/shiqichen17/AdaptVis)  
**Area**: Multimodal VLM  
**Keywords**: Spatial Reasoning, Attention Mechanism, VLM Interpretability, Confidence-aware Decoding, Attention Intervention

## TL;DR

This work investigates the causes of spatial reasoning failures in VLMs from a mechanistic interpretability perspective, finding that image tokens obtain only ~10% of attention despite making up 90% of the input, and that the geometric distribution of attention is the key factor. The authors propose AdaptVis, a training-free decoding method that adaptively adjusts image attention temperature based on runtime confidence, achieving up to a 50% absolute improvement on the WhatsUp dataset.

## Background & Motivation

### Key Challenge

**Background**: VLMs perform surprisingly poorly in spatial reasoning; they often make errors even on simple spatial relationships between two objects, such as "above/below/left/right/front/behind." For example, when a book is "behind" a candle, the VLM might output "left."

While existing research explores the limitations of visual encoders (like CLIP), **how visual and textual tokens interact within internal model states to build spatial understanding** remains under-investigated.

Key discovery: **The issue is not whether the model "looks at" the image, but whether the geometric distribution of attention aligns with the actual object positions.**

## Method

### Attention Analysis: Three Key Findings

**Finding 1: Severe Visual-Textual Attention Imbalance**
Image tokens account for approximately 90% of the input sequence but receive only about 10% of the total attention. Textual priors heavily overpower visual evidence.

**Finding 2: Simply Increasing Image Attention is Ineffective**
Adding a uniform positive constant to image attention logits actually degrades performance. This indicates that the crucial factor is not "how much" attention is given, but "where" it is directed.

**Finding 3: Attention Distribution Strongly Correlates with Answer Correctness**
- When answering correctly: Attention precisely focuses on the relevant entities.
- When answering incorrectly: Attention is dispersed across irrelevant areas.
- Validation using YOLO annotations: The AUROC for attention-annotation overlap is significantly higher in the middle layers (Layers 17-18).
- Early layers "see" visual information (global understanding), while middle layers "process" the information (localized focus).

### ScalingVis: Temperature Scaling

Multiplies the attention logits from the last token to image tokens by a coefficient $\alpha$:

$$A_{n,j}^{(l,h)} = \begin{cases} \alpha \cdot A_{n,j}^{(l,h)} & \text{if } j \in \mathcal{I} \\ A_{n,j}^{(l,h)} & \text{otherwise} \end{cases}$$

Interesting patterns:
- Synthetic data (unfamiliar) $\rightarrow$ $\alpha < 1$ (smoothing attention to explore broader regions) yields better results.
- Real-world data (familiar) $\rightarrow$ $\alpha > 1$ (sharpening attention to reinforce correct focus) yields better results.

### AdaptVis: Confidence-Adaptive Temperature Scaling

Key insight: **The model's confidence reflects the reliability of its attention patterns.**

Validation:
- The model's confidence for "left/right" relationships is significantly higher than for "on/under".
- Increasing $\alpha$ improves "left/right" performance, while decreasing $\alpha$ improves "on/under" performance.

Adaptive Strategy:
- Low confidence $\rightarrow$ $\alpha < 1$: Smooths the attention distribution and broadens the context window.
- High confidence $\rightarrow$ $\alpha > 1$: Sharpens the attention distribution and reinforces the original focus.

Implementation: Applied uniformly across all $H$ heads and $L$ layers, avoiding hyperparameter searching.

## Key Experimental Results

### WhatsUp Dataset


### Main Results

| Model | Cont_A | Cont_B | COCO_one | COCO_two | VG_one | VG_two |
|------|--------|--------|----------|----------|--------|--------|
| LLaVA-1.5 | 60.3 | 73.1 | 53.0 | 58.2 | 35.9 | 40.8 |
| +VCD | 61.5 | 73.4 | 53.3 | 58.2 | 35.8 | 42.5 |
| +DoLa | 61.2 | 73.4 | 53.7 | 57.5 | 36.2 | 42.1 |
| **+AdaptVis** | **84.9** | **83.8** | **53.6** | **59.9** | **42.7** | **48.1** |
| LLaVA-1.6 | 48.2 | 63.0 | 59.7 | 41.8 | 31.6 | 7.3 |
| **+AdaptVis** | **98.2** | **73.4** | **63.1** | **47.7** | **35.2** | **17.2** |

### VSR Dataset


### Ablation Study

| Model | Exact Match | F1 Score |
|------|-------------|----------|
| LLaVA-1.5 | 62.4 | 51.3 |
| **+AdaptVis** | **65.0** | **62.5** |
| LLaVA-1.6 | 58.8 | 29.4 |
| **+AdaptVis** | **62.7** | **39.3** |

### Key Findings

- LLaVA-1.6 performance on Cont_A increases from 48.2 to **98.2** (+50 absolute points).
- VCD and DoLa yield only minor improvements (typically < 2%).
- AdaptVis significantly outperforms ScalingVis on synthetic data, while both perform closely on real-world data.
- Attention visualization confirms that smoothing changes the focal point, while sharpening reinforces correct regions.

## Highlights & Insights

1. **Mechanistic Interpretability-Driven Methodology**: First analyzes the internal attention mechanisms, then designs targeted interventions.
2. **Training-Free Decoding Method**: Zero additional training, modifying only the attention temperature during inference.
3. **Negligible Computational Overhead**: Conducts only simple multiplication operations on attention logits.
4. **Confidence as an Intrinsic Signal**: Ingeniously leverages the model's own uncertainty estimation to guide the direction of intervention.
5. **"Familiar vs Unfamiliar" Dichotomy**: Provides a unified explanation for the differing optimal directions of $\alpha$ on synthetic versus real-world data.

## Limitations & Future Work

- Primarily validated on the LLaVA series; other architectures (e.g., InternVL, Qwen-VL) remain untested.
- Spatial reasoning is limited to simple two-object relationships; complex scenarios with multiple objects are not explored.
- Temperature coefficients still require tuning via a validation set.
- Generalizability to non-spatial reasoning tasks has not been verified.

## Related Work & Insights

- VLM Spatial Reasoning (WhatsUp, VSR)
- Hallucination Mitigation (VCD, DoLa, OPERA)
- Attention Analysis and Intervention
- Confidence Calibration

## Rating

⭐⭐⭐⭐⭐ — An exemplary study originating from a mechanistic interpretability perspective. A training-free method achieving a 50% absolute improvement is highly impressive. It features deep analysis, a concise methodology, and remarkable effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/vlm_reasoning/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](../../NeurIPS2025/vlm_reasoning/mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](../../NeurIPS2025/vlm_reasoning/ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/vlm_reasoning/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](../../ICCV2025/vlm_reasoning/perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)

</div>

<!-- RELATED:END -->
