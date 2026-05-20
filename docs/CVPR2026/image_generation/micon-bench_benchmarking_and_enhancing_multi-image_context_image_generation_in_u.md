---
title: >-
  [Paper Note] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models
description: >-
  [CVPR 2026][Image Generation][Multi-image context generation] This paper proposes MICON-Bench, a multi-image context generation benchmark covering 6 tasks (1,043 cases)…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Multi-image context generation"
  - "unified multimodal models"
  - "benchmark"
  - "dynamic attention rebalancing"
  - "checkpoint evaluation"
date: 2026-05-08
content_hash: 3cf248573ec9567a
---

# MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models

**Conference**: CVPR 2026
**arXiv**: [2602.19497](https://arxiv.org/abs/2602.19497)  
**Code**: [https://github.com/Angusliuuu/MICON-Bench](https://github.com/Angusliuuu/MICON-Bench)  
**Area**: Image Generation / Multimodal Evaluation
**Keywords**: Multi-image context generation, unified multimodal models, benchmark, dynamic attention rebalancing, checkpoint evaluation

## TL;DR
This paper proposes MICON-Bench, a multi-image context generation benchmark covering 6 tasks (1,043 cases), paired with an MLLM-driven Evaluation-by-Checkpoint automated assessment framework. It further introduces DAR (Dynamic Attention Rebalancing), a training-free mechanism that improves generation consistency and quality in unified multimodal models (UMMs) by dynamically adjusting attention weights at inference time.

## Background & Motivation

**Background**: UMMs can process multi-image inputs and produce contextually consistent visual outputs; representative models include Nano-Banana, GPT-Image, BAGEL, and OmniGen2. However, multi-image context generation capabilities lack systematic evaluation.

**Evaluation Gap**: Existing benchmarks (GenEval, T2ICompBench, ImgEdit-Bench) primarily assess text-to-image generation or single-image editing, and do not address cross-image consistency or complex visual relational reasoning. OmniContext involves multiple images but is limited to simple subject composition.

**Limitations of Prior Work**: UMMs tend to **distribute attention uniformly** across all regions of all reference images, including irrelevant regions, leading to hallucinations and inconsistencies.

**Core Idea**: (a) Six standardized tasks coupled with a verifiable checkpoint evaluation system; (b) attention rebalancing to redirect focus at inference time.

## Method

### MICON-Bench Benchmark Design

**6 Tasks** (5 compositional + 1 complex reasoning):

| Task | Description | Cases | Ref. Images |
|------|-------------|--------|-------------|
| Object Composition | Single subject + background composition | 200 | 2–3 |
| Spatial Composition | Multi-object spatial relationship constraints | 200 | 2–3 |
| Attribute Disentanglement | Subject/style/background decoupled recombination | 100 | 3 |
| Component Transfer | Part/accessory transfer across images | 240 | 2–3 |
| FG/BG Composition | Foreground + background fusion | 200 | 2 |
| Story Generation | Causal reasoning story continuation | 103 | 2–3 |
| **Total** | | **1,043** | **2,518** |

### Evaluation-by-Checkpoint Framework
- Each case has verifiable **checkpoints** spanning seven dimensions: instruction following, identity consistency, structure, cross-reference consistency, causality, text anchoring, and overall usability.
- An MLLM (Qwen3-VL-32B) serves as the verifier, judging each checkpoint as pass/fail; the final score is the mean pass rate.
- The Story Generation task additionally employs a predefined answer set to evaluate reasoning ability.

### Dynamic Attention Rebalancing (DAR)

1. **Problem Diagnosis**: UMM attention indiscriminately attends to irrelevant regions in reference images, causing hallucinations.

2. **Efficient Attention Analysis**:
    - Uniformly sample $m \ll L_q$ query tokens (default $m=64$) and compute attention maps against reference image key tokens.
    - Total attention score: $r_k = \sum_{i=1}^{m}\sum_{h=1}^{H} \tilde{A}_{i,h,k}$
    - Min-max normalization yields $\hat{r}_k$.

3. **Dynamic Weight Adjustment**:
    - Dual-threshold three-class partition: $w_k = 1+\gamma$ (if $\hat{r}_k \geq \tau_{high}$), $w_k = 1-\gamma$ (if $\hat{r}_k \leq \tau_{low}$), otherwise $w_k = 1$.
    - Adjusted attention: $A = \text{softmax}\left(\frac{Q(w \odot K_{ref})^\top}{\sqrt{d}}\right)$
    - Defaults: $\gamma=0.15$, $\tau_{high}=0.7$, $\tau_{low}=0.3$.

4. **Design Advantages**: Training-free, plug-and-play, and computationally negligible (only 64 query tokens sampled).

## Key Experimental Results

### Main Results: MICON-Bench Per-Task Scores

| Model | Object | Spatial | Attribute | Component | FG/BG | Story | Avg↑ |
|-------|--------|---------|-----------|-----------|-------|-------|------|
| Nano-Banana | 95.60 | 93.79 | 92.13 | 84.23 | 83.13 | 82.84 | 89.25 |
| GPT-Image | 96.45 | 94.41 | 93.39 | 87.69 | 85.99 | 91.51 | 90.15 |
| UNO | 58.40 | 66.68 | 65.28 | 28.84 | 20.96 | 39.08 | 44.76 |
| DreamOmni2 | 88.24 | 84.76 | 85.28 | 59.64 | 76.16 | 59.58 | 75.56 |
| BAGEL | 87.64 | 89.96 | 89.84 | 52.40 | 64.64 | 65.09 | 73.55 |
| **BAGEL + DAR** | **88.04** | **91.88** | **90.76** | **56.06** | **71.24** | **66.34** | **76.31** |
| OmniGen2 | 89.52 | 80.32 | 81.64 | 44.76 | 57.96 | 60.96 | 67.83 |
| **OmniGen2 + DAR** | **89.84** | **81.00** | **82.12** | **48.72** | **59.28** | **60.73** | **69.21** |

### OmniContext Benchmark

| Method | SINGLE Char/Obj | MULTIPLE Char/Obj | SCENE Char/Obj | Avg↑ |
|--------|-----------------|-------------------|----------------|------|
| OmniGen2 | 8.18/7.33 | 6.56/7.99 | 6.87/7.90 | 7.53 |
| **OmniGen2+DAR** | **8.30/8.19** | **6.64/8.42** | **7.06/7.97** | **7.77** |
| BAGEL | 5.71/6.22 | 3.03/6.90 | 4.24/5.16 | 5.54 |
| **BAGEL+DAR** | **6.26/6.08** | **4.14/7.18** | **4.78/4.84** | **5.80** |

### XVerseBench Benchmark

| Method | Single-Subject Avg↑ | Multi-Subject Avg↑ | Overall↑ |
|--------|--------------------|--------------------|----------|
| OmniGen2 | 52.53 | 49.76 | 51.14 |
| **OmniGen2+DAR** | **53.24** | **50.23** | **51.73** |
| BAGEL | 47.91 | 42.62 | 45.26 |
| **BAGEL+DAR** | **48.54** | **43.91** | **46.23** |

### Key Findings
- MICON-Bench effectively differentiates models: GPT-Image achieves the highest score (90.15), while the diffusion-based UNO scores lowest (44.76).
- DAR yields the most pronounced gains on BAGEL: Avg +2.76 (73.55→76.31), with FG/BG improving by +6.60.
- DAR consistently improves performance across three distinct benchmarks (MICON-Bench, OmniContext, XVerseBench), demonstrating strong generalizability.
- Component Transfer and FG/BG Composition are the most challenging tasks, with even top-tier models scoring only 84–88.
- A substantial gap remains between open-source and closed-source models (BAGEL 73.55 vs. GPT-Image 90.15).

## Highlights & Insights
- **First systematic multi-image context generation benchmark**: 6 tasks spanning a complete difficulty spectrum from simple composition to causal reasoning.
- **Evaluation-by-Checkpoint paradigm**: Fine-grained, quantifiable, and extensible, offering greater objectivity than image-level metrics.
- **DAR is concise yet effective**: Sampling only 64 query tokens with dual-threshold reweighting achieves significant gains at zero training cost.
- The work exposes attention allocation blind spots in UMMs under multi-image reasoning, providing direction for future model design.

## Limitations & Future Work
- The DAR thresholds $\tau_{high}$, $\tau_{low}$, and modulation factor $\gamma$ require manual tuning; adaptive strategies remain unexplored.
- The Story Generation task has a relatively small sample size (103 cases).
- Benchmark data generated via Qwen-Image and GPT-4o may introduce generative model bias.
- Higher-order requirements such as 3D consistency and temporal continuity are not evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-image context generation benchmark + plug-and-play DAR
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7+ models × 3 benchmarks × multiple metrics with comprehensive comparisons
- Writing Quality: ⭐⭐⭐⭐ Task definitions are clear and the evaluation pipeline is well-structured
- Value: ⭐⭐⭐⭐ Benchmark promotes evaluation standardization; DAR is immediately deployable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dualconditioned_di.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation](opro_orthogonal_panel-relative_operators_for_panel-aware_in-context_image_genera.md)

</div>

<!-- RELATED:END -->
