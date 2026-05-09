---
title: >-
  [Paper Note] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images
description: >-
  [AAAI 2026][Segmentation][Remote sensing visual grounding] This paper proposes RSVG-ZeroOV, a training-free framework that integrates cross-attention maps from a VLM and self-attention maps from a diffusion model via an "Overview-Focus-Evolve" three-stage strategy, enabling zero-shot open-vocabulary visual grounding in remote sensing images.
tags:
  - AAAI 2026
  - Segmentation
  - Remote sensing visual grounding
  - zero-shot
  - open-vocabulary
  - diffusion model
  - training-free
date: 2026-05-08
content_hash: 1aab191448490e09
---

# RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images

**Conference**: AAAI 2026
**arXiv**: [2509.18711](https://arxiv.org/abs/2509.18711)
**Code**: N/A
**Area**: Segmentation
**Keywords**: Remote sensing visual grounding, zero-shot, open-vocabulary, diffusion model, training-free

## TL;DR

This paper proposes RSVG-ZeroOV, a training-free framework that integrates cross-attention maps from a VLM and self-attention maps from a diffusion model via an "Overview-Focus-Evolve" three-stage strategy, enabling zero-shot open-vocabulary visual grounding in remote sensing images.

## Background & Motivation

Remote sensing visual grounding (RSVG) aims to localize target objects in remote sensing images based on free-form natural language descriptions. This task has important applications in urban planning, environmental monitoring, and related domains — for example, localizing "the tallest building near the river" or "the factory next to the playground."

Existing methods face three major limitations:

**Closed-set vocabulary constraint**: Current RSVG methods are restricted to predefined categories (e.g., "road," "farmland") and cannot handle free-text descriptions in open-world scenarios. Real-world targets (e.g., "temporary roadside parking area") may not be representable by simple category names, requiring models to understand visual attributes, spatial relationships, and functional roles.

**Reliance on expensive supervision**: The few methods that attempt to leverage foundation models for open-vocabulary RSVG are heavily dependent on high-quality annotated data and time-consuming fine-tuning, resulting in poor scalability.

**Underutilized complementarity of foundation models**: VLMs excel at high-level semantic understanding but have weak spatial perception, while diffusion models (DMs) possess strong structural priors but lack language understanding. Their respective attention maps exhibit complementary properties for object perception, yet this complementarity has not been effectively exploited.

Through systematic exploratory experiments (Guidelines 1–3), the authors derive three empirical guidelines: (1) general-purpose VLMs generalize better than remote-sensing-specialized VLMs; (2) DM self-attention encodes superior structural priors for object localization; (3) cross-attention and self-attention are complementary, and their fusion consistently improves performance. These three guidelines directly inform the framework design.

## Method

### Overall Architecture

RSVG-ZeroOV follows an **"Overview-Focus-Evolve"** three-stage strategy:

1. **Overview stage**: A frozen VLM is used to extract cross-attention maps, capturing semantic associations between the text query and visual regions.
2. **Focus stage**: A frozen diffusion model is used to extract self-attention maps as structural priors; an attention interaction module compensates for shape information missed by the VLM.
3. **Evolve stage**: An attention evolution module suppresses irrelevant activations to produce clean segmentation masks.

The entire framework is **completely training-free**, performing inference using only frozen pretrained models.

### Key Designs

#### 1. **Overview Stage — VLM Cross-Attention Extraction**

- A frozen Qwen2.5-VL is used as the VLM, taking the remote sensing image and text query as input.
- Attention weights $\mathcal{W}^{(t)} \in \mathbb{R}^{H \times 1 \times N}$ are extracted from all Transformer heads of the VLM.
- The image-text relevant attention segment $\mathcal{W}^{(t)}_{p:p'}$ is extracted and averaged across all attention heads and autoregressive steps:

$$\mathcal{A}_C = \frac{1}{T}\sum_{t=1}^{T}\left(\frac{1}{H}\sum_{h=1}^{H}\mathcal{W}^{(t)}_{p:p'}\right)$$

- **Core finding**: Cross-attention maps exhibit two problems — (i) attention concentrates on object boundaries/corners rather than the complete region; (ii) attention is diffuse and often includes irrelevant regions.

**Design Motivation**: The high-level semantic concentration of VLMs causes attention to gravitate toward salient feature points; understanding complex textual expressions requires aggregating contextual cues from multiple visual regions, leading to attention dispersion.

#### 2. **Focus Stage — DM Self-Attention Interaction**

- Multi-scale self-attention maps are extracted from the U-Net of Stable Diffusion V1.4 and fused into a unified structural prior:

$$\mathcal{A}_S = \frac{1}{L}\sum_{l \in L}\mathcal{A}^l_S$$

- The association between cross-attention and self-attention is computed via **cosine similarity**:

$$\mathcal{A}_{CS} = \frac{\mathcal{A}_C \cdot \mathcal{A}_S}{\|\mathcal{A}_C\|_2 \|\mathcal{A}_S\|_2}$$

**Design Motivation**: DM self-attention captures object structure (shape, complete contour) far more effectively than the VLM (whose attention is diffuse) or SAM (which over-focuses on the background). The cosine similarity strategy outperforms anchor-based, multiplicative, and exponential approaches (Tab. 3) by producing initial masks with better semantic consistency.

**Why DM over SAM?** Experiments (Tab. 2) demonstrate that DM self-attention yields the most coherent structural representations — attention is uniformly and densely distributed across the entire object extent — whereas SAM, despite producing sharp boundaries, frequently over-attends to surrounding background due to its purely visual design.

#### 3. **Evolve Stage — Attention Evolution Module**

- The top-K highest-response pixels are selected from the cross-attention map $\mathcal{A}_C$ as seed points:

$$\mathcal{S} = \text{TopK}(\mathcal{A}_C, K)$$

- A **depth-first search (DFS)** is performed recursively from each seed to expand the region; pixel $(u,v)$ is included if and only if:

$$\mathcal{A}_{CS}[u,v] \geq \tau \text{ and } (u,v) \in \text{DFS}(\mathcal{S})$$

- The evolved attention map is binarized to produce the segmentation mask: $\mathbf{M}(i,j) = \mathbb{1}[\mathcal{A}_E(i,j) > \alpha]$

**Design Motivation**: DFS performs region growing from high-confidence seeds, retaining only pixels that are both connected to the seeds and exceed the response threshold, thereby effectively suppressing scattered background activations and producing clean masks.

#### 4. **Optional Refinement Stage**

SAM with box prompts is applied as a post-processing step to further improve mask quality; experiments confirm that box prompts yield the best results.

### Loss & Training

This is a training-free framework requiring no loss functions. Hyperparameters: $K=7$ (seed selection), $\tau=0.3$ (response threshold), $\alpha=0.4$ (binarization threshold), 20-step DDIM sampling. Inference runs on a single RTX-4090 GPU.

## Key Experimental Results

### Main Results

**RRSIS-D Dataset (Test, with refinement)**:

| Method | Type | RSREC Pr@0.5 | RSREC mIoU | RSRES Pr@0.5 | RSRES mIoU |
|--------|------|-------------|------------|-------------|------------|
| QueryMatch | Weakly supervised | 16.22 | 17.21 | 15.54 | 15.73 |
| DiffSegmenter (w/ VLM) | Zero-shot | 25.11 | 28.50 | 19.42 | 23.73 |
| DiffPNG (w/ VLM) | Zero-shot | 21.29 | 24.89 | 17.64 | 20.99 |
| OV-VG | Zero-shot | 16.20 | 21.62 | - | - |
| **RSVG-ZeroOV** | **Zero-shot** | **31.39** | **34.49** | **27.39** | **28.35** |

**RISBench Dataset (Test, with refinement)**:

| Method | RSREC Pr@0.5 | RSREC mIoU | RSRES Pr@0.5 | RSRES mIoU |
|--------|-------------|------------|-------------|------------|
| GroundVLP | 19.91 | 19.19 | 15.82 | 15.58 |
| OV-VG | 22.40 | 22.85 | 17.75 | 16.17 |
| **RSVG-ZeroOV** | **38.90** | **38.87** | **31.03** | **31.84** |

### Ablation Study

| Configuration | RSREC Pr@0.5 | RSREC mIoU | RSRES Pr@0.5 | RSRES mIoU | Note |
|---------------|-------------|------------|-------------|------------|------|
| w/o VLM | 16.22 | 18.82 | 11.43 | 15.81 | Large drop without VLM |
| w/o DM | 21.49 | 26.26 | 1.18 | 6.15 | RSRES nearly fails without DM |
| w/o Evolve | 22.63 | 26.65 | 10.26 | 20.56 | Evolution module is necessary |
| O-F-E (Ours) | **30.15** | **32.92** | **12.84** | **21.85** | Optimal ordering |
| O-E-F | 27.34 | 29.51 | 7.18 | 15.89 | Ordering significantly affects results |

**Self-attention map resolution ablation**:

| Resolution | RSREC mIoU | RSRES mIoU | Note |
|------------|------------|------------|------|
| 32 | 31.97 | 21.11 | Single resolution |
| 64 | 30.76 | 20.13 | Single resolution |
| [32, 64] (Ours) | **32.92** | **21.85** | Multi-scale optimal |
| [16, 32, 64] | 30.51 | 20.36 | Performance drops with too many scales |

**Interaction strategy comparison (Tab. 3)**:

| Strategy | RSREC Pr@0.5 | RSRES mIoU | Note |
|----------|-------------|------------|------|
| Anchor-based + Evolve | 28.73 | 16.58 | Overly simplified |
| Multiplicative + Evolve | 29.26 | 20.75 | Moderate |
| Exponential + Evolve | 27.38 | 14.00 | Excessive amplification |
| **Cosine similarity** + Evolve | **30.15** | **21.85** | Optimal |

### Key Findings

- **Both VLM and DM are indispensable**: Removing the VLM causes a 14.10% drop in RSREC mIoU; removing the DM causes RSRES to collapse from 21.85% to 6.15%.
- **O-F-E ordering outperforms O-E-F**: Focusing first (embedding structural priors) before evolving (region growing) is more effective; reversing the order causes structural information to be prematurely pruned.
- **General-purpose VLM > remote-sensing-specialized VLM**: Qwen2.5-VL (general-purpose) achieves 28.66% Pr@0.5 on zero-shot RSREC, outperforming GeoChat (remote-sensing-specialized, 23.93%).
- **DM self-attention > VLM/SAM self-attention**: DM achieves 30.15%/12.84% Pr@0.5 on RSREC/RSRES respectively, substantially outperforming other self-attention sources.
- Multi-scale self-attention [32, 64] simultaneously preserves high-resolution detail and contextual semantics.

## Highlights & Insights

1. **First zero-shot remote sensing visual grounding framework**: Operating in a completely training-free manner for remote sensing scenarios, offering strong practical utility.
2. **Three empirically-grounded guidelines** derived from systematic exploratory experiments are highly valuable: each guideline is experimentally supported and provides clear guidance for subsequent research.
3. The **DFS region-growing** evolution strategy is concise and effective, suppressing scattered noise without any learnable parameters.
4. The finding of **complementary multi-model attention** is generalizable: VLMs provide semantics but lack structure; DMs provide structure but lack semantics; their fusion is mutually beneficial.

## Limitations & Future Work

- The absolute zero-shot performance remains limited (RSRES mIoU only ~28%), with a considerable gap relative to fully supervised methods.
- Diffusion model inference is time-consuming (20-step DDIM), constraining overall inference speed.
- Hyperparameters of DFS region growing ($K$, $\tau$, $\alpha$) require manual tuning, limiting robustness.
- Validation is conducted only on remote sensing RSVG; generalization to natural images has not been tested.
- The framework's capability for handling complex spatial relationship descriptions (e.g., multi-object relational reasoning) is not explicitly evaluated.

## Related Work & Insights

- The VLM + DM attention fusion paradigm is transferable to other cross-domain zero-shot segmentation tasks.
- DFS region growing can serve as a general-purpose post-processing module for attention maps.
- The three guidelines provide important references for the selection and combination of foundation models in remote sensing.
- The progressive Overview-Focus-Evolve pipeline design can inspire further multimodal perception methods.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First zero-shot open-vocabulary remote sensing visual grounding framework; VLM+DM attention fusion is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two datasets, systematic exploratory experiments, comprehensive ablations, multiple baseline comparisons)
- Writing Quality: ⭐⭐⭐⭐⭐ (Guideline-driven exposition is clear and coherent; figures and tables are high quality)
- Value: ⭐⭐⭐⭐ (Training-free framework offers strong practical utility, though absolute performance still has room for improvement)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[AAAI 2026\] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective](target_refocusing_via_attention_redistribution_for_open-vocabulary_semantic_segm.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)

<!-- RELATED:END -->
