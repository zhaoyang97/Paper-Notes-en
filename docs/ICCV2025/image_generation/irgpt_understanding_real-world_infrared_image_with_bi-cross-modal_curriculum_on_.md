---
title: >-
  [Paper Note] IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark
description: >-
  [ICCV 2025][Image Generation][Infrared Image] This paper proposes IRGPT, the first multimodal large language model grounded in real-world infrared images. It introduces IR-TD…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Infrared Image"
  - "Multimodal Large Language Model"
  - "Curriculum Learning"
  - "Cross-modal Transfer"
  - "Vision-Language Model"
date: 2026-05-08
content_hash: 64de6f5e2ae6b6b8
---

# IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark

**Conference**: ICCV 2025
**arXiv**: [2507.14449](https://arxiv.org/abs/2507.14449)  
**Code**: Dataset is open-sourced  
**Area**: Image Generation
**Keywords**: Infrared Image, Multimodal Large Language Model, Curriculum Learning, Cross-modal Transfer, Vision-Language Model

## TL;DR

This paper proposes IRGPT, the first multimodal large language model grounded in real-world infrared images. It introduces IR-TD, a large-scale infrared-text dataset containing 260K+ image-text pairs, and designs a Bi-cross-modal Curriculum transfer learning strategy. IRGPT achieves state-of-the-art performance across 9 infrared task benchmarks, with zero-shot psum outperforming the baseline InternVL2-8B by 76.35.

## Background & Motivation

**Why do existing VLMs fail on infrared images?** Mainstream vision-language models (LLaVA, Qwen2-VL, InternVL2, etc.) are primarily trained on visible-light images and produce severe hallucinations when applied to infrared images—for example, misidentifying metal signs as non-metallic objects. This problem stems from two root causes:

**Data scarcity**: Large-scale image-text aligned datasets are absent in the infrared domain. Unlike visible-light images, which can be paired with descriptive text harvested from the web, infrared images cannot be collected in the same manner.

**Limitations of synthetic data**: Prior methods (e.g., Infrared-LLaVA) rely on style transfer to generate synthetic infrared images from visible-light counterparts. However, synthetic data is constrained by the training distribution of the generative model and fails to capture the unique characteristics of real infrared images, such as thermally dominant target saliency and imaging differences across spectral bands.

**Why is direct transfer from visible light non-trivial?** Although infrared and visible-light images share certain similarities, direct transfer is challenged by sparse semantic information in infrared images, large inter-band variation (e.g., near-infrared vs. thermal infrared), and inconsistent image-text alignment quality. A progressive, easy-to-hard learning strategy is therefore necessary.

## Method

### Overall Architecture

IRGPT is built upon InternVL2-8B and comprises a visual encoder, a base LLM, a text tokenizer, and a trainable visual projector. Training proceeds in two stages:

- **Incremental Pre-training**: Only the visual encoder and projection layer are trained, with the objective of adapting to infrared image input and achieving text alignment.
- **Supervised Instruction Fine-tuning**: Both the visual projector and the LLM are fine-tuned simultaneously (with LoRA to prevent loss of expressive capacity).

### IR-TD Dataset Construction

To address the scarcity of infrared-text data, the authors aggregate 63 publicly available datasets and explore 5 data generation pipelines:

- **Pipeline IV (LLM-assisted)**: Leverages visible-infrared image pairs; an LLM generates descriptions for the visible images, which are then manually adapted into infrared annotations.
- **Pipeline V (Rule-based generation)**: Produces large-scale image-text pairs from annotation information using rule-based templates.

The final dataset comprises 190K pre-training samples, 33K instruction-tuning samples, and 37K benchmark samples, covering 9 task categories including recognition, localization, relation judgment, pedestrian re-identification, security surveillance, and aerial counting. Compared to Infrared-LLaVA's closed-source IID dataset, IR-TD is open-source, uses real images, and contains no data redundancy.

### Bi-cross-modal Curriculum Transfer Learning

The core innovation lies in two "lessons" that quantify sample difficulty to guide easy-to-hard training:

**Lesson 1: IR-VIS (Infrared–Visible Distance Metric)**

A feature extractor is retrained on both infrared and visible-light grayscale images. The distance from each infrared sample to the visible-light domain is computed via intra-domain distance projection:

$$d_i = (\phi(x_i^{ir}) - c^{ir}) \cdot \frac{c^{vis} - c^{ir}}{\|c^{vis} - c^{ir}\|} + \text{MMD}$$

where $c^{ir}$ and $c^{vis}$ are the domain centroid vectors, and MMD denotes the maximum mean discrepancy. Samples closer to the visible-light domain are easier to learn (near-infrared > thermal infrared).

**Lesson 2: IR-T (Infrared–Text Alignment Metric)**

A warmed-up CLIP model is used to compute the alignment loss between infrared images and text. To distinguish "difficult but valuable samples" from "misaligned samples," a **loss change rate** is introduced:

$$\alpha = \frac{l' - l}{l}$$

where $l$ and $l'$ denote the loss values before and after warm-up, respectively. $\alpha > 0$ indicates degraded alignment (potential noise), while $\alpha \leq 0$ indicates normal convergence.

**Curriculum Scheduling**

The rankings from both lessons are integrated into a composite ranking, and the data are divided into $M$ difficulty tiers. Training is organized in tier order with random sampling within each tier, balancing structured learning with stochasticity. The loss change rate serves as an adaptive weight:

$$w_i = \begin{cases} 1 - \sigma(\alpha_i / \text{Median}(\{\alpha_j | \alpha_j > 0\})) & \alpha_i > 0 \\ 1 + \sigma(-\alpha_i / \text{Median}(\{-\alpha_k | \alpha_k \leq 0\})) & \alpha_i \leq 0 \end{cases}$$

The final weighted cross-entropy loss is $\mathcal{L} = \frac{1}{N}\sum_{i=1}^N w_i \cdot [-\sum_c y_{i,c} \log p_{i,c}]$.

## Key Experimental Results

### Main Results: Performance Comparison across 9 Tasks

| Model | Scene | Rec. | Gro. | Rel. | ReID | Sec. | psum↑ | Loc. | A.C. | P.C. | nsum↓ |
|------|-------|------|------|------|------|------|-------|------|------|------|-------|
| LLaVA1.5-7B (zero-shot) | 35.89 | 23.11 | 11.17 | 23.68 | 2.61 | 18.63 | 115.09 | 83.12 | 42.39 | 117.22 | 242.73 |
| Qwen2-VL-7B (zero-shot) | 49.80 | 59.42 | 25.74 | 56.19 | 4.62 | 30.08 | 225.85 | 50.41 | 18.05 | 89.12 | 157.58 |
| InternVL2-8B (zero-shot) | 46.02 | 77.69 | 30.05 | 53.02 | 7.33 | 38.19 | 252.30 | 43.29 | 21.75 | 60.39 | 125.43 |
| InternVL2-26B (zero-shot) | 62.24 | 72.88 | 39.48 | 55.42 | 8.16 | 43.72 | 281.90 | 38.25 | 21.68 | 64.65 | 124.58 |
| **IRGPT (CL, zero-shot)** | **65.12** | **86.28** | **36.68** | **58.33** | **33.55** | **48.69** | **328.65** | **33.32** | **13.25** | **47.30** | **93.87** |
| InternVL2-26B (fine-tune) | 84.03 | 98.94 | 48.77 | 95.89 | 47.09 | 99.06 | 473.78 | 12.50 | 1.38 | 5.68 | 19.57 |
| **IRGPT (CL, fine-tune)** | **85.12** | **99.79** | **51.58** | **98.69** | **50.79** | **99.82** | **485.79** | **3.32** | **0.25** | **0.82** | **4.39** |

In the zero-shot setting, IRGPT improves psum by 76.35 and reduces nsum by 31.56 over the InternVL2-8B baseline. After fine-tuning, it surpasses the larger InternVL2-26B, achieving psum of 485.79 (+12.01) and nsum of only 4.39 (−15.18).

### Ablation Study: Contribution of Each Curriculum Component

| L1 | anti-L1 | L2 | anti-L2 | α | psum | nsum |
|----|---------|----|---------|----|------|------|
| - | - | - | - | - | 464.49 | 26.85 |
| ✓ | - | - | - | - | 470.23 | 17.29 |
| - | - | ✓ | - | - | 466.81 | 19.33 |
| ✓ | - | ✓ | - | - | 481.36 | 7.08 |
| ✓ | - | ✓ | - | ✓ | **485.79** | **4.39** |
| - | ✓ | - | ✓ | ✓ | 433.24 | 26.50 |

Lesson 2 (image-text alignment) is more sensitive to psum, while Lesson 1 (image quality) is more sensitive to nsum. Reverse curriculum (anti-CL) substantially degrades performance, validating the easy-to-hard training strategy.

### Sampling Schedule Ablation

| Scheduling Strategy | Zero-shot psum | Fine-tune psum | Fine-tune nsum |
|---------|----------------|----------------|----------------|
| Difficulty-Descending | 286.84 | 433.24 | 26.50 |
| Difficulty-Ascending | 309.63 | 474.29 | 19.24 |
| Random | 307.18 | 479.46 | 12.94 |
| **Ascending-Stratified Random** | **328.65** | **485.79** | **4.39** |

The stratified ascending-random strategy achieves the best results, balancing structured learning progression with stochastic variation.

## Highlights & Insights

1. **Real vs. Synthetic Data**: This work is the first to demonstrate that training on real infrared images at scale significantly outperforms approaches based on synthetic infrared data; the authenticity of IR-TD is a key advantage.
2. **Effective Curriculum Design**: The dual-perspective difficulty metric (cross-domain distance + text alignment) offers strong complementarity—L1 focuses on image quality while L2 targets semantic correctness.
3. **Practical Value of Loss Change Rate**: By distinguishing "difficult but learnable" samples from "misaligned" ones, the adaptive weighting scheme protects hard samples while suppressing noise, yielding greater robustness than purely static loss formulations.
4. **Breakthrough on ReID**: Zero-shot ReID improves from 7.33 to 33.55, demonstrating that curriculum learning enables the model to genuinely understand infrared pedestrian identity features.

## Limitations & Future Work

- The model is built on InternVL2-8B and may underperform larger models on highly complex reasoning tasks.
- The domain projection distance computation is an approximation whose accuracy depends on the quality of the feature extractor.
- LLM-generated descriptions in the pre-training subset of IR-TD require manual adaptation, posing a scalability bottleneck.
- Counting-related tasks (Loc/A.C./P.C.) still exhibit considerable error in the zero-shot setting, as the pre-training stage emphasizes alignment over precise understanding.

## Related Work & Insights

- **Infrared-LLaVA**: The most closely related work, but relies on synthetic infrared data and suffers from cross-modal hallucination.
- **InfMAE**: An infrared masked image modeling foundation model; its feature extractor is used in this work to compute domain distances.
- **Curriculum Learning (Baby-step)**: The stratified random sampling strategy directly inherits from classical curriculum learning theory.
- **Insight**: The difficulty quantification methodology for cross-modal transfer is generalizable to other modality pairs (e.g., medical imaging → natural images, satellite imagery → aerial photography).

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — First real-infrared MLLM; novel bi-cross-modal curriculum learning design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 9-task benchmark, both zero-shot and fine-tune evaluation, complete ablation study.
- Value: ⭐⭐⭐⭐ — Open-source dataset with practical applications in security surveillance and autonomous driving.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-motivated dataset construction and methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](../../NeurIPS2025/image_generation/ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](../../NeurIPS2025/image_generation/in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](../../CVPR2026/image_generation/cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)

</div>

<!-- RELATED:END -->
