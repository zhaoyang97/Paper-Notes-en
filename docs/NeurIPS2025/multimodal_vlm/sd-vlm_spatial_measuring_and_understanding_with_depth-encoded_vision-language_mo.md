---
title: >-
  [Paper Note] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Spatial reasoning] This paper proposes MSMU, a large-scale quantitative spatial reasoning dataset (700K QA pairs, 2.5M numerical annotations), and Depth Positional Encoding (DPE)…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Spatial reasoning"
  - "depth encoding"
  - "VLM"
  - "quantitative spatial understanding"
  - "3D perception"
date: 2026-05-08
content_hash: 6fb33176b6dec331
---

# SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.17664](https://arxiv.org/abs/2509.17664)  
**Code**: [GitHub](https://github.com/cpystan/SD-VLM)  
**Area**: Multimodal VLM
**Keywords**: Spatial reasoning, depth encoding, VLM, quantitative spatial understanding, 3D perception

## TL;DR

This paper proposes MSMU, a large-scale quantitative spatial reasoning dataset (700K QA pairs, 2.5M numerical annotations), and Depth Positional Encoding (DPE), enabling VLMs to achieve strong quantitative spatial measurement and understanding without relying on 3D point clouds. SD-VLM outperforms GPT-4o by 26.91% on MSMU-Bench.

## Background & Motivation

VLMs excel at 2D semantic understanding but fall critically short in **quantitative spatial reasoning**—answering questions such as "How large is the table?" or "What is the distance between two objects?" that require precise numerical responses. This capability is essential for real-world applications including robotics, autonomous driving, and augmented reality.

The root cause lies in the fact that images are projections of 3D scenes onto a 2D plane, discarding substantial 3D structural information. Recovering 3D spatial relationships from 2D images theoretically requires a depth map $d$ and camera intrinsics $\mathbf{K}$:

$$\mathbf{P} = d \cdot \mathbf{K}^{-1} \mathbf{p}$$

Two major limitations of existing approaches:

**Data scarcity**: Quantitative annotations in existing spatial datasets (e.g., SpatialVLM, SpatialRGPT) rely on model-estimated pipelines (detection + segmentation + depth estimation + camera calibration), introducing systematic errors.

**Coarse depth integration**: Treating depth maps as additional images or concatenated tokens yields limited gains.

The paper's core insight is that, given sufficiently large and accurate physical measurements, a VLM can implicitly learn camera intrinsics and establish a mapping from 2D to 3D. Theoretically, at least four line segment lengths suffice to constrain camera calibration. Accordingly, the authors (1) construct a large-scale, precisely annotated dataset from real 3D scene data, and (2) design a concise and efficient depth positional encoding scheme.

## Method

### Overall Architecture

SD-VLM is built upon LLaVA-1.5-7B and comprises: a CLIP visual encoder for image feature extraction, a Depth Positional Encoding module for integrating depth information, and an LLM for generating answers from the token sequence. When ground-truth depth maps are unavailable, Depth-Anything-V2 is used for estimation.

### Key Designs

1. **MSMU Dataset Construction**

   Precise spatial annotations are derived from real 3D scenes in ScanNet and ScanNet++. The data generation pipeline proceeds as follows:
   - **Scene graph construction**: Object categories and 3D bounding boxes (center coordinates + dimensions) are extracted from 3D point clouds.
   - **3D-to-2D mapping**: Official tools rasterize 3D instances into 2D image masks, establishing 3D–2D correspondences per object.
   - **Filtering**: Uncommon, occluded, truncated, or semantically ambiguous objects are excluded; Qwen2.5-VL relabels objects with descriptive names (e.g., "white table," "wooden desk").
   - **Templated QA generation**: Seven task categories are covered: scale estimation, object localization, distance measurement, size comparison, reference object reasoning, counting, and existence judgment.
   - **LLM-collaborative CoT augmentation**: Reference objects are sampled randomly; Qwen2.5-VL generates reasoning chains, and DeepSeek-V3 evaluates their quality.

   The final dataset comprises 2K scenes, 25K images, 75K objects, **700K QA pairs, 2.5M numerical annotations**, and 10K CoT reasoning samples.

2. **Depth Positional Encoding (DPE)**

   The core design is remarkably concise. The depth map $\mathbf{D} \in \mathbb{R}^{H \times W \times 1}$ is divided into image patches and mean-pooled to obtain $\mathbf{D}' \in \mathbb{R}^{H' \times W' \times 1}$; depth positional embeddings are then generated via sinusoidal functions:

   $\mathbf{E}^{\text{depth}}(i,j,2t) = \sin\left(\frac{\mathbf{D}'(i,j)}{10000^{2t/d}}\right)$
   $\mathbf{E}^{\text{depth}}(i,j,2t+1) = \cos\left(\frac{\mathbf{D}'(i,j)}{10000^{2t/d}}\right)$

   These embeddings are directly **added** to the image features:

   $\mathbf{E}^{\text{vision}} = \mathbf{E}^{\text{image}} + \mathbf{E}^{\text{depth}}$

   **Design Motivation**: Drawing inspiration from the success of Transformer positional encoding, DPE encodes depth as a positional signal along the z-axis (perpendicular to the image plane), effectively upgrading the model's spatial perception from 2D to 3D. This approach introduces no additional sequence length, no extra modules, and minimal training overhead.

3. **MSMU-Bench Evaluation Benchmark**

   Approximately 1K samples from unseen scenes are held out from MSMU, covering all seven spatial task categories. GPT-4 serves as the judge: qualitative questions are scored on a 0–1 scale, while quantitative questions use the ratio $\delta = \max(\hat{d}/d^*, d^*/\hat{d})$ with a threshold of 1.25 to determine success rate.

### Loss & Training

SD-VLM is fine-tuned from LLaVA-1.5-7B using LoRA for 1 epoch on MSMU. The visual encoder is frozen; the LLM and projector are trained with learning rates of 2e-4 and 2e-5, respectively. Training requires 32 GPU-hours on 8 V100 GPUs.

## Key Experimental Results

### Main Results: MSMU-Bench

| Model | Existence | Scale Est. | Abs. Distance | Ref. Reasoning | Avg. |
|-------|-----------|-----------|--------------|----------------|------|
| GPT-4o | 44.68 | 3.86 | 20.00 | 2.09 | 32.28 |
| Gemini-2 | 38.30 | 23.94 | 12.50 | 18.85 | 35.17 |
| InternVL3-78B | 47.62 | 6.47 | 13.33 | 16.46 | 33.63 |
| SpatialRGPT | 10.64 | 20.08 | 15.00 | 9.95 | 28.98 |
| **SD-VLM** | **87.23** | **51.35** | **40.00** | **46.07** | **56.31** |
| **SD-VLM + CoT** | **87.23** | **51.74** | **50.00** | **49.32** | **59.19** |

### Ablation Study: Depth Integration Strategies

| Method | MSMU-Bench Success Rate | Notes |
|--------|------------------------|-------|
| No depth (Baseline) | 46.73% | LLaVA-1.5-7B |
| + depth as image | 22.64% | Degraded; encoder ill-suited for depth maps |
| + depth as prompt | 48.78% | Marginal improvement |
| + depth as token | 35.72% | Longer sequences are detrimental |
| + DPE (estimated depth) | 55.35% | Significant improvement |
| **+ DPE (sincos)** | **56.31%** | Best configuration |

### Key Findings

- SD-VLM achieves 56.2% on Q-Spatial++ and 33.3% on the quantitative tasks of SpatialRGPT-Bench, both state-of-the-art results.
- Even without spatial data—trained solely on the general LLaVA-mix665k—DPE still yields a 25% relative improvement.
- Performance is nearly identical across different depth estimators (DepthAnything vs. UniDepth: 48.6% vs. 47.6%).
- After adding Gaussian noise ($\delta$=0.7), performance drops only from 56.3% to 51.4%, far surpassing the depth-free baseline of 46.7%.
- Incorporating spatial training data does not degrade general VQA performance (VQA-v2: 79.1% vs. 78.5%).

## Highlights & Insights

- **Minimalist design of DPE**: A single sinusoidal function plus an addition operation—no new parameters, no sequence length increase—yet achieves the best performance among all depth integration strategies.
- Constructing the dataset from real 3D scenes (rather than model estimates) ensures numerical annotation accuracy and eliminates systematic bias.
- MSMU's seven task categories offer comprehensive coverage, ranging from basic scale estimation to reference-object-based reasoning.
- The theoretical analysis is rigorous: the proof that four line segments suffice to calibrate intrinsics provides a principled foundation for the claim that VLMs can implicitly learn camera parameters from sufficient physical measurements.

## Limitations & Future Work

- The approach is validated only on LLaVA-1.5-7B (a relatively dated architecture); DPE has not been evaluated on stronger VLMs such as Qwen2.5-VL or InternVL3.
- The MSMU dataset is limited to indoor scenes (ScanNet), with insufficient coverage of outdoor spatial reasoning.
- The accuracy of the depth estimation model constitutes a performance ceiling; more precise depth estimation would yield further gains.
- DPE compresses depth information via mean pooling, potentially losing local depth variations.

## Related Work & Insights

- SpatialRGPT first introduced depth maps as additional image inputs to VLMs; SpatialBot converts depth information into text form.
- Q-Spatial provides a precise, human-annotated benchmark but at a very small scale.
- Advances in monocular depth estimation models such as Depth-Anything-V2 are a prerequisite for practical deployment of DPE.
- **Insight**: Positional encoding is a lightweight, general-purpose mechanism for integrating auxiliary modality information, potentially extensible to temperature fields, semantic fields, and beyond.

## Rating

- **Novelty**: ⭐⭐⭐⭐ DPE is concise yet effective; MSMU is a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-benchmark evaluation, comprehensive depth integration comparisons, robustness analysis, and generalization verification.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous theoretical analysis with clearly organized experiments.
- **Value**: ⭐⭐⭐⭐⭐ Both the dataset and method are open-sourced, advancing the spatial VLM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)

</div>

<!-- RELATED:END -->
