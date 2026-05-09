---
title: >-
  [Paper Note] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation
description: >-
  [NeurIPS 2025][Autonomous Driving][open-vocabulary segmentation] This paper proposes Vireo, the first single-stage framework that unifies open-vocabulary semantic segmentation (OVSS) and domain-generalized semantic segmentation (DGSS). By introducing GeoText Query to fuse depth-geometric features with linguistic cues, Vireo achieves state-of-the-art performance under both extreme environmental conditions and on unseen categories.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - open-vocabulary segmentation
  - domain generalization
  - depth estimation
  - visual foundation models
  - semantic segmentation
date: 2026-05-08
content_hash: 96417e6056feea9f
---

# Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2506.09881](https://arxiv.org/abs/2506.09881)
**Code**: [GitHub](https://github.com/SY-Ch/Vireo)
**Area**: Autonomous Driving / Semantic Segmentation
**Keywords**: open-vocabulary segmentation, domain generalization, depth estimation, visual foundation models, semantic segmentation

## TL;DR

This paper proposes Vireo, the first single-stage framework that unifies open-vocabulary semantic segmentation (OVSS) and domain-generalized semantic segmentation (DGSS). By introducing GeoText Query to fuse depth-geometric features with linguistic cues, Vireo achieves state-of-the-art performance under both extreme environmental conditions and on unseen categories.

## Background & Motivation

**State of the Field**: OVSS enables recognition of arbitrary text-described categories, while DGSS maintains robustness on unseen domains; each paradigm has complementary strengths.

**Limitations of Prior Work**: The text-visual alignment modules in OVSS degrade significantly under out-of-domain conditions (e.g., nighttime, rain); domain-invariant strategies in DGSS may suppress fine-grained semantic cues, compromising precise response to text queries.

**Root Cause**: How can cross-domain robustness and open-vocabulary recognition be achieved simultaneously? DGSS focuses on encoder-side feature generalization, while OVSS focuses on decoder-side open recognition — the two paradigms are naturally complementary.

**Paper Goals**: Construct a unified OV-DGSS (Open-Vocabulary Domain-Generalized Semantic Segmentation) framework that robustly segments unseen categories under domain shift.

**Starting Point**: Exploit the domain-invariant nature of depth maps (depth and geometric cues are insensitive to illumination and texture variations), combined with the generalization capacity of frozen visual foundation models (VFMs).

**Core Idea**: Inject depth-geometric and text-semantic priors into frozen VFM layers via GeoText Query, augmented by CMPE for enhanced gradient flow and DOV-VEH for multi-modal feature fusion.

## Method

### Overall Architecture

Vireo comprises three core modules:
- **Tunable Vireo + GeoText Query**: Injects and aligns geometric and textual information between layers of the frozen VFM encoder.
- **Coarse Mask Prior Embedding (CMPE)**: Generates coarse prior masks to enhance gradient back-propagation.
- **Domain-Open-Vocabulary Vector Embedding Head (DOV-VEH)**: Fuses visual, geometric, and semantic features to produce final predictions.

Input images are fed simultaneously into a frozen VFM encoder and a frozen DepthAnything V2 encoder for depth estimation; text category labels are encoded into semantic embeddings via a frozen CLIP text encoder.

### Key Designs

1. **GeoText Query**:

    - **Function**: Injects structural-semantic priors between layers of the frozen VFM, progressively refining features layer by layer.
    - **Design Motivation**: Depth features provide domain-invariant spatial constraints that mitigate domain shift in RGB features; text embeddings provide open-vocabulary semantic alignment.
    - **Mechanism**: At each layer, a learnable query $P_l$ interacts with visual features $f_l^V$, depth features $f_l^D$, and text embeddings $\{t_k\}$ via cross-attention:
    $\mathcal{A}_l = \text{CrossAttn}(P_l, f_l^V, f_l^D, \{t_k\})$
      The attention output refines the visual representation $\hat{f}_l^V$ through weighted summation, MLP projection, and residual connection.
    - **Novelty**: Unlike REIN, which performs only prompt tuning, GeoText Query simultaneously integrates depth and text as cross-modal priors.

2. **Coarse Mask Prior Embedding (CMPE)**:

    - **Function**: Generates coarse semantic probability maps to provide denser gradient signals back through the frozen encoder.
    - **Design Motivation**: Freezing the encoder leads to sparse gradients and slow convergence; CMPE injects richer gradient signals to alleviate this.
    - **Mechanism**: Refined features from VFM layers 8/12/16/24 are upsampled to a unified resolution and fused via an Adaptive Attention Gate (AAG). Coarse masks are computed against text embeddings via Einstein summation: $\mathcal{M}(x,y,k) = \langle f^M(x,y), t_k \rangle$. Query priors are further generated via softmax normalization:
    $q_j^{prior} = \sum_k \text{Softmax}(\langle q_j, e_k^{class} \rangle) \cdot e_k^{class}$

3. **DOV-VEH (Domain-Open-Vocabulary Vector Embedding Head)**:

    - **Function**: Fuses multi-scale refined features to generate pixel-level segmentation masks.
    - **Mechanism**: Multi-scale features are enhanced spatially through a Pixel Decoder; a Transformer Decoder then allows GeoText Query to interact with decoded features and text embeddings, yielding mask embeddings $\mathcal{E}_{mask}$ and classification embeddings $\mathcal{E}_{cls}$. The final prediction is:
    $\hat{\mathcal{M}}(x,y,k) = \sum_d \mathcal{E}_{mask}(x,y,d) \cdot \mathcal{E}_{cls}(k,d)$

### Loss & Training

- Optimizer: AdamW with initial learning rate 1e-4 and weight decay 0.05.
- Polynomial learning rate decay over 40K total iterations.
- Data augmentation: multi-scale resizing, random cropping, random horizontal flipping, and photometric distortion.
- Trained on a single RTX A6000 GPU, batch size 8, approximately 14 hours.

## Key Experimental Results

### Main Results

**Domain Generalization (Cityscapes → ACDC + BDD + Mapillary, mIoU %)**:

| Method | Night-ACDC | Fog-ACDC | Rain-ACDC | Snow-ACDC | BDD100k | Mapillary |
|--------|-----------|---------|----------|----------|---------|----------|
| FC-CLIP (OVSS) | 40.8 | 64.4 | 63.2 | 61.5 | 55.9 | 66.1 |
| REIN (DGSS) | 55.9 | 79.5 | 72.5 | 70.6 | 63.5 | 74.0 |
| FADA (DGSS) | 57.4 | 80.2 | 75.0 | 73.5 | 65.1 | 75.9 |
| **Vireo** | **60.6** | **82.3** | **76.3** | **76.2** | **66.7** | **76.0** |

**Open-Vocabulary Generalization (Cityscapes → DELIVER + ADE, mIoU %)**:

| Method | Sun | Night | Cloud | Rain | Fog | ADE150 | ADE847 |
|--------|-----|-------|-------|------|-----|--------|--------|
| CAT-Seg | 28.2 | 20.6 | 26.2 | 26.5 | 24.8 | 20.2 | 7.0 |
| **Vireo** | **35.7** | **27.5** | **32.3** | **31.8** | **32.7** | **21.4** | **7.3** |

### Ablation Study

**Component Ablation (Cityscapes → ACDC + BDD + Map, mIoU %)**:

| Configuration | Snow | Night | Fog | Rain | BDD | Map |
|--------------|------|-------|-----|------|-----|-----|
| REIN (baseline) | 70.6 | 55.9 | 79.5 | 72.5 | 63.5 | 74.0 |
| + DepthAnything V2 | 71.5 | 56.7 | 80.5 | 73.3 | 64.4 | 74.5 |
| + GeoText Query | 74.0 | 58.4 | 81.1 | 74.8 | 65.3 | 75.3 |
| **Vireo (full)** | **76.2** | **60.6** | **82.3** | **76.3** | **66.7** | **76.0** |

**Multi-Backbone Generalization (GTA5 → Citys+BDD+Map, mIoU)**:

| Backbone | REIN | FADA | Vireo | Trainable Params |
|----------|------|------|-------|-----------------|
| EVA02-L | 63.6 | 64.9 | **66.0** | 3.78M |
| DINOv2-L | 64.3 | 66.1 | **67.7** | 3.78M |

### Key Findings

- GeoText Query is the most critical component, yielding approximately +2.5% mIoU in nighttime scenes.
- Depth-geometric features provide the greatest benefit under extreme weather conditions, particularly nighttime and snow.
- Vireo outperforms existing OVSS methods on both seen and unseen categories, with a more pronounced advantage on unseen categories (>+7%).
- Only 3.78M trainable parameters are required — significantly fewer than FADA (11.65M) — while achieving superior performance.

## Highlights & Insights

- **Meaningful Problem Formulation**: This work is the first to define the OV-DGSS problem and propose a unified framework, closely aligned with the practical requirements of autonomous driving.
- **Effective Use of Depth Cues**: Depth maps are inherently domain-invariant; leveraging frozen DepthAnything V2 for geometric feature extraction is a lightweight yet effective strategy.
- **Complementarity Insight**: DGSS optimizes encoder-side generalization while OVSS optimizes decoder-side recognition — Vireo simultaneously advances both ends.
- **Parameter Efficiency**: State-of-the-art performance with only 3.78M trainable parameters, making deployment practical.

## Limitations & Future Work

- Depth estimation quality depends on DepthAnything V2, which may be unreliable in extreme scenes such as dense fog or nighttime.
- Coarse masks generated by CMPE are of limited quality and may introduce noisy priors.
- Absolute performance on ADE847 remains low (7.3% mIoU); ultra-fine-grained segmentation across 847 categories remains challenging.
- Training memory requirements are high (~45 GB GPU memory), and inference efficiency warrants further optimization.
- Validation is limited to autonomous driving scenarios; other OV-DGSS applications (e.g., medical imaging) remain unexplored.

## Related Work & Insights

- **REIN / FADA**: VFM-based DGSS methods that insert learnable modules into frozen VFMs to improve domain generalization.
- **FC-CLIP / CAT-Seg**: OVSS methods that leverage CLIP alignment between vision and text for open-vocabulary recognition.
- **DepthForge**: Demonstrates that injecting depth queries into frozen VFMs improves domain generalization, inspiring this work's use of DepthAnything.
- **Insight**: The paradigm of using depth/geometry as domain-invariant anchors can be extended to video segmentation, 3D scene understanding, and related tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First to define the OV-DGSS problem and propose a unified framework; GeoText Query design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight datasets, five evaluation settings, multi-backbone validation, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, experiments are well-organized, and method descriptions are complete.
- Value: ⭐⭐⭐⭐⭐ Unifying OV and DG is highly practical for autonomous driving scenarios, with strong parameter efficiency.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[NeurIPS 2025\] OpenBox: Annotate Any Bounding Boxes in 3D](openbox_annotate_any_bounding_boxes_in_3d.md)
- [\[NeurIPS 2025\] Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation](extremely_simple_multimodal_outlier_synthesis_for_out-of-distribution_detection_.md)

<!-- RELATED:END -->
