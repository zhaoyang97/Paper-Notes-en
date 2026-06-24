---
title: >-
  [Paper Note] Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset
description: >-
  [AAAI 2026][Medical Imaging][Laparoscopic video desmoking] This paper is the first to categorize surgical smoke into two distinct types — Diffusion Smoke and Ambient Smoke — and proposes STANet, the first smoke-type-aware laparoscopic video desmoking network comprising three sub-networks: semantic soft segmentation, coarse-to-fine disentanglement, and dual-branch reconstruction. It also introduces STSVD, the first large-scale synthetic video desmoking dataset with smoke-type…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Laparoscopic video desmoking"
  - "smoke-type awareness"
  - "smoke mask segmentation"
  - "smoke disentanglement"
  - "surgical video"
date: 2026-05-08
content_hash: 87f1aaa0da4a3205
---

# Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset

**Conference**: AAAI 2026
**arXiv**: [2512.02780](https://arxiv.org/abs/2512.02780)  
**Code**: [GitHub (Dataset)](https://simon-leong.github.io/STSVD/)  
**Area**: Medical Imaging / Video Restoration
**Keywords**: Laparoscopic video desmoking, smoke-type awareness, smoke mask segmentation, smoke disentanglement, surgical video

## TL;DR

This paper is the first to categorize surgical smoke into two distinct types — Diffusion Smoke and Ambient Smoke — and proposes STANet, the first smoke-type-aware laparoscopic video desmoking network comprising three sub-networks: semantic soft segmentation, coarse-to-fine disentanglement, and dual-branch reconstruction. It also introduces STSVD, the first large-scale synthetic video desmoking dataset with smoke-type annotations.

## Background & Motivation

Laparoscopic surgical videos provide real-time visual feedback to surgeons; however, surgical tools such as electrocautery and lasers inevitably generate smoke during tissue ablation, severely degrading video visibility, occluding anatomical structures, and impairing clinical decision-making. Laparoscopic video desmoking is therefore a critical clinical requirement.

**Limitations of Prior Work**:

**Dehazing methods** (DVD, SGDN, DehazeFormer, etc.): Although effective in natural scenes, these methods do not account for the spatiotemporal characteristics of surgical smoke — particularly its motion patterns — and thus perform poorly in surgical settings.

**Desmoking methods** (CycleGAN-based, AALIDNet, SelfSVD, etc.): All treat smoke as a single homogeneous type, ignoring the diversity of smoke motion patterns.

**Key Observation**: The authors identify that surgical smoke can be divided into two fundamentally distinct types based on motion patterns:

- **Diffusion Smoke**: Appears in the early stage of ablation (pre-collision), exhibiting **locality and directionality**, spreading from the tool tip in a specific direction.
- **Ambient Smoke**: Appears in the later stage (post-collision), when smoke collides with the surgical cavity walls and redistributes through turbulence, exhibiting **globality and non-directionality**.

These two smoke types exhibit entirely different spatiotemporal characteristics and therefore require targeted removal strategies. Further complicating the problem, repeated ablation operations cause ambient smoke from previous operations to **entangle** with diffusion smoke from subsequent ones, increasing the difficulty of segmentation and removal.

## Method

### Overall Architecture

STANet is an end-to-end laparoscopic video desmoking network consisting of three sub-networks:

1. **Smoke Feature Awareness Sub-network** (pink region): Extracts and refines smoke video features.
2. **Smoke Mask Segmentation Sub-network** (blue region): Jointly predicts smoke masks and smoke types.
3. **Smoke-Free Video Reconstruction Sub-network** (orange region): Performs dual-branch desmoking guided by the two smoke-type masks.

### Key Designs

1. **Lightweight Non-Rigid Trajectory Attention**: To capture the non-rigid motion characteristics of surgical smoke, the authors draw on the temporal-spatial inversion formulation from SODA and introduce a lightweight non-rigid trajectory attention module. Compared to the original SODA, this module adopts shared projection layers, a window-based attention scheme, and reduced numbers of attention heads and vector dimensions, substantially lowering the computational cost of temporal attention and spatial deformable attention. This enables the network to capture the deformable motion patterns of smoke.

2. **Semantic Soft Segmentation Module (S3M)**: The core component of the smoke mask segmentation sub-network. Soft segmentation of different smoke types is formulated as a **set prediction paradigm**:

    - $N$ learnable queries $q_i$ are used, each predicting a local smoke mask $m_i$ and a smoke type $t_i$.
    - Queries interact with multi-scale spatiotemporal features through three cascaded segmentation blocks (containing mask cross-attention, self-attention, and feed-forward networks).
    - An **attention-weighted mask aggregation mechanism** consolidates local masks into two smoke-type-specific global masks:

   $M^*_{typ} = \sum_i \frac{w_i}{\sum_i w_i} \cdot m_i, \quad \{i \mid t_i = typ\}$

   where $typ$ is either $diff$ (diffusion smoke) or $amb$ (ambient smoke).

3. **Coarse-to-Fine Disentanglement Module (C2FDM)**: The key module for addressing smoke entanglement, embedded after S3M for mask refinement.

    - **Mask Region Selection Sub-module**: Generates three mutually exclusive region masks from $M^*_{diff}$ and $M^*_{amb}$ via binarization and set operations, extracting features for the diffusion smoke region $R^*_{diff}$, the ambient smoke region $R^*_{amb}$, and the entangled region $R^*_{ent}$.
    - **Smoke-Type-Aware Cross-Attention Sub-module**: Features from the three regions are processed via patch embedding and linear projection to produce type-specific queries $Q_{typ}$ and keys $K_{typ}$, along with a shared entanglement value $V_{ent}$. Two cross-attention operations then separate the entangled region into the two smoke types:

   $M'_{typ} = \text{FFN}\left(\text{Softmax}\left(\frac{Q_{typ} K_{typ}^\top}{\sqrt{d}}\right) V_{ent}\right) + M^*_{typ} \cdot B_{typ}$

   - **Iterative Refinement Block**: Consists of four-head self-attention layers, normalization layers, and feed-forward networks that progressively refine the masks to produce the final $M_{diff}$ and $M_{amb}$.

4. **Dual-Branch Smoke-Free Reconstruction Sub-network**: Adopts differentiated strategies based on the distinct characteristics of each smoke type:

    - **Diffusion Smoke Branch** (deformable convolution): Adjacent-frame diffusion smoke masks are concatenated into a temporal composite mask $\bar{M}_{diff}$; a CoordConv layer and channel-reduction attention generate an 18-channel offset field, which is then used by a $3 \times 3$ deformable convolution to extract aligned features along the smoke diffusion trajectory: $F_{diff} = \text{DeformConv}(F, \Delta_{offset})$.
    - **Ambient Smoke Branch** (adaptive dilated convolution): The temporal composite mask $\bar{M}_{amb}$ is used to estimate $K=3$ adaptive dilated sampling position maps; parallel dilated convolutions with dilation rates 1, 2, and 3 are applied and fused: $F_{amb} = \sum_{k=1}^{3} \text{DilatConv}(F, rate_k, map_k)$.
    - Features from both branches are fed into a U-Net decoder to reconstruct the smoke-free video frames. When only one smoke type is present, only the corresponding branch is activated to save computation.

### Loss & Training

A multi-task loss $\mathcal{L}_{mul}$ jointly supervises smoke mask segmentation, smoke type classification, and smoke-free video reconstruction. In addition, a **Smoke High-frequency Wing Loss (SHWL)** $\mathcal{L}_{shwl}$ is introduced:

- A $3 \times 3$ high-pass filter extracts high-frequency components from both the GT mask and the predicted mask.
- Wing Loss is applied to optimize high-frequency errors, placing greater emphasis on otherwise neglected small errors.
- An exponential modulation factor $\phi = 1 + \lambda_g(e^{M_{GT}} - 1)$ increases the penalty in dense smoke regions while reducing overfitting in sparse smoke regions.

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{mul} + \phi \mathcal{L}_{shwl}$

## Key Experimental Results

### Main Results

**Comprehensive comparison across three test sets** (synthetic STSVD + paired real Vivo + unpaired real STSVD-R):

| Method | Type | STSVD PSNR↑ | STSVD SSIM↑ | Vivo PSNR↑ | Vivo SSIM↑ | STSVD-R TOPIQ↑ | STSVD-R MUSIQ↑ |
|--------|------|-------------|-------------|------------|------------|----------------|----------------|
| DehazeFormer | Dehazing | 32.54 | 0.9517 | 23.29 | 0.8648 | 0.3082 | 39.51 |
| SGDN | Dehazing | 32.66 | 0.9674 | 23.15 | 0.8631 | 0.3056 | 40.19 |
| SelfSVD | Desmoking | 29.51 | 0.9588 | 22.13 | 0.8376 | 0.3154 | 38.42 |
| **STANet (Ours)** | **Desmoking** | **33.53** | **0.9733** | **23.84** | **0.8813** | **0.3271** | **40.21** |

Compared to the best prior desmoking method SelfSVD, the proposed method achieves a PSNR gain of **4.02 dB** and SSIM gain of **0.0145** on the synthetic dataset, and a PSNR gain of **1.71 dB** on the paired real dataset.

### Ablation Study

| Config | S3M | C2FDM | SHWL | SVRS | STSVD PSNR↑ | STSVD SSIM↑ | Vivo PSNR↑ | STSVD-R MUSIQ↑ |
|--------|-----|-------|------|------|-------------|-------------|------------|----------------|
| M1 (trained on PSv2rs) | ✗ | ✗ | ✗ | ✗ | 29.30 | 0.9104 | 22.93 | 34.63 |
| M2 (trained on STSVD) | ✗ | ✗ | ✗ | ✗ | 31.19 | 0.9577 | 23.07 | 38.21 |
| M3 | ✓ | ✗ | ✗ | ✗ | 31.90 | 0.9632 | 23.12 | 39.33 |
| M5 | ✓ | ✓ | ✓ | ✗ | 33.09 | 0.9716 | 23.50 | 39.86 |
| M6 (full) | ✓ | ✓ | ✓ | ✓ | 33.53 | 0.9733 | 23.84 | 40.21 |

### Downstream Task Evaluation

| Method | Polyp Detection DSC↑ | Polyp Detection IoU↑ | Instrument Segmentation IoU↑ | Instrument Segmentation mcIoU↑ |
|--------|---------------------|---------------------|------------------------------|-------------------------------|
| Smoky Input | 0.8133 | 0.7813 | 58.51 | 38.59 |
| DehazeFormer | 0.8881 | 0.8203 | 72.77 | 49.34 |
| SelfSVD | 0.8878 | 0.8254 | 68.40 | 45.46 |
| **Ours** | **0.9024** | **0.8394** | **74.81** | **51.90** |

### Key Findings

1. **Importance of the STSVD Dataset**: Switching the training dataset from M1 to M2 yields a substantial PSNR gain of 1.89 dB, demonstrating the critical role of high-quality, diverse, smoke-type-annotated data.
2. **Effectiveness of S3M**: S3M contributes a 0.71 dB PSNR gain in M2→M3, confirming that smoke-type-aware mask prediction significantly improves desmoking quality.
3. **Synergistic Effect of C2FDM + SHWL**: The combination of C2FDM and SHWL contributes a 1.19 dB gain in M3→M5, effectively addressing accurate segmentation in entangled smoke regions.
4. **Downstream Task Generalization**: Desmoking consistently improves both polyp detection and instrument segmentation, with the proposed method achieving best performance on all downstream tasks.

## Highlights & Insights

- **Innovation in Problem Formulation**: The paper is the first to analyze and process surgical smoke as two distinct types, a classification grounded in physical reality (smoke turbulence induced by collision with cavity walls), which is highly well-motivated.
- **End-to-End Type-Aware Design**: The entire pipeline — from mask segmentation to video reconstruction — consistently accounts for smoke-type differences. Deformable convolution is employed for the directional diffusion smoke, while dilated convolution is used for the global ambient smoke, achieving strong alignment between design choices and physical characteristics.
- **Elegant Disentanglement Mechanism**: C2FDM leverages information from non-entangled regions to guide the separation of entangled regions, achieving type-aware disentanglement via cross-attention, which represents the central technical contribution of the method.
- **Complete Data Ecosystem**: Beyond the method, the paper also constructs the first large-scale dataset with smoke-type annotations (120 videos, 12,000 frames, 28 surgical types), substantially advancing the research foundation of this field.

## Limitations & Future Work

1. **Synthetic-to-Real Domain Gap**: Although the method performs well on real data, training still relies on synthetic data, and a gap between synthetic and real smoke may persist.
2. **Simplified Smoke Type Assumption**: Strictly categorizing smoke into two types may not cover all real-world scenarios, as more transitional states may exist in practice.
3. **Computational Efficiency**: With 25.15M parameters and 174.13G FLOPs, the method, while not the most expensive, may still require optimization for real-time surgical applications.
4. **Limited Temporal Modeling Depth**: The current approach only exploits temporal information from adjacent frames, which may be insufficient for modeling long-term smoke variation patterns.
5. **Absence of Real Smoke-Type Annotations**: The STSVD-R real dataset only provides scene-level annotations, lacking pixel-level ground-truth smoke-type labels for more rigorous validation.

## Related Work & Insights

- The concept of smoke-type classification could be extended to other video degradation problems, such as differentiating haze by density or motion state for type-specific processing.
- The disentanglement idea in C2FDM has implications for scenes with mixed degradation types (e.g., rain + haze, noise + blur).
- Constructing task-specific large-scale synthetic datasets with fine-grained annotations (e.g., type labels) is an important infrastructure investment for the continued progress of data-driven methods.
- Evaluating desmoking methods on downstream tasks demonstrates their impact on the broader surgical assistance pipeline, and this evaluation paradigm is worth advocating.

## Rating

| Dimension | Score (1–5) | Remarks |
|-----------|-------------|---------|
| Novelty | ⭐⭐⭐⭐ | First to propose smoke-type classification and a type-aware desmoking framework |
| Practicality | ⭐⭐⭐⭐ | Directly serves surgical assistance systems |
| Theoretical Depth | ⭐⭐⭐ | More engineering-oriented; theoretical analysis is limited |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Three test sets + detailed ablation + downstream task evaluation |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with rich figures and tables |
| Overall | ⭐⭐⭐⭐ | Novel problem formulation, complete methodology, and outstanding dataset contribution |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](../../CVPR2026/medical_imaging/synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)
- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [\[AAAI 2026\] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)
- [\[CVPR 2026\] TRCoRSurg: Temporal-Relational Co-Reasoning for Surgical Video Triplet Recognition](../../CVPR2026/medical_imaging/trcorsurg_temporal-relational_co-reasoning_for_surgical_video_triplet_recognitio.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](../../CVPR2026/medical_imaging/lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)

</div>

<!-- RELATED:END -->
