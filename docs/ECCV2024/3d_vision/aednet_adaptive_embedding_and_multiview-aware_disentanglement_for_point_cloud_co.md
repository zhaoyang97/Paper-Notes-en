---
title: >-
  [Paper Note] AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion
description: >-
  [ECCV 2024][3D Vision][Point Cloud Completion] AEDNet is proposed, which conducts global embedding and local disentanglement of point clouds in the encoder and decoder respectively through the Adaptive Embedding and Multiview-Aware Disentanglement (AED) module. By utilizing 3D viewpoints generated from a unit sphere to observe the point cloud from the outside, it achieves a comprehensive understanding of 3D object geometry, reaching SOTA on the MVP and PCN datasets.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Slot Attention"
  - "Multiview Disentanglement"
  - "Global Embedding"
  - "Geometric Understanding"
date: 2026-05-08
content_hash: 12ca84d6bff9a3a6
---

# AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion

**Conference**: ECCV 2024  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Point Cloud Completion, Slot Attention, Multiview Disentanglement, Global Embedding, Geometric Understanding

## TL;DR

AEDNet is proposed, which conducts global embedding and local disentanglement of point clouds in the encoder and decoder respectively through the Adaptive Embedding and Multiview-Aware Disentanglement (AED) module. By utilizing 3D viewpoints generated from a unit sphere to observe the point cloud from the outside, it achieves a comprehensive understanding of 3D object geometry, reaching SOTA on the MVP and PCN datasets.

## Background & Motivation

**Background**: Point cloud completion is a fundamental task in 3D vision, aiming to infer the missing parts of 3D objects from incomplete point cloud data. This task requires models to both understand the global structure of objects and reconstruct local details. Existing methods typically adopt encoder-decoder architectures, extracting global features through encoders and generating complete point clouds via decoders.

**Limitations of Prior Work**: Existing point cloud completion methods show a key trade-off between global structure understanding and local detail reconstruction. Many approaches either focus excessively on global shapes while neglecting details, or showcase strong detail reconstruction capability but lack global consistency. Traditional methods typically encode the entire point cloud into a single global feature vector, which mixes the geometric information of different parts together, making them difficult to distinguish and reconstruct.

**Key Challenge**: How to simultaneously achieve global perception and local focus for point clouds within a unified framework? Direct encoding of all points fails to effectively distinguish the geometric information of different object parts, leading to a lack of precision in local regions of the generated point clouds.

**Goal**: 1) How to decompose the complete point cloud into multiple meaningful part embeddings; 2) how to effectively disentangle local geometric information from these embeddings; 3) how to flexibly control the number and features of generated points.

**Key Insight**: The authors observe that a more comprehensive geometric understanding can be obtained by observing the point cloud from multiple viewpoints outside the 3D object rather than from the inside. Inspired by Slot Attention, the point cloud is decomposed into multiple "slot" embeddings, where each slot focuses on a specific part of the object, which are then disentangled through multiple 3D viewpoints generated on a unit sphere.

**Core Idea**: Utilizing Slot Attention to embed the point cloud into multiple part representations, and then disentangling geometric information via multiview projection on a unit sphere, thereby achieving a joint global-to-local understanding from the outside in.

## Method

### Overall Architecture

AEDNet adopts an encoder-decoder architecture. The input is an incomplete point cloud, and the output is the completed point cloud. The core lies in the use of the Adaptive Embedding and Disentanglement (AED) module in both the encoder and decoder. The AED module in the encoder decomposes the input point cloud into multiple global embeddings, each focusing on a specific region of the object. The AED module in the decoder then disentangles geometric information from these embeddings to generate the final complete point cloud. The overall process can be summarized as: Incomplete point cloud $\to$ Global embeddings (decomposed into multiple part representations) $\to$ Multiview disentanglement (extracting geometry from spherical viewpoints) $\to$ Point cloud generation.

### Key Designs

1. **Adaptive Point Cloud Embedding**:

    - **Function**: Decomposes the input point cloud into multiple independent embedding representations, with each embedding focusing on a specific part of the object.
    - **Mechanism**: Drawing inspiration from Slot Attention, a global embedding operator is designed. It first initializes a set of learnable slot query vectors, which then establish associations with different parts of the input point cloud through an iterative attention mechanism. Specifically, slots compete for attention weights over different points through a competition mechanism (softmax-normalized attention weights), causing different slots to naturally focus on different regions of the object. This competitive assignment ensures diversity and complementarity among embeddings.
    - **Design Motivation**: Traditional methods encode all points into a single vector, losing part-level structural information. Through the grouping mechanism of Slot Attention, the independent geometric information of each object part can be preserved, providing a foundation for subsequent fine-grained reconstruction.

2. **Multiview-Aware Disentanglement**:

    - **Function**: Disentangles detailed geometric information from global embeddings, supporting flexible generation of point counts.
    - **Mechanism**: A set of 3D viewpoints is uniformly sampled on a unit sphere, with each viewpoint representing an external angle of observation. Then, global embeddings are projected along these viewpoint directions, disentangling geometric features corresponding to the regions via a viewpoint-aware attention mechanism. The key innovation is "observing from the outside" rather than "encoding from the inside"—traditional methods process features directly within the point cloud, while this method yields a comprehensive geometric understanding via external viewpoints. By varying the number of viewpoints, the number of disentangled points and feature complexity can be flexibly controlled.
    - **Design Motivation**: When encoding from inside the point cloud, each point can only perceive its neighboring information, limiting its grasp of the overall geometric structure. By observing from multiple viewpoints outside the sphere, each viewpoint obtains global contour information of the object while focusing on details of specific regions. This design also brings high flexibility—the number of viewpoints is adjustable, directly corresponding to the number of output points.

3. **AED Module**:

    - **Function**: Integrates global embedding and multiview disentanglement and serves as the core component of both the encoder and decoder.
    - **Mechanism**: The AED module connects the two operations in series. In the encoding phase, the module first aggregates the point cloud into global embeddings via Slot Attention, and then extracts local features through multiview disentanglement. In the decoding phase, the AED module is in like manner utilized but in the opposite direction—starting from global embeddings, it gradually generates a complete point cloud via progressive multiview disentanglement. The encoder and decoder share the same module design but with independent parameters, forming a symmetric structure.
    - **Design Motivation**: Reusing the same AED structure in both the encoder and decoder simplifies network design while ensuring consistency in the representation space during encoding and decoding. The symmetric design also contributes to training stability.

### Loss & Training

Training adopts Chamfer Distance (CD) as the primary loss function, which measures the distance between the predicted point cloud and the ground truth complete point cloud. The CD loss simultaneously considers the nearest neighbor distances from predicted-to-ground-truth and ground-truth-to-predicted, ensuring that the generated point cloud covers all regions without excessive redundant points. In addition, a multi-scale supervision strategy is employed at intermediate layers, computing CD loss separately for coarse-level and fine-level predictions to guide the network in learning point cloud completion from coarse to fine.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (AEDNet) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| MVP (CD-l2 ×10^4) | Overall Average | Best | PCN/SnowFlakeNet, etc. | Significant improvement |
| PCN (CD-l1 ×10^3) | Overall Average | SOTA | SeedFormer, etc. | Continuous improvement |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full AEDNet | Best CD | Full model |
| w/o Slot Attention | CD increases | Removes global embedding, degenerating into standard encoding |
| w/o Multiview Disentanglement | CD increases | Removes spherical viewpoint disentanglement, replaced by traditional FoldingNet |
| Different Viewpoint Counts | Smooth variation | Number of viewpoints is positively correlated with generation quality but has a saturation point |
| Different Slot Counts | Affects part granularity | Too few slots lose details, while too many increase redundancy |

### Key Findings
- The global embedding mechanism of Slot Attention contributes the most to completion quality. Removing it leads to a significant increase in CD, indicating that part-level decomposition is the core advantage of the method.
- The number of viewpoints in multiview disentanglement provides a flexible accuracy-efficiency trade-off: increasing viewpoints improves reconstruction accuracy but at the cost of higher computational overhead.
- The model performs particularly well on objects with symmetric structures or regular shapes, as spherical viewpoints are naturally suited to capturing symmetric geometry.
- It still maintains good performance on highly incomplete inputs (high missing ratios), benefited by the global embedding's understanding of the overall structure.

## Highlights & Insights

- **Perspective Shift to External Observation**: Translating traditional "inside-out point cloud encoding" to "looking at the point cloud from external spherical viewpoints," resembling how humans understand a 3D object from multiple angles. This perspective shift enables every viewpoint to capture global information, avoiding the limitations of local receptive fields.
- **Innovative Application of Slot Attention in Point Clouds**: Slot Attention was originally designed for object discovery in 2D images. This work successfully transfers it to 3D point cloud processing, achieving automatic part decomposition via a competitive assignment mechanism without requiring extra part annotations.
- **Flexible Point Count Control**: The output point cloud resolution can be altered simply by adjusting the number of spherical viewpoints. This design is highly practical for real-world applications.

## Limitations & Future Work

- Uniform sampling of spherical viewpoints might not be optimal—for elongated or flat objects, certain viewpoints may provide more information, and adaptive viewpoint sampling based on object shapes could be considered.
- The initialization of Slot Attention relies on learnable parameters; category-specific initialization strategies may be required for categories with extreme shape variance.
- When dealing with highly incomplete point clouds (e.g., only a small fraction is visible), the global embedding may be limited by the lack of sufficient input information.
- The current method might be sensitive to point cloud density, and its robustness under sparse inputs is worth further investigation.

## Related Work & Insights

- **vs SeedFormer**: SeedFormer performs progressive upsampling through seed features, emphasizing local geometry generation. AEDNet simultaneously captures global and local information via global embedding and multiview disentanglement, yielding superior global consistency.
- **vs SnowFlakeNet**: SnowFlakeNet uses a snowflake-like point splitting strategy to progressively increase density. In contrast, AEDNet directly generates the point cloud of target density via viewpoint-driven control, avoiding error accumulation from multi-step upsampling.
- **vs PoinTr**: PoinTr employs a Transformer for point cloud completion, utilizing sequential processing. AEDNet realizes more natural part grouping via Slot Attention, and observation from spherical viewpoints provides unique geometric priors.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Slot Attention and spherical multiview disentanglement to point cloud completion, presenting a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two mainstream datasets, MVP and PCN, with relatively complete ablation studies.
- Writing Quality: ⭐⭐⭐ The methodology is clearly described, though some technical details require reference to the appendix.
- Value: ⭐⭐⭐⭐ Provides a flexible and effective paradigm for point cloud completion, and the idea of spherical viewpoints is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](explicitly_guided_information_interaction_network_for_cross-modal_point_cloud_co.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](../../AAAI2026/3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[ECCV 2024\] Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation](dual-level_adaptive_self-labeling_for_novel_class_discovery_in_point_cloud_segme.md)
- [\[ECCV 2024\] Hiding Imperceptible Noise in Curvature-Aware Patches for 3D Point Cloud Attack](hiding_imperceptible_noise_in_curvature-aware_patches_for_3d_point_cloud_attack.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](../../AAAI2026/3d_vision/dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)

</div>

<!-- RELATED:END -->
