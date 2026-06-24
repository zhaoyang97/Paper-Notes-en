---
title: >-
  [Paper Note] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data
description: >-
  [CVPR 2025][Segmentation][Multimodal Segmentation] This paper proposes the SGMA framework to address three major challenges in incomplete multimodal remote sensing segmentation: modality imbalance, intra-class variations, and cross-modality heterogeneity. Specifically, a Semantic-Guided Fusion (SGF) module constructs global semantic prototypes to estimate modality robustness for adaptively weighted fusion, while a Modality-Aware Sampling (MAS) module dynamically prioritizes t…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Multimodal Segmentation"
  - "Incomplete Modalities"
  - "Semantic Prototypes"
  - "Adaptive Sampling"
  - "Remote Sensing Images"
date: 2026-05-08
content_hash: 7384c6a432d7bc3a
---

# SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data

**Conference**: CVPR 2025  
**arXiv**: [2603.02505](https://arxiv.org/abs/2603.02505)  
**Code**: To be confirmed  
**Area**: Segmentation / Remote Sensing  
**Keywords**: Multimodal Segmentation, Incomplete Modalities, Semantic Prototypes, Adaptive Sampling, Remote Sensing Images

## TL;DR
This paper proposes the SGMA framework to address three major challenges in incomplete multimodal remote sensing segmentation: modality imbalance, intra-class variations, and cross-modality heterogeneity. Specifically, a Semantic-Guided Fusion (SGF) module constructs global semantic prototypes to estimate modality robustness for adaptively weighted fusion, while a Modality-Aware Sampling (MAS) module dynamically prioritizes training fragile modalities.

## Background & Motivation
**Background**: Remote sensing semantic segmentation requires integrating data from multiple sensors (RGB, DSM, NIR, SAR). Multimodal Semantic Segmentation (MSS) has become a key technology for enhancing scene understanding by utilizing complementary multi-source information.

**Limitations of Prior Work**: In practice, sensor failures or incomplete coverage often lead to missing modalities, known as Incomplete Multimodal Semantic Segmentation (IMSS). Existing methods suffer from three critical issues: (1) Modality imbalance—dominant modalities (e.g., RGB) suppress fragile modalities (e.g., DSM, NIR); (2) Large intra-class variations—objects of the same category vary significantly in scale, shape, and orientation (e.g., buildings of different sizes, where small buildings exhibit weak and sparse feature activations); (3) Cross-modality heterogeneity—different modalities generate conflicting semantic responses (e.g., rooftops and grounds share similar colors in RGB but differ in height in DSM, while grounds and grasslands have similar heights in DSM but differ in color in RGB).

**Key Challenge**: Existing methods (such as contrastive learning and joint optimization) either over-align and discard modality-specific information, or bias the training towards dominant modalities, while largely ignoring the problems of intra-class variation and cross-modality heterogeneity. Dropout-based methods fail to learn discriminative representations for fragile modalities; Masked Autoencoders (MAE) focus on low-level reconstruction rather than high-level semantics; and contrastive alignment forces cross-modality consistency, which may weaken modality-specific features.

**Goal**: How to maintain robust segmentation performance under arbitrary modality dropping while simultaneously handling modality imbalance, large intra-class variations, and cross-modality heterogeneity?

**Key Insight**: Compressing multimodal features into global semantic prototypes—class-level intermediate representations—can both reduce intra-class variance and measure the contribution of each modality to each category through attention mechanisms, thereby enabling robustness-aware adaptive fusion.

**Core Idea**: Utilizing semantic prototypes as a cross-modality bridge to estimate modality robustness for adaptive fusion, while conversely using the robustness scores to guide sampling to prioritize training fragile modalities.

## Method

### Overall Architecture
Given multimodal remote sensing images $\{I_m\}_{m\in\mathcal{M}}$ as input, a weight-sharing encoder $F$ independently extracts four scales of features from each modality. During training, the features pass through both SGF and MAS branches simultaneously to generate $\hat{S}_{\text{SGF}}$ and $\hat{S}_{\text{MAS}}$ for joint optimization. During inference, only the SGF branch is retained. Both modules are plug-and-play and compatible with various backbones.

### Key Designs

1. **Semantic-Guided Fusion (SGF)**:

    - **Function**: Compresses multimodal features into global semantic prototypes, estimates the robustness of each modality using these prototypes, and performs adaptively weighted fusion.
    - **Mechanism**: Contains four sub-components: (a) Modality-specific Projector (MP)—maps modality features to a unified semantic space using multi-scale depthwise convolutions (11x11, 7x7, 3x3); (b) Class-aware Semantic Filter (CSF)—compresses channels to the number of classes $K$ via weight-sharing 1x1 convolutions to extract class-level representations; (c) Spatial Perceptron (SP)—utilizes the global semantic prototypes $p_{se}^{i,k}$ as queries and multimodal features as keys/values for multi-head attention to reduce intra-class variance; (d) Robustness Perceptron (RP)—performs attention with the semantic-guided features $f_{se}^i$ as queries to simultaneously obtain fused features and robustness scores for each modality.
    - **Design Motivation**: Serving as class-level intermediate anchors, semantic prototypes associate pixels directly with semantic centroids to reduce intra-class variance. The RP uses semantic-guided features instead of raw features as queries, ensuring that attention weights reflect class-dependent robustness (e.g., high scores for DSM on structural classes, high scores for NIR on vegetation classes).

2. **Modality-Aware Sampling (MAS)**:

    - **Function**: Dynamically adjusts training sampling probabilities based on SGF robustness scores, prioritizing the sampling of fragile modalities.
    - **Mechanism**: Normalizes the inverse of robustness scores as $\hat{r}_m^i = \frac{1/r_m^i}{\sum_{m'}(1/r_{m'}^i)}$, meaning modalities with lower robustness receive higher sampling probabilities. By increasing the training frequency of fragile modalities, their features are trained independently.
    - **Design Motivation**: Decoupling fragile modality training prevents dominant modalities from suppressing the learning of fragile modalities through gradients. It achieves balance via sampling strategies without requiring modality-specific architectural modifications.

3. **Global Semantic Prototype Construction**:

    - **Function**: Extracts class-level global representations from the features of all modalities.
    - **Mechanism**: $\{p_{se}^{i,k}\}_{k=1}^K = [\{c_m^i\}_{m\in\mathcal{M}}] \otimes [\{f_{m\to se}^i\}_{m\in\mathcal{M}}]^T$, where $c_m^i$ is the CSF-compressed class weight, yielding global semantic prototypes for each class through matrix multiplication.
    - **Design Motivation**: Integrating information from all modalities to construct prototypes provides a global receptive field, which enhances class consistency beyond the limitations of local features.

### Loss & Training
$\mathcal{L}_{\text{IMSS}} = \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}(\hat{S}_{\text{SGF}}, S) + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}(\hat{S}_{\text{MAS}}, S) + \lambda_{\text{con}} \sum \mathcal{L}_{\text{con}}(f_m, f_{m'})$

During training, modality dropout is utilized to simulate missing modalities, and the SGF and MAS branches are jointly optimized. During inference, only the SGF branch is used.

## Key Experimental Results

### Main Results
Evaluation on three remote sensing datasets (ISPRS, DFC2023, and another) across seven modality combinations using two backbones:

| Dataset (Pvt-v2-b2) | Metric | Ours (SGMA) | Prev. SOTA (MAGIC) | Gain |
|---|---|---|---|---|
| ISPRS mIoU | Average | 87.84 | 77.37 | **+10.47** |
| ISPRS mIoU | Last-1 (Worst Combination) | 70.36 | 45.14 | **+25.22** |
| ISPRS F1 | Average | 83.51→Full Combination | 81.39 | +2.12 |
| ISPRS F1 | Last-1 | 57.05 | 34.34 | **+22.71** |

Key comparison—single-modality DSM (Pvt-v2-b2): SGMA 70.36 vs MAGIC 45.14 (+25.22), showing that SGMA significantly boosts the independent capability of fragile modalities.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Baseline | - | Without SGF and MAS |
| + SGF only | Significant gain | Effect of semantic prototype fusion |
| + MAS only | Moderate gain | Effect of sampling rebalancing |
| + SGF + MAS | Optimal | Complementarity of both |
| Removal of SP component | Performance drops | Validates the contribution of reducing intra-class variance |
| Removal of RP component | Performance drops | Validates the contribution of robustness estimation |

### Key Findings
- SGMA achieves the largest gains when fragile modalities are used in isolation (over 25 points gain on single-modality DSM), demonstrating that MAS effectively enhances the representation of fragile modalities.
- Improvements are also observed under full modalities (a Top-1 gain of 0.35 points), indicating that SGF's semantic fusion also improves full-scene performance.
- Consistently effective across different backbones (Pvt-v2-b2 and ResNet-50).
- Plug-and-play design: The SGF and MAS modules can be used independently or jointly.

## Highlights & Insights
- **Dual Role of Semantic Prototypes**: Serving as both class-level anchors to reduce intra-class variance and attention weights to quantify modality robustness—solving two issues with a single mechanism.
- **Robustness Closed Loop**: SGF estimates robustness $\to$ MAS guides sampling with robustness $\to$ fragile-modality learning is enhanced $\to$ SGF estimates robustness more accurately, forming a virtuous cycle.
- **Significant Gain in Last-1**: The performance of the worst modality combination is a major bottleneck in IMSS. SGMA improves this metric by over 20 points, demonstrating a true resolution of the fragile modality issue.

## Limitations & Future Work
- Evaluated only in remote sensing scenarios; other multimodal contexts such as medical imaging and autonomous driving have not been tested.
- Semantic prototype computation and dual-branch training increase computational overhead.
- The MAS branch is discarded during inference; however, whether the training signals from MAS can be more effectively transferred to SGF remains worth exploring.
- Prototypes work well when the number of classes $K$ is small, but their discriminative power might decline when the number of classes is extremely large (e.g., 100+ classes).

## Related Work & Insights
- **vs MAGIC**: Classifies modalities into robust/fragile groups for joint optimization and cosine alignment, though alignment may weaken modality-specific information. SGMA performs class-level adaptive fusion via semantic prototypes, preserving modality specificity.
- **vs IMLT**: Employs contrastive learning and masked pretraining, but suffers from contrastive learning's over-alignment issues. SGMA replaces contrastive alignment with semantic prototypes.
- **vs MuSS/M3L**: Earlier methods lag significantly behind in performance, demonstrating that simple dropout or grouping strategies are insufficient.

## Rating
- Novelty: ⭐⭐⭐⭐ Robustness estimation and sample guidance via semantic prototypes is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, multiple backbones, multiple modality combinations, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed methodology description.
- Value: ⭐⭐⭐⭐ A highly practical solution for remote sensing IMSS, offering massive boosts for fragile modalities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](../../CVPR2026/segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[CVPR 2026\] HySeg: Learning Generative Priors for Structure-Aware Remote Sensing Segmentation](../../CVPR2026/segmentation/hyseg_learning_generative_priors_for_structure-aware_remote_sensing_segmentation.md)

</div>

<!-- RELATED:END -->
