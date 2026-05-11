---
title: >-
  [Paper Note] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation
description: >-
  [ICCV 2025][Segmentation][tubular structure segmentation] The first test-time adaptation (TTA) framework specifically designed for tubular structure segmentation (TSS). It adapts to cross-domain topological structural di…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "tubular structure segmentation"
  - "test-time adaptation"
  - "topological continuity"
  - "differential convolution"
  - "domain shift"
date: 2026-05-08
content_hash: 47cabbad0502b0c6
---

# TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation

**Conference**: ICCV 2025
**arXiv**: [2508.00442](https://arxiv.org/abs/2508.00442)
**Code**: None
**Area**: Image Segmentation
**Keywords**: tubular structure segmentation, test-time adaptation, topological continuity, differential convolution, domain shift

## TL;DR
The first test-time adaptation (TTA) framework specifically designed for tubular structure segmentation (TSS). It adapts to cross-domain topological structural differences via Topological Meta-Differential Convolutions (TopoMDCs), and restores topological continuity through a Topological Hard Sample Generation (TopoHG) strategy, achieving an average clDice improvement of 31.81% across 10 datasets.

## Background & Motivation

Tubular structure segmentation (TSS) is critical for applications such as hemodynamic analysis and route planning. Unlike semantic or tumor segmentation, TSS is particularly sensitive to domain shifts—cross-domain differences encompass not only conventional variations such as contrast and noise, but also **topological structural characteristics** including trajectory, curvature, branching patterns, and thickness.

Existing TTA methods fall into two main categories: (1) normalization-based methods (e.g., TENT) that adapt to new domains by updating batch normalization parameters, which are limited under severe pixel-level class imbalance in segmentation tasks; and (2) teacher-student schemes (e.g., CoTTA) that update all parameters but lack targeted handling for tubular structures.

The **root cause** lies in the fact that existing TTA methods are "general-purpose" and fail to perceive two unique challenges inherent to tubular structures:
- **Challenge 1: Inconsistent topological structure.** Tubular structures in the source and target domains may differ substantially in thickness, curvature, and branching patterns, making generic model adaptation strategies insufficient for capturing these cross-domain topological variations.
- **Challenge 2: Fragile topological continuity.** Local features (color, texture, contrast) that distinguish foreground from background shift significantly across domains, causing foreground pixels to be misclassified and disrupting the continuity of tubular structures.

**Core Idea**: Design a two-stage topology-enhanced TTA framework. Stage 1 enhances the model's adaptability to diverse topological structures via 8 directional TopoMDCs without modifying pretrained parameters; Stage 2 improves topological continuity by generating pseudo-break hard samples and aligning predictions.

## Method

### Overall Architecture
TopoTTA comprises two stages: Stage 1 performs Topological Structure Adaptation, and Stage 2 performs Topological Continuity Refinement. Each stage executes 3 iterations, totaling 6 adaptation iterations per image. The model adopts a teacher-student paradigm, with teacher parameters updated via EMA.

### Key Designs

1. **Topological Meta-Differential Convolutions (TopoMDCs)**:

    - **Function**: Replace $3 \times 3$ convolutions in the encoder with 8 directional differential convolutions, enhancing the model's perception of diverse topological patterns without introducing additional parameters.
    - **Mechanism**: Inspired by Central Difference Convolution (CDC), which only computes differences between the center pixel and its neighbors and thus lacks sensitivity to the directionality and continuity of tubular structures. TopoMDCs extend the differential operation to 8 directions (4 orthogonal + 4 diagonal) to simulate various fundamental topological patterns. Taking the upper-left direction $\mathcal{C}_1$ as an example:
    $\mathcal{C}_1(r_x,r_y) = \mathcal{C}_c(r_x,r_y) - \sum_{(\Delta r_x,\Delta r_y)\in\mathcal{R}_1} w(\Delta r_x,\Delta r_y)\cdot x_{\text{in}}(r_x,r_y) + \sum w \cdot x_{\text{in}}(r_x+1,r_y+1)$
    - **Adaptive combination**: The input image is divided into $n \times n$ non-overlapping patches, each learning a set of learnable routing parameters $\boldsymbol{\delta}_j$ (only 8 scalars), controlling the weighting of different directional TopoMDCs. The final output is:
    $\hat{x}_{\text{out}} = \mathcal{C}_0(x_{\text{in}}) - \sum_{j=1}^{n \times n} \boldsymbol{\delta}_j \sum_{i=1}^{8} \mathcal{C}_i(x_{\text{in}}^j)$
    - **Design Motivation**: Tubular structures exhibit similar local topological patterns within small regions but differ substantially across distant regions; assigning different routing parameters per patch therefore enables adaptive enhancement of topological perception across regions.

2. **Topological Hard Sample Generation (TopoHG)**:

    - **Function**: Construct pseudo-breaks on test images to generate challenging samples, compelling the model to learn foreground-background discriminative features in the target domain.
    - **Mechanism** consists of three steps:
        - **Step 1 — Keypoint Selection**: Select foreground points with confidence $> \tau = 0.95$ from teacher model predictions, and randomly sample $N_p = k \cdot |\mathcal{P}|$ keypoints.
        - **Step 2 — Sliding Search**: For each keypoint, use an $s \times s$ window as the foreground window, and slide over adjacent regions to find the background window with the lowest pseudo-label confidence.
        - **Step 3 — Frequency-Domain Pseudo-Break Generation**: Apply FFT to the foreground and background windows, swap the low-frequency components (retaining high-frequency foreground features), to generate hard samples that appear "broken" while preserving key foreground characteristics:
       $$x_p^{\text{swap}} = \text{iFFT}(f_p^{\text{fg}} \cdot (1 - m_{\text{low}}) + f_p^{\text{bg}} \cdot m_{\text{low}})$$
    - **Design Motivation**: By simulating topological breaks, the model is guided to focus on local features that distinguish foreground from background, thereby improving continuity under cross-domain conditions.

3. **Routing Parameter Update and Model Update**:

    - Stage 1 updates only the routing parameters $\boldsymbol{\delta}$ without modifying pretrained network parameters, using an entropy minimization loss: $\mathcal{L}_{\text{EM}} = -\sum_{j=1}^{2} \hat{y}_j \log(\hat{y}_j)$. The parameters $\boldsymbol{\delta}$ are reset to zero upon each new sample.
    - Stage 2 updates all parameters using a weighted cross-entropy consistency loss, with the pseudo-break region weight multiplied by 10: $\mathcal{W}(u,v) = 10$ (break regions), $1$ (other regions).

### Loss & Training
- Stage 1: Entropy minimization loss to update routing parameters $\boldsymbol{\delta}$
- Stage 2: Weighted cross-entropy consistency loss $\mathcal{L}_{\text{CE}}$ to update all student network parameters
- Teacher network updated via EMA
- Each image is independently adapted for 6 iterations (3 per stage)
- TopoTTA is a plug-and-play scheme compatible with CNN-based TSS models

## Key Experimental Results

### Main Results
Cross-domain evaluation across 10 datasets spanning 4 scenarios (retinal vessel, road extraction, microscopic neuron, retinal OCTA):

| Transfer | Metric | TopoTTA | CoTTA (2nd best) | Gain |
|-----------|------|---------|-------------|------|
| DRIVE→CHASE | clDice | **77.05%** | 71.53% | +5.52 |
| DRIVE→STARE | clDice | **62.74%** | 59.86% | +2.88 |
| CHASE→DRIVE | clDice | **70.26%** | 64.80% | +5.46 |
| Neub1→Neub2 | Dice | **66.88%** | 52.21% | +14.67 |
| OCTA500→ROSE | clDice | **78.24%** | 75.65% | +2.59 |

### Ablation Study
Average results on the retinal vessel segmentation scenario:

| Configuration | Dice(%) | clDice(%) | β↓ | Note |
|------|---------|-----------|-----|------|
| Baseline | 65.37 | 61.69 | 82.24 | BN statistics update only |
| + TopoMDCs (Stage 1) | 68.70 | 65.14 | 76.28 | Topological structure adaptation |
| Baseline★ + TopoHG (Stage 2) | 68.82 | 66.61 | 73.63 | Topological continuity refinement |
| TopoTTA (Full) | **69.87** | **67.81** | **73.27** | Two-stage joint |

Ablation over TopoMDC types: the combination of orthogonal ($\mathcal{C}_{1\text{-}4}$) and diagonal ($\mathcal{C}_{5\text{-}8}$) directions performs best, with all 8 directions achieving optimal performance.

### Key Findings
- TopoTTA significantly improves clDice across all 10 cross-domain experiments, with an average gain of 31.81%
- TopoMDCs introduce negligible additional parameters (8 scalars per patch), leaving the network parameter count unchanged
- The frequency-domain low-frequency swapping design in TopoHG outperforms Gaussian blurring, random noise, and spatial image swapping
- Consistent improvements are observed on both UNet and CS2Net baselines (plug-and-play)

## Highlights & Insights
- **First TSS-specific TTA framework**, filling the gap between tubular structure segmentation and test-time adaptation
- The TopoMDCs design is elegant: 8 directional differential convolutions simulate fundamental topological patterns of tubular structures, enhancing topological perception without increasing network parameters
- The TopoHG strategy is novel: pseudo-breaks are generated in the frequency domain to simulate realistic continuity disruptions, guiding the model to "repair" breaks
- The two-stage decoupled design is well-motivated: Stage 1 focuses on topological structure adaptation, Stage 2 on topological continuity refinement
- Routing parameters are reset to zero per sample, ensuring independent adaptation for each input

## Limitations & Future Work
- Currently limited to CNN-based models; applicability to Transformer architectures has not been verified
- Each image requires 6 adaptation iterations, which may be insufficient for latency-sensitive scenarios
- TopoMDCs only replace $3 \times 3$ convolutional layers; other kernel sizes are not considered
- The patch count $n \times n$ is a fixed hyperparameter and does not adapt to varying resolutions or structural densities
- The frequency-domain pseudo-break generation relies on a predefined low-frequency mask, lacking an adaptive frequency selection mechanism

## Related Work & Insights
- The idea of extending CDC directionally can be generalized to other fine-grained visual recognition tasks
- The trend toward task-specific TTA from general-purpose TTA is significant; similar ideas can be applied to other segmentation tasks with distinctive structural characteristics (e.g., bone, crack)
- Frequency-domain data augmentation for test-time adaptation is a promising research direction

## Rating
- Novelty: ⭐⭐⭐⭐ First TSS-specific TTA framework; TopoMDCs and TopoHG are novel designs
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 scenarios, 10 datasets, 2 baseline networks, comprehensive ablation studies
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-motivated problem formulation, though the density of equations is somewhat high
- Value: ⭐⭐⭐⭐ Fills the TSS+TTA gap with clear practical application value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[CVPR 2026\] The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](../../CVPR2026/segmentation/the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)
- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)

</div>

<!-- RELATED:END -->
