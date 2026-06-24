---
title: >-
  [Paper Note] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance
description: >-
  [ECCV 2024][Crowd Counting] Proposes an Auxiliary Point Guidance (APG) strategy and an Implicit Feature Interpolation (IFI) module to stabilize the proposal-target matching instability in point-based crowd counting methods by explicitly generating auxiliary positive and negative samples near ground-truth points, achieving state-of-the-art results on multiple datasets.
tags:
  - "ECCV 2024"
  - "Crowd Counting"
  - "Crowd Localization"
  - "Point Supervision"
  - "Matching Instability"
  - "Implicit Feature Interpolation"
date: 2026-05-08
content_hash: f96571d662f095de
---

# Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance

**Conference**: ECCV 2024  
**arXiv**: [2405.10589](https://arxiv.org/abs/2405.10589)  
**Code**: Mentioned in the paper that it will be released  
**Area**: Human Understanding  
**Keywords**: Crowd Counting, Crowd Localization, Point Supervision, Matching Instability, Implicit Feature Interpolation

## TL;DR

Proposes an Auxiliary Point Guidance (APG) strategy and an Implicit Feature Interpolation (IFI) module to stabilize the proposal-target matching instability in point-based crowd counting methods by explicitly generating auxiliary positive and negative samples near ground-truth points, achieving state-of-the-art results on multiple datasets.

## Background & Motivation

Crowd counting and localization methods can be broadly categorized into three types: density map-based (map-based), detection-based, and point-based. Point-based methods (e.g., P2PNet, CLTR, PET) have attracted attention due to being end-to-end trainable and requiring no complex post-processing, but they suffer from a key problem: **unstable proposal-target matching**.

Specifically, in each training epoch, a large number of target points are matched with different point proposals, which blurs the learning objective of each proposal. The authors define the Instability Rate (IR) to measure this instability and find that existing methods (e.g., Matcher) maintain a high IR during training. The root of this instability lies in the lack of effective learning strategies to guide the network to consistently select the most appropriate proposal, ultimately leading to regional underestimation or overestimation.

## Method

### Overall Architecture

APGCC uses VGG-16 as the backbone network to extract features from conv3 and conv4 layers. After enhancing multi-scale representations through an ASPP module, implicit feature interpolation (IFI) is applied to obtain feature representations at arbitrary positions. Finally, these are fed into prediction heads to output confidence and offset. The overall loss is:

$$\mathcal{L}_{overall} = \mathcal{L}_{point} + \lambda_5 \mathcal{L}_{APG}$$

### Key Designs

1. **Auxiliary Point Guidance (APG)**: Core innovation, designed to solve the matching instability problem. For each ground-truth point $(x,y)$, auxiliary positive samples $A_{pos}$ and auxiliary negative samples $A_{neg}$ are generated in its vicinity:

    - **Auxiliary Positive Samples**: Randomly generated with offsets within the range of $[-n_{pos}, n_{pos}]$ near the ground-truth coordinates. The training objective is a confidence score close to 1, and the predicted offset should steer back to the ground-truth location.
    - **Auxiliary Negative Samples**: Generated within the range of $[-n_{neg}, -n_{pos}] \cup [n_{pos}, n_{neg}]$. The training objective is a confidence score close to 0 and an offset converging to zero, preventing negative samples from "crossing the border" of the ground truth through offsets.
    - **Design Motivation**: By explicitly teaching the network that "proposals close to the ground truth should be positive and those far away should be negative," the matching process is guided to be stable, enabling the same ground-truth point to be consistently matched to the same proposal across different epochs.

2. **Implicit Feature Interpolation (IFI)**: Since auxiliary points are located at arbitrary coordinates rather than grid points, traditional bilinear interpolation is not flexible enough. IFI utilizes implicit functions for continuous feature representation:

    - For any position $(x,y)$, the four nearest latent features $Z_i^*$ are located and the distances $\delta_i^*$ are calculated.
    - The features and distances are concatenated and fed into an MLP $f_\theta$, and positional encoding $\phi(\cdot)$ is introduced to enhance the capture of high-frequency information.
    - The final feature is obtained via area-weighted summation: $F_{proposal}(x,y) = \sum_{i=1}^{4} \frac{S_i}{S} f_\theta(Z_i^*, \delta_i^*, \phi(\delta_i^*))$
    - Compared to traditional upsampling, IFI achieves more precise feature learning with fewer parameters.

### Loss & Training

- **Point Loss** $\mathcal{L}_{point}$: Contains the Cross Entropy classification loss $\mathcal{L}_{cls}$ and the Euclidean regression loss $\mathcal{L}_{loc}$.
- **APG Positive Loss** $\mathcal{L}_{APG}^{pos}$: Maximizes the confidence of auxiliary positive samples and minimizes the distance between their predicted locations and the ground truth.
- **APG Negative Loss** $\mathcal{L}_{APG}^{neg}$: Minimizes the confidence of auxiliary negative samples and constrains their offsets to approach zero.
- **Training Details**: Adam optimizer, learning rate $10^{-4}$ (backbone $10^{-5}$), batch size 8, stride $s=8$, $(k_{pos}, k_{neg}) = (2, 2)$, random ranges $(n_{pos}, n_{neg}) = (2, 8)$.

## Key Experimental Results

### Main Results

| Dataset | Metric | APGCC | PET (Prev. SOTA) | Gain |
|--------|------|-------|-------------|------|
| SHHA | MAE/MSE | **48.8/76.7** | 49.3/78.7 | -0.5/-2.0 |
| SHHB | MAE/MSE | **5.6/8.7** | 6.1/9.6 | -0.5/-0.9 |
| UCF_CC_50 | MAE/MSE | **154.8/205.5** | 159.9/223.7 | -5.1/-18.2 |
| UCF-QNRF | MSE | **136.6** | 144.3 | -7.7 |
| JHU-Crowd++ | MAE/MSE | **54.3/225.9** | 58.5/238.0 | -4.2/-12.1 |
| NWPU | MAE/MSE | **71.7/284.4** | 74.4/328.5 | -2.7/-44.1 |

**Localization Performance (NWPU)**:

| Method | F1(σ_l) | P(σ_l) | F1(σ_s) |
|------|---------|--------|---------|
| PET | 74.2% | 75.2% | 67.5% |
| **APGCC** | **76.4%** | **79.2%** | **68.9%** |

**Localization Performance (SHHA)**:

| Method | F1(σ=4) | F1(σ=8) |
|------|---------|---------|
| CLTR | 43.2% | 74.2% |
| **APGCC** | **48.7%** | **78.4%** |

### Ablation Study

| Configuration | MAE | Description |
|------|-----|------|
| Matcher only | Baseline | Hungarian matching only |
| Nearest Point | Severe underestimation | Multiple ground truths mapped to the same proposal |
| APG only | Suboptimal | Excessive dependence due to the lack of auxiliary point references during inference |
| **Matcher + APG** | **48.8** | Balances allocation and guidance |

**IFI Ablation**:

| Configuration | MAE | Description |
|------|-----|------|
| Nearest neighbor (w/o MLP) | Poor | Insufficient feature context |
| Bilinear interpolation (w/o MLP) | Moderate | Lacks continuous distance transformation |
| IFI with single reference point | Suboptimal | Insufficient reference points |
| IFI w/o positional encoding | Suboptimal | Lost high-frequency information |
| **IFI (Full)** | **Optimal** | All components synergistic |

### Key Findings

- APG significantly reduces the Instability Rate (IR), and a (2,2) ratio of positive to negative samples is sufficient.
- APG is only used during training and adds no inference overhead.
- IFI achieves better performance with fewer parameters compared to traditional upsampling.
- Under a strict localization threshold (σ=4), the F1 score of APGCC improves by 5.5% compared to CLTR.

## Highlights & Insights

- **Precise Problem Definition**: The paper quantifies the long-neglected matching instability in point-based methods using the Instability Rate.
- **Ingenious Auxiliary Point Approach**: Instead of changing the matching mechanism itself, it "teaches" the network how to make the correct choices through additional supervision signals.
- **Decoupled Training/Inference**: APG only functions during training, incurring zero overhead during inference, which is engineering-friendly.
- IFI provides a general feature extraction scheme for arbitrary positions, which can be transferred to other tasks requiring non-grid positional features.

## Limitations & Future Work

- The random ranges $(n_{pos}, n_{neg})$ of auxiliary points require tuning based on the specific dataset.
- The stride is fixed at 8, lacking an adaptive multi-scale stride mechanism.
- The application of the APG strategy to set prediction frameworks like the DETR series detectors has not been explored.
- Although the MLP parameter size of implicit feature interpolation is small, it still introduces some computational cost.

## Related Work & Insights

- P2PNet first proposed the point-based crowd counting framework but did not resolve the matching instability problem.
- Hungarian matching in DETR also exhibits early instability; the concept of APG could inspire potential solutions there.
- The success of implicit functions (e.g., LIIF, NeRF) in continuous representation inspired the design of IFI.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The APG strategy is novel with clear intuition, and the analysis of matching instability is thorough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation on 6 datasets across dual tasks of counting and localization, with exhaustive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-elaborated problem motivation.
- **Value**: ⭐⭐⭐⭐ — Provides substantial improvements for point-based methods, and the APG concept is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[ICLR 2026\] Change Point Localization and Inference in Dynamic Multilayer Networks](../../ICLR2026/others/change_point_localization_and_inference_in_dynamic_multilayer_networks.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/others/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[ECCV 2024\] CLR-GAN: Improving GANs Stability and Quality via Consistent Latent Representation and Reconstruction](clr-gan_improving_gans_stability_and_quality_via_consistent_latent_representatio.md)
- [\[ECCV 2024\] ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)

</div>

<!-- RELATED:END -->
