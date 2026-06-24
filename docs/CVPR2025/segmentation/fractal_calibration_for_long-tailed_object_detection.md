---
title: >-
  [Paper Note] Fractal Calibration for Long-Tailed Object Detection
description: >-
  [CVPR 2025][Segmentation][Long-Tailed Distribution] Proposes FRACAL (FRActal CALibration), a training-free post-processing method that introduces fractal dimension into post-calibration for long-tailed object detection for the first time. By symmetrically calibrating the frequency axis (category frequency) and the spatial axis (category spatial uniformity), it improves rare category mask AP by up to 8.6% on the LVIS dataset, with demonstrated generalization on COCO, V3Det…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Long-Tailed Distribution"
  - "Post-Calibration"
  - "Fractal Dimension"
  - "Logit Adjustment"
  - "Spatial Awareness"
date: 2026-05-08
content_hash: 3bcf3233a489cd40
---

# Fractal Calibration for Long-Tailed Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2410.11774](https://arxiv.org/abs/2410.11774)  
**Code**: [https://github.com/kostas1515/FRACAL](https://github.com/kostas1515/FRACAL)  
**Area**: Object Detection  
**Keywords**: Long-Tailed Distribution, Post-Calibration, Fractal Dimension, Logit Adjustment, Spatial Awareness

## TL;DR

Proposes FRACAL (FRActal CALibration), a training-free post-processing method that introduces fractal dimension into post-calibration for long-tailed object detection for the first time. By symmetrically calibrating the frequency axis (category frequency) and the spatial axis (category spatial uniformity), it improves rare category mask AP by up to 8.6% on the LVIS dataset, with demonstrated generalization on COCO, V3Det, and OpenImages.

## Background & Motivation

**Background**: Object detection models perform well on frequent classes but severely underperform on rare classes when trained on long-tailed data. Existing methods primarily address class imbalance through re-weighting and re-sampling, both of which require modifications to the training pipeline. In image classification, Post-calibration Softmax Adjustment (PCSA) methods have attracted attention due to being training-free and highly compatible.

**Limitations of Prior Work**: Existing PCSA methods only utilize the training class frequencies $p_s(y)$ to adjust logits, completely ignoring the dependency between classes and their spatial distributions $p_s(y, u)$. However, in object detection, classes and locations are highly correlated (e.g., "hats" mainly appear at the top of an image, while "shoes" appear at the bottom), making this information crucial for post-calibration.

**Key Challenge**: Directly modeling the joint class-location distribution $p(y, u)$ faces difficulties in selecting the grid/cell granularity. A grid that is too coarse (e.g., 2x2) loses spatial information, whereas a grid that is too fine (e.g., 64x64) makes spatial statistics for rare classes extremely sparse and noisy. An approach that encodes spatial distribution information without relying on a specific grid granularity is required.

**Goal**: (1) How to incorporate spatial information into post-calibration to complement frequency information? (2) How to address the sparsity of spatial statistics for rare classes?

**Key Insight**: Fractal dimension is a metric independent of specific grid sizes that measures the "uniformity of filling" of a point set in space. $\Phi=0$ represents a single point, while $\Phi=2$ represents fully uniform coverage of the 2D space. Computing the fractal dimension of a category's spatial distribution across multiple grid scales via the box-counting method robustly encodes spatial information without being affected by sparsity.

**Core Idea**: Capturing the spatial uniformity of target categories in image space using fractal dimensions complements traditional frequency calibration. This dual-axis calibration makes the detector more balanced in both the frequency and spatial dimensions.

## Method

### Overall Architecture

FRACAL is an inference-time post-processing method. Its input is the output logit $z_y$ of any pre-trained detector, and the output is the calibrated logit $z_y'$. Calibration is performed in two steps: (1) Classification Calibration C (frequency-based) to adjust foreground logits; (2) Spatial Calibration S (fractal-dimension-based) to further adjust the probabilities. The two steps are combined and then re-normalized. FRACAL's weights are pre-computed offline on the training set and stored in memory, resulting in no extra computational overhead during inference.

### Key Designs

1. **Classification Calibration C (Frequency Calibration)**:

    - **Function**: Adjusts logits via category frequencies to reduce frequent class bias and increase rare class detections.
    - **Mechanism**: Performs $C(z_y) = z_y - \log_\beta(\frac{n_y}{\sum_i^C n_i}) + \log_\beta(\frac{1}{C})$ on foreground logits, where $n_y$ is the number of instances in class $y$, and $\beta$ is a logarithmic base hyperparameter. The target distribution is set to a uniform distribution $p_t(y) = \frac{1}{C}$, as the AP metric evaluates each class independently and rewards balanced detection. Background logits are not adjusted (assuming the spatial distribution of objects in target and training sets is similar, i.e., $p_s(o,u) \approx p_t(o,u)$).
    - **Design Motivation**: Derived from Bayes-optimal classification, frequency calibration is equivalent to shifting the training distribution to a balanced test distribution. This design is independent of the subsequent spatial calibration and can be used on its own.

2. **Spatial-Aware Calibration S (Fractal Dimension Calibration)**:

    - **Function**: Utilizes fractal dimensions to measure the spatial uniformity of each class, down-weighting uniformly distributed classes and promoting sparsely distributed ones.
    - **Mechanism**: First, the fractal dimension $\Phi(y)$ of each class is calculated via the box-counting method: counting the number of grids containing object centers $\nu_y$ across multiple grid granularities $G$. The slope of the fitted line for $(\log G, \log \nu_y)$ is defined as $\Phi(y)$. To handle the sparsity of rare classes, a "quadratic rule" is introduced: calculating only within the range of $G \leq \lfloor\sqrt{n_y}\rfloor$ (ensuring grids could at least theoretically be filled). During inference, $S(z_y) = \frac{\sigma(z_y)}{\Phi(y)^\lambda}$, where $\lambda$ is a hyperparameter. Consequently, the probabilities of uniformly distributed classes (large $\Phi$) are down-weighted, while non-uniformly distributed ones (small $\Phi$) are boosted.
    - **Design Motivation**: The fractal dimension is weakly correlated with frequency (Pearson correlation of 0.35-0.375), providing complementary information. Many rare classes with $\Phi \approx 2$ demonstrate that the quadratic rule is robust for small sample sets. Spatial calibration forces the detector to predict various categories uniformly across all locations, eliminating spatial bias.

3. **Full FRACAL Calibration Formula and Sigmoid Extension**:

    - **Function**: Combines frequency and spatial calibration and supports Sigmoid binary classifiers.
    - **Mechanism**: For Softmax detectors: $F(z_y) = \frac{S(C(z_y))}{\sum_{j=1}^{C+1} S(C(z_j))}$, performing frequency calibration followed by spatial calibration, and finally re-normalizing. For Sigmoid detectors: $F_b(z_i) = \eta(C(z_i) - \log_\beta(\frac{\Phi(y)^\lambda}{\sum_i^C \Phi(i)^\lambda}) + \log_\beta(\frac{1}{C})) \cdot \eta(z_i)$, where both frequency and spatial calibrations are operated in the logit space, with $\eta(z_i)$ serving as a background filter.
    - **Design Motivation**: Binary classifiers perform foreground-background classification and inter-class classification concurrently, which requires decoupling prior to calibration.

### Loss & Training

FRACAL is training-free. All calibration weights are pre-computed offline on the training set. During inference, calibration is applied prior to NMS, with negligible computational overhead. Hyperparameters $\beta$ and $\lambda$ are determined via grid search on the validation set.

## Key Experimental Results

### Main Results

| Method | Backbone | $AP^m$↑ | $AP^m_r$↑ | $AP^m_c$↑ | $AP^m_f$↑ |
|------|----------|---------|-----------|-----------|-----------|
| Baseline | R50 | 25.7 | 15.8 | 25.1 | 30.6 |
| NorCal | R50 | 25.2 | 19.3 | 24.2 | 29.0 |
| GOL | R50 | 27.7 | 21.4 | 27.7 | 30.4 |
| LogN | R50 | 27.5 | 21.8 | 27.1 | 30.4 |
| **FRACAL** | R50 | **28.6** | **23.0** | **28.0** | **31.5** |
| Seesaw | Swin-S | 32.4 | 25.6 | 32.8 | 34.9 |
| **FRACAL** | Swin-S | **34.4** | **27.8** | **34.5** | **36.4** |

When using Swin-S, the relative performance improvement for rare classes reaches 8.6% (25.6 $\rightarrow$ 27.8), and Swin-B+ImageNet22K further yields an absolute gain of 6.6pp for rare classes.

### Ablation Study

| Configuration | $AP^m$ | $AP^m_r$ | Description |
|------|--------|----------|------|
| Frequency-only calibration C | 27.8 | 21.2 | Frequency information is already effective |
| Spatial-only calibration S | 26.1 | 16.3 | Spatial-only calibration is insufficient |
| Grid calibration $C_G$ (fixed G) | ≤27.5 | ≤21.3 | Rare classes are affected by sparsity |
| FRACAL (C + S) | **28.6** | **23.0** | Dual-axis complementarity yields optimal performance |

### Key Findings

- The Pearson correlation between the fractal dimension and frequency is only 0.35 (on LVIS), confirming that they indeed provide complementary information.
- Spatial calibration boosts performance not only for rare classes but also for frequent classes (31.5 vs 30.6 $AP^m_f$), since certain frequent yet spatially non-uniform classes also benefit from spatial bias correction.
- FRACAL can be seamlessly incorporated into various architectures like Mask R-CNN and GFLv2, as well as different backbones like ResNet and Swin.
- It consistently yields improvements on datasets with varying degrees of imbalance (e.g., COCO, V3Det, OpenImages), demonstrating strong generalization.

## Highlights & Insights

- **The introduction of fractal dimension is highly elegant**: It naturally addresses the dilemma of grid granularity selection in spatial statistics and is particularly robust for rare classes (the quadratic rule ensures small samples are not underestimated). This concept of introducing fractal geometry into detection post-calibration is highly novel.
- **The concept of dual-axis calibration is widely applicable**: It is not limited to long-tailed detection; any task exhibiting both "frequency bias" and "spatial bias" (such as class-pixel location imbalance in semantic segmentation) can benefit from this approach.
- **Zero training cost is the primary practical advantage**: As a plug-and-play inference-time post-processing method, FRACAL can be orthogonally combined with any training strategy (such as data augmentation or contrastive learning) without increasing the training burden.

## Limitations & Future Work

- Hyperparameters $\beta$ and $\lambda$ must be searched on the validation set, and different datasets or models may require different values.
- Fractal dimension assumes that the spatial distribution of a category can quantify its "uniformity"; there may be no extra gain for truly location-independent categories.
- It is currently verified only on object detection and instance segmentation, and has not yet been extended to other tasks such as semantic segmentation or panoptic segmentation.
- Spatial calibration is global (one $\Phi$ per class), without considering localized calibration adaptive to image content.
- For extremely rare classes (appearing $<4$ times in the training set), the fractal dimension cannot be calculated and must default to $\Phi=1$, which degenerates to pure frequency calibration.

## Related Work & Insights

- **vs NorCal**: NorCal is also a post-calibration method, but it only normalizes foreground probabilities using frequency statistics, lacking spatial information. FRACAL outperforms NorCal on R50 by 3.4pp in $AP^m$ and 3.7pp in $AP^m_r$.
- **vs LogN**: LogN normalizes prediction statistics using the model's own predictions, requiring forward passes over the entire training set to estimate weights. This is slower and model-dependent compared to FRACAL. FRACAL only requires dataset statistics, making it model-independent.
- **vs Training-time methods (e.g., Seesaw)**: Training-time methods require modifying the training pipeline, making it difficult to combine multiple approaches. FRACAL can be applied directly to any pre-trained model and orthogonally combined with training-time methods for extra gains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of fractal dimension in detection calibration, featuring a highly novel perspective and clear theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple datasets, backbones, and architectures, with exhaustive ablations.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous derivation, with a smooth transition from theory to practice.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play post-processing method with exceptionally high practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2025\] Towards Generalizable Scene Change Detection](towards_generalizable_scene_change_detection.md)

</div>

<!-- RELATED:END -->
