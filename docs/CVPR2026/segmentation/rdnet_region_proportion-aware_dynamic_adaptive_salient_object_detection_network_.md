---
title: >-
  [Paper Note] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images
description: >-
  [CVPR2026][Segmentation][Remote sensing image salient object detection] To address the challenge of large-scale variation in remote sensing images, this paper proposes RDNet, a region proportion-aware dynamic adaptive salient object detection network. RDNet uses a Proportion Guidance mechanism to dynamically select convolution kernel combinations of varying sizes, combined with wavelet frequency-domain interaction and a cross-attention localization module. The method achieves state-of-the-art performance across three ORSI-SOD benchmarks.
tags:
  - CVPR2026
  - Segmentation
  - Remote sensing image salient object detection
  - dynamic adaptive convolution
  - wavelet transform
  - region proportion awareness
  - SwinTransformer
date: 2026-05-08
content_hash: ef8a4a149f68e19f
---

# RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images

**Conference**: CVPR2026
**arXiv**: [2603.12215](https://arxiv.org/abs/2603.12215)
**Code**: To be confirmed
**Area**: Semantic Segmentation / Salient Object Detection
**Keywords**: Remote sensing image salient object detection, dynamic adaptive convolution, wavelet transform, region proportion awareness, SwinTransformer

## TL;DR

To address the challenge of large-scale variation in remote sensing images, this paper proposes RDNet, a region proportion-aware dynamic adaptive salient object detection network. RDNet uses a Proportion Guidance mechanism to dynamically select convolution kernel combinations of varying sizes, combined with wavelet frequency-domain interaction and a cross-attention localization module. The method achieves state-of-the-art performance across three ORSI-SOD benchmarks.

## Background & Motivation

1. **Extreme scale variation in remote sensing targets**: Objects in the same scene may range from very small (aircraft) to very large (stadiums). Fixed-kernel strategies cannot accommodate both cases — large kernels introduce excessive background noise, while small kernels fail to capture the full extent of large objects.
2. **High computational cost of self-attention**: Existing methods perform self-attention directly on full-resolution features for inter-layer interaction, resulting in high computational complexity. Direct mixing of high- and low-frequency information also dilutes target representations.
3. **Lack of global modeling in CNN backbones**: CNN-based feature extractors rely on local convolution kernels and struggle to capture global context and long-range dependencies.
4. **Uniform treatment in existing multi-scale schemes**: Most methods apply identical multi-scale convolution combinations to all samples, without accounting for differences in target region proportions across images.
5. **Insufficient exploitation of mid-level contextual information**: High-level features carry localization semantics and low-level features capture fine details, yet effective and lightweight interaction designs for mid-level contextual features remain lacking.
6. **Complex backgrounds in remote sensing scenes**: Cluttered backgrounds and similar textures make precise boundary segmentation particularly challenging.

## Method

### Overall Architecture

RDNet employs **SwinTransformer** as the backbone to extract five-level features $\{F_i^R\}_{i=1}^{5}$ from 384×384 inputs. Three core modules are designed to process high-, mid-, and low-level features respectively, followed by a bottom-up fusion to produce the final saliency map:

- **RPL module** → High-level features $F_4^R, F_5^R$ → Localization feature $F^A$ + Proportion guidance $F^G$
- **FCE module** → Mid-level features $F_2^R, F_3^R$ → Contextual feature $F^W$
- **DAD module** → Low-level feature $F_1^R$ (guided by $F^G$) → Detail feature $F^P$

### Region Proportion-Aware Localization Module (RPL)

- **Channel attention** is applied to $F_4^R$ and $F_5^R$ separately (GAP + two 1×1 Conv layers + Sigmoid), with cross-multiply-add operations for channel-wise optimization.
- **Spatial attention** is then applied (channel-wise Max Pool + Sigmoid), with cross-multiply-add operations for spatial-wise optimization.
- The results are concatenated and passed through a 3×3 Conv to obtain the localization feature $F^A$.
- **Proportion Guidance (PG) Block**: $F_5^R$ is processed via GAP → two FC layers → output $F^G \in \mathbb{R}^{4 \times 1}$ (representing the target region proportion per batch sample), supervised by MSE Loss.

### Dynamic Adaptive Detail-Aware Module (DAD)

Based on the region proportion output by PG, targets are categorized into three tiers, each with a dynamically selected convolution kernel combination:

| Region Proportion | Kernel Combination | Design Rationale |
|---|---|---|
| **> 50%** | 1×1, 3×3, 5×5, 7×7, 9×9 (5 types) | Large kernels capture overall regions; small kernels refine boundaries |
| **25%–50%** | 1×1, 3×3, 5×5, 7×7 (4 types) | Balanced for medium-scale targets |
| **< 25%** | 1×1, 3×3, 5×5 (3 types) | Avoids excessive background introduction from large kernels |

- **Lower branch (Detail Extractor)**: Multi-kernel convolutions are applied and summed → $F_1^D$
- **Upper branch (Detail Optimizer)**: Channel-wise Max Pool → same kernel combination → summed → 1×1 Conv + Sigmoid → weight $W$
- Final output: $F^P = F_1^D \otimes W \oplus F_1^D$

### Frequency-Matching Context Enhancement Module (FCE)

**Wavelet Interaction Stage**:

- Discrete Wavelet Transform (DWT) is applied to $F_2^R$ and $F_3^R$ to obtain four frequency components (LL/LH/HL/HH).
- Corresponding frequency components interact via matrix multiplication (reshape → transpose → matrix multiply → softmax → multiply back → IDWT), reducing computational complexity to **1/4** of full-resolution attention.

**Feature Enhancement Stage**:

- Interaction results are concatenated with the original features → channel attention → spatial attention → concatenation → 3×3 Conv → $F^W$

### Loss & Training

$$L_{total} = \frac{1}{N} \sum_{i=1}^{N} (L_{bce} + L_{iou} + L_{fm} + L_{mse})$$

- BCE Loss: pixel-level cross-entropy
- IoU Loss: region overlap
- F-measure Loss: harmonic mean of precision and recall
- MSE Loss: supervises region proportion prediction $F^G$

## Key Experimental Results

### Main Results: State-of-the-Art Across Three Benchmarks

| Method | EORSSD M↓ | EORSSD $F_\beta$↑ | EORSSD $E_\xi$↑ | ORSSD M↓ | ORSSD $F_\beta$↑ | ORSSD $E_\xi$↑ | ORSI-4199 M↓ | ORSI-4199 $F_\beta$↑ |
|---|---|---|---|---|---|---|---|---|
| GeleNet | 0.0066 | 0.8367 | 0.9678 | 0.0083 | 0.8879 | 0.9787 | 0.0266 | 0.8711 |
| ADSTNet | 0.0065 | 0.8321 | 0.9633 | 0.0089 | 0.8856 | 0.9800 | 0.0319 | 0.8615 |
| HFCNet | 0.0051 | 0.7845 | 0.9280 | 0.0073 | 0.8581 | 0.9554 | 0.0270 | 0.8272 |
| **RDNet (Ours)** | **0.0049** | **0.8563** | **0.9718** | **0.0066** | **0.9080** | **0.9852** | **0.0254** | **0.8781** |

- On EORSSD, MAE is reduced by 3.9% relative to HFCNet, with an average $F_\beta$ improvement of 9.1%.
- On ORSSD, $F_\beta$ reaches 0.908, a 2.5% gain over ADSTNet.
- t-test p-values against all 21 compared methods are statistically significant.

### Ablation Study

| Configuration | M↓ | $F_\beta$↑ | $S_\alpha$↑ |
|---|---|---|---|
| w/o DAD | 0.0052 | 0.8550 | 0.9273 |
| w/o FCE | 0.0061 | 0.8453 | 0.9224 |
| w/o RPL | 0.0054 | 0.8561 | 0.9329 |
| **Full RDNet** | **0.0049** | **0.8563** | **0.9327** |

- Removing FCE yields the largest MAE increase (0.0061 vs. 0.0049), indicating that mid-level contextual interaction contributes most.
- Backbone comparison: SwinTransformer >> PVT > ResNet > VGG >> ViT (ViT MAE as high as 0.0175).
- The threshold configuration [<25%, 25%–50%, >50%] is validated as optimal.

### Model Efficiency

- FLOPs: 48.7G (vs. GeleNet at 11.7G and PA-KRN at 617.7G)
- Inference speed: 13.6 FPS (moderate speed due to intensive matrix operations)

## Highlights & Insights

1. **Region proportion-guided dynamic kernel selection** is the core innovation, introducing a classification paradigm into detection — predicting the target "size category" first before determining the convolution strategy, thereby avoiding kernel–scale mismatch.
2. **Wavelet-domain frequency-matching interaction** transfers inter-layer feature interaction from the spatial domain to the frequency domain; interacting same-frequency components separately reduces computational cost by 4× while preventing mutual interference between high- and low-frequency information.
3. **Hierarchical three-module design** (high-level localization + mid-level context + low-level detail) is logically coherent, with each module having a clearly defined role.
4. Experiments are highly thorough: comparisons against 21 methods, 7 ablation groups, statistical significance testing via t-test, and failure case analysis.

## Limitations & Future Work

1. **Slow inference speed**: 13.6 FPS is insufficient for real-time remote sensing applications; intensive matrix operations are the primary bottleneck.
2. **Coarse three-tier proportion discretization**: Continuous regression of region proportions may yield finer-grained adaptation than a discrete three-class scheme.
3. **PG Block relies on high-level semantics**: Using only $F_5^R$ for proportion prediction may be inaccurate for very small targets.
4. **Failure cases**: Missed detections persist for extremely small or elongated targets; false detections occur when background textures are similar to targets.
5. **Validation limited to remote sensing datasets**: Generalization to natural image SOD benchmarks has not been tested.
6. **Heavy SwinTransformer backbone**: Limits deployment potential on edge devices.

## Related Work & Insights

- **vs. ADSTNet / GeleNet** (prior SOTA): RDNet comprehensively outperforms both across all three datasets; the core advantage lies in the region proportion-adaptive mechanism.
- **vs. ASTT** (Transformer-based method): $F_\beta$ improves by 13.6%, benefiting from the hierarchical design rather than simple global attention.
- **vs. MCCNet / CorrNet** (context interaction methods): FCE's wavelet-domain interaction is more effective and more lightweight than direct feature concatenation or attention-based interaction.
- **vs. VST** (Vision Transformer): MAE is reduced by 28.9%, demonstrating that Swin's hierarchical window attention is better suited to dense prediction than ViT's flat architecture.

## Rating

- Novelty: ⭐⭐⭐⭐ — Region proportion-guided dynamic kernel selection and wavelet frequency-domain interaction are both practically meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comparisons against 21 methods, 7 ablation groups, statistical significance testing, and failure case analysis make this an exceptionally rigorous evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete mathematical derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Provides an effective solution to the scale problem in remote sensing salient object detection, though real-time performance remains to be improved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_regionguided_selective_optimization_network.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](prompt-driven_lightweight_foundation_model_for_instance_segmentation-based_fault.md)

</div>

<!-- RELATED:END -->
