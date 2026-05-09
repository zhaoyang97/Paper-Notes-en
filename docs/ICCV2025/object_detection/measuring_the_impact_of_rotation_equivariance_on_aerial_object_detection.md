---
title: >-
  [Paper Note] Measuring the Impact of Rotation Equivariance on Aerial Object Detection
description: >-
  [ICCV 2025][Object Detection][Aerial image detection] This paper proposes MessDet, a rotation-equivariant aerial object detector that achieves strict rotation equivariance through a novel downsampling procedure, and introduces rotation-equivariant channel attention (RE-CA) and a multi-branch detection head, attaining state-of-the-art performance on DOTA and other benchmarks with significantly fewer parameters.
tags:
  - ICCV 2025
  - Object Detection
  - Aerial image detection
  - rotation equivariance
  - group equivariant networks
  - channel attention
  - multi-branch detection head
date: 2026-05-08
content_hash: c18c313f972ce0a0
---

# Measuring the Impact of Rotation Equivariance on Aerial Object Detection

**Conference**: ICCV 2025
**arXiv**: [2507.09896](https://arxiv.org/abs/2507.09896)
**Code**: [GitHub](https://github.com/Nu1sance/MessDet)
**Area**: Object Detection
**Keywords**: Aerial image detection, rotation equivariance, group equivariant networks, channel attention, multi-branch detection head

## TL;DR

This paper proposes MessDet, a rotation-equivariant aerial object detector that achieves strict rotation equivariance through a novel downsampling procedure, and introduces rotation-equivariant channel attention (RE-CA) and a multi-branch detection head, attaining state-of-the-art performance on DOTA and other benchmarks with significantly fewer parameters.

## Background & Motivation

The fundamental distinction between aerial object detection and general object detection lies in the fact that, from a bird's-eye-view perspective, objects appear at arbitrary orientations. This imposes the following requirements on detectors:
- **Classification** requires rotation invariance — the classification result should be consistent regardless of object orientation.
- **Regression** requires rotation equivariance — predicted angles should adjust accordingly when the input is rotated.

Existing aerial object detectors primarily learn rotation equivariance implicitly through increased model capacity, data augmentation, novel bounding box representations, or customized loss functions. A small number of works (e.g., ReDet, FRED) have attempted to explicitly achieve rotation equivariance using rotation-equivariant networks (RE-Nets), but a critical issue remains:

**Core Finding**: Mohamed et al. demonstrated that applying a stride-2 convolution on even-sized feature maps results in different sampling locations before and after rotation, thereby **breaking strict rotation equivariance**. Prior works such as ReDet achieve only approximate rotation equivariance. FRED addresses this by converting even dimensions to odd via one-sided zero-padding, but this may introduce feature misalignment artifacts.

**Key Open Question**: Is strict rotation equivariance truly necessary for aerial object detection? How much does it improve over approximate equivariance? This paper provides the first quantitative answer to this question.

## Method

### Overall Architecture

MessDet redesigns the RTMDet architecture using E2CNN to construct a rotation-equivariant backbone (CSPNeXt) and neck (CSPNeXtPAFPN), and introduces three improvements: (1) a novel downsampling procedure that preserves strict rotation equivariance; (2) a rotation-equivariant channel attention mechanism (RE-CA); and (3) a multi-branch detection head network.

### Key Designs

1. **Strictly Rotation-Equivariant Downsampling Procedure**:

    - Function: Ensures strict rotation equivariance during downsampling while preserving output spatial dimensions.
    - Mechanism: A "tuning layer" is inserted before the stride-2 downsampling convolution to convert even-sized feature maps to odd-sized ones:
        - Tuning layer: $k=4, p=1, s=1$, transforming input size $2n$ to $2n-1$
        - Downsampling layer: $k=3, p=1, s=2$, transforming $2n-1$ to $n$
        - Output size: $S_{out} = \lfloor((2n-1)-1)/2\rfloor + 1 = n$

      The tuning layer does not change the final output size but ensures that the downsampling convolution always operates on odd-sized feature maps.
    - Design Motivation: Avoids the feature misalignment caused by FRED's one-sided padding. Users can control whether the model achieves strict rotation equivariance simply by adding or removing the tuning layer, facilitating quantitative comparative experiments.

2. **Rotation-Equivariant Channel Attention (RE-CA)**:

    - Function: Introduces channel attention without breaking rotation equivariance.
    - Mechanism: Rotation-equivariant features $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ can be reshaped to $\mathbb{R}^{\frac{C}{N} \times N \times H \times W}$, where $N$ is the number of rotation orientations. RE-CA produces only $C/N$ weights (rather than $C$), each repeated $N$ times:
    $$\boldsymbol{s} = \sigma(\mathbf{W} \cdot \boldsymbol{z}), \quad \mathbf{W} \in \mathbb{R}^{\frac{C}{N} \times C}$$
      where $\boldsymbol{z}$ is the channel descriptor obtained via global average pooling.
    - Design Motivation: Directly applying standard SENet channel attention to rotation-equivariant features breaks equivariance, as different rotational orientations would receive different weights. RE-CA preserves equivariance through weight sharing, while reducing the parameter count by a factor of $1/N$.

3. **Multi-Branch Detection Head**:

    - Function: Exploits the grouping property of rotation-equivariant features to reduce parameters and improve accuracy.
    - Mechanism: Rotation-equivariant features $\mathbf{X} \in \mathbb{R}^{N \times \frac{C}{N} \times H \times W}$ are split into $N$ groups by orientation, each forwarded through a separate detection head branch, with outputs concatenated at the end.
    - Design Motivation: Rotation-equivariant features are naturally grouped — features generated by the same convolution kernel at different rotational orientations can be processed independently. This design reduces the input channel count per branch to $C/N$, significantly lowering the parameter count of the detection head.

### Loss & Training

- Identical detection losses to RTMDet (GFL classification loss + GIoU regression loss).
- AdamW optimizer; trained for 36 epochs on DOTA-v1.0/v1.5 and DIOR-R.
- Number of rotation orientations $N=8$ (following ReDet).
- Backbone pretrained on ImageNet-1K for 300 epochs.

## Key Experimental Results

### Main Results

| Method | Params | DOTA-v1.0 mAP | Notes |
|--------|--------|---------------|-------|
| RTMDet | 52.3M | 78.85 | Baseline (standard CNN) |
| ReDet | 31.6M | 76.25 | Approximate rotation equivariance |
| LSKNet | 31.0M | 77.49 | Large-kernel convolution |
| PKINet | 30.8M | 78.39 | Current CNN SOTA |
| Appr. MessDet | **15.3M** | 78.45 | Approximate rotation equivariance |
| **Str. MessDet** | **18.1M** | **79.12** | Strict rotation equivariance |

MessDet achieves state-of-the-art performance with only 15.3M–18.1M parameters (approximately 1/3 of RTMDet). The strictly equivariant variant (79.12 mAP) outperforms the approximately equivariant variant (78.45 mAP) by 0.67 mAP.

### Ablation Study

| Configuration | Params | mAP | Notes |
|---------------|--------|-----|-------|
| Str. MessDet + RE-CA | 19.0M | 78.51 | Full configuration |
| Str. MessDet w/o RE-CA | 18.8M | 76.91 | No channel attention, −1.60 |
| Appr. MessDet + RE-CA | 16.2M | 78.15 | Approximate equivariance |
| Appr. MessDet w/o RE-CA | 16.0M | 77.47 | −0.68 |
| RTMDet Head (2 conv) | 2.4M | 78.15 | Standard detection head |
| Multi-branch Head (3 conv) | **1.5M** | **78.45** | 37% fewer params, higher accuracy |

RE-CA contributes a 1.60 mAP improvement to the strictly equivariant model; the multi-branch head achieves a 0.30 mAP gain while reducing parameters by 37%.

### Key Findings

1. **Strict vs. Approximate Rotation Equivariance**: On MessDet (RE-Net), strict equivariance yields a notable improvement over approximate equivariance (+0.67 mAP), whereas the effect is marginal on standard CNNs such as RTMDet (+0.24 mAP), indicating that equivariance is more critical for RE-Nets.
2. **Rotation Equivariance Error During Training**: In approximately equivariant models, the rotation equivariance error in shallow layers decreases over training (the model learns approximate equivariance), while the error in deeper layers may increase.
3. **High Parameter Efficiency**: Weight sharing in RE-Nets (same convolution kernel rotated $N$ times) combined with the multi-branch head makes MessDet the most parameter-efficient SOTA aerial detector to date.
4. State-of-the-art results are also achieved on DOTA-v1.5 (which includes small objects of <10 pixels) and DIOR-R.

## Highlights & Insights

- **First quantitative characterization of the impact of strict vs. approximate rotation equivariance on aerial detection** — a gap left unaddressed by prior work.
- **Elegant engineering**: The tuning layer approach for achieving strict equivariance is simple and effective, avoiding the feature misalignment issues associated with FRED's one-sided padding.
- **Remarkable parameter efficiency**: MessDet achieves 79.12 mAP with only 18.1M parameters, compared to RTMDet's 78.85 mAP at 52.3M — demonstrating the substantial advantage of rotation-equivariant networks in aerial scenarios.
- The multi-branch detection head elegantly exploits the natural grouping structure of rotation-equivariant features.

## Limitations & Future Work

1. Only discrete rotation equivariance under the cyclic group $C_N$ is supported (e.g., $N=8$ corresponds to 45° intervals); continuous rotation equivariance is not handled.
2. The tuning layer introduces additional parameters and computational overhead (18.1M vs. 15.3M), though the overall model remains lightweight.
3. No comparison is made with Transformer-based aerial detectors (e.g., ViT variants).
4. Strict equivariance does not uniformly outperform approximate equivariance across all categories (e.g., the helicopter (HC) category achieves higher accuracy under the approximate variant).

## Related Work & Insights

- MessDet builds on the group equivariant convolution theory from E2CNN and deeply integrates it with a modern detection architecture (RTMDet) for the first time.
- ReDet and FRED are pioneering works in this direction; MessDet advances both the theoretical analysis and architectural design.
- LSKNet and PKINet represent an alternative approach of enhancing rotation robustness through large-kernel convolutions; MessDet achieves comparable or superior accuracy with fewer parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ First quantitative comparison of strict vs. approximate equivariance; RE-CA and multi-branch head are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three datasets (DOTA-v1.0/v1.5, DIOR-R) with comprehensive ablations and rotation error tracking analysis.
- Writing Quality: ⭐⭐⭐⭐ Theoretically clear with well-designed experiments.
- Value: ⭐⭐⭐⭐ Provides important guidance for rotation-equivariant design in aerial detection, with outstanding parameter efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)
- [\[ICCV 2025\] Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability](automated_model_evaluation_for_object_detection_via_prediction_consistency_and_r.md)
- [\[ICCV 2025\] Uncertainty-Aware Gradient Stabilization for Small Object Detection](uncertainty-aware_gradient_stabilization_for_small_object_detection.md)

</div>

<!-- RELATED:END -->
