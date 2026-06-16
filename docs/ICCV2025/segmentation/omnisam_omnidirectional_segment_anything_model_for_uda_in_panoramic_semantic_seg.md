---
title: >-
  [Paper Note] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation
description: >-
  [ICCV 2025][Segmentation][panoramic segmentation] This paper proposes OmniSAM, the first framework to apply SAM2 to unsupervised domain adaptation (UDA) for panoramic semantic segmentation. It partitions panoramic images…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "panoramic segmentation"
  - "SAM2"
  - "unsupervised domain adaptation"
  - "prototypical adaptation"
  - "pseudo label"
date: 2026-05-08
content_hash: 6f178c2f1b20653e
---

# OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation

**Conference**: ICCV 2025
**arXiv**: [2503.07098](https://arxiv.org/abs/2503.07098)  
**Code**: [https://github.com/Ding-Zhong/OmniSAM](https://github.com/Ding-Zhong/OmniSAM)  
**Area**: Image Segmentation
**Keywords**: panoramic segmentation, SAM2, unsupervised domain adaptation, prototypical adaptation, pseudo label

## TL;DR

This paper proposes OmniSAM, the first framework to apply SAM2 to unsupervised domain adaptation (UDA) for panoramic semantic segmentation. It partitions panoramic images into patch sequences via a sliding window and leverages SAM2's memory mechanism to capture cross-patch correspondences. Combined with a FoV-based prototypical adaptation module and a dynamic pseudo-label update strategy, OmniSAM significantly surpasses the state of the art on both indoor and outdoor benchmarks (+10.22% / +6.58%).

## Background & Motivation

The 360°×180° wide field of view (FoV) of omnidirectional cameras offers advantages for applications such as autonomous driving and virtual reality. However, the equirectangular projection of panoramic images introduces unavoidable distortion and object deformation. Due to the difficulty of obtaining large-scale annotated panoramic data, UDA has been proposed to bridge the domain gap between pinhole and panoramic images.

SAM2 demonstrates strong zero-shot capability on pinhole image segmentation, yet directly applying it to panoramic semantic segmentation poses two major challenges:

**FoV Gap**: A substantial FoV discrepancy exists between pinhole images (70°×70°) and panoramic images (180°×360°), causing severe distortion and difficulty in feature alignment.

**Semantic Absence**: SAM2 provides instance-level masks but lacks semantic knowledge.

## Method

### Overall Architecture

The OmniSAM framework consists of the following key components:
- A sliding window that crops source-domain pinhole images and target-domain panoramic images into overlapping patch sequences.
- SAM2's Hiera encoder (fine-tuned via LoRA) as the backbone for multi-scale feature extraction.
- SAM2's memory mechanism to capture cross-patch correspondences.
- A custom semantic decoder replacing the original mask decoder for semantic prediction.
- A FoV-based Prototypical Adaptation (FPA) module for cross-domain feature alignment.

### Key Designs

1. **SAM2 Adaptation**: LoRA fine-tuning is applied to the image encoder (query and value layers only), with fewer than 3 MB of trainable parameters. In the memory mechanism, the memory encoder fuses the current frame's output mask with the lowest-resolution feature $f_{low}$ and stores the result in a fixed-size memory bank ($n$ slots). Memory attention then conditions the current frame's features on past memories through $L$ transformer blocks. The semantic decoder linearly projects multi-scale embeddings $\{f_{high}, f_{med}, f_{con}\}$, upsamples them to a unified resolution, fuses them via convolution, and produces predictions through a linear classification head.

2. **Dynamic Pseudo-Label Update Mechanism**: The adaptation process is divided into multiple epochs, with a small subset of target-domain samples randomly drawn each round to generate pseudo labels. Forward and reverse bidirectional patch sequence processing is applied to sampled images, yielding predictions under different memory contexts. A pixel-level coverage map records how many patches cover each pixel; a label is assigned only when all overlapping patches unanimously vote for a class and the minimum confidence exceeds a threshold, otherwise the pixel is marked as uncertain. This approach is more computationally efficient than full pseudo-label updates and allows timely refinement as the model improves.

3. **FoV-based Prototypical Adaptation (FPA)**: This module is motivated by the assumption that patches at the same position share similar distortion statistics across different input sequences. At each patch prediction step $t$, class prototypes are computed for both source and target domains: $\tau_t^k = \frac{1}{M}\sum_{i,j}^{H,W}(y_{(k,i,j)})_t \cdot (\mathbf{f}_{(i,j)})_t$. A frozen source model is used to iteratively aggregate global source prototypes $\tau_t^{GS}$, and target-domain prototypes are aligned via an MSE loss: $\mathcal{L}_{fpa} = \frac{1}{KC}\|\tau_t^{GS} - \tau_t^T\|_F$.

### Loss & Training

The overall training objective is: $\mathcal{L} = \mathcal{L}_{seg} + \mathcal{L}_{ssl} + \lambda \mathcal{L}_{fpa}$

- $\mathcal{L}_{seg}$: Supervised segmentation loss on the source domain (cross-entropy).
- $\mathcal{L}_{ssl}$: Pseudo-label-based self-supervised loss on the target domain.
- $\mathcal{L}_{fpa}$: FPA prototypical alignment MSE loss.
- Training setup: 2×A800 GPUs, learning rate $6 \times 10^{-5}$, polynomial decay, AdamW optimizer.
- Sliding window crops images into 9-frame sequences of 1024×1024; memory bank size is set to 9.
- Four model variants are provided: Tiny / Small / Base / Large.

## Key Experimental Results

### Main Results

**Indoor Pin2Pan (SPin8 → SPan8)**:

| Method | SPin8 mIoU | SPan8 mIoU | Δ |
|--------|-----------|-----------|-----|
| Trans4PASS+-S | 67.28 | 63.73 | - |
| 360SFUDA++ w/ b2 | - | 68.84 | * |
| OmniSAM-T w/ MA | - | 69.10 | +0.26 |
| OmniSAM-B w/ MA | - | 74.72 | +5.88 |
| **OmniSAM-L w/ MA** | **75.18** | **79.06** | **+10.22** |

**Outdoor Pin2Pan (CS13 → DP13)**:

| Method | CS13 mIoU | DP13 mIoU | Δ |
|--------|-----------|-----------|-----|
| Trans4PASS+-S | 74.52 | 51.48 | - |
| DATR-S | - | 55.88 | * |
| OmniSAM-T w/ MA | - | 59.01 | +3.13 |
| OmniSAM-S w/ MA | - | 60.23 | +4.35 |
| **OmniSAM-B w/ MA** | **77.00** | **62.46** | **+6.58** |

### Ablation Study (Effect of Memory Attention)

**SPin8 → SPan8**:

| Variant | w/o MA | w/ MA | Gain |
|---------|--------|-------|------|
| OmniSAM-T | 68.72 | 69.10 | +0.38 |
| OmniSAM-S | 70.65 | 70.81 | +0.16 |
| OmniSAM-B | 73.09 | 74.72 | +1.63 |
| OmniSAM-L | 78.02 | 79.06 | +1.04 |

**CS13 → DP13**:

| Variant | w/o MA | w/ MA | Gain |
|---------|--------|-------|------|
| OmniSAM-T | 53.73 | 59.01 | +5.28 |
| OmniSAM-S | 57.03 | 60.23 | +3.20 |
| OmniSAM-B | 59.34 | 62.46 | +3.12 |
| OmniSAM-L | 59.02 | 61.63 | +2.61 |

Memory attention yields larger gains in the outdoor setting (+2.6–5.3%) than in the indoor setting (+0.2–1.6%).

### Key Findings

- Increasing model scale consistently improves performance: Tiny→Large raises SPan8 mIoU from 69.10 to 79.06.
- Domain gap analysis: the indoor Pin2Pan gap is small (FoV gap ≈ 0.5%), whereas the outdoor gap is substantially larger (≈ −20%).
- Under the Syn2Real setting (SP13→DP13), OmniSAM-B achieves 45.51 mIoU (+2.34), confirming generalization across synthetic-to-real domain shifts.
- Trainable parameters remain below 26 MB (Large variant), demonstrating strong parameter efficiency.

## Highlights & Insights

- OmniSAM is the first work to introduce SAM2 into UDA for panoramic semantic segmentation, opening a new research direction.
- Treating panoramic images as patch sequences and leveraging the video segmentation memory mechanism represents a particularly elegant perspective shift.
- FoV-based per-frame prototype alignment outperforms conventional full-image prototype alignment by better handling spatially varying distortion.
- The bidirectional sequence processing and coverage-map voting strategy in dynamic pseudo-label updates enhance pseudo-label reliability.
- LoRA fine-tuning achieves excellent parameter efficiency (<3 MB of trainable parameters for the encoder).

## Limitations & Future Work

- The Large and Base variants have substantial model sizes (209.2 M / 72.0 M parameters), which may limit real-time deployment.
- Inference speed is relatively slow due to 9-frame sequence processing; more efficient sequence handling strategies warrant exploration.
- The stride and size of the sliding window require manual tuning for data at different resolutions.
- The outdoor domain gap remains large (approximately −20%), leaving considerable room for improvement compared to the indoor setting.

## Related Work & Insights

- The paradigm of cropping images into sequences and processing them with video models can be generalized to other large-format image analysis tasks, such as remote sensing and medical panoramic imaging.
- The concept of FoV-based prototype alignment can inspire other UDA tasks involving spatially varying domain gaps.
- SAM2's memory mechanism for cross-patch modeling deserves further exploration in a broader range of dense prediction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of SAM2 to panoramic semantic segmentation UDA; elegant perspective shift and novel FPA design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three UDA scenarios, four model scales, comprehensive ablations, and per-class results.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architectural description, well-motivated design choices, and rich illustrations.
- **Value**: ⭐⭐⭐⭐ Substantial improvements over the state of the art, establishing a strong new baseline for panoramic segmentation UDA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] E-SAM: Training-Free Segment Every Entity Model](e-sam_training-free_segment_every_entity_model.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[ICML 2026\] Segment Anything with Robust Uncertainty-Accuracy Correlation](../../ICML2026/segmentation/segment_anything_with_robust_uncertainty-accuracy_correlation.md)
- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](../../AAAI2026/segmentation/segment_anything_across_shots_a_method_and_benchmark.md)

</div>

<!-- RELATED:END -->
