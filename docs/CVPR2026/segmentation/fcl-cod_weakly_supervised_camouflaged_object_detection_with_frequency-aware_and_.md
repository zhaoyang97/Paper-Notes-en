---
title: >-
  [Paper Note] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning
description: >-
  [CVPR 2026][Segmentation][Camouflaged Object Detection] This paper proposes FCL-COD, a framework that injects camouflaged scene knowledge into SAM via Frequency-aware Low-Rank Adaptation (FoRA), enhances foreground-background feature separation through Gradient-aware Contrastive Learning (GCL), and refines boundary-sensitive features with Multi-Scale Frequency Attention (MSFA). Under a weakly supervised setting using only bounding box annotations, FCL-COD surpasses fully supervised state-of-the-art methods.
tags:
  - CVPR 2026
  - Segmentation
  - Camouflaged Object Detection
  - Weakly Supervised
  - SAM
  - Frequency-aware LoRA
  - Contrastive Learning
date: 2026-05-08
content_hash: 1c6b53885bf15e50
---

# FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning

**Conference**: CVPR 2026
**arXiv**: [2603.22969](https://arxiv.org/abs/2603.22969)
**Code**: None
**Area**: Image Segmentation
**Keywords**: Camouflaged Object Detection, Weakly Supervised, SAM, Frequency-aware LoRA, Contrastive Learning

## TL;DR

This paper proposes FCL-COD, a framework that injects camouflaged scene knowledge into SAM via Frequency-aware Low-Rank Adaptation (FoRA), enhances foreground-background feature separation through Gradient-aware Contrastive Learning (GCL), and refines boundary-sensitive features with Multi-Scale Frequency Attention (MSFA). Under a weakly supervised setting using only bounding box annotations, FCL-COD surpasses fully supervised state-of-the-art methods.

## Background & Motivation

Camouflaged Object Detection (COD) requires identifying objects that are highly similar to their backgrounds, posing four major challenges:

**Fully supervised methods** rely on pixel-level annotations, which are costly and may cause models to overlook holistic structural features of targets.

**Weakly supervised methods** exhibit a significant performance gap compared to fully supervised counterparts.

3. SAM-based methods suffer from specific failure modes in camouflaged scenes:
    - (a) Non-camouflaged object response — incorrectly detecting irrelevant objects
    - (b) Partial response — detecting only a portion of the target
    - (c) Extreme response — detection regions that are excessively large or small
    - (d) Lack of fine-grained boundary awareness

This paper systematically addresses each of these four failure modes with dedicated solutions.

## Method

### Overall Architecture

A two-stage framework:
- **Stage 1**: A triadic teacher-student self-training architecture adapts SAM using FoRA and GCL to generate high-quality pseudo-labels.
- **Stage 2**: Pseudo-labels are used to train a lightweight PVT-B4 encoder-decoder with embedded MSFA modules for efficient inference.

### Key Designs

1. **Triadic Teacher-Student Self-training**:

    - Three encoders are maintained: an anchor encoder $f^a$ (frozen original SAM, preserving pretrained knowledge), a student encoder $f^s$ (receiving strongly augmented inputs), and a teacher encoder $f^t$ (receiving weakly augmented inputs, sharing parameters with the student).
    - Student-teacher loss: Focal Loss + Dice Loss supervise the student to learn from teacher pseudo-labels.
    - Anchor loss: Prevents the student and teacher from drifting too far from pretrained SAM knowledge, suppressing pseudo-label error accumulation.
    - Input prompts are bounding boxes derived from GT mask bounding boxes; no pixel-level annotations are used.

2. **Frequency-aware Low-Rank Adaptation (FoRA)**:
   Addresses non-camouflaged object responses. A cascaded transform is inserted between the encoder-decoder path of standard LoRA:
    - **Spatial enhancement $\mathcal{S}_{spa}$**: Aggregates multi-scale context via 1×1, 3×3, and 5×5 convolutions with residual connections.
    - **Frequency modulation $\mathcal{S}_{fre}$**: FFT → frequency-domain 3×3 convolution → IFFT, modeling high-frequency texture differences in camouflaged scenes.
    - Forward pass: $h = W_0 x + W_d \mathcal{S}_{fre}(\mathcal{S}_{spa}(W_e x))$
    - Core Idea: Camouflaged targets and backgrounds are highly similar in the spatial domain but exhibit distinguishable subtle texture differences in the frequency domain.

3. **Gradient-aware Contrastive Learning (GCL)**:
   Addresses partial and extreme responses. The key innovation lies in the sampling strategy:
    - Grad-CAM is applied to teacher feature maps to derive a gradient activation map $G^t$.
    - A gradient-weighted background mask $\tilde{m}_0 = \hat{m}_0 \odot G^t$ is constructed, focusing on hard background regions likely to be confused with the foreground.
    - Masked average pooling constructs foreground instance prototypes and background prototypes for both student and teacher branches.
    - Positive pairs: student-teacher representations of the same instance; negatives: other instances + gradient-weighted background prototypes.
    - InfoNCE contrastive loss pushes foreground representations away from hard background representations.

4. **Multi-Scale Frequency Attention (MSFA)**:
   Addresses the lack of fine-grained boundary awareness. Inserted between the encoder and decoder in Stage 2:
    - Dual-branch design: spatial branch $\mathcal{M}_{spa}$ (stacked 3×3 convolutions) + frequency branch $\mathcal{M}_{fre}$ (FFT → 1×1 convolution → IFFT).
    - Tri-domain attention $\mathcal{T}$: multi-scale features from one domain gate features in the other domain.
    - Multi-scale (S/M/L) spatial and frequency features are cross-gated and then concatenated for fusion.

### Loss & Training

**Stage 1 total loss**:
$$\mathcal{L} = \mathcal{L}_{st}^{dice} + \lambda_1 \mathcal{L}_{anchor} + \lambda_2 \mathcal{L}_{GCL} + \lambda_3 \mathcal{L}_{st}^{focal}$$

Optimal hyperparameters: $\lambda_1$=0.50, $\lambda_2$=1.00, $\lambda_3$=20

**Stage 2 loss**: BCE + uncertainty-aware loss with cosine annealing

Training setup: 2×NVIDIA H20 GPUs, PVT-B4 encoder, SGD (lr=1e-3, momentum=0.9), 60 epochs

## Key Experimental Results

### Main Results

Comparison with fully supervised and weakly supervised methods (SAM-H backbone):

| Method | Supervision | CAMO-MAE↓ | CAMO-$S_m$↑ | COD10K-MAE↓ | COD10K-$S_m$↑ |
|------|------|-----------|-------------|-------------|-------------|
| SARNet | Full | 0.046 | 0.874 | 0.021 | 0.885 |
| CamoFormer-P | Full | 0.046 | 0.872 | 0.023 | 0.869 |
| HitNet | Full | 0.055 | 0.849 | 0.023 | 0.871 |
| SAM-COD | Weak (B) | 0.062 | 0.837 | 0.028 | 0.842 |
| **FCL-COD(H)** | **Weak (B)** | **0.050** | **0.862** | **0.022** | **0.878** |

Under the weakly supervised setting, FCL-COD not only substantially outperforms SAM-COD (MAE reduced by 0.012) but also **surpasses multiple fully supervised methods** (e.g., ZoomNet, CamoFormer-R).

Results across different SAM scales:

| Backbone | CAMO-MAE↓ | COD10K-MAE↓ | NC4K-MAE↓ |
|----------|-----------|-------------|-----------|
| FCL-COD(SAM-B) | 0.060 | 0.027 | 0.041 |
| FCL-COD(SAM-L) | 0.054 | 0.022 | 0.034 |
| FCL-COD(SAM-H) | 0.050 | 0.022 | 0.033 |

### Ablation Study

Incremental ablation of component contributions (COD10K, $E_m$↑):

| FoRA | GCL | MSFA | COD-Train $E_m$ | CHAMELEON $E_m$ | COD10K $E_m$ |
|------|-----|------|-----------------|-----------------|--------------|
| ✗ | ✗ | ✗ | 0.959 | 0.927 | 0.919 |
| ✓ | ✗ | ✗ | 0.963 | 0.928 | 0.923 |
| ✓ | ✓ | ✗ | 0.969 | 0.947 | 0.926 |
| ✓ | ✓ | ✓ | — | **0.954** | **0.938** |

FoRA improves pseudo-label quality → GCL further strengthens foreground-background separation → MSFA refines boundaries during inference.

FoRA sub-ablation: spatial enhancement and frequency modulation each contribute +0.001–0.002 $E_m$; combining both yields +0.004.
GCL sub-ablation: standard CL contributes +0.005; adding gradient awareness yields an additional +0.001.

### Key Findings

- **Frequency-domain information is key to distinguishing camouflaged targets**: camouflaged scenes are highly similar in the spatial domain, but exploitable texture differences exist in the frequency domain.
- Grad-CAM-guided hard negative mining is more effective than random sampling.
- Multi-scale spatial-frequency cross-gating outperforms single-branch designs.
- The method generalizes to weakly supervised Salient Object Detection (SOD), also outperforming fully supervised methods.

## Highlights & Insights

1. **Highly systematic problem decomposition**: The four SAM failure modes in camouflaged scenes (non-camouflaged response / partial response / extreme response / coarse boundaries) correspond respectively to FoRA / GCL / GCL / MSFA, yielding a coherent and principled design.
2. **Multi-level exploitation of frequency-domain priors**: FoRA injects frequency priors during feature adaptation; MSFA leverages frequency branches to refine boundaries during inference, forming a comprehensive frequency-aware system.
3. **Weakly supervised results surpassing fully supervised methods** are compelling, demonstrating that SAM's strong priors combined with proper adaptation can compensate for the lack of dense annotations.
4. **Engineering soundness of the two-stage design**: Stage 1 uses a large SAM model to generate high-quality pseudo-labels; Stage 2 deploys a lightweight model for inference, balancing accuracy and efficiency.

## Limitations & Future Work

- During training, bounding box prompts are derived from GT masks; the acquisition of bounding boxes in practical applications warrants further discussion.
- Inference requires two stages (pseudo-label generation + lightweight detector), making the overall pipeline relatively complex.
- Evaluation on the CHAMELEON dataset (only 76 images) may be subject to statistical variance.
- Extensions to video camouflaged object detection or instance-level camouflaged object detection are not discussed.

## Related Work & Insights

- The spatial-frequency cascaded design of FoRA can be generalized to other LoRA adaptation tasks requiring fine-grained texture discrimination.
- The gradient-aware hard negative mining strategy offers a useful reference for any contrastive learning scenario requiring hard negatives.
- The paradigm of SAM + lightweight adaptation + pseudo-label training is transferable to other weakly supervised dense prediction tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — FoRA and GCL are well-designed; the systematic use of frequency-domain priors is a notable highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, detailed component ablations, hyperparameter analysis, qualitative visualizations, and SOD extension.
- Writing Quality: ⭐⭐⭐⭐ — Problem decomposition is clear, though notation is somewhat dense.
- Value: ⭐⭐⭐⭐ — Weakly supervised results surpassing fully supervised methods are impressive and demonstrate practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Weakly-Supervised Referring Video Object Segmentation through Text Supervision](wsrvos_weakly_supervised_rvos.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation](dss_discover_segment_select_zero_shot_cos.md)
- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)

</div>

<!-- RELATED:END -->
