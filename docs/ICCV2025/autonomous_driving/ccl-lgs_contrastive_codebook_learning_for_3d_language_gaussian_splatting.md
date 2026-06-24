---
title: >-
  [Paper Note] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting
description: >-
  [ICCV 2025][Autonomous Driving][3D Gaussian Splatting] This paper proposes the CCL-LGS framework, which employs a zero-shot tracker for cross-view mask association and a Contrastive Codebook Learning (CCL) module to distill semantically compact intra-class and discriminative inter-class features. The framework addresses cross-view semantic inconsistency in 2D-prior-based 3D semantic field reconstruction caused by occlusion, blur, and viewpoint variation.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D Gaussian Splatting"
  - "Open-Vocabulary Semantic Segmentation"
  - "Contrastive Learning"
  - "Codebook Learning"
  - "Cross-View Consistency"
date: 2026-05-08
content_hash: c25b771c317eeea7
---

# CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2505.20469](https://arxiv.org/abs/2505.20469)  
**Code**: [https://epsilontl.github.io/CCL-LGS/](https://epsilontl.github.io/CCL-LGS/)  
**Area**: 3D Scene Understanding / Autonomous Driving
**Keywords**: 3D Gaussian Splatting, Open-Vocabulary Semantic Segmentation, Contrastive Learning, Codebook Learning, Cross-View Consistency

## TL;DR

This paper proposes the CCL-LGS framework, which employs a zero-shot tracker for cross-view mask association and a Contrastive Codebook Learning (CCL) module to distill semantically compact intra-class and discriminative inter-class features. The framework addresses cross-view semantic inconsistency in 2D-prior-based 3D semantic field reconstruction caused by occlusion, blur, and viewpoint variation.

## Background & Motivation

Recent advances in 3D Gaussian Splatting (3DGS) combined with vision-language models (e.g., CLIP, SAM) have shown remarkable progress in 3D open-vocabulary scene understanding. Existing methods typically adopt a "projection supervision" paradigm: 3D semantics are rendered into 2D views and compared against features extracted by pretrained VLMs. However, this paradigm critically relies on the assumption that 2D semantic features remain consistent across different viewpoints.

Three typical failure cases arise in practice:

**Occlusion**: Different portions of the same object are visible from different viewpoints, causing inconsistent CLIP semantic features.

**Image blur**: Motion blur degrades mask quality and introduces noisy semantic encodings.

**View-dependent variation**: Lighting, reflections, and other factors cause the same object to appear differently across views.

Although existing methods (LangSplat, LEGaussians, GOI, etc.) partially mitigate these issues via 3D geometric consistency, significant input feature inconsistencies still propagate into 3D space due to continued reliance on 2D supervision, resulting in rendering artifacts. The key innovation of CCL-LGS lies in avoiding direct use of imperfect CLIP features and instead explicitly modeling cross-view semantic alignment to address this overlooked fundamental problem.

## Method

### Overall Architecture

CCL-LGS consists of three stages:
1. **Dual-level semantic feature extraction**: Multi-scale masks are generated using SAM, and pixel-level semantic features are extracted via CLIP.
2. **Contrastive Codebook Learning (CCL)**: Mask association is performed using a zero-shot tracker, followed by a contrastive loss to organize and refine semantic features.
3. **3D Gaussian semantic field optimization**: Refined semantic information is integrated into 3DGS and optimized end-to-end using a cross-entropy loss.

### Key Designs

#### 1. Dual-Level Semantic Feature Extraction
- **Function**: Extract precise, multi-scale semantic features from multi-view images.
- **Mechanism**: SAM's 32×32 uniform point prompts generate three mask types (sub-part, part, whole), which are merged into two aggregated mask sets $M_o^{sp}$ (sub-part + part) and $M_o^{wp}$ (whole + part). For each pixel $v$, the semantic feature is:

$$F_i(v) = \text{CLIP}(I_t \odot M_i(v))$$

- **Design Motivation**: Unlike LangSplat, which relies on three independent scales (introducing scale errors), this approach integrates multi-scale information within a unified framework, preserving precise boundaries while reducing computational overhead.

#### 2. Contrastive Codebook Learning (CCL) Module
- **Function**: Align semantic features of the same object across different views while maintaining discriminability between different objects.
- **Mechanism**: The module operates in two steps:
    - **Mask association**: SAM2 propagates $K$ masks from the first frame to all subsequent frames; class labels $y_i \in \{1,2,...,K,-1\}$ are assigned via IoU matching (IoU > 0.5 for a match; otherwise labeled −1).
    - **Contrastive loss**: A codebook $T = \{T_j\}_{j=1}^N$ is constructed; for each feature $F_i$, the most similar prototype is retrieved:

$$j^* = \underset{j}{\text{argmax}} \cos(F_i, T_j)$$

$$L_{\text{max}} = 1 - \cos(F_i, T_{j^*})$$

  - A pull loss brings same-class features closer: $L_{\text{pull}} = 1 - \cos(T_{j_i}, T_{j_k})$ (when $y_i = y_k \neq -1$)
  - A push loss separates different-class features: $L_{\text{push}} = \text{ReLU}(\cos(T_{j_i}, T_{j_k}) - m)$ (when $y_i \neq y_k$, both not −1)
- **Design Motivation**: Compared to autoencoders, the codebook approach implicitly constrains similar features to map to the same entry, providing stronger consistency regularization. The contrastive loss ensures intra-class compactness and inter-class separability, effectively alleviating semantic ambiguity introduced by imperfect masks.

#### 3. 3D Gaussian Semantic Field Optimization
- **Function**: Construct the final 3D semantic field using the trained codebook.
- **Mechanism**: Each pixel's semantic feature is converted into a discrete index map $\mathcal{M} \in \mathbb{R}^{H \times W}$. A lightweight MLP decoder with a softmax layer generates a semantic distribution $\hat{\mathcal{M}} \in \mathbb{R}^{H \times W \times N}$, optimized via cross-entropy loss:

$$\mathcal{L}_{CE} = \text{CE}(\hat{\mathcal{M}}, \mathcal{M})$$

At inference, a relevance map is computed using CLIP-encoded text query $\tau$: $p(\tau|v) = \frac{\exp(\tilde{F}(v) \cdot \varphi(\tau) / \|\tilde{F}(v)\| \|\varphi(\tau)\|)}{\sum_{s \in \mathcal{T}} \exp(\tilde{F}(v) \cdot \varphi(s) / \|\tilde{F}(v)\| \|\varphi(s)\|)}$

### Loss & Training

The total loss is a weighted sum of three terms:

$$L = L_{\text{max}} + \lambda_{\text{pull}} L_{\text{pull}} + \lambda_{\text{push}} L_{\text{push}}$$

Training settings: $\lambda_{\text{pull}} = \lambda_{\text{push}} = 0.25$, margin $m = 0.7$, low-dimensional feature dimension $d_f = 8$, Adam optimizer (lr=0.001), 30,000 training iterations.

## Key Experimental Results

### Main Results

mIoU comparison on the LERF dataset:

| Method | Ramen | Figurines | Teatime | Waldo Kitchen | Avg. |
|--------|-------|-----------|---------|---------------|------|
| LangSplat | 51.2 | 44.7 | 65.1 | 44.5 | 51.4 |
| GOI | 52.6 | 44.5 | 63.7 | 41.4 | 50.6 |
| 3D VL-GS | 61.4 | 58.1 | 73.5 | 54.8 | 62.0 |
| **CCL-LGS** | **62.3** | **61.2** | 71.8 | **67.1** | **65.6** |

mIoU comparison on the 3D-OVS dataset:

| Method | Bed | Bench | Sofa | Lawn | Avg. |
|--------|-----|-------|------|------|------|
| LangSplat | 92.5 | 94.2 | 90.0 | 96.1 | 93.2 |
| 3D VL-GS | **96.8** | **97.3** | **95.5** | **97.9** | **96.9** |
| **CCL-LGS** | 97.3 | 95.0 | 92.3 | 96.1 | 95.2 |

### Ablation Study

Ablation of the CCL module on the LERF dataset:

| Configuration | Ramen | Figurines | Teatime | Kitchen | Avg. | Note |
|---------------|-------|-----------|---------|---------|------|------|
| Baseline | 46.8 | 57.1 | 60.8 | 61.0 | 56.4 | Codebook compression only |
| +Pull loss | 48.0 | 58.0 | 70.1 | 62.0 | 59.5 | Enhanced intra-class consistency |
| +Push loss | 55.1 | 61.0 | 66.0 | 59.3 | 60.4 | Reduced false activations |
| **Full model** | **62.3** | **61.2** | **71.8** | **67.1** | **65.6** | Complementary effect of both losses |

### Key Findings

1. The pull loss yields especially significant intra-class consistency gains for occluded or partially visible objects (e.g., "glass of water").
2. The push loss effectively reduces false activations in ambiguous regions (e.g., around "kamaboko").
3. Data augmentation strategies (as in 3D VL-GS) benefit more from the simpler 3D-OVS dataset with less occlusion and blur, whereas CCL-LGS demonstrates a clearer advantage on the more challenging LERF dataset due to its cross-view consistency mechanism.

## Highlights & Insights

- **Core Insight**: This work is the first to address the overlooked cross-view feature alignment problem in 2D-to-3D semantic field reconstruction, demonstrating that geometric consistency constraints alone are insufficient to resolve semantic inconsistency.
- **Elegant Design**: SAM2's zero-shot tracking capability is leveraged for mask association, establishing cross-view correspondences without additional training.
- **Contrastive Codebook**: The combination of codebook quantization and contrastive learning provides dual benefits — ensuring feature consistency while maintaining inter-class discriminability.

## Limitations & Future Work

- The method depends on the quality of SAM and SAM2; imperfect masks still affect the final results.
- CCL-LGS does not surpass 3D VL-GS's data augmentation strategy on simpler scenes (3D-OVS).
- The codebook size $N$ and the actual number of object categories $K$ are set independently; adaptive determination of this hyperparameter remains an open problem.
- Integration with DINO-based approaches has not been explored.

## Related Work & Insights

- **LangSplat** and **LEGaussians** serve as the primary baselines; they use autoencoders and codebooks for feature compression, respectively, but do not address cross-view consistency.
- Video object segmentation methods (XMem, SAM2) are innovatively introduced into 3D scene understanding to establish cross-view correspondences.
- For downstream tasks requiring high-quality 3D semantics (robotic manipulation, autonomous driving), this method provides a more reliable semantic field.

## Rating

- Novelty: ⭐⭐⭐⭐ First to address cross-view semantic inconsistency and propose a contrastive codebook solution
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablations and sufficient visual analysis, though dataset scale is relatively small
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation
- Value: ⭐⭐⭐⭐ Addresses a practical bottleneck in 3D open-vocabulary understanding

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](../../CVPR2025/autonomous_driving/generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)

</div>

<!-- RELATED:END -->
