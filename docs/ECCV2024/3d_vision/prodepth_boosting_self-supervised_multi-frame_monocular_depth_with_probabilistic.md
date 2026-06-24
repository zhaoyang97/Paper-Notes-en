---
title: >-
  [Paper Note] ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion
description: >-
  [ECCV 2024][3D Vision][Self-Supervised Depth Estimation] Proposes ProDepth, a probabilistic fusion framework that infers dynamic region uncertainty via an auxiliary decoder to adaptively fuse single-frame and multi-frame depth probability distributions using a weighted geometric mean. This directly corrects erroneous matching costs in the cost volume, and combined with an uncertainty-aware loss reweighting strategy, achieves SOTA performance in self-supervised multi-frame mon…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Self-Supervised Depth Estimation"
  - "Multi-Frame Monocular Depth"
  - "Probabilistic Fusion"
  - "Cost Volume Modulation"
  - "Dynamic Object Handling"
date: 2026-05-08
content_hash: 04b3f0d13998ed26
---

# ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion

**Conference**: ECCV 2024  
**arXiv**: [2407.09303](https://arxiv.org/abs/2407.09303)  
**Code**: [sungmin-woo.github.io/prodepth](https://sungmin-woo.github.io/prodepth/)  
**Area**: 3D Vision  
**Keywords**: Self-Supervised Depth Estimation, Multi-Frame Monocular Depth, Probabilistic Fusion, Cost Volume Modulation, Dynamic Object Handling  

## TL;DR

Proposes ProDepth, a probabilistic fusion framework that infers dynamic region uncertainty via an auxiliary decoder to adaptively fuse single-frame and multi-frame depth probability distributions using a weighted geometric mean. This directly corrects erroneous matching costs in the cost volume, and combined with an uncertainty-aware loss reweighting strategy, achieves SOTA performance in self-supervised multi-frame monocular depth estimation.

## Background & Motivation

- **Background**: Self-supervised multi-frame monocular depth estimation relies on geometric consistency between consecutive frames under the static scene assumption, using a cost volume to evaluate the probability of each depth candidate, which outperforms single-frame methods in overall performance.
- **Limitations of Prior Work**: Moving objects in dynamic scenes violate the static scene assumption, leading to misaligned feature matching in the cost volume, which generates erroneous depth probability distributions. Meanwhile, the reprojection-based photometric loss provides misleading training supervision in dynamic regions.
- **Key Challenge**: Multi-frame methods predict more accurately in static regions, whereas single-frame methods handle moving objects better due to their independence from the cost volume. Their strengths are complementary, but adaptively selecting which cue to trust at the pixel level remains a difficult challenge.
- **Core Problem**: (1) How to accurately identify dynamic objects (without relying on extra semantic segmentation networks); (2) How to directly correct the corrupted matching cost distributions in the cost volume caused by dynamic objects; (3) How to mitigate the impact of erroneous supervision in dynamic regions on training.
- **Key Insight**: Existing methods either supervise multi-frame depth with single-frame depth at the loss level (ManyDepth) or adjust dynamic object positions with single-frame depth at the input level (DynamicDepth), neither of which directly corrects errors within the cost volume itself.
- **Core Idea**: Represent both single-frame depth and multi-frame cost volume as probability distributions over depth candidates, and adaptively fuse the two distributions using a weighted geometric mean based on inferred uncertainty to directly modulate the cost volume.

## Method

### Overall Architecture

ProDepth comprises three main components, as shown in Figure 2:
1. **Auxiliary Depth Estimation and Uncertainty Inference**: A single-frame decoder estimates a Gaussian distribution of depth; an auxiliary cost volume decoder estimates the corrupted depth, which is compared against the single-frame depth to infer uncertainty.
2. **Probabilistic Cost Volume Modulation (PCVM)**: Adaptively fuses single-frame and multi-frame depth probability distributions based on uncertainty.
3. **Uncertainty-Aware Loss Reweighting**: Reduces erroneous supervision in dynamic regions during training based on uncertainty.

### Key Designs

**Module 1: Auxiliary Depth Estimation and Uncertainty Inference**

The single-frame depth estimation network $\theta_{\text{single}}$ outputs the mean $D_{\text{single}}$ and variance $\sigma_p^2$ of a Gaussian distribution, trained by maximizing the log-likelihood:

$$\mathcal{L}_p^{\log}(D_{\text{single}}) = \frac{(\mathcal{L}_p(D_{\text{single}}))^2}{\sigma_p^2} + \log \sigma_p^2$$

The auxiliary decoder $\psi_{\text{dec}}$ estimates the depth $D_{\text{cv}}$ from the corrupted cost volume features. Leveraging the structural awareness of depth estimation (consistent pixel depth within the same object), it amplifies pixel-level inconsistency into clear object-level errors. Uncertainty is calculated from the difference between the two depths:

$$U = 1 - e^{-\beta |D_{\text{single}} - D_{\text{cv}}|}, \quad \beta = 0.6$$

$U \in [0,1]$ represents the probability of each pixel being in a dynamic region.

**Module 2: Probabilistic Cost Volume Modulation (PCVM)**

Convert single-frame depth into a probability distribution over depth candidates (Gaussian PDF):

$$p_{\text{single}}(d_i|x) = \frac{1}{\sqrt{2\pi\sigma_p^2(x)}} \exp\left(-\frac{(d_i - D_{\text{single}}(x))^2}{2\sigma_p^2(x)}\right)$$

Convert multi-frame cost volume matching costs into a probability distribution via softmax:

$$p_{\text{cv}}(d_i|x) = \frac{\exp(-\mathcal{C}(x,i))}{\sum_{j=1}^k \exp(-\mathcal{C}(x,j))}$$

Fuse the two distributions using a weighted geometric mean:

$$P(d|x) = p_{\text{single}}(d|x)^{U(x)} \cdot p_{\text{cv}}(d|x)^{1-U(x)}$$

High uncertainty (dynamic pixels) $\rightarrow$ fusion results favor the single-frame distribution; low uncertainty (static pixels) $\rightarrow$ favor the multi-frame distribution. After fusion, min-max normalization is used to restore the original scale of the cost volume.

**Module 3: Uncertainty-Aware Loss Reweighting**

$$\mathcal{L}_{up} = M \odot (1 - U) \odot \mathcal{L}_p, \quad M = [U < \gamma]$$

Combining a binary mask $M$ (to exclude high-uncertainty regions) and a continuous weight $(1-U)$ (to reduce the loss weight of likely dynamic regions based on probability) is more granular than a pure binary mask.

### Loss & Training

Total loss function:

$$\mathcal{L} = \sum_x \left[\mathcal{L}_{up,s}(D_{\text{multi}}) + \lambda_1 \mathcal{L}_{up,s}^{log}(D_{\text{single}}) + \lambda_2 \mathcal{L}_p(D_{\text{cv}}) + \lambda_3 \mathcal{L}_c\right]$$

- Multi-frame and single-frame depths utilize the uncertainty-aware loss $\mathcal{L}_{up}$ to prevent overfitting in dynamic regions.
- Cost volume depth uses standard $\mathcal{L}_p$ (**deliberately allowing** erroneous supervision in dynamic regions), enabling the auxiliary decoder to learn to produce erroneous depths, thereby enhancing uncertainty inference.
- Gradients of $\mathcal{L}_p(D_{\text{cv}})$ flow back only to the auxiliary cost volume decoder parameters, without affecting the cost volume itself.

## Key Experimental Results

### Main Results

**Cityscapes Dataset** (with many dynamic objects):

| Method | Extra Semantics | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | $\delta < 1.25$ ↑ |
|------|---------|-----------|----------|--------|-------------------|
| ManyDepth | None | 0.114 | 1.193 | 6.223 | 0.875 |
| DynamicDepth | ✓ | 0.103 | 1.000 | 5.867 | 0.895 |
| **ProDepth** | **None** | **0.095** | **0.876** | **5.531** | **0.908** |

**KITTI Dataset**:

| Method | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | $\delta < 1.25$ ↑ |
|------|-----------|----------|--------|-------------------|
| DepthFormer | 0.090 | 0.661 | 4.149 | 0.905 |
| DualRefine | 0.090 | 0.658 | 4.237 | 0.912 |
| **ProDepth** | **0.086** | **0.629** | **4.139** | **0.918** |

### Ablation Study

Component ablation on Cityscapes (Abs Rel ↓):

| # | Uncertainty Inference | PCVM | Loss Strategy | Abs Rel |
|---|-----------|------|---------|---------|
| 1 | consistency mask | None | masking | 0.107 |
| 3 | segmentation mask | None | masking | 0.100 |
| 7 | Weighted Uncertainty $U$ | None | masking | 0.100 |
| 8 | Weighted Uncertainty $U$ | ✓ | masking | 0.098 |
| 9 | Weighted Uncertainty $U$ | None | reweighting | 0.097 |
| **10** | **Weighted Uncertainty $U$** | **✓** | **masking+reweighting** | **0.095** |

### Key Findings

1. Combining PCVM with segmentation masks actually degrades performance (Row #4), because segmentation masks include static objects, causing multi-frame cues to be discarded.
2. Weighted uncertainty + PCVM yields the best performance, as probabilistic fusion is more effective than selection based on binary criteria.
3. Loss reweighting is superior to pure binary masking because it can handle boundary regions with ambiguous uncertainty.
4. Generalization experiments on Waymo demonstrate that ProDepth's cross-dataset transferability outperforms DynamicDepth, which requires semantic segmentation.

## Highlights & Insights

- **Weighted Geometric Mean Fusion**: Compared to a weighted arithmetic mean (addition), the multiplicative nature better preserves the peak positions of each distribution.
- **Bootstrap-style Uncertainty Inference**: Leverages errors within the cost volume itself to back-infer dynamic regions, requiring no external networks.
- **Probabilistic Continuity**: Uncertainty is continuous within $[0,1]$, which is more fine-grained than binary masks.
- **End-to-End Unification**: Uncertainty inference, cost volume correction, and loss adjustment are tightly coupled.

## Limitations & Future Work

- Uncertainty inference relies on the quality of single-frame depth; if the single-frame depth itself is inaccurate, the inferred uncertainty may be biased.
- PCVM may lose precision when the discretization of depth candidates is sparse.
- Experiments are only validated on autonomous driving scenes (Cityscapes/KITTI/Waymo); applicability to indoor scenes remains unknown.
- Optical flow information could be integrated to further improve the detection accuracy of dynamic objects.

## Related Work & Insights

- **ManyDepth**: First introduced adaptive cost volumes to resolve scale ambiguity and proposed a binary consistency mask $\rightarrow$ This work points out its pixel-level independence and lack of structural awareness.
- **DynamicDepth**: Utilizes pretrained segmentation networks to identify movable objects and adjust input images $\rightarrow$ This work points out that segmentation masks contain unnecessary static objects.
- **DepthFormer**: Replaces traditional similarity metrics with attention mechanisms to improve feature matching $\rightarrow$ Still fails to resolve errors within the cost volume itself.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Probabilistic cost volume modulation is a novel direct correction strategy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluations across three datasets, detailed ablations, and generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with thorough problem analysis.
- **Value**: ⭐⭐⭐⭐ — Deployment-friendly, as it does not rely on auxiliary semantic networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [\[ECCV 2024\] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [\[ECCV 2024\] PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion](pcf-lift_panoptic_lifting_by_probabilistic_contrastive_fusion.md)
- [\[CVPR 2026\] Fusion of Depth and Semantics for Probabilistic Floorplan Localization](../../CVPR2026/3d_vision/fusion_of_depth_and_semantics_for_probabilistic_floorplan_localization.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)

</div>

<!-- RELATED:END -->
