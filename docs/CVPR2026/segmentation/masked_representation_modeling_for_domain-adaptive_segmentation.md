---
title: >-
  [Paper Note] Masked Representation Modeling for Domain-Adaptive Segmentation
description: >-
  [CVPR 2026][Segmentation][Masked modeling] This paper proposes Masked Representation Modeling (MRM), which performs masking and reconstruction in latent space rather than pixel space as a plug-and-play auxiliary task for UDA segmentation, yielding an average gain of +2.3 mIoU across 4 baselines on GTA→Cityscapes.
tags:
  - CVPR 2026
  - Segmentation
  - Masked modeling
  - representation reconstruction
  - domain-adaptive segmentation
  - auxiliary task
  - plug-and-play
date: 2026-05-08
content_hash: 3c8706b40d83bf59
---

# Masked Representation Modeling for Domain-Adaptive Segmentation

**Conference**: CVPR 2026
**arXiv**: [2509.13801](https://arxiv.org/abs/2509.13801)
**Code**: [GitHub](https://github.com/Wenlve-Zhou/MRM)
**Area**: Segmentation / Unsupervised Domain Adaptation
**Keywords**: Masked modeling, representation reconstruction, domain-adaptive segmentation, auxiliary task, plug-and-play

## TL;DR

This paper proposes Masked Representation Modeling (MRM), which performs masking and reconstruction in latent space rather than pixel space as a plug-and-play auxiliary task for UDA segmentation, yielding an average gain of +2.3 mIoU across 4 baselines on GTA→Cityscapes.

## Background & Motivation

Unsupervised domain adaptation (UDA) for semantic segmentation aims to overcome domain shift by leveraging labeled source-domain data alongside unlabeled target-domain data. Auxiliary self-supervised tasks represent an effective avenue for enhancing UDA; contrastive learning has been extensively explored with notable success, yet another powerful family of self-supervised methods—masked image modeling (MIM)—remains largely unexplored in UDA segmentation.

Two core reasons why MIM has not been adopted:

**Input structure constraints**: MIM requires masking image patches, which disrupts the input structure of segmentation networks such as DeepLab and DAFormer.

**Optimization conflicts**: MIM reconstructs low-level pixels/tokens, which is misaligned with the high-level semantic objectives of segmentation tasks.

The paper's core idea: **perform masking and reconstruction in feature space (rather than input space)**, thereby preserving the input pipeline while aligning the reconstruction objective with the segmentation objective—since the reconstructed features are fed directly into the segmentation decoder for pixel classification.

## Method

### Overall Architecture

MRM is inserted as an auxiliary task into an existing UDA pipeline. A masking operation is applied to the encoder output features $f^t = E(x^t)$; a lightweight Rebuilder $R(\cdot)$ then reconstructs the masked features, which are subsequently passed to the decoder $D(\cdot)$ for classification. At inference time, the Rebuilder is removed entirely, incurring **zero additional inference overhead**. The overall optimization objective is $\mathcal{L}_{overall} = \mathcal{L}_{sup} + \mathcal{L}_{uda} + \lambda\mathcal{L}_{mrm}$.

### Key Designs

1. **Representation-space masked reconstruction (MRM)**: The encoder output features $f^t \in \mathbb{R}^{C \times H \times W}$ are first passed through a representation embedding layer that rescales them to $C' \times H' \times W'$ (unifying feature dimensions across different architectures). Subsequently, 40% of spatial locations are randomly masked and replaced with learnable mask tokens. After reconstruction, a projection layer restores the original dimensions, and the result is fused with the original features as $f^r = M^s \odot f^o + (1 - M^s) \odot f^t$. **The key distinction from MIM**: the reconstruction target is not pixel values but correct semantic predictions by the decoder on the reconstructed features, $\mathcal{L}_{mrm} = -\sum_{i,j,c} \tilde{y}_{ijc} \log D(R(E(x^t)))_{ijc}$, supervised with pseudo-labels $\tilde{y}$.

2. **Lightweight Rebuilder module**: An asymmetric design inspired by the MAE decoder, comprising: (a) a representation embedding layer—linear mapping for channel adjustment plus bilinear interpolation for spatial adjustment; (b) a masking layer—uniformly random sampling to generate a binary mask; (c) Transformer blocks (only 1–2) with absolute positional encoding; and (d) a Projector consisting of transposed convolutions to restore original dimensions. The overall design is highly lightweight, requiring only a minimal number of Transformer blocks to be effective.

3. **Multi-scale adaptation for hierarchical architectures**: For hierarchical architectures such as DAFormer, rather than instantiating a separate Rebuilder at each stage (which would be prohibitively expensive), only the final-stage representation is processed by the Transformer, and independent upsampling operations subsequently generate multi-scale features for each target resolution. This design is inspired by the finding in ViTDet that multi-scale features can be obtained from the final representation via simple upsampling.

### Loss & Training

- $\mathcal{L}_{mrm}$: cross-entropy loss supervised with target-domain pseudo-labels
- Balancing coefficient $\lambda = 1.0$
- Rebuilder configuration: 2 Transformer blocks, embedding dim = 512, $H'=W'=16,\ C'=512$
- Masking ratio: 40% (lower than MAE's 60–75%, as visible tokens in MRM undergo shallower processing)
- Rebuilder learning rate: $2 \times 10^{-4}$
- Training hardware: single NVIDIA RTX 3090

## Key Experimental Results

### Main Results

| Dataset | Metric (mIoU) | +MRM (on MIC) | MIC Original | Gain |
|--------|------|------|----------|------|
| GTA → Cityscapes | mIoU | **77.5** | 75.9 | +1.6 |
| Synthia → Cityscapes | mIoU | **68.1** | 67.3 | +0.8 |

| Baseline | GTA→City Original | +MRM | Gain |
|---------|------|------|------|
| DACS | 52.1 | 55.9 | **+3.8** |
| DAFormer | 68.3 | 70.3 | **+2.0** |
| HRDA | 73.8 | 75.4 | **+1.6** |
| MIC | 75.9 | 77.5 | **+1.6** |

### Ablation Study

| Configuration | mIoU | Notes |
|------|---------|------|
| Masking ratio 20% | 54.3 | Excessive information; reconstruction task too easy |
| Masking ratio 40% | **55.9** | Optimal |
| Masking ratio 60% | 55.2 | Reduced diversity of reconstruction signal |
| Masking ratio 80% | 54.1 | Excessive masking harms semantic consistency |
| Transformer blocks n=1 | 55.4 | Already effective with a single block |
| Transformer blocks n=2 | **55.9** | Optimal |
| Transformer blocks n=4 | 55.6 | Additional blocks yield no further gain |

### Key Findings

- MRM is **model-agnostic**: consistent gains are observed across 4 different baselines (DACS/DAFormer/HRDA/MIC).
- MIC + MRM achieves 77.5 mIoU, surpassing all prior GTA→Cityscapes SOTA results (+1.4).
- Consistent effectiveness on Synthia→Cityscapes (average +2.8 mIoU) demonstrates cross-domain stability.
- Improvements are particularly pronounced for fine-grained categories (traffic sign, rider, motorbike), indicating that MRM enhances high-level semantic discrimination in the decoder.
- The optimal masking ratio of 40% is substantially lower than standard MIM (60–75%), reflecting the distinct characteristics of operating in feature space.

## Highlights & Insights

- **Key conceptual contribution**: Transferring MIM from input space to feature space simultaneously resolves both the architectural compatibility issue and the optimization conflict with segmentation.
- **Truly plug-and-play**: The Rebuilder is used only during training and is entirely removed at inference, incurring zero additional overhead—a critically important property for practical deployment.
- MRM is complementary to contrastive learning: whereas contrastive learning primarily strengthens the encoder, MRM jointly enhances both the encoder and the decoder.
- Lightweight design: only 1–2 Transformer blocks are required, demonstrating that deep models are unnecessary for effective feature-space reconstruction.

## Limitations & Future Work

- Performance depends on pseudo-label quality; label noise may impose an upper bound on MRM's effectiveness.
- Although the multi-scale adaptation scheme is efficient, it reconstructs only from the final-layer features, potentially losing shallow-layer details.
- Joint use with contrastive learning remains unexplored, despite the complementary nature of the two approaches.
- The masking strategy employs simple uniform random sampling, without leveraging semantic or domain-specific information to guide mask placement.

## Related Work & Insights

- **MAE/ConvNeXtV2**: Classical MIM methods that reconstruct in input space and cannot be directly applied to segmentation.
- **PiPa/GANDA/QuadMix**: Recent SOTA UDA methods that MRM could further enhance as an auxiliary task.
- **IFVD/CWD/CIRKD**: Segmentation knowledge distillation methods that primarily strengthen the encoder; MRM is distinctive in simultaneously strengthening the decoder.
- Inspiration: The concept of feature-space reconstruction can be generalized to domain adaptation in other dense prediction tasks, such as detection and depth estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ Transferring masked modeling from input space to representation space is conceptually clear with a well-defined contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two major benchmarks, 4 baselines, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The comparative diagram of three auxiliary task paradigms is highly intuitive, and the method description is clear.
- Value: ⭐⭐⭐⭐ A concise and effective plug-and-play strategy with practical value for the UDA segmentation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](boundary_segment_action_segmentation.md)
- [\[CVPR 2026\] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)

</div>

<!-- RELATED:END -->
