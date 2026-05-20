---
title: >-
  [Paper Note] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification
description: >-
  [CVPR2026][Object Detection][Optical-SAR Cross-Modal Matching] SDF-Net is proposed to exploit the rigid-body geometric structure of ships as a cross-modal invariant anchor. It enforces structural consistency via gradient…
tags:
  - "CVPR2026"
  - "Object Detection"
  - "Optical-SAR Cross-Modal Matching"
  - "Ship Re-Identification"
  - "Feature Disentanglement"
  - "Structural Consistency"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 8a122baccb676cfa
---

# SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification

**Conference**: CVPR2026
**arXiv**: [2603.12588](https://arxiv.org/abs/2603.12588)  
**Code**: [cfrfree/SDF-Net](https://github.com/cfrfree/SDF-Net)  
**Area**: Object Detection / Cross-Modal Re-Identification
**Keywords**: Optical-SAR Cross-Modal Matching, Ship Re-Identification, Feature Disentanglement, Structural Consistency, Vision Transformer

## TL;DR

SDF-Net is proposed to exploit the rigid-body geometric structure of ships as a cross-modal invariant anchor. It enforces structural consistency via gradient energy extracted from intermediate layers, and disentangles modality-shared/specific features at the terminal layer with additive residual fusion, achieving SOTA on HOSS-ReID (All mAP 60.9%, surpassing TransOSS by 3.5%).

## Background & Motivation

**State of the Field — Large optical-SAR modality gap**: Optical images rely on passive reflection while SAR relies on active microwave backscattering, resulting in severe nonlinear radiometric distortion (NRD) between the two modalities; direct appearance alignment is mathematically ill-posed.

**Limitations of Prior Work — Physical priors ignored**: Mainstream methods treat cross-modal ReID as a pure statistical distribution alignment problem, failing to explicitly exploit the physical property of ships as rigid bodies.

**Limitations of Prior Work — Generative methods are costly and introduce artifacts**: Image translation methods such as CycleGAN reduce statistical discrepancy but incur high computational costs and may introduce hallucination artifacts that obscure identity-critical features.

**Limitations of Prior Work — Pedestrian ReID methods cannot be directly transferred**: VI-ReID methods focus on body pose alignment and deformable parts, whereas ships are rigid bodies whose geometric structure remains stable across modalities while radiometric appearance changes drastically — the design assumptions are fundamentally different.

**Root Cause — Geometric structure as a cross-modal invariant**: The hull contour, aspect ratio, and spatial layout of ships are highly consistent across optical and SAR imagery, whereas texture and intensity responses are modality-specific.

**Starting Point — Intermediate-layer features as optimal structural probes**: Raw pixels are severely corrupted by SAR speckle noise; high-level semantics are too abstract and lose spatial topology; intermediate layers retain geometric layout while being abstract enough to filter low-level noise.

## Method

### Overall Architecture

SDF-Net is built on a ViT-B/16 backbone and comprises four stages:

- **(a) Input Stage** — Cross-modal dual-head Tokenizer: optical/SAR images are mapped to a unified $C$-dimensional latent space via independent linear projection heads, neutralizing low-level sensor discrepancies.
- **(b) Intermediate Stage** — Structure-aware Consistency Learning (SCL): gradient energy is extracted from the $B_s$-th Transformer block to align cross-modal structural prototypes.
- **(c) Terminal Stage** — Disentangled Feature Learning (DFL): the final representation is decomposed into modality-shared identity feature $\mathbf{f}_{sh}$ and modality-specific feature $\mathbf{f}_{sp}$, fused via additive residual combination.
- **(d) Inference Stage** — Bidirectional cross-modal retrieval using the fused feature $\mathbf{f}_{fuse}$.

### Key Design 1: Structure-Aware Consistency Learning (SCL)

**Intermediate-layer gradient energy extraction**: First-order partial derivatives along horizontal and vertical directions are computed from the feature map $\mathbf{F}^{(B_s)}$ at layer $B_s=6$:

$$\mathbf{G}_x(h,w) = \mathbf{F}(h,w+1) - \mathbf{F}(h,w-1)$$

Spatial integration yields a channel-level gradient energy descriptor $\mathbf{f}_{struct} = \mathbf{e}_x + \mathbf{e}_y \in \mathbb{R}^{B \times C}$; global aggregation effectively suppresses interference from isolated strong scatterers such as SAR corner reflectors.

**Scale-invariant Instance Normalization**: Instance Normalization is applied to $\mathbf{f}_{struct}$ along the channel dimension, mapping the high dynamic range of SAR and the narrow-band reflectance of optical imagery to a standardized manifold, stripping modality-specific "style" while preserving geometric "content."

**Prototype-level consistency loss**: For each identity $i$ in a mini-batch, optical/SAR structural prototypes $\mathbf{c}_i^o$ and $\mathbf{c}_i^s$ are computed, and their Euclidean distance is minimized:

$$\mathcal{L}_{struct} = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \|\mathbf{c}_i^o - \mathbf{c}_i^s\|_2^2$$

### Key Design 2: Disentangled Feature Learning with Residual Fusion (DFL)

The terminal representation $\mathbf{F}^{(L)}$ is decomposed via two independent linear projection heads into:
- $\mathbf{f}_{sh}$: modality-invariant shared identity features (regularized by $\mathcal{L}_{struct}$)
- $\mathbf{f}_{sp}$: modality-specific features (retaining SAR corner reflector responses, optical color texture, etc.)

An **orthogonality constraint** ensures subspace independence: $\mathcal{L}_{orth} = \mathbb{E}[|\langle \bar{\mathbf{f}}_{sh}, \bar{\mathbf{f}}_{sp} \rangle|]$

**Additive residual fusion**: $\mathbf{f}_{fuse} = \mathbf{f}_{sh} + \mathbf{f}_{sp}$, which is parameter-free and does not expand the feature dimension; modality-specific features serve as residuals that supplement fine-grained identity discriminative information.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{id} + \lambda_{orth} \mathcal{L}_{orth} + \lambda_{struct} \mathcal{L}_{struct}$$

where $\mathcal{L}_{id}$ consists of label-smoothed cross-entropy and weighted triplet loss; $\lambda_{orth}=10.0$, $\lambda_{struct}=1.0$.

## Key Experimental Results

### Dataset and Setup

- **HOSS-ReID benchmark**: training set of 1,063 images (574 optical + 489 SAR); test set evaluated under three protocols: All-to-All, Optical-to-SAR, and SAR-to-Optical.
- Single RTX 3090, ViT-B/16 backbone, input resolution 256×128, SGD optimizer, 100 epochs, P×K sampling (8 identities × 4 instances, 2 optical + 2 SAR per identity).

### Main Results: Comparison with SOTA on HOSS-ReID

| Method | All mAP | All R1 | O→S mAP | O→S R1 | S→O mAP | S→O R1 |
|--------|---------|--------|---------|--------|---------|--------|
| TransReID (ICCV21) | 48.1 | 60.8 | 27.3 | 18.5 | 20.9 | 11.9 |
| VersReID (TPAMI24) | 49.3 | 59.7 | 25.7 | 13.8 | 27.7 | 17.9 |
| D2InterNet (SIGIR25) | 50.2 | 59.1 | 33.0 | 21.5 | 28.8 | 25.4 |
| TransOSS (ICCV25) | 57.4 | 65.9 | 48.9 | 33.8 | 38.7 | 29.9 |
| **SDF-Net (Ours)** | **60.9** | **69.9** | **50.0** | **35.4** | **46.6** | **38.8** |

### Ablation Study

**Module effectiveness**: SCL alone improves SAR→Optical mAP (44.5→46.6); DFL substantially improves All R1 (67.6→69.9); combining both achieves the optimal balance (All mAP 60.9).

**Structure extraction layer selection**: $B_s=6$ yields the best performance; shallow layers (2/4) suffer from noise interference, while deep layers (8/10/12) exhibit spatial semantic collapse.

**Fusion strategy comparison**:

| Strategy | All mAP | All R1 | S→O mAP |
|----------|---------|--------|---------|
| Modality-specific only $\mathbf{f}_{sp}$ | 58.7 | 67.6 | 43.9 |
| Shared only $\mathbf{f}_{sh}$ | 59.2 | 68.2 | 43.1 |
| Concatenation (Cat) | 59.5 | 68.8 | 45.1 |
| **Additive fusion (Ours)** | **60.9** | **69.9** | **46.6** |

### Key Findings

- **High computational efficiency**: SDF-Net has identical parameter count to TransOSS (86.24M); FLOPs increase by only 0.17G (<0.8%), yet yield gains of 3.5% mAP and 4.0% R1.
- The SAR→Optical scenario shows the most significant improvement (mAP +7.9%), validating the effectiveness of geometric structure anchors in combating SAR radiometric distortion.
- Hyperparameter sensitivity analysis shows stable performance near $\lambda_{orth}=10.0$ and $\lambda_{struct}=1.0$, with mild rather than catastrophic degradation upon deviation.

## Highlights & Insights

- **Physics-prior-driven design**: This work is the first to establish rigid-body geometric invariance as a core learning objective in optical-SAR ship ReID, rather than relying on implicit statistical alignment.
- **Zero additional parameters**: Both SCL (gradient energy + IN) and DFL (additive fusion) are parameter-free operations, achieving extreme efficiency.
- **Intermediate-layer structural probe**: The method cleverly leverages the property of intermediate Transformer layers to filter low-level noise while preserving spatial topology.
- **Prototype-level alignment**: Structural alignment is performed at the identity level rather than the instance level, avoiding overfitting to single-sample noise.
- Grad-CAM visualizations and layer-by-layer feature evolution analyses provide substantial interpretability support.

## Limitations & Future Work

- Gradient energy methods may fail for extremely low-resolution SAR targets where structural contours are entirely submerged in dense speckle.
- The current framework assumes nadir or near-vertical observation geometry; 3D structural distortions (layover, foreshortening) caused by extreme incidence angles in practical satellite imagery are not yet addressed.
- Validation is conducted on a single dataset (HOSS-ReID); generalizability requires confirmation on additional benchmarks.
- The training set is relatively small (1,063 images); large-scale pretraining and data augmentation strategies remain unexplored.

## Related Work & Insights

- **Cross-modal pedestrian ReID (VI-ReID)**: Pose alignment methods for visible-infrared matching such as DEEN, Hi-CMD, and VersReID are not applicable to rigid-body ships.
- **Disentangled representation learning**: Hi-CMD's hierarchical disentanglement and orthogonal subspace projection; this work treats modality-specific features as residual complements rather than discarding them as noise.
- **Optical-SAR matching**: Handcrafted structural descriptors such as HOPC operate at the raw pixel level and are vulnerable to speckle; this work shifts operations to the intermediate latent space.
- **Strongest baseline TransOSS (ICCV25)**: ViT-based cross-modal tokenization relying on implicit self-attention alignment, lacking explicit physical constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of physical priors, intermediate-layer gradient energy, and parameter-free fusion is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation (modules/layers/fusion/hyperparameters/computation), rich visualizations, but only a single dataset.
- Writing Quality: ⭐⭐⭐⭐ — Physical motivation is clearly articulated; mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ — Pioneers a physics-guided, structure-aware paradigm for cross-modal matching in remote sensing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)
- [\[CVPR 2026\] DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[CVPR 2026\] Towards Intrinsic-Aware Monocular 3D Object Detection](towards_intrinsic-aware_monocular_3d_object_detection.md)

</div>

<!-- RELATED:END -->
