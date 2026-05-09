---
title: >-
  [Paper Note] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification
description: >-
  [CVPR 2026][Remote Sensing][Optical-SAR cross-modal] This paper proposes SDF-Net, a physics-guided structure-aware disentangled feature learning network that enforces cross-modal geometric consistency via intermediate-layer gradient energy (SCL) and decouples shared/modality-specific features at the terminal layer (DFL) with parameter-free additive fusion, achieving 60.9% mAP (+3.5% vs. SOTA TransOSS) on HOSS-ReID.
tags:
  - CVPR 2026
  - Remote Sensing
  - Optical-SAR cross-modal
  - ship re-identification
  - structure-aware
  - feature disentanglement
  - gradient energy
date: 2026-05-08
content_hash: 7219d1a13b56e179
---

# SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification

**Conference**: CVPR 2026
**arXiv**: [2603.12588](https://arxiv.org/abs/2603.12588)
**Code**: [github.com/cfrfree/SDF-Net](https://github.com/cfrfree/SDF-Net)
**Area**: Remote Sensing / Cross-Modal Retrieval
**Keywords**: Optical-SAR cross-modal, ship re-identification, structure-aware, feature disentanglement, gradient energy

## TL;DR

This paper proposes SDF-Net, a physics-guided structure-aware disentangled feature learning network that enforces cross-modal geometric consistency via intermediate-layer gradient energy (SCL) and decouples shared/modality-specific features at the terminal layer (DFL) with parameter-free additive fusion, achieving 60.9% mAP (+3.5% vs. SOTA TransOSS) on HOSS-ReID.

## Background & Motivation

**Background**: Optical and SAR sensors are complementary in maritime surveillance — optical imagery provides high-resolution visual detail while SAR enables all-weather, all-day observation. Cross-modal ship ReID is a fundamental task for fusing these two heterogeneous data sources. Existing methods fall into three categories: implicit attention-based alignment (TransOSS), statistical/generative alignment (CycleGAN), and handcrafted geometric descriptors (HOPC).

**Limitations of Prior Work**: Optical and SAR imagery exhibit severe nonlinear radiometric distortion (NRD) — passive visible-light reflection versus active microwave backscattering results in completely different textural appearances for the same target. Implicit alignment methods ignore physical priors; generative synthesis introduces artifacts and incurs high cost; person ReID assumptions about deformable bodies do not apply to rigid ship structures.

**Key Challenge**: Ships are rigid bodies whose geometric structure remains stable across modalities, yet texture is highly modality-dependent. Existing methods attempt to align all features without distinguishing structure from texture.

**Goal**: To explicitly exploit geometric structure as a physics-grounded "anchor" for cross-modal association, enforcing strict structural consistency while tolerating modality-specific appearance variation.

**Key Insight**: Gradient energy is extracted from intermediate-layer features — abstract enough to filter low-level noise (e.g., SAR speckle) while preserving spatial topological information. At the terminal layer, shared and modality-specific features are disentangled and fused via additive residuals to maintain discriminability.

**Core Idea**: Use intermediate-layer gradient energy statistics as scale-invariant structural descriptors to enforce cross-modal geometric consistency, while disentangling and fusing modality-invariant and modality-specific features at the terminal layer.

## Method

### Overall Architecture

ViT-B/16 dual-head tokenizer encoder → intermediate-layer (Block 6) structure-aware consistency learning (SCL) → terminal-layer disentangled feature learning (DFL) → parameter-free additive residual fusion → identity classification. Training jointly optimizes identity loss, structural consistency loss, and orthogonality constraint loss.

### Key Designs

1. **Structure-Aware Consistency Learning (SCL)**:
    - **Function**: Extracts cross-modal geometrically invariant structural features from the ViT intermediate layer (Block 6).
    - **Mechanism**: Computes spatial gradients of intermediate feature maps $\mathbf{G}_x(h,w) = \mathbf{F}(h,w+1) - \mathbf{F}(h,w-1)$; spatially integrates them into a gradient energy descriptor $\mathbf{f}_{struct} = \mathbf{e}_x + \mathbf{e}_y \in \mathbb{R}^{B \times C}$; applies Instance Normalization to eliminate inter-modal magnitude differences; constructs identity-level cross-modal prototypes and enforces alignment via Euclidean distance constraints.
    - **Design Motivation**: The intermediate layer balances spatial detail and semantic abstraction — shallow layers are corrupted by speckle noise, while deep layers lose spatial information due to global aggregation. The gradient operator naturally acts as a high-pass filter insensitive to SAR multiplicative intensity differences. The macro-level geometric skeleton of rigid ships remains consistent across modalities under near-vertical observation.

2. **Disentangled Feature Learning with Additive Fusion (DFL)**:
    - **Function**: Decomposes the terminal representation into modality-invariant shared features and modality-specific features, fused in a parameter-free manner.
    - **Mechanism**: Two parallel linear projection heads map $\mathbf{F}^{(L)}$ to $\mathbf{f}_{sh}$ and $\mathbf{f}_{sp}$; an orthogonality constraint $\mathcal{L}_{orth} = \mathbb{E}[|\langle \bar{\mathbf{f}}_{sh}, \bar{\mathbf{f}}_{sp} \rangle|]$ ensures subspace independence; additive fusion $\mathbf{f}_{fuse} = \mathbf{f}_{sh} + \mathbf{f}_{sp}$.
    - **Design Motivation**: Unlike person ReID which discards modality-specific features, SAR corner-reflector responses and optical paint reflections in ships contain identity-discriminative cues. Additive fusion treats modality-specific features as residual corrections with zero additional parameters.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{id} + 10.0 \cdot \mathcal{L}_{orth} + 1.0 \cdot \mathcal{L}_{struct}$$

- $\mathcal{L}_{id}$: Label-smoothed cross-entropy + weighted triplet loss
- SGD, weight decay 1e-4, batch size 32 (P=8 identities × K=4 images, strictly 2 optical + 2 SAR per identity)
- 100 epochs, linear warmup, single RTX 3090

## Key Experimental Results

### Main Results

| Method | Type | All mAP | All R1 | O→S mAP | S→O mAP |
|--------|------|---------|--------|---------|---------|
| TransReID | Single-modal ReID | 48.1% | 60.8% | 27.3% | 20.9% |
| D2InterNet | Single-modal ReID | 50.2% | 59.1% | 33.0% | 28.8% |
| DEEN | Cross-modal ReID | 43.8% | 58.5% | 31.3% | 27.4% |
| VersReID | Cross-modal ReID | 49.3% | 59.7% | 25.7% | 27.7% |
| TransOSS | Optical-SAR specific | 57.4% | 65.9% | 48.9% | 38.7% |
| **SDF-Net** | **Optical-SAR specific** | **60.9%** | **69.9%** | **50.0%** | **46.6%** |

### Ablation Study

| SCL | DFL | All mAP | All R1 | O→S mAP | S→O mAP |
|-----|-----|---------|--------|---------|---------|
| ✗ | ✗ | 58.6% | 67.6% | 46.5% | 44.5% |
| ✓ | ✗ | 59.2% | 66.5% | 47.6% | 46.6% |
| ✗ | ✓ | 59.8% | 69.9% | 49.3% | 41.4% |
| ✓ | ✓ | **60.9%** | **69.9%** | **50.0%** | **46.6%** |

| Fusion Strategy | All mAP | All R1 | S→O mAP |
|-----------------|---------|--------|---------|
| $\mathbf{f}_{sp}$ only | 58.7% | 67.6% | 43.3% |
| $\mathbf{f}_{sh}$ only | 59.2% | 68.8% | 43.1% |
| Concatenation | 59.5% | 69.3% | 44.1% |
| **Additive Sum** | **60.9%** | **69.9%** | **46.6%** |

| Extraction Layer $B_s$ | All mAP | S→O mAP |
|------------------------|---------|---------|
| 2 | 59.7% | 46.0% |
| 4 | 60.4% | 45.3% |
| **6** | **60.9%** | **46.6%** |
| 8 | 58.4% | 45.5% |
| 10 | 58.7% | 44.7% |

### Key Findings

- The most challenging SAR-to-Optical scenario shows the largest improvement (+7.9% mAP), validating the critical role of geometric anchoring in bridging the active/passive modality gap.
- SCL and DFL contribute complementarily: SCL improves the S→O direction (geometric alignment) while DFL improves Rank-1 (identity discriminability), and their combination yields the best overall performance.
- Additive fusion outperforms concatenation — zero additional parameters with optimal performance, confirming the effectiveness of treating modality-specific features as residual corrections.
- Block 6 is the optimal structural extraction layer — too shallow (Block 2) suffers from speckle noise contamination, while too deep (Block 8+) collapses spatial information.

## Highlights & Insights

- Physics-guided design philosophy — leveraging rigid-body geometric invariance as a prior rather than purely data-driven alignment, yielding +7.9% mAP in the hardest SAR→Optical scenario.
- The combination of gradient energy and Instance Normalization to construct scale-invariant structural descriptors is elegant and generalizable to other cross-modal matching tasks.
- The parameter-free additive fusion design is concise and effective, avoiding the computational overhead and artifact issues associated with generative methods.

## Limitations & Future Work

- Validation is limited to the single HOSS-ReID dataset with only 1,063 training images, which is a relatively small scale.
- The method assumes near-vertical observation — 3D SAR distortions (layover/foreshortening) at extreme incidence angles are not addressed.
- In very low-resolution SAR imagery, structural contours may be entirely overwhelmed by speckle.
- Multi-scale structural extraction is not explored; only a single layer (Block 6) is used.

## Related Work & Insights

- **TransOSS (ICCV 2025)**: ViT-based optical-SAR baseline, 57.4% mAP → Ours 60.9%; lacks explicit physical constraints.
- **HOPC (classical remote sensing)**: Handcrafted local geometric descriptors operating at the pixel level; this work elevates the concept to the intermediate latent space.
- **Hi-CMD (VI-ReID)**: Discards modality-specific features; this paper demonstrates that modality-specific information should be retained as residual corrections for rigid-body targets.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Physics-guided gradient energy structural features combined with disentangled residual fusion; theoretical motivation is well-grounded.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Three-protocol evaluation with three-dimensional ablation (modules / fusion strategies / extraction layers) and comprehensive baselines.
- ⭐⭐⭐⭐ **Writing Quality**: The correspondence between physical motivation and method design is explicit, with a complete logical chain.
- ⭐⭐⭐⭐ **Value**: Practically valuable for cross-modal remote sensing retrieval; the gradient energy concept is transferable to other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](pretrained_image_matchers_for_sar_optical_satellite_registration.md)
- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[CVPR 2026\] ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery](acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)

</div>

<!-- RELATED:END -->
