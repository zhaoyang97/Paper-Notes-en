---
title: >-
  [Paper Note] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement
description: >-
  [CVPR 2026][Multimodal VLM][multimodal registration] This paper proposes HRNet, which learns clean shared representations via cross-scale disentanglement and adaptive projection (CDAP), and jointly predicts rigid and non-rigid transformations in a unified coarse-to-fine pipeline without iteration, achieving state-of-the-art performance on four multimodal datasets.
tags:
  - CVPR 2026
  - Multimodal VLM
  - multimodal registration
  - hybrid transformation
  - feature disentanglement
  - cross-scale consistency
  - Mamba
date: 2026-05-08
content_hash: 4aef0928433013fe
---

# Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement

**Conference**: CVPR 2026
**arXiv**: [2603.19623](https://arxiv.org/abs/2603.19623)
**Code**: [GitHub](https://github.com/Chunlei0913/HRNet)
**Area**: Multimodal VLM / Multimodal Image Registration
**Keywords**: multimodal registration, hybrid transformation, feature disentanglement, cross-scale consistency, Mamba

## TL;DR
This paper proposes HRNet, which learns clean shared representations via cross-scale disentanglement and adaptive projection (CDAP), and jointly predicts rigid and non-rigid transformations in a unified coarse-to-fine pipeline without iteration, achieving state-of-the-art performance on four multimodal datasets.

## Background & Motivation

**State of the Field**: Multimodal image registration (e.g., RGB-thermal infrared, RGB-SAR) is foundational for cross-modal fusion. Existing deep learning methods commonly adopt multi-scale strategies to improve accuracy, but are typically limited to a single transformation type.

**Limitations of Prior Work**: (a) Most multi-scale frameworks support only rigid or non-rigid transformations — rigid transformations cannot handle local deformations, while non-rigid transformations distort structural integrity under large global offsets; (b) existing hybrid registration methods employ serial cascades (rigid followed by non-rigid), where rigid and non-rigid components are estimated in different feature spaces, making coordination difficult and causing downstream stages to inherit upstream errors; (c) shared feature extraction methods alleviate modality discrepancy, but constraints primarily act on the shared component, while modality-private information still leaks into the shared space.

**Root Cause**: How to simultaneously estimate global rigid alignment and local non-rigid deformation within a unified feature space, without interference from modality-private information?

**Paper Goals**: Design a unified framework that simultaneously addresses: (a) modality-private leakage into the shared feature space; (b) coordinated estimation of rigid and non-rigid transformations.

**Starting Point**: "Disentangle-then-Align" — first learn clean multi-scale shared representations via CDAP, then jointly predict hybrid transformations in HPPM.

**Core Idea**: Representation disentanglement (cross-scale gating + adaptive projection) + hybrid parameter prediction (rigid + non-rigid within a unified coarse-to-fine pipeline) = a unified deformation field produced in a single forward pass.

## Method

### Overall Architecture
Given fixed image $I_f$ and moving image $I_m$, the pipeline proceeds in three stages: (1) a shared backbone with MSBN extracts multi-scale features $F, M$; (2) the CDAP module disentangles these into clean shared features $(\hat{F}^s, \hat{M}^s)$; (3) HPPM predicts the hybrid transformation $\phi$ in a coarse-to-fine manner, which is used to warp $I_m$.

### Key Designs

1. **CDAP: Cross-Scale Disentanglement and Adaptive Projection**:

    - **Function**: Follows a decompose-gate-project pipeline to learn clean multi-scale shared representations.
    - **Mechanism**:
        - **Decompose**: At each scale, a shared-weight extractor $E_{sh}^i$ extracts modality-agnostic shared components, while modality-specific extractors $E_{pf/m}^i$ extract private components.
        - **Gate (ILDA)**: Cross-scale attention gating uses semantics from adjacent scales: $\widetilde{F}_i^s = \alpha_i^s \odot F_i^s - \gamma^i \alpha_i^p \odot F_i^p$, where $\alpha_i^s, \alpha_i^p$ are computed via cross-scale attention.
        - **Project (DSS)**: Data-adaptively generates approximately orthogonal bases $W_i^s = Gen^i(z_i^s)$, and projects as $\hat{F}_i^s = \widetilde{F}_i^s W_i^{s\top}$.
    - **Design Motivation**: Decomposition alone cannot prevent private-to-shared leakage (explicit gating suppression is required); fixed projections lack flexibility (adaptive bases are needed); cross-scale attention exploits complementary semantic information from adjacent scales.

2. **HPPM: Hybrid Parameter Prediction Module**:

    - **Function**: Jointly predicts rigid and non-rigid transformations non-iteratively within a unified pipeline.
    - **Mechanism**: Five scales process features from coarse to fine. At the coarsest scale, HRB estimates global rigid parameters $H$ (via GAP + FC) and encodes them as a coarse deformation field. At each subsequent scale, the previous transformation $\phi_{i-1}$ is upsampled to warp the current moving features; the concatenated features are fed into HRB to estimate an incremental deformation $\phi_i' = \text{conv}(f_i)$, accumulating updates as $\phi_i = \text{upsample}(\phi_{i-1}) + \phi_i'$. Each HRB contains two RSSBs (Residual State Space Blocks, based on Mamba) to model long-range dependencies.
    - **Design Motivation**: Unlike serial cascades with separate estimation, HPPM jointly estimates both transformations within the same shared feature space, with rigid predictions immediately encoded as flow and progressively refined at subsequent scales. Mamba's state space model captures long-range dependencies at low computational cost.

3. **Structured Regularization**:

    - **Function**: Three complementary regularizers shape the shared feature space.
    - **Mechanism**:
        - $L_{ccd}$ (cross-covariance decorrelation): $\|\text{Cov}(\hat{F}_i^s, F_i^p)\|_F^2$ → reduces shared-private coupling.
        - $L_{bo}$ (basis orthogonality): $\|W^{(i)}W^{(i)\top} - I\|_F^2$ → prevents subspace degeneracy.
        - $L_{cs}$ (cross-scale directional consistency): $1 - \cos(\hat{F}_i^s, \hat{F}_{i+1}^s)$ → maintains cross-scale semantic consistency.
        - $L_{tri}$ (triplet loss): pulls together cross-modal co-located shared features and pushes away private interference.
    - **Design Motivation**: The four losses constrain shared space quality from complementary perspectives: decoupling, non-redundancy, consistency, and alignment.

### Loss & Training
- Total loss: $L = \alpha_r L_r + \alpha_n L_n + \alpha_s L_s + \alpha_{tri} L_{tri} + \alpha_{cs} L_{cs} + \alpha_{ccd} L_{ccd} + \alpha_{bo} L_{bo}$
- **Three-stage curriculum training**: warmup (10%) → mid (50%) → late (40%), with progressive adjustment of loss weights (e.g., $\alpha_n$: 6→10→12).
- Adam optimizer, lr=1e-4, batch size=8, 100 epochs, images resized to 256×256.

## Key Experimental Results

### Main Results (Rigid Registration)

| Method | RGB-NIR RE↓ | RGB-TIR RE↓ | RGB-IR RE↓ | RGB-SAR RE↓ |
|--------|-------------|-------------|------------|-------------|
| IHN | 3.887 | 3.006 | 5.684 | 7.087 |
| MMRNet | 3.179 | 2.472 | 4.406 | 7.075 |
| **HRNet (Ours)** | **0.785** | **0.744** | **0.578** | **3.161** |

RE reduction relative to MMRNet: 75.3%, 69.9%, 86.9%, 55.3% (average ~72%).

### Main Results (Non-Rigid Registration)

| Method | RGB-NIR RE↓ | RGB-TIR RE↓ | RGB-IR RE↓ | RGB-SAR RE↓ |
|--------|-------------|-------------|------------|-------------|
| ADRNet (hybrid) | - | - | - | - |
| MMRNet | - | - | - | - |
| **HRNet (Ours)** | **best** | **best** | **best** | **best** |

RE reduction relative to ADRNet: 61.2%, 62.5%, 66.9%, 23.3%.

### Ablation Study

| Configuration | Key Effect | Remarks |
|---------------|------------|---------|
| w/o CDAP | Shared features contain private noise | Registration accuracy degrades |
| w/o ILDA gating | Private information leaks | Insufficient disentanglement |
| w/o DSS projection | Cross-modal alignment unstable | Feature space less compact |
| Rigid only | Cannot handle local deformation | Poor structural integrity |
| Non-rigid only | Distortion under large offsets | Insufficient global alignment |
| **Full HRNet** | **Unified rigid + non-rigid** | **Best across all metrics** |

### Key Findings
- **Large advantage of hybrid registration**: On RGB-IR, RE drops from 4.406 to 0.578 (86.9%↓), demonstrating that joint estimation is far superior to single-paradigm approaches.
- RGB-SAR is the most challenging setting (largest modality gap), yet HRNet still achieves significant improvements.
- Progressively increasing the non-rigid weight ($\alpha_n$: 6→12) during three-stage curriculum training is critical.

## Highlights & Insights
- **Unified hybrid framework**: For the first time, rigid and non-rigid transformations are jointly estimated non-iteratively within a single pipeline, producing a single unified deformation field.
- **Comprehensive disentanglement**: The decompose-gate-project pipeline in CDAP combined with four structured regularizers fundamentally resolves private information leakage.
- **Mamba for registration**: RSSBs provide long-range dependency modeling at low computational overhead, well-suited for global structure awareness in registration.

## Limitations & Future Work
- Current validation is conducted at 256×256 resolution; efficiency and performance at higher resolutions remain to be tested.
- Curriculum training hyperparameter tuning may require adjustment for different modality pairs.
- Robustness to extreme occlusion or fully non-overlapping regions is not discussed.

## Related Work & Insights
- **vs. ADRNet (serial hybrid)**: ADRNet estimates transformations in stages, causing downstream stages to inherit upstream errors; HRNet jointly estimates within a unified feature space.
- **vs. Shi et al. (feature disentanglement)**: Existing methods only constrain the shared component; HRNet explicitly suppresses private leakage via ILDA gating and structured regularization.
- **Insight**: Cross-scale feature interaction via adjacent-scale gating is a generalizable idea worth exploring in other multi-scale tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified hybrid registration framework, CDAP disentanglement design, and integration of Mamba are all meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four multimodal datasets, both rigid and non-rigid settings, with detailed ablation and curriculum training analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and complete derivations, though notation density is high.
- Value: ⭐⭐⭐⭐ Provides a general template for multimodal registration, though the application scope is relatively specialized.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](ebmc_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)

<!-- RELATED:END -->
