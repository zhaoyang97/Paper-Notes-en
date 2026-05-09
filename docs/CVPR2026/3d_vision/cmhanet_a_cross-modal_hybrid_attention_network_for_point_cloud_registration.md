---
title: >-
  [Paper Note] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration
description: >-
  [CVPR2026][3D Vision][Point cloud registration] CMHANet is proposed to deeply integrate 2D image texture-semantic features with 3D point cloud geometric features via a cross-modal hybrid attention mechanism, combined with a contrastive learning objective, achieving state-of-the-art point cloud registration performance on 3DMatch/3DLoMatch.
tags:
  - CVPR2026
  - 3D Vision
  - Point cloud registration
  - cross-modal fusion
  - hybrid attention
  - contrastive learning
  - multimodal features
date: 2026-05-08
content_hash: 056460d489e548f0
---

# CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration

**Conference**: CVPR2026
**arXiv**: [2603.12721](https://arxiv.org/abs/2603.12721)
**Code**: [DongXu-Zhang/CMHANet](https://github.com/DongXu-Zhang/CMHANet)
**Area**: 3D Vision
**Keywords**: Point cloud registration, cross-modal fusion, hybrid attention, contrastive learning, multimodal features

## TL;DR

CMHANet is proposed to deeply integrate 2D image texture-semantic features with 3D point cloud geometric features via a cross-modal hybrid attention mechanism, combined with a contrastive learning objective, achieving state-of-the-art point cloud registration performance on 3DMatch/3DLoMatch.

## Background & Motivation

1. **Point cloud registration is a fundamental 3D vision task**: widely applied in large-scale 3D reconstruction, augmented reality, and scene understanding, yet noise, sparsity, and low overlap in real-world scenes severely limit registration accuracy.
2. **Traditional methods rely solely on geometric information**: ICP and its variants are sensitive to initialization and prone to local optima; deep learning-based methods have improved but mostly still exploit only a single 3D geometric modality.
3. **2D images and 3D point clouds are naturally complementary**: point clouds provide precise geometry but lack texture descriptions, while 2D images offer dense texture semantics but lack explicit 3D information; fusing both enables more comprehensive scene understanding.
4. **Prevalence of RGB-D sensors**: paired RGB-D cameras are now widely available, making multimodal data straightforward to acquire.
5. **Existing multimodal fusion methods lack fine-grained modeling**: approaches such as IMFNet and CMIGNet attempt fusion but employ generic strategies without carefully modeling the 2D/3D feature interaction.
6. **Attention mechanisms excel at capturing long-range dependencies**: Transformer architectures can model global context, yet how to fully leverage their advantages in cross-modal settings remains an open problem.

## Method

### Overall Architecture

CMHANet follows a four-stage pipeline: (1) feature extraction and downsampling → (2) hybrid attention-based superpoint matching → (3) dense correspondence refinement → (4) rigid transformation estimation.

### Feature Extraction and Downsampling

- **Point cloud encoder**: A KPConv-FPN backbone extracts geometric features from source/target point clouds and downsamples them into superpoints $S^P, S^Q$; raw dense points are associated with superpoints via Nearest-Superpoint Aggregation.
- **Image encoder**: A ResUNet-50 backbone extracts visual features $\hat{F}^n, \hat{F}^m$ from the corresponding 2D images.

### Hybrid Attention Module

Three attention mechanisms are applied iteratively for $N$ rounds to progressively refine features:

1. **Geometric Self-Attention**: models global structural relationships within a single point cloud. Keys incorporate both learned features and relative geometric embeddings (distance embedding + angular embedding), endowing the attention with spatial awareness.
2. **Geometric Aggregation-Attention**: the core cross-modal fusion component. 3D superpoints serve as Queries to retrieve relevant visual context from the 2D image domain; 3D coordinate position embeddings and 2D pixel position embeddings are injected into both Queries and Keys to enforce cross-modal geometric consistency; aggregated image features are incorporated into point cloud features via residual connections.
3. **Geometric Cross-Attention**: enables interaction between the source and target point clouds, with the source providing Queries and the target providing Keys/Values, establishing consistent cross-cloud correspondences.

### Superpoint Matching and Dense Correspondence

- **Superpoint matching**: a similarity matrix is computed from the fused features; a learnable dustbin parameter handles non-overlapping points (outliers); the Sinkhorn algorithm (50 iterations) performs doubly stochastic normalization; top-$k$ selection yields the final superpoint match pairs.
- **Dense correspondence refinement**: point-level similarity is computed within each matched superpoint pair; Sinkhorn and top-$k$ selection are applied again to extract fine-grained point-to-point correspondences.

### Transformation Estimation

- **Local stage**: a weighted SVD computes a local rigid transformation for each superpoint pair (differentiable closed-form solution).
- **Global stage**: a Local-to-Global verification strategy counts the number of spatial inliers for each candidate transformation over the full correspondence set, selecting the transformation with the most inliers as the final result (threshold $\tau_a = 5$ cm), avoiding the non-differentiability of RANSAC.

### Loss & Training

Three terms are jointly optimized: $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_f + \lambda \mathcal{L}_{cmc}$ ($\lambda = 0.5$)

- **Coarse matching loss $\mathcal{L}_c$**: metric learning based on overlap-aware circle loss; point pairs with overlap ratio >10% are treated as positives, weighted by the square root of their overlap ratio.
- **Fine matching loss $\mathcal{L}_f$**: minimizes the alignment error of dense point correspondences within matched superpoint pairs.
- **Cross-modal contrastive loss $\mathcal{L}_{cmc}$**: constructs contrastive learning between geometric and image features at the superpoint level, with diagonal entries as positives and off-diagonal entries as negatives; effective even with batch size = 1.

## Key Experimental Results

### Main Results

Registration Recall (%) on 3DMatch and 3DLoMatch benchmarks:

| Method | 3DMatch (5000) | 3DLoMatch (5000) |
|--------|---------------|-----------------|
| Predator | 89.0 | 61.2 |
| CoFiNet | 89.3 | 67.5 |
| GeoTransformer | — | — |
| OIF-PCR | — | — |
| **CMHANet** | **92.4** | **75.5** |

Registration accuracy (RANSAC-free): RRE 1.764°, RTE 0.060 m (3DMatch); RRE 2.839°, RTE 0.084 m (3DLoMatch), both best in class.

### Zero-Shot Generalization (TUM RGB-D)

Trained on 3DMatch and directly evaluated on 8 TUM sequences, achieving a mean RMSE of 0.76 (×10⁻²), substantially outperforming Robust ICP (1.69) and DGR (1.44).

### Ablation Study

| Configuration | 3DMatch RR | 3DLoMatch RR |
|---------------|-----------|-------------|
| Loss only (no HA, no IM) | 89.9 | 71.9 |
| No HA (with Loss + IM) | 90.5 | 72.4 |
| No Aggre-Att | 91.3 | 73.6 |
| **Full CMHANet** | **92.4** | **75.5** |

- Removing hybrid attention: 3DLoMatch RR drops by 3.1%.
- Removing aggregation attention (core cross-modal fusion): 3DLoMatch RR drops by 1.9%.
- Removing the image module (geometry only): 3DMatch RR drops to 89.9%.

Image encoder comparison: ResNet-34 < ResUNet-50 ≈ ResNet-101; ResUNet-50 achieves the best accuracy–efficiency trade-off.

## Highlights & Insights

- **Fine-grained cross-modal fusion design**: the aggregation attention injects 3D coordinate embeddings and 2D pixel embeddings into Queries and Keys, enabling geometrically-aware cross-modal retrieval that surpasses naive concatenation.
- **End-to-end registration without RANSAC**: the Local-to-Global strategy replaces RANSAC, is differentiable, and is more than 100× faster.
- **Strong generalization**: zero-shot evaluation on TUM RGB-D surpasses all baselines by a large margin.
- **Efficient contrastive loss design**: the superpoint-level cross-modal contrastive loss remains effective at batch size = 1, requiring no large batches.

## Limitations & Future Work

- Registration quality degrades under extremely low overlap (<10%) or on entirely textureless planar surfaces.
- **Increased inference overhead**: cross-modal encoding and fusion lead to longer feature extraction times compared to unimodal methods.
- **Dependency on RGB-D paired data**: requires extrinsically calibrated point cloud–image correspondences, limiting applicability to LiDAR-only scenarios.
- **No decoupling of rotation and translation**: the authors suggest that separately estimating $R$ and $t$ could further improve alignment accuracy in future work.
- **Not validated on outdoor large-scale scenes**: experiments are confined to indoor datasets; outdoor benchmarks such as KITTI remain untested.

## Related Work & Insights

- **vs. GeoTransformer**: GeoTransformer employs only geometric self-attention and cross-attention; CMHANet additionally introduces image aggregation attention, yielding lower RRE (1.764° vs. 1.772°).
- **vs. IMFNet / PCR-CG**: both are multimodal methods, yet CMHANet outperforms PCR-CG by 9.2% and IMFNet by 27.1% in 3DLoMatch RR, validating the superiority of hybrid attention fusion.
- **vs. CoFiNet**: both adopt a coarse-to-fine strategy, but CMHANet's cross-modal information provides a significant advantage under low overlap (75.5% vs. 67.5%).
- **vs. traditional ICP variants**: on TUM zero-shot evaluation, CMHANet achieves a mean RMSE of only 0.76 compared to 1.69 for Robust ICP.

## Rating

- Novelty: ⭐⭐⭐⭐ — The cross-modal positional embedding design in the aggregation attention is original; the superpoint-level construction of positive/negative pairs for the contrastive loss is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three datasets (3DMatch/3DLoMatch/TUM); ablation studies are complete (modules/backbones/estimators); both quantitative and qualitative analyses are provided.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, consistent notation, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — A practical direction for multimodal point cloud registration; SOTA results with open-source code offer meaningful reference for subsequent work.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds](s2am3d_scale-controllable_part_segmentation_of_3d_point_cloud.md)
- [\[CVPR 2026\] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs](hg-i2p_bridging_modalities_for_generalizable_image-to-point-cloud_registration_v.md)
- [\[CVPR 2026\] CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](clipoint3d_language-grounded_few-shot_unsupervised_3d_point_cloud_domain_adaptat.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)

<!-- RELATED:END -->
