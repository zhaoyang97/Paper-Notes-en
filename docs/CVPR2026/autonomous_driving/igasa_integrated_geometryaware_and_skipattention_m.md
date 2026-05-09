---
title: >-
  [Paper Note] IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration
description: >-
  [CVPR 2026][Autonomous Driving][point cloud registration] IGASA is a point cloud registration framework that combines a Hierarchical Pyramid Architecture (HPA), Hierarchical Cross-Layer Attention (HCLA) with skip-attention fusion, and Iterative Geometry-Aware Refinement (IGAR) with dynamic consistency weighting. It achieves 94.6% Registration Recall on 3DMatch (SOTA), 100% RR on KITTI, with a total inference time of only 2.763s.
tags:
  - CVPR 2026
  - Autonomous Driving
  - point cloud registration
  - hierarchical pyramid architecture
  - skip-attention
  - geometry-aware iterative refinement
  - coarse-to-fine
date: 2026-05-08
content_hash: 0393348ffd9ae4be
---

# IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration

**Conference**: CVPR 2026
**arXiv**: [2603.12719](https://arxiv.org/abs/2603.12719)
**Code**: [https://github.com/DongXu-Zhang/IGASA](https://github.com/DongXu-Zhang/IGASA)
**Area**: 3D Vision / Point Cloud Registration / Autonomous Driving
**Keywords**: [point cloud registration, hierarchical pyramid architecture, skip-attention, geometry-aware iterative refinement, coarse-to-fine]

## TL;DR
IGASA is a point cloud registration framework that combines a Hierarchical Pyramid Architecture (HPA), Hierarchical Cross-Layer Attention (HCLA) with skip-attention fusion, and Iterative Geometry-Aware Refinement (IGAR) with dynamic consistency weighting. It achieves 94.6% Registration Recall on 3DMatch (SOTA), 100% RR on KITTI, with a total inference time of only 2.763s.

## Background & Motivation
Point cloud registration (PCR) is a fundamental task for autonomous driving, robotic navigation, and environment modeling. Traditional ICP and its variants rely on nearest-neighbor distance minimization, making them sensitive to initialization and prone to local minima. Deep learning methods—particularly Transformer-based approaches such as GeoTransformer, RoITr, and SIRA-PCR—have made notable progress in capturing long-range dependencies and global context. However, a **semantic gap** persists: as networks deepen to acquire high-level semantics, fine-grained geometric details are diluted through downsampling. Naive skip-connection fusion strategies (concatenation or summation) fail to effectively bridge the discrepancy between low-level geometric cues and high-level semantic embeddings. Furthermore, the refinement stage of coarse-to-fine frameworks typically relies on RANSAC or hard-threshold outlier rejection, which is computationally expensive and may discard valid correspondences.

## Core Problem
How to effectively bridge the semantic gap in multi-scale feature extraction—preserving fine-grained geometry while capturing global semantics—and suppress outliers more robustly during the fine registration stage?

## Method

### Overall Architecture
The pipeline consists of three stages: (a) the HPA module constructs a three-level feature pyramid ($F_\text{ordinary}$, $F_\text{minor}$, $F_\text{primary}$) for source point cloud $P$ and target point cloud $Q$, progressively enlarging the receptive field; (b) the HCLA module uses SGIRA to fuse global semantics with local geometry and SAIGA to refine features for coarse matching; (c) the IGAR module iteratively refines coarse matches by dynamically updating correspondence weights to suppress outliers, yielding the final transformation $\{R, t\}$.

### Key Designs
1. **Hierarchical Pyramid Architecture (HPA)**: A three-level KPConv-based encoder. The ordinary layer uses base voxel size $dl_0$ (influence radius $2.5 \cdot dl_0$) to capture fine-grained local geometry; the minor layer uses voxel size $2 \cdot dl_0$ for semi-global structure; the primary layer uses voxel size $4 \cdot dl_0$ (convolution radius $10 \cdot dl_0$) for global semantics. Feature dimensions are 64, 128, and 256, respectively, with progressively increasing channel depth.
2. **Hierarchical Cross-Layer Attention (HCLA)**: Composed of two sub-modules—(i) **SGIRA** (Skip-Guided Inter-Resolution Attention): leverages primary-layer global semantic features to guide minor-layer feature fusion via a gated fusion mechanism (dual-branch convolution + adaptive gating weights + residual correction), then generates enhanced features $F_\text{minor}^{+}$ using attention combining semantic similarity, geometric distance compensation, and skip residuals; (ii) **SAIGA** (Skip-Augmented Intrinsic Geometry Attention): performs self-attention on $F_\text{minor}^{+}$, integrating geometric distance weights (controlled by learnable $\alpha$) and skip attention scores to sharpen the discriminability of local spatial features, yielding $F_\text{minor}^{++}$. The two modules operate in series: SGIRA acts as a "semantic filter" and SAIGA as a "geometric sharpener."
3. **Iterative Geometry-Aware Refinement (IGAR)**: Introduces dynamic geometric consistency weighting in the fine registration stage, replacing hard RANSAC rejection. At each iteration $k$, correspondence weights are recomputed based on the current transformation $(R^{(k)}, t^{(k)})$ as $w_{ij} = \exp(-\text{residual}^2/\sigma^2) \times \mathbb{1}[\text{residual} < \tau]$, and registration error is minimized in a weighted manner. Optimal $R^*$ and $t^*$ are solved via weighted centroid computation and SVD of the weighted cross-covariance matrix. With $N=5$ iterations by default, this achieves soft suppression of outliers.

### Loss & Training
- **Total loss**: $L_\text{total} = L_\text{mat} + L_\text{key} + L_\text{den}$
- $L_\text{mat} = \lambda_p \cdot L_p + \lambda_c \cdot L_c$: multi-level matching probability loss + weighted cross-entropy loss (distinguishing true matches from outliers)
- $L_\text{key} = \lambda_f \cdot L_f + \lambda_k \cdot L_k + \lambda_i \cdot L_i$: InfoNCE descriptor contrastive loss + keypoint displacement $L_2$ loss + confidence binary cross-entropy
- $L_\text{den} = \lambda_t \cdot L_t + \lambda_r \cdot L_r$: translation $L_2$ loss + rotation Frobenius norm loss
- Training: AdamW, lr=$10^{-4}$, weight decay=$10^{-4}$, lr decay $0.95$/epoch; 15 epochs on 3DMatch, 30 on KITTI, 10 on nuScenes; single RTX 3090, batch size 1

## Key Experimental Results

| Dataset | Metric | Ours (IGASA) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| 3DMatch (5000 pts) | RR | **94.6%** | SIRA-PCR 93.6% | +1.0% |
| 3DMatch (5000 pts) | IR | **87.9%** | RoITr 82.6% / SIRA-PCR 70.8% | +5.3% / +17.1% |
| 3DMatch (250 pts) | RR | **94.3%** | SIRA-PCR 92.4% | +1.9% |
| 3DLoMatch (5000 pts) | RR | **76.5%** | GeoTransformer 75.5% | +1.0% |
| 3DLoMatch (5000 pts) | IR | **61.6%** | RoITr 54.3% / SIRA-PCR 43.3% | +7.3% / +18.3% |
| KITTI | RR | **100.0%** | Predator/GeoTransformer 99.8% | +0.2% |
| KITTI | RTE | **4.6 cm** | OIF-Net 6.5 cm | −1.9 cm |
| KITTI | RRE | **0.24°** | OIF-Net 0.23° | on par |
| nuScenes | RR | **99.9%** | HRegNet 99.9% | on par |
| nuScenes | RTE | **0.12 m** | HRegNet 0.18 m | −33% |
| nuScenes | RRE | **0.21°** | HRegNet 0.45° | −53% |
| Inference time (3DMatch) | Total | 2.763 s | GeoTransformer 2.701 s | +0.062 s |

### Ablation Study
- **Contribution of HCLA**: baseline RR 89.6% → +HCLA 92.8% (+3.2%), validating the importance of cross-layer semantic alignment.
- **Contribution of IGAR**: +HCLA 92.8% → +HCLA+IGAR 94.6% (+1.8%); IR increases from 79.2% to 87.9% (+8.7%), confirming that iterative geometric refinement is critical for outlier suppression.
- **SGIRA vs. SAIGA**: SGIRA alone achieves FMR 96.2%; SAIGA alone achieves IR 84.2%; using both yields FMR 98.2% and IR 87.9%—demonstrating the synergy between semantic filtering and geometric sharpening.
- **Loss functions**: Each individual loss term performs poorly (IR < 75%); joint optimization of all three achieves IR 87.9%, confirming that multi-task supervision is indispensable.
- **Computational efficiency**: Total inference time of 2.763 s is highly competitive with GeoTransformer (2.701 s) and CoFiNet (2.660 s); the additional overhead of HCLA+IGAR is only ~0.1 s.

## Highlights & Insights
- IGAR's soft-suppression strategy replaces RANSAC's hard rejection, offering improved robustness and differentiability.
- The dual-unit design of HCLA (semantic filter SGIRA + geometric sharpener SAIGA) effectively bridges the semantic gap.
- IR improvements are substantial (87.9% on 3DMatch, 61.6% on 3DLoMatch), indicating a significant gain in correspondence quality.
- When the number of sampled points drops from 5000 to 250, RR decreases only marginally from 94.6% to 94.3%, demonstrating strong robustness.
- IGASA achieves 100% RR on KITTI with RTE of only 4.6 cm.

## Limitations & Future Work
- FMR on 3DLoMatch (80.5%) falls short of RoITr (89.6%) and SIRA-PCR (88.8%), indicating room for improvement in feature robustness under sparse sampling.
- Iterative refinement ($N=5$) incurs additional computational cost; fewer iterations may be necessary for real-time applications.
- Adaptability to highly dynamic environments has not been validated.
- Comparisons with recent diffusion-based registration methods (e.g., PointDifformer) under large-rotation scenarios are not sufficiently explored.

## Related Work & Insights
- **vs. GeoTransformer**: GeoTransformer (2022) models global dependencies with geometric Transformers but employs a simple fusion strategy without cross-layer attention; its IR on 3DMatch ranges from 71.9% to 85.1%, compared to IGASA's 87.9%. Inference times are nearly identical (2.701 s vs. 2.763 s).
- **vs. RoITr**: RoITr (2023) uses rotation-invariant Transformers, achieving IR of 82.6%–83.0%, which outperforms GeoTransformer but remains below IGASA's 87.9%. RoITr achieves higher FMR (89.6% vs. 82.1%), suggesting that IGASA's feature recall warrants further improvement.
- **vs. SIRA-PCR**: SIRA-PCR (2023) adopts sim-to-real adaptation, achieving RR of 93.6% on 3DMatch (vs. IGASA's 94.6%) and 73.5% on 3DLoMatch (vs. IGASA's 76.5%); the performance gap is more pronounced in low-overlap scenarios.

The design principle of SGIRA—using deep-layer semantics to guide shallow-layer feature fusion—is transferable to skip-connection optimization in U-Net-style architectures for medical image segmentation. IGAR's dynamic geometric consistency weighting is applicable to other outlier-rejection tasks such as visual localization and Structure-from-Motion. The multi-scale pyramid combined with cross-layer attention paradigm parallels EPT (another paper from the same batch), both emphasizing that information at different granularities requires differentiated processing.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-unit HCLA design and IGAR soft-suppression strategy are genuinely novel, though the overall approach remains within the coarse-to-fine paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets (3DMatch, 3DLoMatch, KITTI, nuScenes) with comprehensive ablation and runtime analysis.
- Writing Quality: ⭐⭐⭐ Content is thorough but some formulas and notation are redundant; the related work section cites several papers with limited relevance to the core topic.
- Value: ⭐⭐⭐⭐ Advances the state of the art across the board in point cloud registration, with particularly significant IR gains; open-source code facilitates broader adoption.

---

- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value to Me: ⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)
- [\[CVPR 2026\] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](buildanypoint_3d_building_structured_abstraction_from_diverse_point_clouds.md)
- [\[CVPR 2026\] U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences](u4d_uncertainty-aware_4d_world_modeling_from_lidar_sequences.md)

<!-- RELATED:END -->
