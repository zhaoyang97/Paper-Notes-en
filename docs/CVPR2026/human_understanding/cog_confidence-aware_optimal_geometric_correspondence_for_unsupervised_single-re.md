---
title: >-
  [Paper Note] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation
description: >-
  [CVPR2026][Human Understanding][novel object pose estimation] This paper proposes COG, a framework that models cross-view correspondences as a confidence-aware optimal transport (OT) problem. By predicting per-point conf…
tags:
  - "CVPR2026"
  - "Human Understanding"
  - "novel object pose estimation"
  - "optimal transport"
  - "confidence learning"
  - "unsupervised learning"
  - "point cloud registration"
  - "visual foundation models"
date: 2026-05-08
content_hash: 63422d5c60435114
---

# COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation

**Conference**: CVPR2026
**arXiv**: [2603.00493](https://arxiv.org/abs/2603.00493)
**Code**: [YC-Che/COG](https://github.com/YC-Che/COG)
**Area**: Human Understanding / 6DoF Object Pose Estimation
**Keywords**: novel object pose estimation, optimal transport, confidence learning, unsupervised learning, point cloud registration, visual foundation models

## TL;DR

This paper proposes COG, a framework that models cross-view correspondences as a confidence-aware optimal transport (OT) problem. By predicting per-point confidence scores as transport marginal constraints, COG suppresses contributions from non-overlapping regions and outliers, achieving unsupervised single-reference 6DoF novel object pose estimation on par with supervised methods.

## Background & Motivation

**Task Definition**: Estimating the 6DoF pose (rotation + translation) of arbitrary novel objects from a single reference RGB-D image — a fundamental task for robotics, AR, and 3D scene understanding.

**Limitations of Prior Work**: Traditional methods rely on CAD models or multi-view references, limiting practical scalability. Under the single-reference setting, large viewpoint changes and partial observations make the problem severely ill-posed.

**Drawbacks of Discrete Matching**: Existing methods (e.g., UnoPose) construct discrete one-to-one correspondences via argmax, which tends to collapse onto a few dominant keypoints, leaving the majority of points unused.

**Non-differentiability**: Discrete matching breaks the gradient flow, preventing end-to-end unsupervised training.

**OT Post-processing Issue**: Existing OT-based methods (RPM-Net, Robust OT) adopt uniform marginals, with confidence used only as post-hoc calibration rather than being jointly optimized with correspondences end-to-end.

**Semantic Ambiguity**: Pure geometric matching is ambiguous; semantic priors are needed to distinguish different object parts.

## Method

### Overall Architecture

COG adopts a **coarse-to-fine** two-stage architecture:

- **Preprocessing**: UnoSeg segments object masks → depth maps are back-projected into 3D point clouds → DINO extracts per-pixel RGB features as semantic descriptors.
- **Coarse Stage**: Farthest point sampling yields sparse point clouds (256 points); a geometric Transformer encoder-decoder predicts per-point confidence and features; Sinkhorn OT computes soft correspondences; weighted SVD estimates the coarse pose.
- **Fine Stage**: The query point cloud is transformed using the coarse pose, and the full point cloud (1024 points) with positional encodings is used for fine-grained alignment to produce the final pose.
- **Inference Iteration**: At inference, the estimated pose is used to iteratively transform the query point cloud for refinement (default: 1 iteration).

### Key Designs: Confidence-aware Optimal Transport

**Core Idea**: Cross-view correspondences are modeled as an OT problem in which per-point confidence scores serve directly as the target marginal constraints of the transport plan.

1. **Affinity Kernel Construction**: Geometric similarity (cosine) and semantic similarity (denoised DINO features) are fused: $\mathbf{K}_{[i,j]} = \exp(\frac{1}{\tau}\langle \mathbf{G}_{p[i]}, \mathbf{G}_{q[j]}\rangle_{\cos}) \cdot (1 + \langle \mathbf{S}_{p[i]}, \mathbf{S}_{q[j]}\rangle_{\cos})^{\lambda/\tau}$
2. **Confidence as Marginals**: An MLP confidence head outputs $\mathbf{c}_p, \mathbf{c}_q \in [0,1]^n$, normalized to $\mathbf{w}_p = \mathbf{c}_p / \overline{\mathbf{c}_p}$, which serve as the target marginal distributions for the Sinkhorn algorithm.
3. **Soft Correspondence Matrix**: Row-normalizing the transport plan $\Pi$ yields $\mathbf{M}_{pq}$ and $\mathbf{M}_{qp}$, which act as soft projection operators generating corresponding points via convex combination.
4. **Pose Estimation**: Bidirectional correspondences and original point clouds are concatenated; confidence-weighted SVD (Umeyama algorithm) jointly solves for the rigid transformation.

**Semantic Denoising**: The STEGO self-label refinement strategy is applied to DINO features via energy-based clustering, reducing cross-view feature inconsistencies.

### Loss & Training

| Loss | Role | Core Formulation |
|------|------|-----------------|
| $\mathcal{L}_{\text{pose}}$ | Confidence-weighted Chamfer distance for pose alignment | Gaussian RBF kernel measuring geometric distance between transformed and target point clouds |
| $\mathcal{L}_{\text{cycl}}$ | Cycle-consistency constraint to reinforce overlap-region correspondences | Points should reconstruct their original positions after bidirectional projection |
| $\mathcal{L}_{\text{sem}}$ | Semantic consistency constraint to prevent semantically mismatched correspondences | Penalizes correspondences assigned to semantically dissimilar points |
| $\mathcal{L}_{\text{conf}}$ | Pseudo-label supervision for confidence learning | BCE loss; pseudo-label = product of geometry × pose × semantic RBF kernel responses (stop-gradient) |

Total loss: $\mathcal{L} = \gamma_{\text{pose}}\mathcal{L}_{\text{pose}} + \gamma_{\text{cycl}}\mathcal{L}_{\text{cycl}} + \gamma_{\text{sem}}\mathcal{L}_{\text{sem}} + \gamma_{\text{conf}}\mathcal{L}_{\text{conf}}$

**Unsupervised Confidence Learning**: Continuous pseudo-labels (non-binary) are generated from Gaussian RBF kernel responses measuring geometric reconstruction, pose alignment, and semantic consistency, and used to train the confidence branch via BCE loss. The supervised variant replaces the Chamfer distance in $\mathcal{L}_{\text{pose}}$ with per-point distances to GT-transformed points.

## Key Experimental Results

### Main Results: Single-reference Novel Object Pose Estimation (BOP Benchmark)

| Method | Supervision | LM-O | TUD-L | YCB-V | Mean |
|--------|-------------|------|-------|-------|------|
| FreeZe | None | 45.5 | 68.3 | 65.5 | 59.8 |
| Robust OT | None | 45.5 | 66.3 | 66.0 | 59.3 |
| Dustbin OT | None | 50.2 | 67.6 | 65.4 | 61.1 |
| **COG (Unsupervised)** | **None** | **56.7** | **73.8** | **75.9** | **68.8** |
| UnoPose | GT Pose | 58.7 | 71.0 | 83.1 | 70.9 |
| **COG (Supervised)** | **GT Pose** | **60.8** | **80.0** | **80.5** | **73.8** |

### Overlap Region Prediction (TUD-L IoU)

| Method | Dragon | Frog | Watering Can | Mean |
|--------|--------|------|--------------|------|
| UnoPose (Supervised) | 70.0 | 72.2 | 59.1 | 67.1 |
| COG (Unsupervised) | 71.2 | 64.4 | 81.2 | 72.3 |
| COG (Supervised) | 72.9 | 68.3 | 83.9 | 75.0 |

### Ablation Study

**Correspondence Mechanism Ablation** (YCB-V):

- Argmax + all losses → Mean 73.1; Softmax → 73.0; Uniform OT → 75.2; **Confidence OT → 75.9**
- OT-based methods outperform discrete matching by approximately 2–3%; confidence marginals provide further improvement.

**OT Parameter Ablation**:

- Injecting semantic priors improves mAP from 73.2 to 75.9 and reduces ENT from 23.0 to 10.5, yielding more compact correspondences.
- More than 2 Sinkhorn iterations slightly degrades performance (correspondences become too diffuse); 2 iterations are adopted.

### Key Findings

1. Unsupervised COG lags behind the supervised SOTA UnoPose by only 2.1% on average, and surpasses it by 2.8% on TUD-L.
2. Confidence-weighted OT marginals consistently outperform uniform marginals, validating the effectiveness of non-uniform marginals.
3. Semantically denoised DINO features significantly reduce ENT, focusing correspondences on semantically consistent regions.
4. A single refinement iteration improves performance by over 1%; diminishing returns thereafter, balancing accuracy and speed (~4 seconds per sample).

## Highlights & Insights

- **Elegant Mathematical Formulation**: Confidence scores are embedded directly into OT marginal constraints rather than used as post-processing, enabling end-to-end joint optimization of correspondences and confidence.
- **Remarkable Unsupervised Performance**: Using only geometric, semantic, and cycle-consistency cues to generate pseudo-labels, COG matches supervised methods, demonstrating strong generality.
- **Interpretable Confidence**: Visualizations confirm that the model accurately assigns low confidence to non-overlapping regions and outliers.
- **Well-rounded Design**: The coarse-to-fine architecture, semantic denoising, cycle-consistency, and pseudo-label confidence learning form a complementary and complete system.

## Limitations & Future Work

- In highly occluded or cluttered scenes (LM-O, YCB-V), the unsupervised variant still lags behind the supervised counterpart (4.1% gap on LM-O), indicating room for improvement in complex environments.
- The method depends on UnoSeg for initial mask prediction; segmentation failures propagate directly to pose estimation.
- Inference takes approximately 4 seconds per sample (including segmentation and DINO feature extraction), limiting real-time applicability.
- An inherent trade-off exists between marginal precision and correspondence sharpness in Sinkhorn iterations, requiring manual balancing.
- Evaluation is limited to rigid objects; extension to articulated or deformable objects remains unexplored.

## Related Work & Insights

- **Novel Object Pose Estimation**: UnoPose (SE(3)-invariant framework with discrete matching), SAM-6D (DINO + SAM segmentation), MegaPose (CAD retrieval + refinement).
- **OT for Point Cloud Registration**: RPM-Net (uniform marginal Sinkhorn), Robust OT, Dustbin OT (dustbin row/column).
- **Visual Foundation Models**: DINOv2 for semantic features; STEGO for semantic denoising.
- **Unsupervised Pose**: Equi-Pose (SE(3)-equivariant backbone), OP-Align (articulated objects), Zero-shot Pose (semantic feature alignment).

## Rating

- Novelty: ⭐⭐⭐⭐ (embedding confidence into OT marginals is a natural and original formulation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3 BOP benchmarks + detailed ablations + visualizations)
- Writing Quality: ⭐⭐⭐⭐ (clear mathematical derivations and expressive overall architecture diagrams)
- Value: ⭐⭐⭐⭐ (unsupervised performance matching supervised methods is a practically significant direction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [\[AAAI 2026\] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation](../../AAAI2026/human_understanding/coordar_one-reference_6d_pose_estimation_of_novel_objects_via_autoregressive_coo.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[AAAI 2026\] Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](../../AAAI2026/human_understanding/modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)

</div>

<!-- RELATED:END -->
