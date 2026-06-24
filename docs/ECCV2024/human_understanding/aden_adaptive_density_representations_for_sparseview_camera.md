---
title: >-
  [Paper Note] ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation
description: >-
  [ECCV 2024][Human Understanding][Sparse-view] The ADen framework is proposed to unify pose regression and probabilistic estimation paradigms by employing a generator to yield multiple pose hypotheses and a discriminator to score and select the best hypothesis. With only 500 adaptive samples, this approach outperforms methods requiring 500K uniform samples while achieving real-time inference.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Sparse-view"
  - "Camera Pose"
  - "Generator-Discriminator"
  - "Adaptive Sampling"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: d50fc1ca182d1757
---

# ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation

**Conference**: ECCV 2024  
**arXiv**: [2408.09042](https://arxiv.org/abs/2408.09042)  
**Area**: 3D Vision / Camera Pose Estimation  
**Keywords**: Sparse-view, Camera Pose, Generator-Discriminator, Adaptive Sampling, Contrastive Learning  

## TL;DR

The ADen framework is proposed to unify pose regression and probabilistic estimation paradigms by employing a generator to yield multiple pose hypotheses and a discriminator to score and select the best hypothesis. With only 500 adaptive samples, this approach outperforms methods requiring 500K uniform samples while achieving real-time inference.

## Background & Motivation

- **Background**: Two mainstream approaches exist for sparse-view camera pose estimation: regression methods (unimodal prediction) and probabilistic methods (uniform sampling of the SO(3) space).
- **Limitations of Prior Work**: Regression methods assume a unimodal distribution, leading to poor performance on symmetric objects. Probabilistic methods, such as RelPose, require 500K uniform samples to achieve sufficient accuracy, which incurs a prohibitive computational cost, while the curse of dimensionality limits joint rotation and translation modeling.
- **Key Challenge**: The contradiction between the need for dense sampling to achieve accuracy and the runtime constraints dictated by sampling efficiency. Uniform grids are computationally infeasible in high-dimensional spaces.
- **Goal**: To achieve high-precision, multi-modal-aware pose estimation using an extremely small number of samples.
- **Key Insight**: In real-world scenarios, pose distributions are highly skewed with only a few dominant modes. Thus, adaptive sampling is far superior to uniform sampling.
- **Core Idea**: The generator learns to sample a small number of high-quality hypotheses from the conditional distribution, while the discriminator employs contrastive learning to select the best hypothesis.

## Method

### Overall Architecture

ResNet extracts image-wise features → Transformer fuses multi-view features → A shared backbone branches into two heads: the Pose Generator (for multi-hypothesis generation) and the Pose Discriminator (for contrastive ranking).

### Key Designs

**Design 1: Multi-Hypothesis Pose Generator**
- **Function**: Generates $M$ pose hypotheses, each represented as a 7D vector (quaternion + translation).
- **Mechanism**: $M$ learnable query embeddings are mapped through an MLP and combined with the fused features to generate $M$ hypotheses. Regression loss is applied only to the hypothesis closest to the ground truth (GT) (i.e., with the minimum geodesic distance), while the remaining hypotheses receive no loss.
- **Design Motivation**: To prevent mode collapse by avoiding regressing all hypotheses to the same GT, thereby allowing the model to freely explore multiple modes.

**Design 2: Contrastive Discriminator**
- **Function**: Evaluates the probability of correctness for each generated hypothesis.
- **Mechanism**: During training, the GT pose is included as a positive sample, and the discriminator is trained using a contrastive negative log-likelihood loss to distinguish the GT from the generated hypotheses. During inference, the GT is not used, and the hypothesis with the highest probability is selected.
- **Design Motivation**: Formulates the pose selection as a contrastive learning task, thereby dodging the curse of dimensionality associated with uniform sampling.

**Design 3: Joint Training Stabilization Strategy**
- **Function**: Injects Gaussian noise into the query embeddings.
- **Mechanism**: Applies stabilization training techniques akin to GAN training to prevent the discriminator gradient from vanishing due to an overly proficient generator.
- **Design Motivation**: Addresses the classic training instability inherent in generator-discriminator frameworks.

### Loss & Training

$\mathcal{L} = \mathcal{L}_g + \mathcal{L}_d$. Generator loss: geodesic rotation distance of the closest hypothesis + L2 translation distance. Discriminator loss: contrastive negative log-likelihood. Trained for 2000 epochs, using Adam optimizer with lr=1e-4.

## Key Experimental Results

### Main Results

**CO3D Dataset Rotation Accuracy (Acc@15°)**

| Method | 2-view(seen) | 5-view(seen) | 8-view(seen) |
|------|-------------|-------------|-------------|
| RelPose++ | 81.8 | 84.7 | 85.5 |
| PoseDiff | 76.0 | 77.7 | 78.5 |
| **ADen** | **84.3** | **86.5** | **87.3** |

**Tight Threshold Accuracy (Acc@5°)**

| Method | Seen | Unseen |
|------|------|--------|
| RelPose++ | 39.5 | 27.8 |
| **ADen** | **51.2** | **36.5** |

### Ablation Study

| Configuration | Acc@15° |
|------|---------|
| Regression Only (w/o multi-hypotheses) | 82.1 |
| Generator Only (w/o discriminator) | 83.5 |
| Full ADen | **84.3** |
| Without Query Noise | 83.0 |

### Key Findings

1. The advantages of ADen are even more pronounced under tight thresholds (5°/10°), as it is not constrained by grid resolution.
2. ADen also achieves state-of-the-art (SOTA) results in zero-shot transfer on Objectron and Niantic, demonstrating strong generalization capability.
3. Using only 500 samples outperforms 500K uniform samples, validating the high efficiency of adaptive sampling.

## Highlights & Insights

1. Elegantly unifies the regression and probabilistic paradigms, combining the strengths of both.
2. Naturally extends to high-dimensional spaces (joint R+t) without increasing the sample counts.
3. Achieves real-time inference speed, multiple times faster than RelPose++.

## Limitations & Future Work

1. The diversity of the generator depends on the initialization of learning queries, which might be insufficient.
2. The discriminator may still encounter difficulties with highly symmetric objects.
3. The possibility of using a diffusion model as the generator has not been explored.

## Related Work & Insights

- RelPose/RelPose++ pioneered the probabilistic pose estimation paradigm, while ADen revolutionizes the sampling scheme through adaptive sampling.
- Key insight: Real-world pose distributions are sparse, discarding the necessity to uniformly sample the entire space.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ★★★★☆ |
| Practicality | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★★ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)
- [\[ICLR 2026\] From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper](../../ICLR2026/human_understanding/from_sparse_to_dense_spatio-temporal_fusion_for_multi-view_3d_human_pose_estimat.md)
- [\[ECCV 2024\] AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)
- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)

</div>

<!-- RELATED:END -->
