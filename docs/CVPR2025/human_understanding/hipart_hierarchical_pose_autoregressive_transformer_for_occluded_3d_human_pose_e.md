---
title: >-
  [Paper Note] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][3D Pose Estimation] HiPART proposes an autoregressive generation scheme that generates hierarchical dense 2D poses (48→96 joints) from sparse 2D poses (17 joints), replacing complex temporal/visual encoders with rich skeletal context to address occlusion. It achieves SOTA performance on single-frame 3D HPE and surpasses most multi-frame methods, while requiring fewer parameters and less computation.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "3D Pose Estimation"
  - "Occlusion Handling"
  - "Hierarchical Densification"
  - "VQ-VAE"
  - "Autoregressive Generation"
  - "Center-to-Periphery"
  - "Sparse-to-Dense"
date: 2026-05-08
content_hash: e9cb0377f6f963eb
---

# HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.23331](https://arxiv.org/abs/2503.23331)  
**Code**: None  
**Area**: Human Understanding / 3D Pose Estimation  
**Keywords**: 3D Pose Estimation, Occlusion Handling, Hierarchical Densification, VQ-VAE, Autoregressive Generation, Center-to-Periphery, Sparse-to-Dense

## TL;DR

HiPART proposes an autoregressive generation scheme that generates hierarchical dense 2D poses (48→96 joints) from sparse 2D poses (17 joints), replacing complex temporal/visual encoders with rich skeletal context to address occlusion. It achieves SOTA performance on single-frame 3D HPE and surpasses most multi-frame methods, while requiring fewer parameters and less computation.

## Background & Motivation

**Background**: 3D Human Pose Estimation (HPE) is typically decoupled into a two-stage pipeline: 2D detection + 3D lifting. To combat occlusion, existing methods often introduce temporal context (VideoPose, MixSTE) or visual cues (Lifting by Image) during the lifting stage. However, these methods rely on complex temporal or image encoders, incurring high parameter and computational overhead.

**Limitations of Prior Work**: (1) Previous methods focus heavily on "stacking info" at the lifting stage, ignoring the fundamental limitation at the input level—the sparse 2D skeleton representation (only 17 joints) itself is a bottleneck; (2) Under occlusion, sparse joints lack sufficient local context to infer occluded parts; for instance, when the wrist is occluded, there is only the elbow joint as a reference; (3) Temporal-based methods require a large number of consecutive frames (243 frames) while vision-based methods require extra image encoders.

**Key Challenge**: The sparsity of input representation vs. the demand for rich local context in occluded scenes. When lifting hierarchical dense 2D poses coarsened from GT meshes (96→48 joints), MPJPE drops sharply from 37.6mm to 17.5mm (a 55% improvement), proving the value of dense inputs. However, 3D GT meshes are unavailable in real-world scenarios.

**Key Insight**: Leveraging a generative approach to "imagine" dense 2D poses from sparse 2D poses—designing an autoregressive strategy tailored specifically for skeletal topology (not standard raster scan) to provide rich skeletal context without requiring additional temporal or visual inputs.

**Core Idea**: Utilizing an autoregressive Transformer to generate hierarchical dense 2D poses (from 17 joints to 48+96 joints) to provide rich skeletal context for 3D lifting to combat occlusion.

## Method

### Overall Architecture

HiPART consists of two stages: **Stage 1 (MSST)**: Multi-Scale Skeleton Tokenization—using VQ-VAE-2 to progressively quantize the GT dense 2D pose (96 joints) into hierarchical discrete tokens (17 sparse tokens + 48 dense tokens), reinforced by Skeleton-aware Alignment to strengthen cross-scale token connections; **Stage 2 (HiARM)**: Hierarchical Autoregressive Modeling—starting from sparse 2D poses, a center-to-periphery and sparse-to-dense strategy is used to autoregressively generate all tokens. Finally, the generated hierarchical 2D poses are fed into a vanilla spatial transformer for 2D-to-3D lifting.

### Key Designs

1. **Multi-Scale Skeleton Tokenization (MSST)**:

    - **Function**: Compresses high-dimensional dense 2D poses into hierarchical discrete token representations
    - **Mechanism**: Similar to a VQ-VAE-2 architecture, employing an MLP-Mixer to implement encoders and decoders. Two encoders, $\mathcal{E}_f$ and $\mathcal{E}_d$, progressively encode the 96-joint fine pose into 48-joint dense embeddings and 17-joint sparse embeddings. These are quantized using a sparse codebook $C_s$ and a dense codebook $C_d$, respectively. The decoder reconstructs them in reverse.
    - **Design Motivation**: Discrete tokenization enables subsequent autoregressive modeling of pose distributions, and the multi-scale design preserves skeletal information from coarse to fine.

2. **Skeleton-aware Alignment**:

    - **Function**: Enhances semantic consistency across tokens of different scales
    - **Mechanism**: Employs two alignment strategies—(1) **Part-wise Local Alignment (LA)**: utilizes an InfoNCE contrastive loss to match a sparse token $\hat{z}_s^i$ with the average of its corresponding $r$ dense tokens as positive pairs, while treating other parts as negative pairs; (2) **Action-wise Global Alignment (GA)**: concatenates all tokens and projects them for action label classification, aligning them via cross-entropy loss.
    - **Design Motivation**: LA ensures semantic consistency of sparse and dense tokens belonging to the same local body part; GA ensures global coherent action semantics, providing a consistent token space for autoregressive generation.

3. **Hierarchical Autoregressive Modeling (HiARM)**:

    - **Function**: Autoregressively generates hierarchical dense tokens from sparse 2D poses
    - **Mechanism**: Replaces standard next-token sequence with two skeleton-specific strategies—**(1) Center-to-periphery**: progressively generates from the body center (root joint) toward the limbs, as joints farther from the root exhibit higher depth uncertainty; **(2) Sparse-to-dense**: predicts sparse tokens $q_s^i$ (global coarse-grained) followed by parallel prediction of the corresponding $r$ dense tokens $q_d^{(i,j)}$ (local fine-grained), compressing what would be $1+r$ steps into 2 steps for accelerated inference. The model comprises LSAB (Local Self-Attention Block, modeling multi-scale token interactions within the same body part) $\rightarrow$ GCSAB (Global Causal Self-Attention Block, modeling causal relationships across parts/joints) $\rightarrow$ PH (Prediction Head).
    - **Design Motivation**: Human skeletons hold non-Euclidean structures, making sequential raster scans unsuited. Center-to-periphery matches the spatial distribution of pose uncertainty; Sparse-to-dense leverages the assumption that "local details can be derived from global abstractions."

### Loss & Training

- **Stage 1**: $\mathcal{L}_1 = \|x_f - \hat{x}_f\|^2 + \|x_d - \hat{x}_d\|^2 + \text{VQ losses} + \lambda_l \mathcal{L}_{local} + \lambda_g \mathcal{L}_{global}$
- **Stage 2**: $\mathcal{L}_2 = \text{CE}(q_s, p_s) + \lambda_d \cdot \text{CE}(q_d, p_d)$
- The codebook is updated using EMA.
- Hierarchical pose GT is obtained through 3D meshes of Human3.6M (6890 vertices $\rightarrow$ 96 $\rightarrow$ 48 joints) + camera projection.
- The lifting stage uses a vanilla spatial transformer without requiring temporal or visual encoders.

## Key Experimental Results

### Main Results

Human3.6M (MPJPE↓, single-frame methods):

| Method | Type | Avg MPJPE↓ |
|------|------|-----------|
| SemGCN + visual | visual | 57.6 |
| Lifting by Image | visual | 51.0 |
| DiffPose | single | 49.7 |
| **HiPART (ours)** | **hierarchical** | **49.3** |

Comparison with multi-frame methods (HiPART is single-frame):

| Method | Frames | Avg MPJPE↓ | GFLOPs |
|------|------|-----------|--------|
| VideoPose | 243 | 47.1 | High |
| MixSTE | 243 | 40.9 | High |
| **HiPART** | **1** | **49.3** | **Low** |
| HiPART + MixSTE | 243 | 39.0 | - |

Occlusion scenario (3DPW-Occ):

| Method | Protocol 1↓ | Protocol 2↓ |
|------|------------|------------|
| DiffPose | 87.7 | 59.1 |
| **HiPART** | **82.4** | **56.8** |

### Ablation Study

| Configuration | MPJPE↓ |
|------|--------|
| Sparse only (17 joints) | 51.9 |
| + Dense (48 joints) | 50.1 |
| + Fine (96 joints) | **49.3** |
| Without LA | 50.2 |
| Without GA | 49.8 |
| Standard order (raster scan) | 50.5 |
| **Center-to-periphery** | **49.3** |

### Key Findings

- HiPART achieves SOTA performance among single-frame methods with an MPJPE of 49.3mm. While this is an improvement over the next-best DiffPose (49.7mm), its primary advantage lies in its lower complexity.
- A gap still exists compared to the 243-frame MixSTE (40.9mm). However, combining HiPART with MixSTE yields 39.0mm, proving that HiPART is complementary to temporal-based methods.
- The advantage is more pronounced in the occluded scenario 3DPW-Occ (82.4 vs 87.7), verifying the robustness of dense skeletons against occlusion.
- The progressive introduction of hierarchical dense 2D poses continuously improves results: 51.9 $\rightarrow$ 50.1 $\rightarrow$ 49.3, indicating that each additional density level contributes to performance.
- Center-to-periphery scanning reduces error by 1.2mm compared to raster scanning, and all ablation items for Skeleton-aware Alignment show positive contributions.

## Highlights & Insights

1. **Disruptive Problem Formulation**: While most methods stack information on the lifting end, HiPART directly targets the sparsity of the input representation—a fundamental but overlooked bottleneck. The toy experiment (where MPJPE drops from 37.6 to 17.5) directly quantifies the potential of dense inputs.
2. **Autoregressive Strategy for Non-Euclidean Skeletons**: Center-to-periphery + sparse-to-dense serves as an autoregressive scanning order designed specifically for human skeletal topology, which is more effective than generic raster scanning.
3. **Lightweight and Efficient**: Achieving single-frame SOTA with only a vanilla spatial transformer for lifting indicates that the richness of the input representation is more critical than model complexity.
4. **Orthogonal Complementarity**: The further improvement when combining HiPART with temporal methods (MixSTE) proves that the two strategies are mutually beneficial rather than conflicting.

## Limitations & Future Work

- Hierarchical GT poses require 3D meshes (coarsened from Human3.6M GT mesh), limiting direct application to datasets lacking mesh annotations.
- The current setup is fixed to three levels (sparse/dense/fine: 17/48/96 joints); the optimal level configuration is not fully explored.
- Inference requires multiple steps (VQ-VAE + Autoregressive + Decoding + Lifting), yielding higher complexity than direct lifting.
- Under extreme occlusion (where >70% of the body is occluded), the sparse input itself becomes highly unreliable, making the foundation for densification unstable.

## Related Work & Insights

- Comparison with PCT (VQ classification paradigm): PCT frame-shifts pose estimation into a classification task, whereas HiPART's autoregressive modeling of token distribution is more sensible.
- Comparison with Pose2Mesh, HGN: While these models treat hierarchical 3D pose prediction as an auxiliary task, HiPART independently optimizes 2D hierarchical pose generation.
- Inspiration for generative densification: Can similar ideas be applied to other sparse skeleton tasks, such as hand gesture estimation or animal pose estimation?

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐⭐: Inspiring problem definition (densifying inputs vs. enhancing models) and an elegantly customized autoregressive strategy for skeletons.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Evaluated extensively across benchmarks (Human3.6M, 3DPW, 3DPW-Occ) with convincing comparative and complementary experiments with multi-frame methods.
- **Writing Quality** ⭐⭐⭐⭐: Clear overall workflow, though Stage 1 and Stage 2 contain many intricate details that require careful reading.
- **Value** ⭐⭐⭐⭐: Highly practical, offering single-frame SOTA performance with low complexity and orthogonal compatibility with temporal methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](../../NeurIPS2025/human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [\[CVPR 2025\] Pose Priors from Language Models](pose_priors_from_language_models.md)
- [\[CVPR 2025\] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
