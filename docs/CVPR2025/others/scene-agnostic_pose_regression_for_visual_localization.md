---
title: >-
  [Paper Note] Scene-Agnostic Pose Regression for Visual Localization
description: >-
  [CVPR 2025][Visual Localization] Proposed a new task paradigm called "Scene-Agnostic Pose Regression" (SPR), which regresses the relative poses of subsequent frames using the first frame of the sequence as the coordinate origin. This avoids the need for retraining in APR, database retrieval in RPR, and cumulative drift in VO. A large-scale dataset, 360SPR, containing 200K panorama images, and a dual-branch SPR-Mamba model are established.
tags:
  - "CVPR 2025"
  - "Visual Localization"
  - "Pose Regression"
  - "Scene-Agnostic"
  - "Panorama"
  - "Mamba"
date: 2026-05-08
content_hash: 5b43707d4b320cfa
---

# Scene-Agnostic Pose Regression for Visual Localization

**Conference**: CVPR 2025  
**arXiv**: [2503.19543](https://arxiv.org/abs/2503.19543)  
**Code**: [https://github.com/junweizheng93/SPR](https://github.com/junweizheng93/SPR)  
**Area**: LLM Evaluation  
**Keywords**: Visual Localization, Pose Regression, Scene-Agnostic, Panorama, Mamba

## TL;DR

Proposed a new task paradigm called "Scene-Agnostic Pose Regression" (SPR), which regresses the relative poses of subsequent frames using the first frame of the sequence as the coordinate origin. This avoids the need for retraining in APR, database retrieval in RPR, and cumulative drift in VO. A large-scale dataset, 360SPR, containing 200K panorama images, and a dual-branch SPR-Mamba model are established.

## Background & Motivation

**Background**: Visual localization mainly consists of three paradigms: Absolute Pose Regression (APR), which directly predicts the 6DoF pose relative to the scene coordinate system from images; Relative Pose Regression (RPR), which predicts the relative pose between a query image and a reference image; and Visual Odometry (VO), which predicts the current pose based on the previous frame's pose and the current frame.

**Limitations of Prior Work**: APR learns scene-specific features and cannot generalize to unseen environments, requiring retraining for any new scene. RPR generalizes better but requires a large-scale database of reference images for retrieval during inference, and its performance degrades rapidly when overlap between reference and query images is insufficient. VO suffers from unavoidable cumulative drift in open-loop trajectories.

**Key Challenge**: A trilemma of "generalization ability vs. inference dependence vs. cumulative drift" exists among the three paradigms. No single method can simultaneously achieve no retraining, no database requirement, and zero cumulative drift.

**Goal**: (1) Define a new SPR task paradigm to solve all three problems simultaneously; (2) Establish a large-scale panorama dataset; (3) Design an effective SPR model.

**Key Insight**: Define the first frame of the sequence as the coordinate origin and regress the pose of the query frame relative to the first frame (rather than the previous frame). This decouples the coordinate system from the scene (it is not an absolute scene coordinate). Consequently, the model learns relative features between frames instead of scene-specific features. Furthermore, because each frame's pose is directly regressed relative to the first frame without relying on the prediction of the previous frame, cumulative drift is eliminated.

**Core Idea**: Use the first frame of the sequence as the origin, and leverage the information of all predecessor frames in the sequence to regress the pose of any query frame relative to the origin, while using panoramas to maximize visual information and frame-to-frame overlap.

## Method

### Overall Architecture

SPR-Mamba takes a panorama sequence captured along a trajectory, I_2, ..., I_q$ as input, and outputs the 6DoF camera pose $\mathbf{T}_q$ of the query frame $ relative to the first frame $. The model consists of a frozen DINO feature extractor and two complementary branches (local branch and global branch).

### Key Designs

1. **Local Branch**:

    - **Function**: Learning step-by-step relative poses between adjacent frames.
    - **Mechanism**: Compute the differences between DINO features of adjacent frames ($帧产生-1$ difference vectors) and process them through multi-layer linear layers. An auxiliary translation and rotation head is attached during training to output inter-frame relative poses as additional supervision. The auxiliary head can be removed during inference.
    - **Design Motivation**: The local branch provides fine-grained, short-range motion information, compensating for the high-frequency displacement details that the global branch might lose during direct long-distance regression.

2. **Global Branch**:

    - **Function**: Learning the global relative pose from any query frame to the first frame.
    - **Mechanism**: Multiple Mamba blocks are stacked to process DINO features of the entire sequence. Utilizing the SSM property of Mamba, the last hidden state of the final Mamba block aggregates all information from $ to $. This hidden state is fused with the output features of local branch, and the final pose is output through translation and rotation heads.
    - **Design Motivation**: Mamba can process variable-length sequences sequentially with linear inference complexity, making it suitable for continuously receiving new frames. The global branch directly regresses the pose relative to the origin, thereby avoiding the cumulative drift of VO.

3. **360SPR Large-Scale Dataset**:

    - **Function**: Providing large-scale training and evaluation data for the SPR task.
    - **Mechanism**: Utilize the Habitat simulator to collect over 200K panorama images and 3.6M pinhole images across 270 indoor scenes. Three robot heights (0.1m/0.5m/1.7m) are simulated, with trajectory lengths of 3-20m. At each sampling point, 18 pinhole images are stitched into a panorama. Cross-validation of quality is performed by three inspectors, with the cleaning process taking more than 300 hours.
    - **Design Motivation**: The existing panorama localization dataset, 360Loc, contains only 4 scenes and fewer than 10K panorama images, which is far from sufficient for robust localization requirements.

### Loss & Training

The training loss includes the translation and rotation regression losses of the global branch, as well as the inter-frame pose supervision on the auxiliary head of the local branch. DINO features are frozen after extraction, and only the Mamba blocks and regression heads are trained. SPR-Mamba supports inference on sequences of arbitrary lengths.

## Key Experimental Results

### Main Results

360SPR Dataset (Unseen Scenes):

| Paradigm | Method | Median TE (m) ↓ | Median RE (°) ↓ |
|------|------|-----------|-----------|
| APR | PoseNet | 30.25 | 47.15 |
| APR | Marepo | 27.98 | 48.12 |
| RPR | PanoPose | 10.91 | 20.01 |
| RPR | FAR | 11.85 | 21.04 |
| **SPR** | **SPR-Mamba** | **~3-4** | **~5-7** |

On unseen scenes, SPR-Mamba reduces translation error by 7m+ and rotation error by 16°+ compared to APR and RPR.

### Ablation Study

| Configuration | Description |
|------|------|
| Global branch only | Lacks local fine-grained information, leading to higher errors |
| Local branch only | Suffers from VO-like cumulative effects, leading to moderate errors |
| Dual-branch fusion | Complementary local + global, resulting in the lowest error |
| Trained on 360Loc → Tested on Generalization | +4.08m / +4.67°, insufficient data volume and diversity |

### Key Findings

- Panoramas significantly reduce localization errors compared to narrow-perspective images—as FoV increases, errors decrease across all three paradigms: APR, RPR, and SPR.
- The scene diversity in the 360SPR dataset is crucial.
- Training with multiple heights improves robustness—training at a fixed height leads to a significant performance drop at other heights.
- The SPR paradigm exhibits zero cumulative drift on open-loop trajectories.

## Highlights & Insights

- **The definition of the SPR task itself is the most significant contribution**: The concept of using the first frame as the origin is simple yet elegant, solving the core issues of the APR, RPR, and VO paradigms in one go.
- **Mamba for Pose Regression**: The sequential processing characteristics of SSMs allow the final hidden state to naturally aggregate global information, enabling linear complexity during inference as new frames are continuously received.
- **Quantitative validation of the necessity of panoramas**: A complete curve showing the decline in error as FoV increases from small to large is provided.

## Limitations & Future Work

- The dataset is simulator-based, which introduces a domain gap with the real world.
- SPR requires a complete trajectory sequence as input and is not applicable to single-frame localization.
- The selection of the first frame impacts the results but has not been thoroughly discussed.
- Performance in large-scale outdoor scenes remains unknown.
- Comparison with feature matching-based methods has not been conducted.

## Related Work & Insights

- **vs. PoseNet (APR)**: PoseNet learns scene-specific features and fails when shifting to another scene; SPR learns relative features between frames and is scene-agnostic.
- **vs. RelPose-GNN (RPR)**: RPR requires database retrieval to find reference images; SPR only requires the sequence itself.
- **vs. DeepVO (VO)**: VO suffers from cumulative drift; SPR directly regresses each frame to the origin.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The task definition is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compares multiple baselines with a large dataset scale.
- Writing Quality: ⭐⭐⭐⭐⭐ The comparison of paradigms is clear at a glance.
- Value: ⭐⭐⭐⭐ Both the dataset and paradigm significantly advance the visual localization community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Practical Solutions to the Relative Pose of Three Calibrated Cameras](practical_solutions_to_the_relative_pose_of_three_calibrated_cameras.md)
- [\[NeurIPS 2025\] MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation](../../NeurIPS2025/others/metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)
- [\[CVPR 2025\] VinaBench: Benchmark for Faithful and Consistent Visual Narratives](vinabench_benchmark_for_faithful_and_consistent_visual_narratives.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](../../ICML2025/others/sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)

</div>

<!-- RELATED:END -->
