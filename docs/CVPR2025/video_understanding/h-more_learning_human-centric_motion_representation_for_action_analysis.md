---
title: >-
  [Paper Note] H-MoRe: Learning Human-centric Motion Representation for Action Analysis
description: >-
  [CVPR 2025][Video Understanding][Human-centric motion representation] This paper proposes H-MoRe (Human-centric Motion Representation), a joint self-supervised learning framework with skeleton constraints and boundary constraints. It learns precise, human-centric motion representations (world-local flows) from real-world scenes, significantly outperforming traditional optical flow methods across gait recognition (CL@R1 +16.01%), action recognition (Acc@1 +8.92%)…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Human-centric motion representation"
  - "optical flow"
  - "self-supervised learning"
  - "gait recognition"
  - "action recognition"
date: 2026-05-08
content_hash: d94f69c184f20cd7
---

# H-MoRe: Learning Human-centric Motion Representation for Action Analysis

**Conference**: CVPR 2025  
**arXiv**: [2504.10676](https://arxiv.org/abs/2504.10676)  
**Code**: [https://github.com/haku-huang/h-more](https://github.com/haku-huang/h-more)  
**Area**: Video Understanding  
**Keywords**: Human-centric motion representation, optical flow, self-supervised learning, gait recognition, action recognition

## TL;DR

This paper proposes H-MoRe (Human-centric Motion Representation), a joint self-supervised learning framework with skeleton constraints and boundary constraints. It learns precise, human-centric motion representations (world-local flows) from real-world scenes, significantly outperforming traditional optical flow methods across gait recognition (CL@R1 +16.01%), action recognition (Acc@1 +8.92%), and video generation (FVD -67.07%).

## Background & Motivation

Understanding human motion is a fundamental challenge in computer vision. Current motion representations mainly fall into two categories:

1. **Optical Flow**: Encodes motion and shape information in a matrix format, which is easily processed by CNNs/ViTs. However, it calculates offsets indiscriminately for all pixels—in scenes with dynamic backgrounds, human motion is overwhelmed by noise. Furthermore, optical flow methods are typically trained on synthetic data, lacking real biological entities.
2. **Human Pose**: Represents motion using 2D/3D skeleton joints. It is highly precise for describing human movement but discards body shape and contour details, which are critical for shape-dependent tasks such as gait recognition.

The motivation of H-MoRe is: Is it possible to design a motion representation that preserves shape information (like optical flow in a matrix format, easily integrated into CNNs/ViTs) while focusing on human motion (filtering out background noise like Pose)? Furthermore, inspired by kinematics, this work introduces "motion relative to the subject itself" (local flow) to provide richer motion semantics.

## Method

### Overall Architecture

H-MoRe consists of two core components:
1. **World flow $M_w$**: Computes human motion relative to the environment between adjacent frames using an optical flow estimation network $\Phi$ (based on RAFT-small), optimized in a self-supervised manner via a joint constraint learning framework.
2. **Local flow $M_l$**: Estimates the overall motion trend of the subject $v_s$ using a lightweight network $\Psi$, and then derives the motion relative to the subject itself as $M_l = M_w - v_s$.

### Key Designs

1. **Skeleton Constraint $\mathcal{F}$**:
    - **Function**: Constrains motion direction and intensity using pose information, ensuring the motion of each body point complies with kinematic principles.
    - **Mechanism**: Extracts 17 joint points using a 2D pose estimator to construct skeleton displacement $\vec{K} = K_{t+1} - K_t$. For each body point $p$ on the flow map $M$, it matches the nearest joint point $\hat{q}$ and applies two sub-constraints: ① Angle constraint $\mathcal{F}_A$: verifies if the angle between the estimated motion $u_p$ and the skeleton displacement $k_{\hat{q}}$ exceeds a threshold $\vartheta_a$; ② Intensity constraint $\mathcal{F}_I$: verifies if the motion magnitude falls within $[\vartheta_i^l, \vartheta_i^h]$ times the skeleton displacement. The complete constraint is formulated as $\mathcal{F} = \frac{1}{hw} \sum_{p} [\mathcal{F}_A + \beta \cdot \mathcal{F}_I]$.
    - **Design Motivation**: Skeleton displacement provides a "global-range" prior of human motion—the direction and intensity of a body point's movement should not deviate excessively from its nearest joint. This constraint enables the model to learn correct motion directions and intensities even without optical flow ground truth.

2. **Boundary Constraint $\mathcal{G}$**:
    - **Function**: Refines motion details using human boundary priors, ensuring that the learned flow maintains a clear body contour.
    - **Mechanism**: Computes the Chamfer distance between the flow edge $s$ and the human boundary $e$ (obtained via U2Net semantic segmentation + Canny edge detection). To ensure efficient computation, a **patch-centroid distance** approximation is proposed: the edge curves are split into multi-scale patches, approximating the Chamfer distance with the distance between patch centroids: $\mathcal{C}(\mathcal{P}_s, \mathcal{P}_e) \approx \mathcal{D}(c_{\mathcal{P}_s}, c_{\mathcal{P}_e})$. The final loss is formulated as $\mathcal{G} = \frac{1}{n_{ms}} \sum_{ms} \frac{1}{n_\mathcal{P}} \sum_{\mathcal{P}} \mathcal{C}(\mathcal{P}_s, \mathcal{P}_e)$.
    - **Design Motivation**: While the skeleton constraint defines the "global scope" of motion, it lacks fine local details (e.g., fingers, feet). The boundary constraint supplements shape information by aligning flow edges with the human silhouette, resulting in a motion representation featuring both precise movement and clean shape boundaries.

3. **World-Local Flow Estimation**:
    - **Function**: Provides two complementary perspectives of motion—absolute motion (world) and relative motion (local).
    - **Mechanism**: Inspired by the Galilean transformation, world flow $M_w$ is the motion of body points relative to the environment (blue vector), while local flow $M_l$ is the motion relative to the subject itself (red vector). They are converted via the subject's overall motion trend $v_s$ (brown vector) as: $M_l = M_w - v_s$. $v_s$ is estimated by a lightweight network $\Psi$ (with 4 layers of cross-attention) from $M_w$ and input frames.
    - **Design Motivation**: Certain tasks (e.g., gait recognition) focus more on the motion of body parts relative to the body itself (e.g., arm swinging relative to the torso). Local flow provides this "self-referential" motion information. By obtaining local flow via vector decomposition instead of an additional heavy estimation network, inference efficiency is maintained (34 fps).

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{F}(M, X_t, X_{t+1}) + \alpha \cdot \mathcal{G}(M, X_t)$, where $\alpha=0.1$, $\beta=0.01$. The thresholds are set to $\vartheta_a=15°$, $\vartheta_i^l=0.8$, and $\vartheta_i^h=1.2$. The model is trained for 8 epochs with a batch size of 64 using AdamW with an exponentially decaying learning rate initialized at $1\times10^{-4}$ across 16 RTX 6000 Ada GPUs. Network $\Phi$ is based on RAFT-small with 2 self-attention blocks, and $\Psi$ consists of 4 cross-attention layers.

## Key Experimental Results

### Main Results

**Gait Recognition (CASIA-B, GaitBase)**:

| Motion Representation Method | Params (M) | FLOPs (G) | NM@R1 | BG@R1 | CL@R1 |
|-------------|---------|----------|-------|-------|-------|
| w/o Flow | - | - | 96.51 | 91.50 | 78.02 |
| RAFT | 5.25 | 1780.4 | 96.91 | 93.12 | 80.52 |
| FlowFormer++ | 16.15 | 3048.1 | 96.66 | 94.31 | 85.70 |
| **H-MoRe** | **5.57** | **861.5** | **98.26** | **95.62** | **87.66** |

**Action Recognition (Diving48) + Video Generation (MHAD)**:

| Method | Acc@1↑ | Acc@5↑ | SSIM↑ | FVD↓ |
|------|--------|--------|-------|------|
| w/o Flow | 64.07 | 95.08 | 0.9463 | 329.22 |
| VideoFlow | 71.45 | 96.72 | 0.9564 | 165.63 |
| **H-MoRe** | **72.99** | **97.62** | **0.9574** | **108.38** |

### Ablation Study

**Joint Constraints Ablation**:

| Skeleton Constraint $\mathcal{F}$ | Boundary Constraint $\mathcal{G}$ | CL@R1 | Acc@1 | Note |
|:-:|:-:|-------|-------|------|
| ✓ | | 83.01 | 72.13 | Correct motion direction but lacks shape details |
| | ✓ | 84.93 | 68.17 | Precise edges but high motion deviation |
| ✓ | ✓ | **85.25** | **72.99** | Complementary combination achieves optimal performance |

**World-Local Flow Ablation**:

| World $M_w$ | Local $M_l$ | CL@R1 | Acc@1 | Note |
|:-:|:-:|-------|-------|------|
| ✓ | | 80.78 | 70.91 | Absolute motion only |
| | ✓ | 82.82 | 72.64 | Relative motion is more effective for most tasks |
| ✓ | ✓ | **85.25** | **72.99** | Complementary combination achieves optimal performance |

### Key Findings

- Under the most challenging clothing change (CL) conditions, H-MoRe improves performance by 9.64% over the baseline without flow, and by 1.96% compared to the state-of-the-art optical flow method.
- Using local flow alone generally outperforms world flow, validating the critical importance of "relative motion."
- H-MoRe demonstrates superior robustness in subject-overlapping scenarios: at 40% overlap, its accuracy drop is considerably smaller than that of RAFT.
- Compared to Pose-based methods, H-MoRe outperforms 3D Pose (41M parameters) with only 5.7M parameters, highlighting the value of shape information.

## Highlights & Insights

- **Paradigm Innovation**: Instead of simply "estimating more accurate optical flow," this work defines a completely new "human-centric motion representation"—jointly encoding motion and shape while learning from real-world data in a self-supervised manner.
- **Physics-Inspired**: The world-local flows directly stem from the Galilean transformation in kinematics, elegantly avoiding dual-network estimation via vector decomposition.
- **Plug-and-Play**: H-MoRe outputs a matrix-formatted representation, allowing it to directly replace optical flow as an input channel for any CNN/ViT.
- **Patch-centroid distance**: The technique of approximating Chamfer distance with centroid distance is highly valuable for other applications.

## Limitations & Future Work

- Currently validated only in 2D, without extension to 3D environments.
- Constrained by computational resources, only a limited number of subjects are supported per scene.
- The skeleton constraint relies heavily on the accuracy of the 2D Pose estimator (e.g., ED-Pose); errors in pose estimation will affect the quality of H-MoRe.
- Training requires a semantic segmentation network (U2Net) to provide boundary priors, increasing overall system complexity.

## Related Work & Insights

- **Relation to Optical Flow**: H-MoRe can be conceptualized as "human-targeted optical flow," specializing generic optical flow via skeleton and boundary constraints.
- **Relation to Pose**: H-MoRe maintains the motion precision of Pose-based approaches while preserving shape details in a matrix format that Pose representations typically lack.
- **Insights**: Designing "domain-specific motion representations" for targeted categories (e.g., humans, vehicles) may be a more promising direction than pursuing generic optical flow.

## Rating

- Novelty: ⭐⭐⭐⭐ Incorporating kinematic concepts into motion representation design with world-local flows is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies spanning three core tasks: gait recognition, action recognition, and video generation.
- Writing Quality: ⭐⭐⭐⭐ Rich and clear illustrations, with rigorous mathematical formulations for the constraint designs.
- Value: ⭐⭐⭐⭐ Introduces a novel "plug-and-play" motion representation paradigm with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[CVPR 2025\] HuMoCon: Concept Discovery for Human Motion Understanding](humocon_concept_discovery_for_human_motion_understanding.md)
- [\[CVPR 2025\] Temporally Consistent Object-Centric Learning by Contrasting Slots](temporally_consistent_object-centric_learning_by_contrasting_slots.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)

</div>

<!-- RELATED:END -->
