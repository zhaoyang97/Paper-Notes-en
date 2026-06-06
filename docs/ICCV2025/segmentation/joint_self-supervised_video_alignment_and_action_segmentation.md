---
title: >-
  [Paper Note] Joint Self-Supervised Video Alignment and Action Segmentation
description: >-
  [ICCV 2025][Segmentation][Video Alignment] This paper proposes the VAOT/VASOT framework, which integrates Gromov-Wasserstein optimal transport with structural priors to unify self-supervised video alignment and action se…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Video Alignment"
  - "Action Segmentation"
  - "Optimal Transport"
  - "Self-Supervised Learning"
  - "Gromov-Wasserstein"
date: 2026-05-08
content_hash: 223c95086c465609
---

# Joint Self-Supervised Video Alignment and Action Segmentation

**Conference**: ICCV 2025
**arXiv**: [2503.16832](https://arxiv.org/abs/2503.16832)  
**Code**: [https://retrocausal.ai/research/](https://retrocausal.ai/research/)  
**Area**: Image Segmentation
**Keywords**: Video Alignment, Action Segmentation, Optimal Transport, Self-Supervised Learning, Gromov-Wasserstein

## TL;DR

This paper proposes the VAOT/VASOT framework, which integrates Gromov-Wasserstein optimal transport with structural priors to unify self-supervised video alignment and action segmentation within a single model for the first time. The framework surpasses existing methods on video alignment and achieves state-of-the-art performance on action segmentation.

## Background & Motivation

**Video alignment** (frame-to-frame matching) and **action segmentation** (frame-to-action-label assignment) both require fine-grained temporal understanding of video, yet these two tasks have never been studied jointly.

Existing limitations:

**Video alignment**: VAVA employs standard Kantorovich optimal transport with an optimality prior, but struggles to balance multiple losses and cannot handle repeated actions.

**Action segmentation**: Methods such as TOT and UFSA exhibit degraded performance under order variations, imbalanced segmentation, and repeated action scenarios.

**ASOT** addresses action segmentation via fused GW optimal transport but does not consider video alignment.

Core observation: Both tasks require fine-grained temporal understanding, and multi-task learning can share representations and promote mutual improvement. In particular, video alignment can substantially benefit action segmentation.

## Method

### Overall Architecture

Two methods are proposed:
- **VAOT** (single-task): Self-supervised video alignment based on fused GW optimal transport.
- **VASOT** (multi-task): A unified optimal transport framework for joint video alignment and action segmentation.

### Key Designs

1. **Video Alignment Optimal Transport (VAOT)**:

    - Based on Fused Gromov-Wasserstein (FGW) optimal transport:
    $\mathcal{F}_{FGW} = (1-\alpha)\mathcal{F}_{KOT}(\mathbf{C}, \mathbf{T}) + \alpha \mathcal{F}_{GW}(\mathbf{C}^x, \mathbf{C}^y, \mathbf{T})$
    - **Visual cues** (KOT): The cost matrix $\mathbf{C}_{ij} = 1 - \frac{\mathbf{x}_i^\top \mathbf{y}_j}{\|\mathbf{x}_i\| \|\mathbf{y}_j\|}$ measures inter-frame visual similarity.
    - **Structural prior** (GW): Temporal consistency constraints are defined via $\mathbf{C}^x$ and $\mathbf{C}^y$, penalizing pairings that map temporally adjacent frames to temporally distant ones.
    - The structural prior is carefully designed: mapping temporally proximate frames within radius $r$ to distant frames incurs a cost of $1/r$.
    - Naturally handles order variations and repeated actions.

2. **Efficient Numerical Solver**:

    - Entropic regularization $-\epsilon H(\mathbf{T})$ is added and the problem is solved via projected mirror descent.
    - The sparse structure of $\mathbf{C}^x$ and $\mathbf{C}^y$ is exploited, yielding $O(NM)$ complexity per iteration.
    - Convergence is typically achieved within 25 iterations, enabling efficient GPU-based training.

3. **Background / Redundant Frame Handling**:

    - A dummy frame is appended to each of $X$ and $Y$.
    - If a frame's matching probability to all counterpart frames falls below threshold $\zeta$, it is matched to the dummy frame.
    - Dummy frames and their associated frames are excluded from loss computation.

4. **VASOT — Joint Multi-Task Framework**:

    - Integrates VAOT (video alignment) and ASOT (action segmentation) into a unified framework.
    - Video alignment performs frame-to-frame matching $(X \leftrightarrow Y)$; action segmentation performs frame-to-action matching $(X \leftrightarrow A,\ Y \leftrightarrow A)$.
    - Both tasks share the frame encoder parameters $\theta$ and action embeddings $\mathbf{A}$.

### Loss & Training

**VAOT loss**: Cross-entropy loss aligning normalized similarity $\mathbf{P}$ with pseudo-labels $\mathbf{T}^*$:

$$\mathcal{L} = -\sum_{i=1}^{N}\sum_{j=1}^{M} \mathbf{T}_{ij}^* \log \mathbf{P}_{ij}$$

**VASOT joint loss**:

$$\mathcal{L}_{joint} = w_{align}\mathcal{L}_{xy} + w_{seg}(\mathcal{L}_{xa} + \mathcal{L}_{ya})$$

- Setting $w_{align} = w_{seg} = 1$ yields good results for both tasks.
- Gradients are not back-propagated through $\mathbf{T}^*$.
- Pseudo-labels are computed using the augmented cost matrix $\tilde{\mathbf{C}} = \mathbf{C} + \rho\mathbf{R}$, where $\mathbf{R}$ is the temporal prior.
- Action embeddings $\mathbf{A}$ are initialized via K-Means.
- Video alignment uses a ResNet-50 encoder; action segmentation uses a 2-layer MLP encoder.

## Key Experimental Results

### Main Results

**Video alignment results** (IKEA ASM):

| Method | Acc@0.1 | Acc@0.5 | Acc@1.0 | AP@5 | AP@10 | AP@15 |
|--------|---------|---------|---------|------|-------|-------|
| TCC | 22.70 | 25.04 | 25.63 | 18.03 | 17.53 | 17.20 |
| VAVA | 29.12 | 29.95 | 29.10 | 26.42 | 25.73 | 25.80 |
| **VAOT** | **33.73** | **36.42** | **38.64** | **31.49** | **31.92** | **32.01** |

**Action segmentation results**:

| Method | Breakfast MoF/F1/mIoU | 50 Salads (Eval) MoF/F1/mIoU | Desktop MoF/F1/mIoU |
|--------|-----------------------|------------------------------|---------------------|
| ASOT | 56.1/38.3/18.6 | 59.3/53.6/30.1 | 70.4/68.0/45.9 |
| **VASOT** | **57.5/39.0/18.8** | **60.6/57.4/34.5** | **70.9/75.1/49.3** |

### Ablation Study

**Design choice ablation (IKEA ASM)**:

| Variant | Acc@0.1 | Acc@0.5 | Acc@1.0 | AP@5 | AP@10 | AP@15 |
|---------|---------|---------|---------|------|-------|-------|
| w/o structural prior | 30.29 | 35.52 | 37.81 | 27.54 | 27.33 | 27.15 |
| w/o temporal prior | 17.84 | 17.84 | 17.84 | 15.63 | 15.64 | 15.56 |
| w/o balance constraint | 17.84 | 20.71 | 25.24 | 15.49 | 15.69 | 15.78 |
| w/o dummy frames | 30.16 | 34.49 | 36.10 | 29.57 | 29.24 | 28.87 |
| **All** | **33.73** | **36.42** | **38.64** | **31.49** | **31.92** | **32.01** |

### Key Findings

- **Asymmetry in multi-task benefit**: Action segmentation provides minimal benefit to video alignment, whereas video alignment substantially improves action segmentation. This may be attributed to the finer granularity of the frame-to-frame alignment task, which yields better representations for segmentation.
- The temporal prior $\mathbf{R}$ is the most critical component; its removal causes a dramatic performance drop.
- Balanced constraints outperform unbalanced ones, as the number of video frames greatly exceeds the number of action categories, making the alignment problem inherently more balanced.
- VAOT is robust to hyperparameters $r$ and $\alpha$, with Acc@1.0 and Progress remaining stable over a wide range.
- The largest gains over VAVA are observed on the in-the-wild dataset (IKEA ASM).

## Highlights & Insights

- This work is the first to unify video alignment and action segmentation within a single optimal transport framework, offering both theoretical elegance and practical utility.
- The FGW structural prior is cleverly designed to handle order variations, background frames, and repeated actions within a single formulation.
- The discovery of asymmetric mutual benefit in multi-task learning is insightful: a fine-grained task (alignment) can improve a coarser-grained task (segmentation), but not vice versa.
- The simple dummy-frame design effectively handles background and redundant frames present in real-world videos.

## Limitations & Future Work

- The number of action categories $K$ must be pre-specified as the ground-truth value, limiting fully unsupervised applicability.
- The multi-task weights $w_{align}$ and $w_{seg}$ are set to equal values; more sophisticated multi-task weight learning strategies could yield further improvements.
- The ResNet-50 encoder may constrain video representation capacity; video foundation models are worth exploring.
- The framework addresses alignment and segmentation only within a single activity; cross-activity scenarios remain unexamined.

## Related Work & Insights

- The approach extends the optimal transport formulation of ASOT from action segmentation to video alignment, demonstrating an elegant transfer of methodology.
- Comparison with VAVA confirms that FGW with structural priors outperforms Kantorovich optimal transport with optimality priors.
- Promising future directions include deep supervision, advanced multi-task weight learning, and joint keypoint matching with clustering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First unification of video alignment and action segmentation; the adaptation of FGW to video alignment is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 7 datasets with comprehensive ablation studies and hyperparameter sensitivity analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and the overall structure is clear.
- **Value**: ⭐⭐⭐⭐ Solid theoretical contributions; the asymmetric multi-task benefit finding offers meaningful insights to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](../../NeurIPS2025/segmentation/exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [\[ICCV 2025\] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation](clot_closed_loop_optimal_transport_for_unsupervised_action_segmentation.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ICCV 2025\] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss](prompt_guidance_and_human_proximal_perception_for_hot_prediction_with_regional_j.md)
- [\[NeurIPS 2025\] Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](../../NeurIPS2025/segmentation/self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)

</div>

<!-- RELATED:END -->
