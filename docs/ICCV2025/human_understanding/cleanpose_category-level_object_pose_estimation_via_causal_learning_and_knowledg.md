---
title: >-
  [Paper Note] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation
description: >-
  [ICCV 2025][Human Understanding][Category-level pose estimation] This work is the first to introduce causal reasoning into category-level object pose estimation (COPE). It eliminates spurious correlations induced by data bias via a front-door adjustment-based causal reasoning module, and provides unbiased categorical semantic supervision through residual knowledge distillation from the 3D foundation model ULIP-2. The method achieves 61.7% on the strict 5°2cm metric on REAL275, surpassing the state of the art by 4.7%.
tags:
  - ICCV 2025
  - Human Understanding
  - Category-level pose estimation
  - causal reasoning
  - knowledge distillation
  - front-door adjustment
  - data bias
date: 2026-05-08
content_hash: 623117983bbe6ae9
---

# CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation

**Conference**: ICCV 2025
**arXiv**: [2502.01312](https://arxiv.org/abs/2502.01312)
**Code**: [https://github.com/chrislin0621/CleanPose](https://github.com/chrislin0621/CleanPose)
**Area**: Human Understanding
**Keywords**: Category-level pose estimation, causal reasoning, knowledge distillation, front-door adjustment, data bias

## TL;DR

This work is the first to introduce causal reasoning into category-level object pose estimation (COPE). It eliminates spurious correlations induced by data bias via a front-door adjustment-based causal reasoning module, and provides unbiased categorical semantic supervision through residual knowledge distillation from the 3D foundation model ULIP-2. The method achieves 61.7% on the strict 5°2cm metric on REAL275, surpassing the state of the art by 4.7%.

## Background & Motivation

Category-level object pose estimation (COPE) aims to predict the 9DoF pose (rotation, translation, and size) of arbitrary objects within predefined categories without requiring high-quality CAD models, in contrast to instance-level methods. Recent approaches have predominantly focused on model architecture design to capture robust category-level features, yet performance gains have begun to plateau.

The authors identify a fundamental yet overlooked issue: **inherent biases in training datasets**. These manifest as:
- Repeated training samples with limited pose diversity
- Similar scene environments that cause models to overfit to specific appearance and pose patterns
- Annotation biases and sensor noise from data collection

Such biases cause models to learn **spurious correlations** rather than genuine causal relationships. For instance, if objects of a certain category consistently appear at similar angles in training data, the model may rely on background similarity rather than object structure as a basis for pose estimation. Constructing a fully unbiased dataset is impractical given the high cost of 3D annotation, necessitating a model-side solution.

The authors draw inspiration from human perception: humans effectively handle intra-class variation by leveraging **analogical reasoning** to infer structural features and causal relationships, maintaining stable perception of novel objects across varying environments and viewpoints.

## Method

### Overall Architecture

CleanPose augments the AG-Pose baseline with two core modules:
1. **Causal Reasoning Module** (based on front-door adjustment): eliminates the influence of hidden confounding variables
2. **Residual Knowledge Distillation Module**: transfers unbiased 3D semantic knowledge from the ULIP-2 foundation model

Input RGB-D images are segmented to obtain cropped images and point clouds, which are processed by PointNet++ and DINOv2 respectively for feature extraction before being concatenated and fed into the causal reasoning module.

### Key Designs

#### 1. Structural Causal Model
- **Function**: Establishes a causal graph over key variables in the COPE task
- **Mechanism**: Defines four variables — visual input $\mathcal{X}$, output pose $\mathcal{Y}$, mediator variable $\mathcal{M}$ (structural information / keypoints), and hidden confounding variable $\mathcal{U}$ (data bias)
    - **Front-door path** $\mathcal{X} \rightarrow \mathcal{M} \rightarrow \mathcal{Y}$: humans first identify structural information (keypoints), then infer pose via analogical reasoning
    - **Confounding path** $\mathcal{X} \leftarrow \mathcal{U} \rightarrow \mathcal{Y}$: data bias simultaneously influences input sampling and pose distribution
- **Design Motivation**: Conventional methods model $P(\mathcal{Y}|\mathcal{X})$, which conflates the influence of confounders. Causal reasoning models $P(\mathcal{Y}|do(\mathcal{X}))$ via the do-operator, severing the confounding path.

#### 2. Front-Door Adjustment-Based Causal Reasoning
- **Function**: Eliminates the influence of confounding variables by intervening at both stages of the front-door path ($\mathcal{X} \rightarrow \mathcal{M}$ and $\mathcal{M} \rightarrow \mathcal{Y}$)
- **Mechanism**: The front-door adjustment formula is:

$$P(\mathcal{Y}|do(\mathcal{X})) = \sum_{x'} P(x') \sum_m P(\mathcal{Y}|m, x') P(m|\mathcal{X})$$

The expectation is approximated via a query mechanism:

$$\mathbb{E}_{x'}[\boldsymbol{x}'] \approx \sum_i \frac{\exp(\boldsymbol{g}_1 \boldsymbol{x}_i'^T)}{\sum_j \exp(\boldsymbol{g}_1 \boldsymbol{x}_j'^T)} \boldsymbol{x}_i'$$

where $\boldsymbol{g}_1 = q_1(x)$ is an embedding function. The final output is obtained via multi-head self-attention ($\mathcal{F}_s$) and cross-attention ($\mathcal{F}_c$):

$$\mathcal{F}_s = SA(\mathcal{F}_{kpt}), \quad \mathcal{F}_c = CA(\mathcal{F}_{kpt}, \mathcal{F}_{samp})$$
$$\mathcal{F}_f = LN(\mathcal{F}_s + \mathcal{F}_c)$$

- **Mediator Variable Implementation**: Keypoint features $\mathcal{F}_{kpt}$ from 96 keypoints uniformly distributed over the object surface
- **Cross-Sample Feature Sampling**: Adopts a MoCo-style dynamic queue ($N_c \times N_q$), initialized with features extracted by ULIP-2's 3D encoder and updated via a FIFO strategy. At each step, $N_s=12$ features are randomly sampled from the queue.

- **Adaptive Weight Fusion**:

$$w_a = \sigma(\mathcal{F}_f W_f + \mathcal{F}_{kpt} W_k)$$
$$\mathcal{F}_f \leftarrow w_a \odot \mathcal{F}_f + (1-w_a) \odot \mathcal{F}_{kpt}$$

- **Design Motivation**: Intervening on the input alone is insufficient to fully block the confounding effect due to the presence of mediator variables; intervention must be applied at both stages of the front-door path.

#### 3. Residual Knowledge Distillation
- **Function**: Transfers unbiased categorical semantic knowledge from the 3D foundation model ULIP-2
- **Mechanism**: ULIP-2 is pretrained on large-scale diverse data, implicitly learning robust debiased feature representations. The frozen ULIP-2 3D encoder (PointBERT) extracts teacher features $\mathcal{F}_P^{ULIP}$, and the student model's point cloud features are transformed via a residual network before alignment:

$$\hat{\mathcal{F}}_P^{avg} = \mathcal{F}_P^{avg} + \mu \times K_2(\delta(K_1(\mathcal{F}_P^{avg})))$$

$$\mathcal{L}_{KD} = \frac{1}{B} \sum_i^B \| \mathcal{F}_P^{ULIP} - \psi(\hat{\mathcal{F}}_P^{avg}) \|_2$$

- **Design Motivation**: $K_2$ is initialized to zero so that only the original features are retained in the early training phase, avoiding the introduction of additional confounders. Progressive updates balance feature learning and knowledge transfer.

### Loss & Training

Total loss: $\mathcal{L}_{all} = \alpha_1 \mathcal{L}_{pose} + \alpha_2 \mathcal{L}_{KD}$

- $\mathcal{L}_{pose} = \|\mathcal{R}_{gt} - \mathcal{R}\|_2 + \|t_{gt} - t\|_2 + \|s_{gt} - s\|_2$
- $\alpha_1 = 1, \alpha_2 = 0.01, \mu = 0.1$
- Adam optimizer, learning rate from 2e-5 to 5e-4 (triangular2 cyclical schedule), 120K iterations, batch size 24

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on REAL275:

| Method | Prior Required | IoU75* | 5°2cm | 5°5cm | 10°2cm | 10°5cm |
|--------|---------------|--------|-------|-------|--------|--------|
| AG-Pose (CVPR24) | ✗ | 61.3 | 57.0 | 64.6 | 75.1 | 84.7 |
| GCE-Pose (CVPR25) | ✓ | - | 57.0 | 65.1 | 75.6 | 86.3 |
| SecondPose (CVPR24) | ✗ | 49.7 | 56.2 | 63.6 | 74.7 | 86.0 |
| **CleanPose** | **✗** | **62.7** | **61.7** | **67.6** | **78.3** | **86.3** |

On HouseCat6D, CleanPose surpasses AG-Pose across all metrics (IoU50: +2.9%, 5°2cm: +1.1%, 10°5cm: +2.1%).

### Ablation Study

Effect of causal learning and knowledge distillation on REAL275:

| Causal Reasoning | KD | 5°2cm | 5°5cm | 10°2cm | 10°5cm |
|-----------------|-----|-------|-------|--------|--------|
| ✗ | ✗ | 57.0 | 64.6 | 75.1 | 84.7 |
| ✓ | ✗ | 59.7 | 66.5 | 77.5 | 86.0 |
| ✗ | ✓ | 57.9 | 65.5 | 76.1 | 85.2 |
| **✓** | **✓** | **61.7** | **67.6** | **78.3** | **86.3** |

Comparison of feature storage and update strategies:

| Storage | Update Strategy | 5°2cm | 5°5cm | Note |
|---------|----------------|-------|-------|------|
| Dynamic Queue | FIFO | **61.7** | **67.6** | Best: tracks dynamic feature evolution |
| Dynamic Queue | No Update | 57.1 | 66.1 | Features become stale |
| Dynamic Queue | Similarity Update | 61.0 | 67.1 | Approximates but underperforms FIFO |
| Memory Bank | No Update | 57.6 | 66.4 | Cannot capture training dynamics |

### Key Findings

1. The causal reasoning module alone contributes +2.7% (5°2cm), knowledge distillation alone contributes +0.9%, and their combination yields +4.7%, demonstrating strong complementarity.
2. Dynamic queue + FIFO strategy improves over static memory banks by 4.1% (5°2cm), confirming the importance of dynamic feature updates for causal reasoning.
3. Residual distillation outperforms direct concatenation (Concat) and contrastive learning, as the residual connection balances original features with newly acquired knowledge.
4. Even without shape priors, CleanPose surpasses GCE-Pose, which relies on such priors.

## Highlights & Insights

- **Precise Problem Diagnosis**: The first work to analyze COPE performance bottlenecks from the perspective of data bias, rather than continuing to refine model architectures.
- **Well-Grounded Causal Modeling**: Modeling data bias as a hidden confounding variable with a theoretically justified choice of front-door adjustment.
- **MoCo-Style Dynamic Queue**: Creatively repurposes contrastive learning techniques to address the challenge of cross-sample feature sampling in causal reasoning.
- **Theory-Practice Integration**: The approach proceeds from human perceptual mechanisms, through causal graph modeling, to a concrete attention-based implementation.

## Limitations & Future Work

- The mediator variable $\mathcal{M}$ in the causal graph is fixed to keypoint features; alternative structural representations remain unexplored.
- The dynamic queue size $N_q=80$ and sampling count $N_s=12$ are empirically determined, lacking adaptive adjustment mechanisms.
- Validation is limited to 6 categories on REAL275; scalability to settings with more categories remains to be verified.
- ULIP-2 is used as a fixed teacher model; ensembling multiple teachers or self-distillation has not been explored.

## Related Work & Insights

- **AG-Pose** (CVPR24) serves as the direct baseline, explicitly extracting local and global keypoint information.
- The dynamic queue concept from **MoCo** is innovatively transplanted into a causal reasoning context.
- Causal reasoning has been applied to object detection and image captioning, but 3D point cloud pose estimation represents an entirely new application domain.
- The proposed methodology is generalizable to other 3D vision tasks affected by data bias, such as 3D reconstruction and point cloud classification.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of front-door adjustment causal reasoning to COPE, opening a new research direction
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across three datasets with well-designed ablations, though the number of categories is limited
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations, though LaTeX rendering issues somewhat affect readability
- Value: ⭐⭐⭐⭐⭐ Achieves a breakthrough 61.7% on the 5°2cm metric on REAL275, surpassing the state of the art by 4.7%

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](../../AAAI2026/human_understanding/vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)

<!-- RELATED:END -->
