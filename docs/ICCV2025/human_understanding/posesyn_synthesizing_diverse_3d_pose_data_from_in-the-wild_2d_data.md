---
title: >-
  [Paper Note] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data
description: >-
  [ICCV 2025][Human Understanding][3D human pose estimation] This paper proposes the PoseSyn framework, which identifies hard samples for a target pose estimator (TPE) from in-the-wild 2D pose data via an Error Extraction Module (EEM), then expands inaccurate pseudo-labels into diverse motion sequences via a Motion Synthesis Module (MSM). A human animation model subsequently renders these sequences into realistic training images with accurate 3D annotations…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "3D human pose estimation"
  - "data synthesis"
  - "motion generation"
  - "hard sample mining"
  - "data augmentation"
date: 2026-05-08
content_hash: e3ea84f769998ebd
---

# PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data

**Conference**: ICCV 2025
**arXiv**: [2503.13025](https://arxiv.org/abs/2503.13025)  
**Code**: None  
**Area**: Human Understanding
**Keywords**: 3D human pose estimation, data synthesis, motion generation, hard sample mining, data augmentation

## TL;DR

This paper proposes the PoseSyn framework, which identifies hard samples for a target pose estimator (TPE) from in-the-wild 2D pose data via an Error Extraction Module (EEM), then expands inaccurate pseudo-labels into diverse motion sequences via a Motion Synthesis Module (MSM). A human animation model subsequently renders these sequences into realistic training images with accurate 3D annotations, improving 3D pose estimation accuracy by up to 14% across multiple real-world benchmarks.

## Background & Motivation

3D human pose estimation has broad applications in action recognition, virtual reality, and motion analysis. However, obtaining accurate 3D pose annotations requires multi-camera systems or motion capture equipment, which is costly and restricted to controlled indoor environments. Existing 3D pose datasets (e.g., Human3.6M, MuCo, MPI-INF-3DHP) suffer from the following core issues:

**Domain gap**: The visual discrepancy between indoor and real-world scenes (background, lighting, subject appearance) causes poor generalization to in-the-wild scenarios.

**Insufficient coverage of hard poses**: Controlled environments struggle to encompass the complex, dynamic pose configurations found in the real world.

**Limitations of existing data augmentation approaches**:
- **Geometric transformation methods** (PoseAug, AdaptPose): Operate only at the keypoint level and cannot alter image context.
- **Synthetic rendering methods** (PoseGen): Use GANs to generate poses and NeRF to render images, but NeRF-rendered subjects lack realistic backgrounds and appearance diversity.
- **Text-guided methods**: Text descriptions provide insufficient precise control over complex poses.

The core insight of this paper is that abundant in-the-wild 2D pose data can serve as a bridge: by identifying real images on which the TPE performs poorly, training data can be synthesized in a targeted manner around those hard poses. Since hard images lack 3D annotations and directly using TPE pseudo-labels is inaccurate, motion synthesis is employed to expand pose variants that better approximate the true hard poses.

## Method

### Overall Architecture

PoseSyn consists of three stages:
1. **EEM (Error Extraction Module)**: Uses the discrepancy between 2D ground-truth annotations and the 2D projection of TPE's 3D predictions to identify hard and easy samples.
2. **MSM (Motion Synthesis Module)**: Expands the inaccurate pseudo-labels of hard samples into motion sequences, jointly leveraging text descriptions and initial pose information.
3. **Video Generation and Filtering**: Employs a human animation model (Champ) to synthesize realistic training videos from motion sequences and reference images of easy samples; low-quality frames are filtered before fine-tuning the TPE.

### Key Designs

1. **Error Extraction Module (EEM)**:

    - **Function**: Automatically identifies samples on which the TPE performs poorly from in-the-wild 2D pose datasets.
    - **Mechanism**: For each sample in the 2D dataset, the TPE predicts a 3D pose $\hat{J}^{\text{3D}}$, which is projected to 2D as $\hat{J}^{\text{2D}}$. A weighted error against the 2D ground-truth pose is computed:
    $Err = \sum_{n=2}^{N_{\text{2D}}} \mathbf{w}_n \left|(\hat{J}^{\text{2D},n} - \hat{J}^{\text{2D},1}) - (J_{\text{GT}}^{\text{2D},n} - J_{\text{GT}}^{\text{2D},1})\right|$
      The $K_C$ samples with the largest errors form the hard dataset $\mathcal{D}_C$, while the $K_{NC}$ samples with the smallest errors form the non-hard dataset $\mathcal{D}_{NC}$. Arm and leg joints are assigned higher weights $\mathbf{w}_n$ due to their greater motion variability.
    - **Design Motivation**: Unlike PoseGen, which identifies hard poses from synthetic images, EEM evaluates TPE performance directly on real images, thereby capturing cases that genuinely affect real-world deployment. Hard samples tend to contain dynamic and complex poses, while non-hard samples are predominantly static standing poses.

2. **Motion Synthesis Module (MSM)**:

    - **Function**: Expands the inaccurate pseudo-labels $\hat{J}_C^{\text{3D}}$ of hard samples into motion sequences containing diverse pose variants.
    - **Mechanism**:
        - A VLM first generates text descriptions for each hard image (e.g., "person kneeling down and chopping with an axe").
        - The pseudo-label pose $\hat{J}_C^{\text{3D}}$ is replicated $T$ times to form an initial motion representation $\mathcal{MR}_{\text{init}} = F(\hat{J}_C^{\text{3D}} \otimes T)$.
        - The Motion VQ-VAE encoder of T2M-GPT encodes this into initial motion indices $\mathcal{S}_{\mathcal{MR}}$.
        - Under the joint guidance of text embeddings $\mathbf{e}_{\text{text}}$ and initial motion indices $\mathcal{S}_{\mathcal{MR}}$, motion sequences are autoregressively generated:
       $$p(\mathcal{S}_C | \mathbf{e}_{\text{text}}, \mathcal{S}_{\mathcal{MR}}) = \prod_{i=0}^{|\mathcal{S}_C|} p(s^i | \mathbf{e}_{\text{text}}, \mathcal{S}_{\mathcal{MR}}, s^{<i})$$
        - Decoding yields a motion sequence $\mathcal{M}_C$ consisting of $L$ frames of 3D poses.
    - **Design Motivation**: Generation using text alone (without $\mathcal{MR}_{\text{init}}$) introduces ambiguity and fails to precisely capture the geometric details of hard poses; using pseudo-labels alone is inaccurate. By generating motion sequences rather than isolated frames, the method produces multiple plausible variants near the hard pose, increasing the probability of approximating the true difficult pose.

3. **Motion-Guided Video Generation and Training**:

    - **Function**: Converts motion sequences into training image–pose pairs with realistic appearance and backgrounds.
    - **Mechanism**: The off-the-shelf human animation model Champ is conditioned on reference images $I_{NC}$ from non-hard data for appearance and on motion sequences $\mathcal{M}_C$ for driving, generating realistic human animation videos. For each frame, the TPE predicts a 3D pose, and frames are filtered based on the error against the motion sequence's 3D pose:
    $Err_{\text{3D},l} = \sum_{n=2}^{N_{\text{3D}}} |(\hat{J}_l^{\text{3D},n} - \hat{J}_l^{\text{3D},1}) - (J_{\text{C},l}^{\text{3D},n} - J_{\text{C},l}^{\text{3D},1})|$
      Frames with errors exceeding threshold $\tau$ are discarded; high-quality retained samples are merged with original real data to fine-tune the TPE.
    - **Design Motivation**: Unlike PoseGen's NeRF rendering (which lacks background and appearance diversity), Champ preserves the realistic subject appearance and background from reference images, yielding more photorealistic training data.

### Loss & Training

PoseSyn does not introduce new loss functions; it generates augmented data for fine-tuning the TPE, which uses its own original training loss (e.g., MPJPE loss). Key hyperparameters: $K_C = 500$ (hard samples), $K_{NC} = 200$ (non-hard reference images), yielding approximately 27,000 synthesized 3D pose data samples in total.

## Key Experimental Results

### Main Results

Performance comparison of three TPEs on real-world datasets (MPJPE↓mm / PA-MPJPE↓mm):

| TPE | Method | 3DPW MPJPE↓ | EMDB MPJPE↓ | CMU_171204 MPJPE↓ | HuMMan MPJPE↓ | Mean MPJPE↓ |
|-----|--------|------------|------------|-------------------|--------------|------------|
| 3DCrowdNet | Real-only | 81.7 | 115.8 | 108.8 | 98.9 | 103.2 |
| 3DCrowdNet | PoseGen | 80.0 | 113.1 | 104.0 | 94.5 | 99.7 |
| 3DCrowdNet | **PoseSyn** | **77.4** | **111.0** | **101.0** | **93.1** | **97.5** |
| HyBrik | Real-only | 88.0 | 155.4 | 117.5 | 119.7 | 121.2 |
| HyBrik | **PoseSyn** | **78.4** | **129.9** | **100.3** | **95.3** | **104.6** |
| 4DHumans | Real-only | 81.3 | 116.3 | 115.1 | 106.1 | 106.8 |
| 4DHumans | **PoseSyn** | **77.0** | **108.6** | **104.1** | **98.0** | **99.1** |

PoseSyn achieves the best performance across all TPEs and datasets, with MPJPE improvements of 6–14% and PA-MPJPE improvements of 5–9%.

### Ablation Study

MSM effectiveness analysis (PA-MPJPE↓mm, 100 hard samples from EMDB):

| Method | Mean±Std | Min | Note |
|--------|---------|-----|------|
| (a) Pseudo-label $\hat{J}^{\text{3D}}$ | 181.7 ± 0.0 | 181.7 | Single inaccurate prediction |
| (b) w/o $\mathcal{MR}_{\text{init}}$ | 222.3 ± 36.4 | 151.1 (−16.8%) | Text only; worse mean but better minimum |
| (c) **Full MSM** | **209.3 ± 36.5** | **140.8 (−22.5%)** | Text + initial pose; lowest minimum error |

Impact of MSM on TPE training (3DCrowdNet, Mean MPJPE↓):

| Configuration | Mean MPJPE↓ | Mean PA-MPJPE↓ | Note |
|---------------|------------|----------------|------|
| Real-only | 103.2 | 66.2 | Baseline |
| Pseudo-label $\hat{J}^{\text{3D}}$ only | 99.6 | 65.4 | Limited improvement |
| w/o $\mathcal{MR}_{\text{init}}$ | 98.5 | 64.4 | Missing pose prior |
| **Full PoseSyn** | **97.5** | **62.9** | Best with both components |

### Key Findings

- **Motion sequences are more effective than single-frame generation**: By expanding pseudo-labels into motion sequences, MSM generates at least one variant closer to the true hard pose than the original pseudo-label (Min error reduced by 22.5%).
- **Initial motion representation $\mathcal{MR}_{\text{init}}$ is critical**: Without it, Min error decreases by only 16.8% (vs. 22.5%), demonstrating that text alone cannot precisely locate hard poses.
- **Identifying hard poses from real images outperforms using synthetic images**: Ours-N (same EEM+MSM but with NeRF rendering) already surpasses PoseGen, while PoseSyn (with Champ animation) achieves further gains, reaching a maximum MPJPE improvement of 14%.
- **Model-agnostic effectiveness**: Consistent improvements are observed across three TPEs with different architectures and scales: 3DCrowdNet, HyBrik, and 4DHumans.

## Highlights & Insights

1. **Hard-sample-driven data synthesis**: Rather than augmenting all data indiscriminately, PoseSyn precisely targets the TPE's weak spots for focused augmentation — a strategy particularly effective in the continuous pose space.
2. **Motion sequences as a natural extension of poses**: Expanding isolated inaccurate poses into continuous motion sequences leverages motion priors to naturally produce pose variants, avoiding the implausible poses that may arise from random perturbations directly in keypoint space.
3. **Closed-loop augmentation without 3D annotations**: The entire pipeline (hard sample identification → motion synthesis → image generation → automatic 3D annotation) is driven solely by 2D ground-truth annotations, entirely bypassing the need for costly 3D labeling.
4. **Critical role of image quality**: Compared to NeRF rendering, the Champ animation model preserves realistic backgrounds and subject appearance, which is essential for training image-to-3D pose estimators.

## Limitations & Future Work

- The pipeline depends on the quality of text descriptions generated by the VLM; in scenes with complex occlusions, the descriptions may be insufficiently precise.
- The Champ human animation model may produce subject-background blending artifacts; although a filtering mechanism is in place, it may discard an excessive number of useful samples.
- Hard samples are currently collected only from the MPII dataset; extending to additional in-the-wild datasets could yield further improvements.
- Motion synthesis relies on T2M-GPT, whose performance ceiling is bounded by the diversity of its pre-training motion data.

## Related Work & Insights

- The hard sample mining strategy of EEM is generalizable to other regression tasks requiring data augmentation (e.g., depth estimation, optical flow estimation).
- The concept of using motion synthesis as an intermediate representation may be applicable to other tasks that recover 3D information from 2D signals.
- Advances in video generation models such as Champ will directly enhance the quality of PoseSyn's synthesized data, creating a virtuous cycle.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of hard-sample-driven augmentation and motion sequence expansion is novel, though individual modules rely on existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three TPEs and six datasets; ablation study is rigorously designed.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated; the method pipeline diagram is intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a generalizable data augmentation paradigm that requires no 3D annotations, offering practical value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/human_understanding/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[ICCV 2025\] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](../../CVPR2026/human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](../../NeurIPS2025/human_understanding/learning_dense_hand_contact_estimation_from_imbalanced_data.md)

</div>

<!-- RELATED:END -->
