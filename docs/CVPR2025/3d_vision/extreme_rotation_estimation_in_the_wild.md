---
title: >-
  [Paper Note] Extreme Rotation Estimation in the Wild
description: >-
  [CVPR 2025][3D Vision][Extreme Rotation Estimation] This paper proposes an extreme 3D rotation estimation method for real-world Internet images, constructing the ExtremeLandmarkPairs (ELP) benchmark dataset. Through a progressive learning scheme (panoramic cropping $\rightarrow$ FoV + appearance augmentation $\rightarrow$ real data fine-tuning) and an auxiliary-channel-enhanced Transformer model, the proposed method significantly outperforms existing methods on non-overlappin…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Extreme Rotation Estimation"
  - "Camera Pose Estimation"
  - "Non-overlapping Views"
  - "Internet Images"
  - "Progressive Learning"
date: 2026-05-08
content_hash: 19e2270cdcc413bd
---

# Extreme Rotation Estimation in the Wild

**Conference**: CVPR 2025  
**arXiv**: [2411.07096](https://arxiv.org/abs/2411.07096)  
**Code**: [https://tau-vailab.github.io/ExtremeRotationsInTheWild/](https://tau-vailab.github.io/ExtremeRotationsInTheWild/)  
**Area**: 3D Vision  
**Keywords**: Extreme Rotation Estimation, Camera Pose Estimation, Non-overlapping Views, Internet Images, Progressive Learning

## TL;DR

This paper proposes an extreme 3D rotation estimation method for real-world Internet images, constructing the ExtremeLandmarkPairs (ELP) benchmark dataset. Through a progressive learning scheme (panoramic cropping $\rightarrow$ FoV + appearance augmentation $\rightarrow$ real data fine-tuning) and an auxiliary-channel-enhanced Transformer model, the proposed method significantly outperforms existing methods on non-overlapping Internet image pairs.

## Background & Motivation

**Background**: Estimating relative 3D rotation between two images is a fundamental task in camera localization and 3D reconstruction. Traditional methods rely on pixel-level correspondences (such as SIFT, LoFTR) to compute relative poses, performing well under scenes with sufficient overlap. Recent works (such as DenseCorrVol, CascadedAtt) have also begun to explore rotation estimation under extreme, non-overlapping views.

**Limitations of Prior Work**: (1) Traditional feature matching methods fail under extreme views (few or no overlap) because they cannot extract valid correspondences; (2) Existing extreme rotation estimation methods (DenseCorrVol, CascadedAtt) are trained and evaluated only on controlled images cropped from panoramas—featuring a fixed 90° FoV, consistent illumination, and identical camera intrinsics—and fail to generalize to real-world Internet photos; (3) Real-world Internet images pose severe appearance diversity challenges: varying illumination, weather, seasons, dynamic objects, and diverse camera intrinsics.

**Key Challenge**: Existing methods perform well on "simulated" extreme views, but the extremity of real-world Internet images far exceeds that of cropped panoramic images—they are extreme not only in view but also in appearance and camera parameters. Meanwhile, most camera views in Internet image sets capture overlapping areas (since SfM requires dense imagery), leaving the number of genuinely non-overlapping image pairs very limited, leading to insufficient training data.

**Goal**: (1) How to construct a real-world Internet extreme-view image pair dataset? (2) How to enable the model to progressively generalize from controlled panoramic cropped data to real-world Internet data? (3) How to reason about relative rotation using implicit cues (vanishing points, shadow directions, skylines, etc.) in non-overlapping cases?

**Key Insight**: Identify rotation-dominated image pairs by constructing a Mutual Nearest Neighbors (MNN) graph combined with FoV-adaptive overlap estimation; bridge the data scarcity gap with progressive learning—first learning baseline capabilities on panoramic cropped data, then approaching the real data distribution via FoV augmentation and diffusion-based appearance augmentation, and finally fine-tuning on real data.

**Core Idea**: Elevate extreme rotation estimation from controlled panoramic environments to real in-the-wild Internet photos via a progressive cross-domain learning scheme and an auxiliary-channel-augmented Transformer.

## Method

### Overall Architecture

Input a pair of potentially non-overlapping Internet images. Use a pre-trained LoFTR to extract image features, combine them with auxiliary channels (keypoint masks, match masks, semantic segmentation maps), and reshape them into token sequences. After adding learnable Euler angle position embeddings, the sequences are fed into a rotation estimation Transformer (Decoder architecture). The output Euler angle tokens are concatenated with the average image token, and three independent MLP heads are used to predict the 360-bin probability distributions of roll/pitch/yaw.

### Key Designs

1. **ExtremeLandmarkPairs (ELP) Dataset Construction**:

    - Function: Provides a training and evaluation benchmark for real-world Internet extreme-view image pairs.
    - Mechanism: Starts from Internet image sets such as MegaScenes, MegaDepth, and Cambridge Landmarks. (a) Identifies rotation-dominant image pairs by constructing an MNN graph ($K=5$ nearest neighbors)—adjacent images in dense regions are usually dominated by rotation with small translations, while images in sparse regions are excluded from the MNN graph. (b) Since real images have varying FoVs, overlap cannot be judged solely by the rotation angle as in panoramic cropped datasets. A FoV-adaptive overlap classification is designed: $|\gamma| < \frac{fov_x^1+fov_x^2}{4}$ and $|\beta| < \frac{fov_y^1+fov_y^2}{4}$ indicates large overlap, while more than twice the corresponding threshold indicates no overlap. (c) Restricts the FoV difference of the image pairs to no more than 5°, filters out images with roll > 10°, and excludes aerial views. Eventually, about 34K non-overlapping pairs are obtained.
    - Design Motivation: Previous benchmarks only used panoramic cropped images, which do not reflect real-world challenges. ELP provides two test sets: sELP (single camera, constant illumination) and wELP (genuine in-the-wild Internet photos), supporting hierarchical evaluation.

2. **Progressive Learning Scheme**:

    - Function: Progressively generalizes from controlled data to real-world Internet data.
    - Mechanism: Three-stage training. Stage 1 (Initialization): Train baseline capabilities on StreetLearn panoramic cropped data (~1M image pairs, fixed 90° FoV). Stage 2 (Data Augmentation Training): Two types of augmentations—(a) FoV augmentation ($\Delta$FoV): analyze the FoV distribution of the ELP training set, sample FoV from $\mathcal{N}(\mu, 1.5\sigma)$ to crop panoramic images, allow up to 5° FoV differences between image pairs, and use varying aspect ratios; (b) Appearance augmentation ($\Delta$Im): apply text prompts such as "Make it snowy/sunset/night/busy street" to part of the data using InstructPix2Pix to generate diverse appearance variants. Stage 3 (Real Data Fine-Tuning): Fine-tune on the ELP training set, prioritizing training on non-overlapping pairs.
    - Design Motivation: Real extreme image pairs are scarce (only ~34K non-overlapping pairs), insufficient for training from scratch. The progressive scheme allows the model first to learn basic rotation estimation and then adapt to FoV and appearance diversity, before finally fitting the real data distribution. Ablations show that each stage makes an indispensable contribution.

3. **Auxiliary Channel Augmentation**:

    - Function: Provides the Transformer with structured reasoning cues beyond pixel appearance.
    - Mechanism: In addition to the image features extracted by LoFTR, three types of auxiliary channels are concatenated: (a) Keypoint masks—marking the spatial distribution of local feature points in the image; (b) Match masks—marking the positions of successfully matched keypoints between the image pair, providing alignment cues for small overlap cases; (c) Semantic segmentation maps—segmenting the image into categories like sky, building, and road, helping to identify implicit cues such as skylines and dynamic objects used to reason about non-overlapping pairs.
    - Design Motivation: For non-overlapping image pairs, pixel-level features can rarely provide rotation information directly. However, semantic cues (such as comparing skyline heights between two images) and matching patterns (matches indicate overlap, no matches suggest potential non-overlap) can assist reasoning. Combined with pre-trained LoFTR features (which encode knowledge of Internet image pairs), this yields a stronger feature representation.

### Loss & Training

Use cross-entropy loss to train 360-bin classification for each of the three Euler angles (roll, pitch, yaw). During inference, the bin with the highest probability is taken as the estimated angle. During evaluation, Top-1 and Top-5 prediction results are reported (Top-5 considers the top 5 peaks of yaw predictions). In progressive training, non-overlapping batches are processed before overlapping batches.

## Key Experimental Results

### Main Results

| Method | wELP-None MGE↓ | wELP-None RRA30↑ | wELP-Small MGE↓ | wELP-Large MGE↓ |
|------|----------|------------|----------|----------|
| SIFT | 122.84 | 2.0 | 7.27 | 2.94 |
| LoFTR | 56.54 | 33.0 | 6.80 | 2.13 |
| DenseCorrVol | 82.04 | 13.7 | 125.73 | 120.53 |
| CascadedAtt | 78.60 | 20.8 | 139.14 | 170.62 |
| DUSt3R | 81.21 | 26.9 | 2.80 | 1.01 |
| **Ours** | **26.97** | **50.7** | **4.47** | **2.41** |

On the wELP test set (real in-the-wild images) with non-overlapping pairs, the proposed method achieves an MGE of 26.97°, which is vastly superior to all baselines (DUSt3R is 81.21°, possibly affected by its training data limits), and improves the RRA30 from 26.9% to 50.7%.

### Ablation Study

| Training Data | wELP-None MGE↓ | wELP-None RRA30↑ | wELP-Small MGE↓ | wELP-Large MGE↓ |
|---------|----------|------------|----------|----------|
| Panorama Cropped [8] Only | 74.94 | 25.3 | 55.28 | 13.65 |
| +ΔFoV | 61.62 | 38.4 | 12.91 | 4.61 |
| +ΔIm | 68.31 | 36.1 | 11.46 | 4.46 |
| +ELP (Full) | **26.97** | **50.7** | **4.47** | **2.41** |

### Key Findings

- Models trained on panoramic cropped data (DenseCorrVol, CascadedAtt) degrade severely on real Internet images—exceeding 120° MGE on wELP-Large (equivalent to random prediction), indicating an immense domain gap from panoramic to real scenes.
- Both FoV and appearance augmentations contribute significantly: adding only FoV augmentation reduces None-MGE from 74.94 to 61.62; adding only appearance augmentation reduces Small-MGE from 55.28 to 11.46. They are complementary.
- Fine-tuning on the ELP real dataset brings the largest improvement: None-MGE drops from 68.31 to 26.97 (a 60% reduction), proving real-world data is irreplaceable.
- The model has only 80M parameters, much smaller than DUSt3R's 577M, yet far outperforms it in non-overlapping extreme scenarios.
- LoFTR is more effective as a feature extractor than an ImageNet-pretrained CNN, because LoFTR itself is trained on feature matching tasks across Internet image pairs.
- In evaluations on panoramic cropped images (Table 3), the proposed model is comparable to previous methods (None RRA10 = 96.4%), showing that generalizing to real-world data does not sacrifice performance in controlled scenarios.

## Highlights & Insights

- **High Practical Value of the Problem**: Relative rotation estimation for Internet photo collections is an important component in 3D reconstruction pipelines. Existing methods rely on feature matching when overlap is present and fail completely when there is no overlap. This paper fills this gap.
- **Exquisitely Designed Progressive Learning Scheme**: Through FoV distribution matching and diffusion-based appearance augmentation, the method elegantly bridges the domain gap between panoramic cropped data and real Internet data. Using InstructPix2Pix for appearance augmentation (e.g., turning day into night, adding snow) is a highly valuable data augmentation trick.
- **Systematic ELP Dataset Construction**: Rotation-dominant image pairs are automatically extracted from dense SfM reconstructions. The design of FoV-adaptive overlap classification is reasonable, providing a standardized benchmark for future research.

## Limitations & Future Work

- The MGE in non-overlapping scenes is still 26.97° (13.62° on sELP), leaving room for accuracy improvements, especially for non-overlapping pairs in wELP.
- The method assumes that the image pairs are dominated by rotational motion (with negligible translation) and is not applicable to scenes with significant translational motion.
- It only handles outdoor scenes; structural cues for indoor scenes (such as skylines) are not applicable.
- Filtering out images with roll > 10° and pairs with large FoV differences restricts the scope of application.
- The three-stage progressive training increases training complexity and hyperparameter tuning effort.

## Related Work & Insights

- **vs DenseCorrVol/CascadedAtt**: These two approaches were pioneering efforts in extreme rotation estimation but were trained/tested entirely on panoramic cropped data, leading to over 100° MGE on real Internet images. This paper adapts the model to real scenarios using progressive learning and auxiliary channels.
- **vs DUSt3R/Mast3R**: DUSt3R is a powerful general 3D reconstruction method that performs excellently in overlapping scenes. However, it is based on CroCo pretraining (assuming overlap), showing a poor MGE of 81.21° on non-overlapping pairs, while having 7 times more parameters than the proposed method.
- **vs LoFTR**: LoFTR is an excellent feature matching method in overlapping scenes, but fails to yield reliable estimates in small-overlap and non-overlapping scenes. This paper utilizes LoFTR as a feature extractor (rather than a matcher), leveraging its capability to encode knowledge of Internet image pairs.

## Rating

- Novelty: ⭐⭐⭐⭐ Meaningful problem definition, systematic progressive learning and dataset construction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple test sets (sELP/wELP/panorama cropped), rich ablation studies and baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed description of dataset construction.
- Value: ⭐⭐⭐⭐ The ELP dataset and progressive learning scheme drive the research on extreme pose estimation forward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[ICCV 2025\] PersPose: 3D Human Pose Estimation with Perspective Encoding and Perspective Rotation](../../ICCV2025/3d_vision/perspose_3d_human_pose_estimation_with_perspective_encoding_and_perspective_rota.md)
- [\[CVPR 2025\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
