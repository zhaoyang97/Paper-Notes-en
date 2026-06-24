---
title: >-
  [Paper Note] WildAvatar: Learning In-the-Wild 3D Avatars from the Web
description: >-
  [CVPR 2025][Human Understanding][3D human avatar] Proposes an automated annotation pipeline and filtering protocols to construct WildAvatar—a large-scale, in-the-wild 3D avatar creation dataset containing over 10,000 human subjects, which is over 10 times larger than previous datasets and outperforms existing SMPL annotation methods on the EMDB benchmark.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "3D human avatar"
  - "in-the-wild dataset"
  - "automatic annotation"
  - "SMPL estimation"
  - "large-scale data"
date: 2026-05-08
content_hash: 8f6beb31ad83dc13
---

# WildAvatar: Learning In-the-Wild 3D Avatars from the Web

**Conference**: CVPR 2025  
**arXiv**: [2407.02165](https://arxiv.org/abs/2407.02165)  
**Code**: [https://wildavatar.github.io/](https://wildavatar.github.io/)  
**Area**: Human Understanding / 3D Human Reconstruction  
**Keywords**: 3D human avatar, in-the-wild dataset, automatic annotation, SMPL estimation, large-scale data

## TL;DR
Proposes an automated annotation pipeline and filtering protocols to construct WildAvatar—a large-scale, in-the-wild 3D avatar creation dataset containing over 10,000 human subjects, which is over 10 times larger than previous datasets and outperforms existing SMPL annotation methods on the EMDB benchmark.

## Background & Motivation

1. **Background**: 3D human avatar creation has widespread applications in fields like VR/AR, film production, and the metaverse. With the emergence of NeRF, significant progress has been made in methods that reconstruct 3D avatars from 2D observations and synthesize novel views/poses. However, existing methods are primarily validated on **laboratory datasets**.

2. **Limitations of Prior Work**: Current avatar datasets primarily rely on expensive laboratory systems—multi-view camera arrays, depth sensors, IMUs, laser scanners, professional actors, and lighting stages. These setups are (a) highly costly and difficult to scale; (b) limited in scene diversity, exhibiting a domain gap with the real world; and (c) even the largest dataset (such as HUMBI with 772 subjects) remains negligible compared to real-world diversity.

3. **Key Challenge**: A vast amount of free, real-world human motion video exists on the internet, but these videos vary in quality and lack the precise annotations (SMPL parameters, camera parameters, human segmentation masks) required for avatar creation. A few attempts to collect in-the-wild data (such as the TikTok dataset) heavily rely on manual filtering, which cannot scale.

4. **Goal**: Design a fully automated annotation pipeline and filtering protocols to extract high-quality human motion data from web videos to construct a large-scale, in-the-wild avatar dataset.

5. **Key Insight**: Utilize a four-stage pipeline composed of recent SOTA models (such as YOLO, SAM, and HMR 2.0) for automatic annotation, combined with multi-consistency check protocols to automatically filter out low-quality samples.

6. **Core Idea**: Replace manual intervention with a pipelined automated approach. Through a two-step "annotation + filtering" strategy, high-quality human data can be extracted from YouTube at an extremely low cost, achieving a 10x increase in dataset scale.

## Method

### Overall Architecture
The pipeline is divided into four stages: Stage I detects and tracks human bounding boxes -> Stage II extracts human segmentation masks (SAM) -> Stage III coarsely estimates single-frame SMPL and camera parameters -> Stage IV refines SMPL and camera parameters across frames (temporal smoothing). Four filtering protocols are interleaved among these four stages to automatically reject unqualified videos.

### Key Designs

1. **Four-Stage Automatic Annotation Pipeline**:
    - **Function**: Automatically obtain high-quality annotations (bounding boxes, human masks, SMPL parameters, camera parameters) from raw web videos.
    - **Mechanism**: Stage I uses YOLO or similar models to detect and track human bounding boxes; Stage II employs SAM (Segment Anything) to generate high-quality human segmentation masks based on bounding box inputs, eliminating the need for manual static background frame selection; Stage III estimates SMPL parameters frame-by-frame using HMR 2.0 or similar; Stage IV refines SMPL and camera parameters over video sequences via gradient descent, introducing 2D keypoints and SAM masks as additional supervision to enhance temporal consistency.
    - **Design Motivation**: Each stage utilizes the current best-performing method, and the refinement in subsequent stages compensates for the coarse estimation of the previous stages. In particular, the temporal smoothing in Stage IV significantly improves the alignment accuracy of limbs (feet, arms, head).

2. **Four Automated Filtering Protocols**:
    - **Function**: Automatically reject unqualified video clips to guarantee dataset quality.
    - **Mechanism**:
        - Protocol I (Clear Body & Significant Motion): Filter out videos with low confidence in detection/pose estimation, and eliminate clips with severe occlusion or excessively small motion amplitude.
        - Protocol II (Annotation Ensemble of Experts): Utilize prediction results from multiple different SOTA models (detection, 2D pose, SMPL estimation) to compute standard deviations, retaining only videos with high consistency across models.
        - Protocol III (2D Keypoint Consistency): Project the 3D keypoints of SMPL onto 2D to compare with pose estimation results, compute PCK values, and eliminate low-consistency samples.
        - Protocol IV (SMPL & SAM Mask Consistency): Check whether the SMPL projected mask is covered by the SAM mask (since SMPL represents a naked body and SAM includes clothing), eliminating samples with GSM (overlap) values that are too low.
    - **Design Motivation**: Web video quality is highly variable, and since ground truth is unavailable, multiple cross-validations are crucial for quality assurance. The four protocols tighten progressively, filtering down from 465,801 candidate clips to 10,647 high-quality ones.

3. **Dataset Statistics & Diversity Analysis**:
    - **Function**: Verify the diversity advantages of the dataset in terms of poses, viewpoints, and clothing.
    - **Mechanism**: Analyzing body pose distributions via t-SNE reveals that WildAvatar poses are significantly more diverse than those in laboratory datasets; camera viewpoints are no longer restricted to fixed angles; SSIOU (SMPL and Segment Inverse IoU) is introduced to measure clothing richness, demonstrating that WildAvatar clothing variety significantly exceeds that of laboratory data.
    - **Design Motivation**: Prove that scaling up and in-the-wild collection can overcome the "scene paucity" issue of laboratory datasets.

### Loss & Training
- Stage IV SMPL refinement uses 2D keypoint reprojection error + SAM mask alignment loss + temporal smoothing constraints for gradient descent optimization.
- The annotation pipeline itself does not require training new models; it entirely utilizes pre-trained, off-the-shelf models.

## Key Experimental Results

### Main Results

Comparison of SMPL annotation quality on the EMDB benchmark:

| Method | PA-MPJPE↓ | MPJPE↓ | PVE↓ |
|------|-----------|--------|------|
| HMR2.0 | 60.6 | 98.0 | 120.3 |
| HybrIK | 65.6 | 103.0 | 122.2 |
| CLIFF | 68.1 | 103.3 | 128.0 |
| **Ours** | **59.9** | **94.9** | **115.5** |

Progressive quality improvement via filtering protocols:

| Filtering Stage | PCK↑ | SOIOU↓ | Retained Clip Count |
|----------|------|--------|-----------|
| No Filtering | 0.282 | 0.760 | 465,801 |
| +Protocol I | 0.762 | 0.214 | 43,824 |
| +Protocol II | 0.839 | 0.146 | 25,392 |
| +Protocol III | 0.882 | 0.129 | 12,482 |
| +Protocol IV | 0.902 | 0.052 | 10,647 |
| +Stage IV Refinement | 0.921 | 0.028 | 10,647 |

### Ablation Study

| Configuration | Description |
|------|------|
| Protocol I is most critical | Filters out 90% of low-quality videos, with PCK jumping from 0.282 to 0.762 |
| Protocol II | PCK further improves by 10.1%, demonstrating a significant effect of multi-expert ensemble |
| Stage IV Refinement | Keeps the number of samples unchanged but improves PCK from 0.902 to 0.921 |
| Final PCK=0.921 | Only 1.7% lower than the 0.937 of 3DPW GT |

### Key Findings
- Protocol I is the most critical preprocessing step, filtering out a large number of videos with no human, severe occlusions, or minimal motions.
- The multi-expert ensemble (Protocol II) contributes majorly to annotation reliability, improving PCK by 10.1%.
- Temporal refinement in Stage IV does not change the sample count but significantly improves annotation quality.
- The final dataset achieves PCK=0.921, which is close to the 0.937 of 3DPW manual annotations.
- Training on generalizable avatar methods yields performance improvements of up to 7% compared to using laboratory datasets.

## Highlights & Insights
- **Fully Automated Pipeline Design**: From YOLO/SAM/HMR 2.0 to temporal refinement, the entire process requires no manual intervention and is highly cost-effective. This paradigm of "off-the-shelf model combination + multiple verification" can be generalized to other data collection tasks.
- **Clever Filtering Protocol Design**: In the absence of ground truth, replacing traditional evaluation with "multi-model cross-consistency" serves as a generalizable strategy for data quality assurance.
- **SMPL-SAM Cross-Verification Idea**: Utilizing the prior constraint that the SMPL projection (naked body) should be covered by the SAM mask (clothed contour) for quality checks is highly ingenious.
- **Scale is Key**: Demonstrates that a 10x scale improvement in data can yield up to a 7% performance boost for generalizable methods, once again validating the "scaling law".

## Limitations & Future Work
- Since data is sourced from YouTube, potential copyright and privacy issues exist; consequently, only video IDs rather than raw videos are released.
- The automated pipeline still has performance upper bounds—it cannot handle severe occlusions, extreme poses, or motion blur.
- Annotation quality is bounded by the expressive power of the SMPL model itself, making it incapable of modeling detailed hands or faces.
- Although clothing diversity in the dataset is superior to that of laboratories, it is still biased by the distribution of YouTube content.
- Future work can integrate SMPL-X or more detailed parametric models to enhance annotation granularity.

## Related Work & Insights
- **vs NeuMan**: NeuMan jointly disentangles the avatar and the scene but requires precise global alignment, limiting its applicability to only 6 scenes. WildAvatar bypasses the challenges of scene decoupling by purely focusing on the human region, achieving a scale of over ten thousand.
- **vs TikTok Dataset**: The TikTok dataset contains 340 subjects but relies on manual filtering and exhibits minimal viewpoint changes. WildAvatar is fully automated, scales to 10k+ subjects, and features free viewpoints, comprehensively outperforming it.
- **vs ZJU-MoCap**: ZJU-MoCap features 9 subjects in 6 laboratory scenes vs. 10k+ subjects in the wild for WildAvatar. WildAvatar demonstrates that large-scale, in-the-wild data can compensate for (or even surpass) the lack of precise laboratory equipment.

## Rating
- Novelty: ⭐⭐⭐ Methodologically, the work is a combination of existing tools; the primary contribution lies in the dataset itself.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluations on the EMDB benchmark, step-by-step filtering ablation, and validations on downstream tasks are extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured, and dataset statistics and visualizations are well executed.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in large-scale, in-the-wild avatar datasets, providing an important driving force for the entire field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2026\] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision](../../CVPR2026/human_understanding/flexavatar_learning_complete_3d_head_avatars_with_partial_supervision.md)
- [\[CVPR 2025\] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)
- [\[CVPR 2025\] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)

</div>

<!-- RELATED:END -->
