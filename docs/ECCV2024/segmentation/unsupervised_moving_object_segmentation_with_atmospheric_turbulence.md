---
title: >-
  [Paper Note] Unsupervised Moving Object Segmentation with Atmospheric Turbulence
description: >-
  [ECCV 2024][Segmentation][Unsupervised segmentation] This paper proposes an unsupervised method to segment moving objects in atmospheric turbulence videos using a "detect-then-grow" strategy. It first separates real motion from turbulent motion using epipolar geometric consistency checks based on Sampson distance, then performs region growing from high-confidence seed pixels to generate segmentation masks, and finally refines the masks with a spatio-temporal consistency loss.…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Unsupervised segmentation"
  - "Atmospheric turbulence"
  - "Motion segmentation"
  - "Region growing"
  - "Epipolar geometry"
date: 2026-05-08
content_hash: 5ef0c19e841835e7
---

# Unsupervised Moving Object Segmentation with Atmospheric Turbulence

**Conference**: ECCV 2024  
**arXiv**: [2311.03572](https://arxiv.org/abs/2311.03572)  
**Code**: None (Dataset page: [https://turb-research.github.io/DOST/](https://turb-research.github.io/DOST/))  
**Area**: Segmentation  
**Keywords**: Unsupervised segmentation, Atmospheric turbulence, Motion segmentation, Region growing, Epipolar geometry

## TL;DR
This paper proposes an unsupervised method to segment moving objects in atmospheric turbulence videos using a "detect-then-grow" strategy. It first separates real motion from turbulent motion using epipolar geometric consistency checks based on Sampson distance, then performs region growing from high-confidence seed pixels to generate segmentation masks, and finally refines the masks with a spatio-temporal consistency loss. It substantially outperforms existing methods on DOST, the first real turbulence video dataset (achieving a 60.1% IoU gain).

## Background & Motivation

**Background**: Moving object segmentation (MOS) is a fundamental task in video understanding. Learning-based methods (including supervised ones like RVOS and unsupervised ones like TMO, DS-Net, etc.) have achieved excellent performance on standard video datasets. These methods typically rely on optical flow to estimate motion, assuming a static background and rigid object motion.

**Limitations of Prior Work**: When videos are affected by atmospheric turbulence, all the aforementioned assumptions collapse. Atmospheric turbulence introduces irregular, time-varying pixel displacements—implying that even a fully static background exhibits "motion", which severely corrupts optical flow estimation. Worse still, long-range videos captured with telephoto lenses are particularly sensitive to turbulence, while also being highly susceptible to camera shake. Existing motion segmentation methods suffer degraded performance under normal turbulence and fail almost completely under severe turbulence (IoU < 0.25). Supervised methods are also helpless because annotated real-world turbulence video datasets are virtually non-existent.

**Key Challenge**: The fundamental challenge lies in distinguishing among three types of motion: (1) real rigid motion of the target; (2) random atmospheric turbulence perturbations; and (3) global camera shake. Optical flow cannot distinguish among these three, and existing methods rely on either optical flow or appearance features (which are also blurred by turbulence).

**Goal**: (1) How to extract the real object motion signal from the turbulence-corrupted optical flow? (2) How to generate precise segmentation masks in the absence of annotated training data? (3) How to guarantee the spatio-temporal consistency of the segmentation masks?

**Key Insight**: The authors present a key insight—object motion violates the epipolar geometric constraint of the scene (since epipolar geometry assumes a static scene), whereas turbulent motion, although also violating epipolar geometry, consists of random, small-magnitude perturbations that can be eliminated through temporal averaging. Therefore, using the Sampson distance (a metric of epipolar geometric consistency) can effectively disentangle real object motion from turbulent motion.

**Core Idea**: Disentangle real motion from turbulent motion using Sampson distance, and generate unsupervised segmentation masks through region growing and spatio-temporal consistency refinement.

## Method

### Overall Architecture
The input is a turbulence-affected video sequence $\{I_t\}_{t=1}^T$. The method consists of three stages: (1) Epipolar geometry-based motion disentanglement—calculating bidirectional optical flows, generating motion feature maps $\{M_t\}$ via optical flow stabilization and Sampson distance, and marking potential motion regions; (2) Region growing-based segmentation—selecting high-confidence seed pixels on the motion feature maps, generating coarse segmentation masks $\{\beta_t^m\}$ using a region growing algorithm, and unifying object IDs across multiple frames using K-means; (3) Spatio-temporal refinement—employing a U-Net (Refine-Net) with pixel cross-entropy consistency loss and grouping functions to iteratively refine the masks $\{\alpha_t^m\}$.

### Key Designs

1. **Epipolar Geometry-based Motion Disentanglement**:

    - Function: To extract the real target motion signal from the turbulence-corrupted optical flow.
    - Mechanism: A two-step process is adopted. The first step is optical flow stabilization: averaging the optical flows within a short temporal window $\hat{F}_t^j = \frac{1}{|\mathcal{K}_j|} \sum_{i \in \mathcal{K}_j} \frac{F_{t \to t+i}}{i}$, leveraging the assumption that "real motion is consistent over short intervals while turbulent perturbations are random and cancellable" to eliminate turbulence noise. The second step is the geometric consistency check: using the stabilized optical flow to estimate the fundamental matrix $\mathbf{F}$ between adjacent frames (via LMedS regression), and then computing the Sampson distance $M_t^j(\mathbf{p}_1, \mathbf{p}_2) = \frac{(\mathbf{p}_2^T \mathbf{F} \mathbf{p}_1)^2}{(\mathbf{F}\mathbf{p}_1)_1^2 + (\mathbf{F}\mathbf{p}_1)_2^2 + (\mathbf{F}^T\mathbf{p}_2)_1^2 + (\mathbf{F}^T\mathbf{p}_2)_2^2}$. Static background pixels satisfy the epipolar constraint (producing small Sampson distances), while moving objects violate it (producing large Sampson distances). Finally, all available Sampson distance maps are averaged to get the frame-level motion feature map $M_t$.
    - Design Motivation: Performing motion detection directly with optical flow is entirely unfeasible under turbulence (as optical flow estimation itself is severely disrupted). The Sampson distance cleverly exploits the statistical difference between "the systematic epipolar constraint violation of moving objects" and "the random epipolar constraint violation of turbulence"—the former persists and accumulates, whereas the latter occurs randomly and can be averaged out.

2. **Region Growing-based Segmentation**:

    - Function: To generate complete target segmentation masks from motion feature maps.
    - Mechanism: First, seed regions are identified on the motion feature map using a sliding window $D \times D$—windows satisfying the mean condition $\bar{M}_t(W_k) > \delta_1$ and variance condition $\sigma^2 < \delta_2$ are selected as seeds. Then, the mask is grown outward from these seeds: neighboring pixels are included in the mask when the difference between their motion feature value and the seed's value is smaller than a threshold $\delta_{seed} = 0.2 \times M_t(\mathbf{p}_{seed})$: For multi-object scenarios where different seeds yield different mask IDs, K-means clustering is applied on the mask centroids across frames to unify object IDs across the video sequence: $\arg\min_{\mu_m} \sum_m \sum_t \|c_t^m - \mu_m\|^2$.
    - Design Motivation: The motion feature map is a continuous-valued "heatmap" rather than a binary mask—direct thresholding would lose boundary details. Region growing starts from high-confidence regions and expands progressively, naturally adapting to the object's shape, and the threshold can adaptively scale based on turbulence intensity (larger thresholds are used for severe turbulence to obtain more conservative masks).

3. **Spatio-Temporal Refinement**:

    - Function: To eliminate mask noise and frame-to-frame inconsistencies generated during region growing.
    - Mechanism: A Refine-Net $\Phi_\theta$ (with a U-Net backbone) is designed, which takes the concatenation of video frame $I_t$ and motion feature map $M_t$ as input and outputs the refined mask $\alpha_t^m$. The training loss consists of three components: (a) $\mathcal{L}_1$—the pixel cross-entropy between the refined mask and the coarse mask; (b) $\mathcal{L}_2^g$—the bidirectional consistency loss between the refined mask and the coarse mask warped by optical flow; (c) $\mathcal{L}_3^g$—the bidirectional consistency loss between the refined mask and its own warped version. Additionally, a K-means grouping function is introduced to update the reference mask $\beta_t^m$ every 3 epochs, aggregating the pixel motion values and spatial coordinates to reassign foreground/background, thereby eliminating spatial discontinuities.
    - Design Motivation: Masks generated by region growing might suffer from gaps or leaking. The bidirectional consistency loss utilizes optical flow to establish correspondence across frames, forcing masks of adjacent frames to align after warping. The grouping function constrains the spatial coherence of masks from a global perspective.

### Loss & Training
Refine-Net training is conducted in two stages. Initialization phase (20-30 epochs): $\mathcal{L}_{ini} = \gamma_1 \mathcal{L}_1 + \gamma_2 \sum_g \mathcal{L}_2^g + \gamma_3 \sum_g \mathcal{L}_3^g$. Refinement phase (10 epochs): using the same loss but updating the reference masks via the grouping function every 3 epochs. Optical flow is estimated using RAFT, with a maximum frame interval of 4. The method is completely unsupervised and requires no annotated training data.

## Key Experimental Results

### Main Results

| Method | Normal Turb. $\mathcal{J}$ | Severe Turb. $\mathcal{J}$ | Overall $\mathcal{J}$ | Overall $\mathcal{F}$ | Overall $\mathcal{G}$ |
|--------|------|------|----------|------|------|
| TMO | 0.643 | 0.235 | 0.439 | 0.536 | 0.487 |
| DSprites | 0.427 | 0.101 | 0.264 | 0.374 | 0.319 |
| DS-Net | 0.361 | 0.191 | 0.276 | 0.327 | 0.302 |
| **Ours** | **0.851** | **0.557** | **0.703** | **0.723** | **0.713** |

### Ablation Study

| Configuration | IoU ($\mathcal{J}$) | Description |
|------|---------|------|
| Full pipeline (A+B+C) | 0.703 | Full model |
| Region growing only (A) | ~0.55 | No refinement, masks contain gaps |
| A+B (w/o grouping loss) | ~0.65 | Improved spatio-temporal consistency |
| w/o flow stabilization | 0.354 | Sampson distance corrupted by turbulence noise |
| w/o geometric consistency check | 0.685 | Incomplete motion/turbulence separation |
| Flow interval = 1 | ~0.60 | Insufficient information |
| Flow interval = 4 | 0.703 | Optimal |
| Flow interval = 5 | ~0.70 | Gain saturation |

### Key Findings
- **Optical flow stabilization is crucial**—removing it causes the IoU to plummet from 0.703 to 0.354, indicating that raw optical flow is completely unreliable under turbulence.
- The advantage is most pronounced under **severe turbulence**: TMO drops to 0.235 IoU while the proposed method retains 0.557, as the epipolar geometry method possesses inherent resistance to random turbulence.
- The method is also robust to **camera shake** (IoU 0.712), because camera motion is a global rigid motion that conforms to the epipolar geometric assumption.
- Foundation models like SAM also fail under strong turbulence, indicating that turbulence is indeed a unique challenge that cannot be solved solely by scaling up model size.
- In turbulence-free scenarios, TMO slightly outperforms the proposed method—this is expected because TMO leverages richer visual features.

## Highlights & Insights
- **Disentangling motion with epipolar geometry + temporal averaging** is the core contribution of this work—it exploits the fundamental difference between the statistical properties of turbulence motion (random, small-amplitude, cancelable) and real motion (persistent, large-amplitude, non-cancelable). This idea can be extended to other scenarios with global random perturbations (such as underwater imaging or rain/fog).
- **Fully unsupervised** is a major advantage—the method does not require any annotated data or pre-training. This is particularly important given that annotating turbulence video is extremely difficult.
- The contribution of the **DOST dataset** shouldn't be overlooked—it is the first real turbulence video dataset with ground-truth motion segmentation masks (38 videos, 1719 frames), filling a data gap in this field.

## Limitations & Future Work
- The inference speed is only 0.95 FPS because of the need to perform optical flow calculation, region growing, and Refine-Net training from scratch for each video—preventing real-time processing.
- The $\delta_{seed}$ threshold must be manually adjusted based on turbulence intensity (0.1 for normal, 0.3 for severe), lacking an automatic adaptation mechanism.
- Limited ability to separate overlapping moving objects—the non-overlapping constraint of region growing may lead to one object being assigned to another's mask.
- The method assumes a sufficient number of frames to stabilize optical flow (at least 4-5 frames), which may yield suboptimal results on extremely short videos.
- It does not utilize any appearance features (color, texture), relying solely on geometric motion information, which limits the detection of objects that are static but suddenly start moving.

## Related Work & Insights
- **vs TMO**: TMO prioritizes visual features over motion, performing well on normal videos but failing under turbulence because turbulence destroys appearance consistency; the proposed pure geometric method is more robust under turbulence.
- **vs Deformable Sprites**: Employs flow-guided grouping loss for segmentation, but its optical flow is unreliable under turbulence; the proposed method stabilizes the flow before usage.
- **vs SAM**: Large-scale pre-trained segmentation foundation models also fail under strong turbulence, demonstrating that turbulence is a model-agnostic, low-level challenge.
- **vs Traditional turbulence restoration methods**: Pipelines that restore first and segment later may introduce reconstruction errors; the proposed method segments directly on turbulence videos, avoiding cascaded errors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically address moving object segmentation under atmospheric turbulence; the epipolar geometry-based disentanglement is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Provides the first dedicated dataset DOST, with comprehensive comparisons and ablations.
- Writing Quality: ⭐⭐⭐⭐ The methodology pipeline is clear, with well-motivated modules.
- Value: ⭐⭐⭐⭐ Opens up a new direction for turbulence video understanding, and the dataset is a major contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DIMOS: Disentangling Instance-level Moving Object Segmentation](../../CVPR2026/segmentation/dimos_disentangling_instance-level_moving_object_segmentation.md)
- [\[ECCV 2024\] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[ECCV 2024\] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](un-evimo_unsupervised_event-based_independent_motion_segmentation.md)
- [\[ICML 2025\] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning](../../ICML2025/segmentation/unmore_unsupervised_multi-object_segmentation_via_center-boundary_reasoning.md)

</div>

<!-- RELATED:END -->
