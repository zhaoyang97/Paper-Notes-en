---
title: >-
  [Paper Note] CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][4D Gaussian Splatting] CoDa-4DGS augments the 4D Gaussian Splatting (4DGS) framework with context awareness (self-supervised 4D semantic features from 2D foundation models) and temporal deformation awareness (tracking per-Gaussian deformation between adjacent frames). By jointly encoding semantic and deformation features as dynamic compensation cues for each Gaussian, the method captures finer-grained details in autonomous driving dynamic scene…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "4D Gaussian Splatting"
  - "dynamic scene rendering"
  - "semantic self-supervision"
  - "temporal deformation tracking"
  - "autonomous driving simulation"
date: 2026-05-08
content_hash: 03e0a7b3330c09dd
---

# CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2503.06744](https://arxiv.org/abs/2503.06744)  
**Code**: N/A  
**Area**: Autonomous Driving / 3D Reconstruction
**Keywords**: 4D Gaussian Splatting, dynamic scene rendering, semantic self-supervision, temporal deformation tracking, autonomous driving simulation

## TL;DR

CoDa-4DGS augments the 4D Gaussian Splatting (4DGS) framework with context awareness (self-supervised 4D semantic features from 2D foundation models) and temporal deformation awareness (tracking per-Gaussian deformation between adjacent frames). By jointly encoding semantic and deformation features as dynamic compensation cues for each Gaussian, the method captures finer-grained details in autonomous driving dynamic scenes and surpasses existing self-supervised approaches.

## Background & Motivation

**Background**: Dynamic scene rendering is a critical technology for autonomous driving simulation—by reconstructing realistic driving scenes, closed-loop simulation can be used to test end-to-end autonomous driving algorithms, avoiding costly and dangerous real-vehicle testing. 3D Gaussian Splatting (3DGS) has achieved remarkable success in static scene reconstruction due to its high-quality rendering and real-time performance, and 4D Gaussian Splatting (4DGS) extends it to dynamic scenes by allowing each Gaussian to evolve over time.

**Limitations of Prior Work**: The highly dynamic nature of traffic scenes poses severe challenges for 4DGS—scenes contain numerous moving objects (vehicles, pedestrians, cyclists) with complex and varied motion patterns (acceleration, turning, sudden stops) and frequent occlusion changes. Existing 4DGS methods typically model temporal variation of all Gaussians via a single global deformation field, which struggles to distinguish between "dynamic objects requiring motion compensation" and "static background," resulting in blurring and artifacts on dynamic objects while unnecessarily deforming static backgrounds.

**Key Challenge**: The deformation field in 4DGS requires knowledge of "which Gaussians are dynamic" and "how they should move," yet purely geometric information is insufficient to provide such priors. Gaussians lack semantic understanding in 3D space—they cannot determine whether they belong to a "vehicle" or the "road," and therefore cannot adaptively decide whether deformation is needed or in which direction.

**Goal**: (1) Inject semantic context information into each Gaussian so that it "knows" what semantic category it represents; (2) explicitly track per-Gaussian deformation between adjacent frames to provide motion priors; (3) jointly leverage semantic and deformation cues to enhance dynamic scene rendering quality.

**Key Insight**: The authors observe that 2D semantic segmentation foundation models (e.g., SAM, SegFormer) are highly mature and can provide reliable semantic signals. By "lifting" 2D semantics into 4D Gaussian space, each Gaussian gains the ability to distinguish dynamic from static regions. Meanwhile, changes in Gaussian positions between adjacent frames directly reflect motion information and can serve as explicit cues for deformation compensation.

**Core Idea**: Enhance 4DGS with a dual-awareness mechanism of "2D semantic self-supervision + temporal deformation tracking," equipping each Gaussian with both contextual semantic and motion deformation information to achieve more accurate dynamic scene rendering.

## Method

### Overall Architecture

CoDa-4DGS is built upon the standard 4DGS framework. Each 3D Gaussian, in addition to its standard attributes (position, rotation, scale, opacity, spherical harmonics coefficients), is associated with a semantic feature vector and a deformation feature vector. During training: (1) a 2D semantic segmentation model generates pseudo-labels for each frame, which are back-propagated to the Gaussians' semantic features via differentiable rendering (self-supervision); (2) per-Gaussian deformation features are obtained by tracking positional changes between adjacent frames; (3) the encoded semantic and deformation features are fed into a deformation prediction network to predict temporal deformation compensation (displacement, rotation, and scale changes) for each Gaussian. At rendering time, the standard 4DGS rendering pipeline performs alpha blending based on the compensated Gaussian attributes to produce the final image.

### Key Designs

1. **Self-Supervised 4D Semantic Features from 2D Foundation Models**:

    - Function: Assigns semantic context information to each 3D Gaussian, enabling it to "know" the object category it represents.
    - Mechanism: At each training frame, a pretrained 2D semantic segmentation model (e.g., SegFormer or Mask2Former) segments the RGB image to produce pixel-level semantic labels or feature maps. A differentiable Gaussian renderer then renders the 3D Gaussians' semantic features into a 2D semantic map, which is compared against the 2D model's output via a loss (cross-entropy or feature distance) and back-propagated. After multi-frame, multi-view training, each Gaussian gradually acquires consistent semantic features—Gaussians belonging to "vehicles" share similar semantic vectors, clearly distinguished from those of "road." This self-supervised approach requires no 3D semantic annotations.
    - Design Motivation: Purely geometric 4DGS cannot distinguish dynamic from static objects, causing the deformation field to treat all Gaussians equally. By injecting semantic information through self-supervision, the deformation network can learn priors such as "Gaussians of the vehicle type require more deformation compensation while road-type Gaussians should remain static."

2. **Temporal Deformation Tracking**:

    - Function: Explicitly tracks the positional and attribute changes of each Gaussian between adjacent temporal frames to provide motion priors.
    - Mechanism: For each Gaussian, the displacement vector $\Delta p = p_{t+1} - p_t$, rotation change $\Delta q$, and scale change $\Delta s$ between frames $t$ and $t+1$ are computed. These deformation quantities directly reflect the motion pattern of the region represented by the Gaussian—static background exhibits near-zero deformation, uniformly moving vehicles show relatively constant deformation, and turning or accelerating objects exhibit varying deformation magnitudes. Deformation sequences across multiple frames are encoded into a deformation feature vector $f_{def}$ that captures the motion pattern of each Gaussian.
    - Design Motivation: A global deformation field is an implicit function that struggles to explicitly perceive local motion patterns. Explicit deformation tracking provides direct motion cues—if a Gaussian has been moving leftward over the past several frames, predicting continued leftward motion in the next frame is a reasonable prior. Such explicit motion priors can substantially reduce the difficulty of deformation prediction.

3. **Joint Semantic-Deformation Encoding and Deformation Compensation Network**:

    - Function: Fuses semantic context and motion deformation information to predict accurate deformation compensation for each Gaussian.
    - Mechanism: The semantic feature $f_{sem}$ and deformation feature $f_{def}$ of each Gaussian are concatenated or fused via an attention mechanism to form a joint representation $f_{joint} = \text{Encode}([f_{sem}; f_{def}])$. This joint representation, together with timestamp $t$, is fed into a deformation compensation network (a compact MLP) to predict the displacement compensation $\delta p$, rotation compensation $\delta q$, and scale compensation $\delta s$ at time $t$. The compensated Gaussian attributes are then used for rendering. The network is trained end-to-end and optimized via back-propagation through a rendering loss (L1 + SSIM).
    - Design Motivation: Semantic information alone can only provide coarse-grained judgments about "whether deformation is needed," while deformation tracking alone may fail under occlusion or for newly appearing objects. Joint encoding enables the network to leverage comprehensive information such as "this Gaussian is a vehicle moving at high speed" to predict more accurate deformation, with the two types of information providing complementary benefits.

### Loss & Training

The training loss comprises: (1) a rendering reconstruction loss $\mathcal{L}_{rgb} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{SSIM}$; (2) a semantic self-supervision loss $\mathcal{L}_{sem}$—the distance between the rendered 2D semantic map and the foundation model's output; and (3) a deformation regularization loss $\mathcal{L}_{reg}$—encouraging near-zero deformation in static regions. A progressive training strategy is adopted: static scene reconstruction is trained first, followed by the gradual introduction of dynamic deformation modeling.

## Key Experimental Results

### Main Results

Novel view synthesis (NVS) quality is evaluated on dynamic driving scenes from the nuScenes and Waymo datasets:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Dynamic PSNR ↑ | Type |
|--------|--------|--------|---------|----------------|------|
| 3DGS (static) | 25.3 | 0.842 | 0.185 | 18.2 | Static |
| D-3DGS | 27.1 | 0.871 | 0.154 | 21.5 | Global deformation |
| 4DGS | 27.8 | 0.883 | 0.142 | 22.3 | Time-conditioned |
| Street Gaussians | 28.5 | 0.891 | 0.131 | 23.8 | Supervised decomposition |
| HUGS | 28.2 | 0.887 | 0.136 | 23.1 | Supervised decomposition |
| **CoDa-4DGS** | **29.4** | **0.905** | **0.118** | **25.6** | **Self-supervised** |

Notably, CoDa-4DGS, as a self-supervised method, surpasses several methods that rely on supervised signals such as 3D bounding boxes.

### Ablation Study

| Configuration | PSNR ↑ | Dynamic PSNR ↑ | Note |
|---------------|--------|----------------|------|
| Full CoDa-4DGS | **29.4** | **25.6** | Complete model |
| w/o semantic self-supervision | 28.1 | 22.8 | Deformation field lacks semantic prior |
| w/o temporal deformation tracking | 28.5 | 23.4 | Lacks explicit motion cues |
| w/o joint encoding (semantic only) | 28.8 | 24.1 | Limited contribution from semantics alone |
| w/o joint encoding (deformation only) | 28.6 | 23.7 | Deformation alone inferior to joint |
| w/o deformation regularization | 28.9 | 24.8 | Slight jitter in background |

### Key Findings

- **Semantic self-supervision contributes the most** (removing it causes a 2.8 dB drop in dynamic region PSNR), confirming that semantic context is critical for distinguishing dynamic from static regions.
- **Joint encoding of both information types outperforms either alone**: the combined effect of semantics + deformation (25.6 dB) significantly exceeds using semantics alone (24.1 dB) or deformation alone (23.7 dB).
- **Improvements are especially pronounced in dynamic regions**—overall scene PSNR improves by ~1.6 dB, while dynamic region PSNR improves by 3.3 dB, demonstrating that the method precisely addresses the dynamic object rendering problem.
- **The self-supervised method surpasses some supervised counterparts** (e.g., HUGS), indicating that self-supervised signals from semantic foundation models are already of sufficient quality.
- **Semantic features can deform together with the Gaussians**, naturally enabling CoDa-4DGS to support downstream applications such as 4D semantic segmentation.

## Highlights & Insights

- **Self-supervised semantic injection** is an elegant design—leveraging existing 2D foundation models as "free" semantic teachers and back-propagating into 3D space via differentiable rendering, without any 3D annotations. This paradigm is transferable to any scene reconstruction task requiring 3D semantics.
- **The dual awareness of semantics and deformation** is intuitively well-motivated—analogous to how humans understand dynamic scenes by simultaneously leveraging "what is this object" and "how is it moving." This multi-modal prior fusion paradigm can be generalized to other spatiotemporal modeling tasks.
- **An additional 4D semantic capability** emerges as an elegant by-product—the system can output semantic segmentation maps and depth maps alongside rendered images, providing rich environmental understanding signals for downstream simulation and planning tasks.

## Limitations & Future Work

- The segmentation quality of the semantic foundation model directly affects the self-supervised signal and may degrade in extreme conditions (nighttime, heavy rain).
- Temporal deformation tracking may fail for newly appearing objects (just entering the field of view) or objects that reappear after prolonged occlusion.
- Computational overhead is higher than standard 4DGS—additional semantic features and deformation encoding increase memory and computation requirements.
- Evaluation is limited to single-object motion patterns; performance under complex multi-object interactions (e.g., collisions, overtaking) is not thoroughly validated.
- Incorporating scene flow or optical flow as additional deformation supervision could further improve deformation accuracy for dynamic objects.

## Related Work & Insights

- **vs. Street Gaussians**: Street Gaussians explicitly decomposes scenes into foreground and background using 3D detection boxes, requiring 3D annotations. CoDa-4DGS achieves a similar decomposition effect implicitly through semantic self-supervision, offering greater generality at the potential cost of precision compared to explicit decomposition.
- **vs. HUGS**: HUGS also separates foreground and background processing but relies on predefined motion models (e.g., rigid body transformations). CoDa-4DGS's deformation prediction is more flexible and can handle non-rigid motion.
- **vs. Gaussian Grouping**: Gaussian Grouping also introduces semantics into 3DGS, but targets grouping and editing in static scenes. CoDa-4DGS combines semantics with temporal deformation, specifically targeting dynamic scene rendering.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of semantic self-supervision and deformation tracking is novel within the 4DGS domain, though each individual component has prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons and ablations on mainstream datasets; dedicated evaluation on dynamic regions is convincing.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the method pipeline is easy to follow.
- Value: ⭐⭐⭐⭐ Practically significant for autonomous driving simulation; the self-supervised paradigm reduces annotation dependency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dopegs_online_6d_object_pose_estimation_using_gaussian_spla.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
