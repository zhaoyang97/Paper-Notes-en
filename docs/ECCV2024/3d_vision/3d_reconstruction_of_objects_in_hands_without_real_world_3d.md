---
title: >-
  [Paper Note] 3D Reconstruction of Objects in Hands without Real World 3D Supervision
description: >-
  [ECCV 2024][3D Vision][hand-object reconstruction] This paper proposes the HORSE framework, which trains an occupancy network to reconstruct the 3D shape of hand-held objects from a single RGB image. This is achieved by extracting multi-view 2D mask supervision from in-the-wild videos (using hand pose as an object pose proxy) and learning a 2D slice adversarial shape prior from a synthetic 3D shape collection. Without using any real-world 3D annotations…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "hand-object reconstruction"
  - "visual hull"
  - "shape prior"
  - "occupancy network"
  - "synthetic-to-real"
date: 2026-05-08
content_hash: b2ac3ba97212c038
---

# 3D Reconstruction of Objects in Hands without Real World 3D Supervision

**Conference**: ECCV 2024  
**arXiv**: [2305.03036](https://arxiv.org/abs/2305.03036)  
**Code**: None (project page available)  
**Area**: 3D Vision / Hand-held Object Reconstruction / Weakly Supervised Learning  
**Keywords**: hand-object reconstruction, visual hull, shape prior, occupancy network, synthetic-to-real  

## TL;DR
This paper proposes the HORSE framework, which trains an occupancy network to reconstruct the 3D shape of hand-held objects from a single RGB image. This is achieved by extracting multi-view 2D mask supervision from in-the-wild videos (using hand pose as an object pose proxy) and learning a 2D slice adversarial shape prior from a synthetic 3D shape collection. Without using any real-world 3D annotations, it outperforms 3D-supervised methods by 11.6% on the MOW dataset.

## Background & Motivation
The 3D reconstruction of hand-held objects is crucial for AR/VR and robot learning, but existing methods rely on paired image and 3D shape data for training. Collecting such real-world paired data is extremely challenging: visual scanning devices require the object to be fully visible (which is violated under hand occlusion), synthesizing realistic hand-object interactions remains an open problem, and manual alignment of template shapes is both expensive and imprecise. Consequently, existing datasets suffer from a limited variety of objects (e.g., only 10 classes in HO3D and 20 classes in DexYCB), leading to poor generalization of trained models when encountering novel objects in-the-wild.

However, a massive number of in-the-wild videos exhibiting hand-object interactions exist on the internet (e.g., EPIC-Kitchens, Ego4D), alongside large-scale synthetic 3D shape collections (e.g., ShapeNet). Both types of data have their limitations: videos feature realistic interactions but lack 3D annotations, while shape collections provide 3D geometry but lack authentic hand grasps. The core motivation lies in how to organically fuse and utilize these two types of indirect 3D cues to break through the bottleneck of 3D supervision.

## Core Problem
How to train a model to reconstruct 3D shapes of hand-held objects from a single RGB image without real-world 3D supervision? Two key sub-problems need to be addressed: (1) How to extract effective 3D learning signals from in-the-wild videos without 3D annotations? (2) How to compensate for the shape incompleteness caused by hand occlusion?

## Method

### Overall Architecture
Input: Single RGB image $\rightarrow$ FrankMocap estimates hand pose parameters (joint pose $\theta_a$, global rotation $\theta_w$, weak perspective camera $\theta_c$) $\rightarrow$ ResNet-50 extracts global features and pixel-aligned local features $\rightarrow$ Composed with 3D point representations in the hand joint coordinate system $\rightarrow$ Occupancy network $\mathcal{F}$ predicts the occupancy value of each 3D query point $\rightarrow$ Marching Cubes extracts the mesh.

The training phase jointly utilizes three signal sources: VISOR in-the-wild videos providing multi-view 2D mask supervision, and the ObMan synthetic dataset providing 3D shape priors and direct 3D supervision.

### Key Designs
1. **2D Mask-guided 3D Sampling (Visual Hull Supervision)**: Leverages multi-view frames $\{I_1,...,I_n\}$ of the same object and their segmentation masks $\{M_1,...,M_n\}$ from VISOR videos. The crucial assumption is that when a person grasps a rigid object, the hand and the object move rigidly together, meaning that **the hand pose can serve as a proxy for the object pose**. FrankMocap is used to estimate the 6DoF hand pose of each frame to register different views. 3D points are sampled in the hand coordinate system. If a point projects inside the object masks in all views (and is not inside the hand mask or part of the MANO hand vertices), it is labeled as occupied; otherwise, it is labeled as unoccupied. A rejection sampling strategy (up to 50 rounds) is adopted to ensure a balance of positive and negative samples. This is essentially a visual hull algorithm used during training, while requiring only a single image at test times.

2. **2D-Slice 3D Discriminator (Shape Prior)**: Visual hulls cannot recover parts occluded by the hand. To address this, an adversarial training framework is introduced, where a discriminator $\mathcal{D}$ is trained using 3D shapes of over 2500 hand-held objects from ObMan. The key innovation is using **2D slices instead of 3D voxels** as input to the discriminator: a 2D plane passing through the origin is sampled in the hand coordinate system, randomly rotated to be non-axis-aligned, and uniformly sampled on the plane to obtain occupancy values that form a 2D cross-section image. The benefits of this approach are: (a) high computational efficiency (no need to sample $64^3$ points), and (b) exposure to different random slices during training allows learning fine-grained shape distinctions (e.g., cross-sections of a sphere are always circular, while those of a cylinder are mostly elliptical).

3. **Wild Objects in Hands Dataset Construction**: Filters rigid object interaction segments with hand-object contact from the VISOR dataset. First, 14,768 trajectories are selected based on contact annotations. Manual filtering is then applied to retain 604 videos of single-hand rigid object manipulations. Hand pose estimation inaccuracies are automatically filtered out using the uncertainty estimation of FrankMocap (standard deviation across 5 translation-augmented versions). This yields 473 videos covering 144 object classes, which is 4 times larger in scale than existing datasets.

### Loss & Training
Total loss: $\mathcal{L}_{\mathcal{F}} = \lambda_v \mathcal{L}_{\text{visual-hull}} + \lambda_c \mathcal{L}_{\text{consistency}} + \lambda_f \mathcal{L}_{\text{adv}}^{\mathcal{F}}$

- $\mathcal{L}_{\text{visual-hull}}$: Cross-entropy loss, supervised by the occupancy labels generated from multi-view masks.
- $\mathcal{L}_{\text{consistency}}$: Cross-view consistency loss, minimizing the prediction discrepancy for the same 3D point across different view inputs.
- $\mathcal{L}_{\text{adv}}$: Adversarial loss in LSGAN form, where the discriminator distinguishes between ObMan ground-truth occupancy slices and predicted occupancy slices.

Training setup: Pre-trained on ObMan first, followed by joint training with an ObMan:VISOR ratio of 1:2, batch size = 64, learning rate = 1e-5 on 4×A40 GPUs, with $\lambda_v=\lambda_c=1$, $\lambda_f=\lambda_d=0.25$, and alternating training between $\mathcal{F}$ and $\mathcal{D}$ ($\mathcal{F}$ trained for 2 steps, $\mathcal{D}$ for 1 step).

## Key Experimental Results

| Dataset | Method | F@5↑ | F@10↑ | CD(mm)↓ |
|--------|------|------|-------|---------|
| MOW (Object Gen.) | AC-OCC (ObMan 3D) | 0.095 | 0.179 | 8.69 |
| MOW (Object Gen.)| AC-SDF (ObMan 3D) | 0.108 | 0.199 | 7.82 |
| MOW (Object Gen.)| AC-SDF (ObMan+HO3D+HOI4D 3D) | 0.095 | 0.193 | 7.43 |
| MOW (Object Gen.)| DDFHO | 0.094 | 0.166 | 3.06 |
| MOW (Object Gen.)| **HORSE (Ours)** | **0.121** | **0.220** | 6.76 |
| HO3D (Object Gen.)| AC-SDF (ObMan) | 0.17 | 0.33 | 3.72 |
| HO3D (Object Gen.)| **HORSE (Ours)** | **0.20** | **0.35** | **3.39** |
| HO3D (View Gen.) | GF (HO3D 3D) | 0.12 | 0.24 | 4.96 |
| HO3D (View Gen.) | **HORSE (Ours)** | **0.23** | **0.43** | **1.41** |

### Ablation Study
- **Contribution of Loss Functions**: $\mathcal{L}_{\text{visual-hull}}$ is the most critical component (alone it improves F@5 from 0.095 to 0.111), $\mathcal{L}_{\text{consistency}}$ alone has a limited effect but yields a significant boost when combined with the visual hull (0.117), and $\mathcal{L}_{\text{shape-prior}}$ further enhances the performance (0.121).
- **2D Slice vs. 3D Voxel Discriminator**: 2D slice (F@5=0.121) outperforms all 3D voxel resolutions ($10^3\rightarrow32^3$, with F@5 dropping from 0.120 to 0.104). Denser 3D sampling actually degrades performance because a large fraction of points fall outside the object.
- **Sampling Strategy**: Rejection sampling (0.117) far outperforms uniform sampling (0.093); incorporating negative samples from the hand mask is beneficial (0.117 vs. 0.113).
- **Hand Pose Filtering**: Automatically filtering out low-quality poses is crucial (improves F@5 from 0.213 to 0.234 after filtering), while using ground truth hand poses offers only a slight additional boost (0.243).
- **Occupancy vs. SDF**: Under weakly supervised settings, occupancy is more stable than SDF (continuous-value regression of SDF is unstable under weak supervision).
- **Performance Drop with More 3D Supervision**: Incorporating 3D annotations from HO3D/HOI4D into AC-SDF degrades generalization on MOW. This is because the limited variety of objects in these datasets triggers overfitting.

## Highlights & Insights
- **Using hand pose as a proxy for object pose** is a highly clever insight. By exploiting the rigid grasp hypothesis, it transforms scenarios where SfM fails (due to scarce object feature points) into a solved problem (hand pose estimation), which is simple yet effective.
- The **2D slice discriminator** avoids the computational overhead and sampling imbalance issues of a 3D voxel discriminator, while achieving stronger fine-grained shape discrimination capabilities through randomly rotated slice planes.
- Proves a counter-intuitive conclusion: **adding limited lab-controlled 3D annotation data is less effective than increasing highly diverse, weakly supervised in-the-wild data**. The diversity advantage of 144 classes versus fewer than 20 classes overrides the drawback of lower supervision quality.
- The entire pipeline requires no real-world 3D annotations, significantly lowering the barrier for data collection.

## Limitations & Future Work
- **Inaccurate hand poses prevent sampling points from fully covering the object**, leading to missing regions. Future work could jointly optimize hand pose and object shape.
- **In-the-wild videos typically do not provide a 360° view**. The visual hull can only cover visible perspectives, relying on shape priors for reconstructing the back side.
- **The rigidity assumption limits its scope**: Non-rigid objects (e.g., towels, dough) or in-hand manipulation scenarios are not supported.
- The shape prior originates from ObMan's synthetic object set, which might lack priors for uncommon shapes; training on larger-scale 3D datasets (such as Objaverse) could mitigate this.
- At test time, the system still heavily relies on FrankMocap to accurately estimate hand poses and camera parameters.

## Related Work & Insights
- **vs. AC-SDF [Ye et al. CVPR 2022]**: AC-SDF trains an SDF network using paired 3D supervision, whereas HORSE employs occupancy and indirect 2D supervision. HORSE systematically outperforms AC-SDF in object generalization (even though AC-SDF uses more 3D annotated datasets); the key differentiator lies in the diversity of training data rather than the precision of supervision.
- **vs. DDFHO [Zhang et al. NeurIPS 2023]**: DDFHO outperforms HORSE in the CD metric (3.06 vs. 6.76) but yields a lower F-score (0.094 vs. 0.121). DDFHO utilizes a conditional directional distance field, which concentrates reconstruction precision but lacks sufficient coverage.
- **vs. HO/GF [Hasson/Karunratanakul]**: These methods are trained on HO3D using full 3D supervision, yet HORSE outperforms them without requiring HO3D's 3D annotations, thanks to stronger feature representations (pixel-aligned + hand joint features) and more diversified training data.

## Insights & Connections
- The concept of the 2D slice discriminator can be generalized to other scenarios requiring 3D shape priors (e.g., NeRF regularization, 3DGS shape constraints).
- The core idea of "using hand pose as a proxy for object pose" can be extended to other human-object interaction scenarios, such as inferring the placement of contacted objects using body pose in full-body pose estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ The hand pose proxy and 2D slice discriminator represent clever engineering innovations, though the core ideas (visual hull + GAN prior) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation study comprehensively covers individual design choices, and the multiple evaluation protocols (object/view generalization) are convincing.
- Writing Quality: ⭐⭐⭐⭐ The logic is precise, the motivation is fully articulated, and the illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Offers a practical framework for weakly supervised hand-held object reconstruction, though it is inferior to DDFHO in the CD metric, and practical deployment remains dependent on the accuracy of the hand pose estimator.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians](reconstruction_and_simulation_of_elastic_objects_with_spring-mass_3d_gaussians.md)
- [\[CVPR 2026\] 2D-LFM: Lifting Foundation Model without 3D Supervision](../../CVPR2026/3d_vision/2d-lfm_lifting_foundation_model_without_3d_supervision.md)
- [\[CVPR 2026\] ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects](../../CVPR2026/3d_vision/ictpolarreal_a_polarized_reflection_and_material_dataset_of_real_world_objects.md)
- [\[ECCV 2024\] PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects](pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)
- [\[CVPR 2026\] JRM: Joint Reconstruction Model for Multiple Objects without Alignment](../../CVPR2026/3d_vision/jrm_joint_reconstruction_model_for_multiple_objects_without_alignment.md)

</div>

<!-- RELATED:END -->
