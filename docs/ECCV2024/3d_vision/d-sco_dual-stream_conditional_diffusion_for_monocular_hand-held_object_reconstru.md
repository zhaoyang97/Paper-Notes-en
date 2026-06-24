---
title: >-
  [Paper Note] D-SCo: Dual-Stream Conditional Diffusion for Monocular Hand-Held Object Reconstruction
description: >-
  [ECCV2024][3D Vision][Hand-held Object Reconstruction] This work proposes D-SCo, a dual-stream conditional diffusion model for hand-held object point cloud reconstruction from a single RGB image. By combining unified hand-object semantic embeddings and hand-joint geometric embeddings, two branches provide semantic and geometric priors, respectively. Paired with a hand-constrained centroid-fixing strategy to stabilize the diffusion process, D-SCo achieves an F-5 score of 0.61…
tags:
  - "ECCV2024"
  - "3D Vision"
  - "Hand-held Object Reconstruction"
  - "Diffusion Model"
  - "Hand-Object Interaction"
  - "Dual-Stream Denoising"
  - "Centroid Fixing"
date: 2026-05-08
content_hash: d8368ad62951ad3e
---

# D-SCo: Dual-Stream Conditional Diffusion for Monocular Hand-Held Object Reconstruction

**Conference**: ECCV2024  
**arXiv**: [2311.14189](https://arxiv.org/abs/2311.14189)  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: Hand-held Object Reconstruction, Diffusion Model, Hand-Object Interaction, Dual-Stream Denoising, Centroid Fixing

## TL;DR
This work proposes D-SCo, a dual-stream conditional diffusion model for hand-held object point cloud reconstruction from a single RGB image. By combining unified hand-object semantic embeddings and hand-joint geometric embeddings, two branches provide semantic and geometric priors, respectively. Paired with a hand-constrained centroid-fixing strategy to stabilize the diffusion process, D-SCo achieves an F-5 score of 0.61 on ObMan (outperforming DDF-HO by 10.9%) and also leads significantly on real-world datasets such as HO3D/MOW.

## Background & Motivation

**Background**: Monocular hand-held 3D object reconstruction is highly challenging due to severe hand occlusion and object diversity. Existing approaches include SDF-based methods (gSDF), implicit representation methods (iHOI), and diffusion-based methods (DDF-HO).

**Limitations of Prior Work**: (a) Hand occlusion results in less than 50% of the object being visible; (b) Existing diffusion methods only condition on image features, failing to fully leverage hand geometry—the hand pose and joint positions actually tightly constrain the location and shape of the object; (c) Point cloud centroid drift during the diffusion process leads to unstable reconstruction.

**Key Challenge**: The hand is the source of occlusion but also serves as a strong spatial and geometric prior for the object—how to transform the hand from a "distractor" into a "helpful signal"?

**Goal**: (1) How to leverage hand pose to provide semantic and geometric constraints for object reconstruction? (2) How to stabilize the point cloud centroid during the diffusion process?

**Key Insight**: Hand-object interaction provides two levels of information: a semantic level (hand occlusion regions can be compensated by hand surface features) and a geometric level (hand joint coordinate systems provide multiple reference frames for the object), which are handled by two separate denoising branches.

**Core Idea**: A dual-stream conditional denoiser where the semantic stream processes the unified hand-object projection features, and the geometric stream processes the multi-coordinate transformation features based on hand joints. This is accompanied by a centroid prediction network to fix the point cloud center during the diffusion process.

## Method

### Overall Architecture
Input a single RGB image → Hand pose estimation (off-the-shelf) → Centroid prediction network to estimate the object center $\hat{\mathcal{M}}$ → Centroid-fixing diffusion (maintaining the centroid at $\hat{\mathcal{M}}$ during both forward noise addition and reverse denoising) → Dual-stream denoiser to fuse semantic and geometric conditions → Output object point cloud.

### Key Designs

1. **Centroid Fixing**:

    - **Function**: Specifically trains a centroid prediction network (PointNet + ResNet-18) to predict the 3D centroid of the object; the point cloud is recentered to this centroid at each diffusion step.
    - **Mechanism**: Forward process $X_0 \leftarrow X_0 - \bar{X}_0 + \hat{\mathcal{M}}$, zero-centering the noise $\epsilon \leftarrow \epsilon - \bar{\epsilon}$, and each reverse step $X_t \leftarrow X_t - \bar{X}_t + \hat{\mathcal{M}}$.
    - **Design Motivation**: Ablation results show that removing centroid fixing drops the F-5 score from 0.61 to 0.44 (-28%), and further to 0.32 without centroid prediction. Centroid drift is the core bottleneck in diffusion-based point cloud reconstruction.

2. **Unified Hand-Object Semantic Embedding ($X_t^{HO}$)**:

    - **Function**: Projects both the object point cloud and hand vertices into the image feature space, using one-hot encoding to distinguish between the hand and the object.
    - **Mechanism**: The projection operation $X_t^O = \pi(\mathcal{R}(X_t), \mathcal{F})$ extracts image features. After incorporating hand vertices, $X_t^{HO} \in \mathbb{R}^{(N+N_h) \times (C+1)}$.
    - **Design Motivation**: The hand-occluded object regions correspond to the pixels of the hand in the image. Incorporating the image features of hand vertices allows the denoiser to "see" semantic information in the regions occluded by the hand.

3. **Hand Joint Geometric Embedding ($X_t^A$)**:

    - **Function**: Transforms each object point into the local coordinate systems of 15 hand joints, obtaining $X_t^A \in \mathbb{R}^{N \times 45}$ (15 joints $\times$ 3D coordinates).
    - **Mechanism**: The relative coordinates of hand joints provide precise spatial relations of the object within the hand reference frames, serving as strong geometric constraints.
    - **Design Motivation**: When an object is grasped by a hand, the hand joint positions implicitly contain information about the object's size, shape, and grasp style.

4. **Dual-Stream Denoiser**:

    - Semantic branch: $f_\theta^1([X_t, X_t^{HO}]) \rightarrow \mathcal{F}_\theta^1$
    - Geometric branch: $f_\theta^2([X_t, X_t^A]) \rightarrow \mathcal{F}_\theta^2$
    - Fusion: $\epsilon_\theta = g_\theta([\mathcal{F}_\theta^1, \mathcal{F}_\theta^2])$

### Loss & Training
$\mathcal{L} = \mathcal{L}_{denoise} + \eta_1 \mathcal{L}_{mask}$, where the denoising loss is the standard $\|\epsilon - \epsilon_\theta\|$, and the mask loss is supervised via rendering consistency. The centroid network is trained independently: 3D centroid loss + 2D projection loss + projection consistency loss. The model is primarily trained on ObMan (synthetic data, 141K frames) and fine-tuned on HO3D/MOW/DexYCB (real-world data).

## Key Experimental Results

### Main Results

**ObMan (Synthetic)**:

| Method | F-5↑ | F-10↑ | CD(mm)↓ |
|------|------|-------|---------|
| iHOI | 0.42 | 0.63 | 1.02 |
| gSDF | 0.44 | 0.66 | — |
| DDF-HO | 0.55 | 0.67 | 0.14 |
| **D-SCo** | **0.61** | **0.81** | **0.11** |

**HO3D (Real, Fine-tuned)**:

| Method | F-5↑ | F-10↑ | CD↓ |
|------|------|-------|-----|
| DDF-HO | 0.27 | 0.40 | 0.86 |
| **D-SCo** | **0.41** | **0.63** | **0.34** |

On DexYCB, F-5 is 0.63 vs 0.44 for gSDF (+43%).

### Ablation Study

| Configuration | ObMan F-5 | HO3D F-5 |
|------|----------|----------|
| Full Model | 0.61 | 0.41 |
| w/o mask loss | 0.57 | 0.36 |
| w/o dual-stream (single-stream) | 0.54 | 0.34 |
| w/o all hand embeddings | 0.48 | 0.28 |
| w/o semantic embeddings | 0.51 | 0.33 |
| w/o geometric embeddings | 0.51 | 0.30 |
| w/o centroid fixing | 0.44 | 0.27 |
| w/o centroid prediction | 0.32 | 0.23 |

### Key Findings
- **Centroid fixing is the most critical component**: Removing it drops the F-5 score from 0.61 to 0.44 (-28%), proving that centroid stability is key in diffusion-based point cloud reconstruction.
- **Dual-stream outperforms single-stream**: The two branches provide complementary information. Fusing them achieves a 13% higher F-5 score compared to the single-stream counterpart.
- **Semantic and geometric embeddings are equally important**: Removing either yields similar performance drops (both to 0.51), while removing both drops the score to 0.48.
- **Robustness to occlusion**: The model maintains high performance under <50% visibility, outperforming iHOI and DDF-HO.
- **Oracle Sampling**: Taking the best out of 5 samples reaches F-5 = 0.67, indicating that the stochastic nature of the diffusion model can be leveraged via multi-sampling strategies.

## Highlights & Insights
- **Turning Hand from Distractor into Cue**: Traditional methods treat hand occlusion as a major bottleneck. D-SCo reformulates this by converting hand pose into a dual prior (semantic + geometric). This "turning adversity into advantage" design philosophy is highly instructive.
- **Simplicity and Effectiveness of Centroid Fixing**: Simply performing a mean subtraction and offset adjustment at each diffusion step incurs zero additional computation but delivers massive gains (+39% F-5). This technique can generalize to any diffusion-based 3D generation task.
- **Novel Multi-coordinate Geometric Encoding**: Transforming object points into 15 hand-joint coordinate systems acts as a "hand-aware positional encoding" for the point cloud, directly leveraging the kinematic structure of the human hand.

## Limitations & Future Work
- **Dependence on Hand Pose Accuracy**: Error in hand pose estimation will propagate to centroid prediction and geometric embeddings.
- **Constraining Only Centroid but Not Orientation**: The method does not exploit hand pose to infer the full 6DoF pose of the object (only 3DoF translation is handled).
- **Synthetic-to-Real Domain Gap**: Zero-shot transfer from ObMan training to HO3D performs significantly worse than fine-tuning (F-5 of 0.27 vs. 0.41).
- **Future Work**: (1) Incorporate hand orientation to constrain object pose; (2) Train on larger-scale real-world datasets; (3) Extend to bimanual or multi-object scenarios.

## Related Work & Insights
- **vs DDF-HO**: While also being a diffusion-based method, DDF-HO only uses image conditioning. By incorporating dual-stream hand conditioning, D-SCo increases F-5 from 0.55 to 0.61 on ObMan, and from 0.27 to 0.41 on real HO3D data.
- **vs gSDF**: As a deterministic SDF-based method, gSDF lacks generative diversity. D-SCo, being a probabilistic method, can further improve performance through multi-sampling (oracle 0.67).
- **vs iHOI**: Implicit methods require trade-offs between resolution and memory, whereas the point cloud representation of D-SCo is more flexible.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-stream conditional design and centroid-fixing strategy are highly creative, offering a unique way to integrate hand pose into a diffusion prior.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on synthetic and 3 real-world datasets, accompanied by detailed ablation studies, occlusion analysis, and Oracle experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological derivation and well-designed ablation configurations.
- Value: ⭐⭐⭐⭐ Establishes a new SOTA for hand-held object reconstruction. The centroid-fixing technique is easily generalizable to other diffusion-based 3D generation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](../../ICCV2025/3d_vision/hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](../../CVPR2026/3d_vision/arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[ICML 2026\] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](../../ICML2026/3d_vision/physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)
- [\[CVPR 2026\] ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos](../../CVPR2026/3d_vision/forehoi_feed-forward_3d_object_reconstruction_from_daily_hand-object_interaction.md)
- [\[ECCV 2024\] NOVUM: Neural Object Volumes for Robust Object Classification](novum_neural_object_volumes_for_robust_object_classification.md)

</div>

<!-- RELATED:END -->
