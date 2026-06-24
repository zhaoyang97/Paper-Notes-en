---
title: >-
  [Paper Note] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding
description: >-
  [CVPR 2025][Human Understanding][3D Gaze Estimation] This paper proposes the GA3CE method, which encodes the subject's 3D pose and scene object locations into a subject-centric egocentric space, and designs a direction-distance-decomposed D3 position encoding. This allows a Transformer to learn the spatial relationships between the 3D gaze direction and the scene context, reducing the 3D gaze angular error by 13%–37% under unconstrained settings.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "3D Gaze Estimation"
  - "Egocentric Transformation"
  - "Position Encoding"
  - "Scene Understanding"
  - "Transformer"
date: 2026-05-08
content_hash: 3bc37061868cac3a
---

# GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding

**Conference**: CVPR 2025  
**arXiv**: [2505.10671](https://arxiv.org/abs/2505.10671)  
**Code**: [https://woven-visionai.github.io/ga3ce-project](https://woven-visionai.github.io/ga3ce-project) (Project Page)  
**Area**: Human Understanding  
**Keywords**: 3D Gaze Estimation, Egocentric Transformation, Position Encoding, Scene Understanding, Transformer

## TL;DR
This paper proposes the GA3CE method, which encodes the subject's 3D pose and scene object locations into a subject-centric egocentric space, and designs a direction-distance-decomposed D3 position encoding. This allows a Transformer to learn the spatial relationships between the 3D gaze direction and the scene context, reducing the 3D gaze angular error by 13%–37% under unconstrained settings.

## Background & Motivation

1. **Background**: 3D gaze estimation has important applications in scenarios such as surveillance and retail analysis. Existing methods typically rely on 2D appearance features or make limited use of depth info in post-processing to constrain the gaze direction.

2. **Limitations of Prior Work**: Clear eye features are unavailable when the subject is far from the camera or facing away. Existing methods either ignore scene spatial relationships (such as GAFA) or only utilize depth in non-learnable post-processing steps (such as GFIE), failing to learn the spatial relationship among the subject, scene, and gaze in an end-to-end manner.

3. **Key Challenge**: Estimating 3D gaze direction from 2D observations is inherently ill-posed—the same 3D scene produces vastly different 2D appearances and 3D gaze directions under different camera poses, which increases the learning difficulty. No prior method has systematically addressed the complexity brought by such camera pose variations.

4. **Goal** (three sub-problems): (i) What are effective representations for the subject and the scene? (ii) How can the variation caused by different camera poses be eliminated? (iii) How can the spatial relationship among the subject, scene, and gaze be modeled?

5. **Key Insight**: Human vision research shows that the direction and distance of objects in the subject's field of view strongly influence gaze behavior. Inspired by this, the authors align the 3D context to a subject-centric egocentric space and encode direction and distance in a decomposed manner.

6. **Core Idea**: Use 3D poses and object locations as intermediate representations, eliminate camera pose variations through egocentric transformation, and then use direction-distance-decomposed position encoding in a Transformer to learn spatial relationships for predicting the 3D gaze direction.

## Method

### Overall Architecture
The input consists of an RGB image, a depth map, and camera intrinsics, and the output is the subject's 3D gaze direction (a unit vector). The method consists of three steps: (1) Use pre-trained models to extract 3D human pose keypoints and scene object 3D positions as intermediate representations; (2) Transform these 3D contexts into the egocentric space and encode them using GA3CE (Gaze-Aware 3D Context Encoding); (3) Use a Transformer encoder-decoder to learn the spatial relationships between objects and output the residual gaze direction, which is finally inversely transformed to obtain the 3D gaze direction.

### Key Designs

1. **Subject & Object Representation**:

    - **Function**: Abstraction of the subject and the scene from 2D images to 3D intermediate representations.
    - **Mechanism**: For the subject representation, a pre-trained 3D pose estimator (MotionBERT) is used to extract $N_{pose}=15$ 3D keypoints $P_{pose}$ from the cropped full-body image, combined with the gaze direction $\mathbf{v}$ predicted by a head appearance estimator as a prior. For the scene representation, MobileSAM is used in a "segment-everything" manner to obtain all object instance masks, which are then back-projected to the 3D space using the depth map and camera intrinsics, taking the median to obtain the object 3D positions $P_{object}$.
    - **Design Motivation**: Replacing 2D appearance with a 3D representation avoids the issue of drastic 2D appearance changes under different camera angles, serving as the foundation for subsequent geometric normalization. Using SAM enables class-agnostic object detection without requiring predefined object categories.

2. **Egocentric Transformation**:

    - **Function**: Normalizing all 3D contexts into a unified coordinate system centered at the subject's head, with the gaze direction aligned to the z-axis.
    - **Mechanism**: The pose and object locations are first translated to have the head position at the origin, and then rotated to align the gaze direction $\mathbf{v}$ with $\mathbf{z}=[0,0,1]$. To maintain rotation consistency, a cyclotorsion rotation is designed. Inspired by the eye's counter-rolling motion, it constrains the rotation matrix $R=\text{Euler}(\theta,\phi,0)$ so that the horizontal axis remains horizontal, avoiding the z-axis inconsistency caused by simple axis-angle rotations. After the transformation: $P'_{pose}=sR(P_{pose}-\mathbf{t}_{pose})$, $P'_{object}=R(P_{object}-\mathbf{t}_{object})$.
    - **Design Motivation**: Geometric normalization eliminates the variations in 2D representations caused by camera pose changes, enabling the network to focus on learning simpler spatial relationships in the egocentric space, which dramatically reduces the learning difficulty.

3. **Direction-Distance-Decomposed PE (D3 PE)**:

    - **Function**: Encoding 3D points into high-dimensional features that simultaneously capture direction and distance similarities.
    - **Mechanism**: For a 3D point $\mathbf{p}$, it is decomposed into a direction component $\mathbf{p}/\|\mathbf{p}\|$ and a distance component $\|\mathbf{p}\|$, then concatenated after applying sinusoidal position encoding to each: $\tilde{\gamma}(\mathbf{p})=\gamma(\mathbf{p}/\|\mathbf{p}\|)\oplus\gamma(\|\mathbf{p}\|)$. Unlike standard position encodings that only have high similarity near the reference point, D3 encoding gradually increases similarity along the direction from the origin to the reference point (forming a radial gaze pattern), which aligns better with gaze behavior characteristics.
    - **Design Motivation**: Human gaze fixation is typically concentrated near the center of the field of view, and direction and distance are the key factors influencing gaze. D3 encoding explicitly captures information in these two dimensions, making it more suitable than standard PE for gaze estimation tasks.

### Loss & Training
The loss function is the angular error between the predicted gaze direction and the ground truth: $\mathcal{L}=\arccos(\mathbf{g}^T\mathbf{g}_{GT})$. The AdamW optimizer is used with a learning rate of 0.0014, and the model is trained for 20 epochs on a single A10G GPU. The weights of SAM, the pose estimator, and the gaze direction estimator are frozen during training. The Transformer encoder and decoder both have 3 layers.

## Key Experimental Results

### Main Results

| Dataset | Metric (3D MAE ↓) | GA3CE | GFIE (Prev. SOTA) | Gain |
|--------|--------------|-------|-----------------|------|
| GFIE | 3D MAE (°) | **11.1** | 17.7 | **37%** |
| GFIE (Zero-Shot Depth) | 3D MAE (°) | 12.3 | 17.7 | 31% |
| GFIE + GFM | 3D MAE (°) | **10.6** | 16.4 | 35% |
| CAD-120 (Cross-Domain) | 3D MAE (°) | **25.2** | 27.3 | 8% |
| CAD-120 + GFM | 3D MAE (°) | **15.8** | 19.8 | 20% |
| GAFA (Full Scene Average) | 3D MAE (°) | **19.9** | 22.9 | **13%** |

### Ablation Study

| Configuration | GFIE 3D MAE | GAFA 3D MAE | Description |
|------|-------------|-------------|------|
| Appearance only | 19.4 | 22.9 | Appearance only (Baseline) |
| + Pose | 13.1 | 20.3 | Add 3D Pose |
| + Pose + Object | **11.1** | **19.9** | Full Model |
| w/o ECT (No Egocentric Transformation) | 15.3 | - | Transformation contributes significantly |
| Standard PE (Replacing D3 PE) | 12.5 | - | D3 PE is effective |

### Key Findings
- 3D pose is the most important context, decreasing the error from 19.4° to 13.1° (GFIE) and contributing the most.
- Object positions contribute significantly to performance on the GFIE dataset (13.1° $\to$ 11.1°), as the subject frequently interacts with objects in these scenes. It contributes less on GAFA (20.3° $\to$ 19.9°), because GAFA contains fewer subject-object interactions.
- Replacing the real depth sensor with a zero-shot depth estimator (Depth Anything) leads to a performance drop of only 1.2°, demonstrating the method's robustness to depth quality.
- Visualization of the Transformer decoder's attention indicates that objects closer to the ground-truth gaze target receive the highest attention weights.

## Highlights & Insights
- **Geometric Normalization via Egocentric Transformation**: Camera pose variation is eliminated using geometric priors instead of learning, which is simple, efficient, and transferable to any 3D understanding task requiring face/viewer-perspective inputs.
- **D3 Position Encoding**: The direction-distance-decomposed design elegantly integrates human vision characteristics into feature encoding, generating a radial gaze pattern naturally suited for gaze tasks. This decomposition philosophy can be extended to other tasks requiring spatial relationship modeling (e.g., pointing estimation, attention prediction).
- **No 3DMM or Extra Annotation Required**: Automated 3D context extraction is achieved using SAM and pre-trained pose estimators, lowering the deployment barrier.
- **Cyclotorsion rotation** is an elegant engineering design that solves the geometric rotation inconsistency issue during gaze alignment.

## Limitations & Future Work
- Currently, only object positions are utilized without exploiting object appearance features (e.g., CLIP semantics). Incorporating semantic information could yield further improvements.
- The method depends on the quality of pre-trained models (SAM, pose estimator) and may degrade under extreme occlusion or rare poses.
- Single-frame setup; temporal information is not utilized. The temporal version of GAFA has shown that temporal sequence helps (21.7° vs. 22.9°).
- 3D pose estimation might be inaccurate on the GAFA dataset (subjects at a distance), which limits the performance ceiling.
- Gaze estimation in multi-person scenes remains unexplored.

## Related Work & Insights
- **vs GFIE**: GFIE first estimates the 2D gaze point and then constrains the 3D gaze direction with depth, which is a decoupled two-step approach. GA3CE directly learns 3D gaze end-to-end, without requiring a 2D gaze-following module.
- **vs GAFA**: GAFA utilizes temporal RGB and 2D body flow to estimate gaze. GA3CE uses 3D pose and object positions as more robust intermediate representations, outperforming GAFA using only a single frame.
- The concept of egocentric transformation can be applied to other social signal understanding tasks (e.g., gesture direction estimation, interaction intention prediction).

## Rating
- Novelty: ⭐⭐⭐⭐ Egocentric transformation and D3 PE are novel designs, though the overall framework is still a combination of mature components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very thorough, with evaluations on three datasets, extensive ablations, and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, and detailed method description.
- Value: ⭐⭐⭐⭐ Significant progress in unconstrained scene gaze estimation, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](../../ECCV2024/human_understanding/3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)
- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)

</div>

<!-- RELATED:END -->
