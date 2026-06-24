---
title: >-
  [Paper Note] Shape-Guided Configuration-Aware Learning for Endoscopic-Image-Based Pose Estimation of Flexible Robotic Instruments
description: >-
  [ECCV 2024][Medical Imaging][Flexible Robots] By leveraging the 3D shape prior of flexible robots to guide image feature learning, this work extracts part-level geometric representations and applies a dynamic shape deformation mechanism. This achieves highly accurate pose estimation of flexible robots from endoscopic images, significantly outperforming baseline methods such as keypoint detection, skeleton extraction, and direct regression in predicting both external orientati…
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Flexible Robots"
  - "Pose Estimation"
  - "3D Shape Prior"
  - "Endoscopic Surgery"
  - "Shape-guided Learning"
date: 2026-05-08
content_hash: 213f00b5300d4c28
---

# Shape-Guided Configuration-Aware Learning for Endoscopic-Image-Based Pose Estimation of Flexible Robotic Instruments

**Conference**: ECCV 2024  
**Institution**: The Chinese University of Hong Kong / Agilis Robotics / The University of Hong Kong  
**Code**: [https://github.com/Yiyao-Ma/PoseFlex](https://github.com/Yiyao-Ma/PoseFlex)  
**Area**: Medical Images  
**Keywords**: Flexible Robots, Pose Estimation, 3D Shape Prior, Endoscopic Surgery, Shape-guided Learning

## TL;DR

By leveraging the 3D shape prior of flexible robots to guide image feature learning, this work extracts part-level geometric representations and applies a dynamic shape deformation mechanism. This achieves highly accurate pose estimation of flexible robots from endoscopic images, significantly outperforming baseline methods such as keypoint detection, skeleton extraction, and direct regression in predicting both external orientation and internal bending angles.

## Background & Motivation

**Background**: Flexible robots are playing an increasingly important role in intraluminal procedures (e.g., gastrointestinal endoscopic surgery). Accurately estimating their external orientation (roll, pitch, and yaw) and internal bending angles is essential for intraoperative navigation and control. Traditional approaches primarily fall into sensor-based methods and image-based methods.

**Limitations of Prior Work**: Sensor-based methods (e.g., electromagnetic tracking, fiber-optic sensors) are limited by high cost, environmental restrictions (electromagnetic interference), and integration difficulties. While image-based methods do not require additional physical sensors, existing schemes perform poorly when dealing with the geometrical complexity of flexible robots. Specifically, keypoint detection methods struggle to locate stable keypoints on highly flexible structures with high degrees of freedom; skeleton extraction methods cannot handle cases with drastic shape changes; and direct regression methods lack mechanisms to effectively model shape deformations.

**Key Challenge**: The shape of a flexible robot dynamically changes alongside its bending angles, making 2D image representations insufficient for capturing its complex 3D geometric relations. Relying solely on 2D image features fails to fully comprehend the 3D spatial state of the robot.

**Goal**: (1) How to inject effective 3D geometric information into image feature learning? (2) How to handle the discrepancy between the shape prior and the actual robot shape in images? (3) How to utilize initial estimates to further refine pose predictions?

**Key Insight**: Inspired by recent advances in joint 2D-3D representation learning, it is observed that although flexible robots are highly deformable, their geometric structures possess stable prior knowledge at the part level (such as assemblies of cylindrical segments). Injecting such a 3D prior into image features could bridge the gap between 2D observations and 3D states.

**Core Idea**: Part-level geometric features from the 3D shape prior are utilized to query and enhance image feature representations. Through a dynamic deformation mechanism, the shape prior is-adaptively matched with the actual observations, thereby improving the accuracy of flexible robot pose estimation.

## Method

### Overall Architecture

PoseFlex adopts a two-stage framework: the first stage, **PoseEst.**, performs shape-guided pose estimation, while the second stage, **PoseRefine.**, handles shape deformation and pose refinement based on the initial pose. The input is a single endoscopic image and a predefined 3D shape prior model of the flexible robot, and the output consists of four pose parameters (roll, pitch, yaw, and the bending angle).

In the PoseEst. stage, the system extracts part-level geometric representations from the 3D shape prior, aligns these representations with image features via a cross-attention mechanism, and then regresses the pose using the enhanced features. In the PoseRefine. stage, based on the initial pose predicted by PoseEst., the shape prior is dynamically deformed by modeling skeleton curves and instantiating cylinders. The deformed shape is then used to re-guide image feature extraction, achieving more accurate pose estimation.

### Key Designs

1. **Part-level 3D Shape Representation Extraction (Part-level Shape Representation)**:

    - **Function**: Extract configuration-aware geometric features from the predefined 3D shape prior.
    - **Mechanism**: First, based on the configuration of the flexible robot (such as the number of segments, segment lengths, etc.), the point cloud of the 3D shape prior is segmented and labeled by part (e.g., individual cylindrical segments, joints). Then, a PointNet++-style encoder extracts geometric features for each part. Consequently, each part feature encodes local geometric information of that part, while introducing global configuration priors through the part labels. Compared to global features, part-level features can capture the local shape differences of each robot segment.
    - **Design Motivation**: Since flexible robots consist of multiple bendable segments, and different segments exhibit different deformation patterns, part-level representations describe the 3D structure of the robot more finely than global representations. Ablation studies demonstrate that removing configuration information leads to a continuous drop in accuracy.

2. **Shape-guided Image Feature Enhancement**:

    - **Function**: Query and enhance the robot representation in the image using the part-level features of the 3D shape prior.
    - **Mechanism**: The part-level features of the 3D shape are used as queries, while the feature maps extracted by the image backbone serve as keys and values, interacting via a cross-attention mechanism. Specifically, each part's shape token searches for its corresponding regions on the image feature maps to perform attention aggregation. In this way, shape tokens absorb visual information from corresponding image areas, while image features obtain structured guidance from the 3D geometry. The enhanced features are then fed into a pose regression head to predict pose parameters.
    - **Design Motivation**: Directly regressing poses from image features lacks 3D geometric constraints. Utilizing the 3D shape as a structured query allows the network to "know which areas to focus on," similar to the concept of using learnable queries for object detection in DETR.

3. **Configuration-aware Shape Deformation**:

    - **Function**: Deform the static shape prior into a version closer to the actual robot shape in the image based on the initial pose estimate.
    - **Mechanism**: Given the initial pose parameters predicted in the PoseEst. stage, a 3D skeleton curve is first parameterized using the bending angle (based on a constant curvature arc model), and cylinders are then instantiated along the skeleton curve to reconstruct the 3D model of the robot. The deformed shape prior matches the robot in the actual image much better. This deformed model provides more accurate shape guidance in a subsequent round of part-level feature extraction, thereby refining the pose estimation further.
    - **Design Motivation**: The initial shape prior has a static, "default" pose, which differs significantly from the actual bent robot shape. Without deformation, the image areas queried by the shape tokens might not correctly match the robot parts, degrading the guidance effect. Dynamic deformation bridges the gap between the prior shape and the actual shape.

### Loss & Training

A probabilistic model is adopted for pose prediction, employing the Matrix Fisher distribution to model the uncertainty of the rotation matrix. The loss function combines the NLL loss (negative log-likelihood) for rotation parameters and the L1 loss for the bending angle. Concurrently, it outputs uncertainty estimation; predictions with high uncertainty tend to have larger errors, serving as a reliable indicator of pose quality.

## Key Experimental Results

### Main Results

Evaluation was conducted on a general flexible robotic platform designed for intraluminal procedures, which contains various illumination, occlusion, and motion blur conditions.

| Method | Roll(Mean°) | Pitch(Mean°) | Yaw(Mean°) | Bend(Mean°) | Acc5° | Acc10° |
|------|------------|--------------|------------|-------------|-------|--------|
| Keypoint (KP) | Higher | Higher | Higher | Higher | Low | Low |
| Skeleton (SKL) | Higher | Higher | Higher | Higher | Low | Low |
| Direct Regression (DR) | Medium | Medium | Medium | Medium | Medium | Medium |
| SimPS | Medium | Medium | Medium | Medium | Medium | Medium |
| **PoseEst. (Ours)** | **Lower** | **Lower** | **Lower** | **Lower** | **Higher** | **Higher** |
| **PoseRefine. (Ours)** | **Lowest** | **Lowest** | **Lowest** | **Lowest** | **Highest** | **Highest** |

PoseEst. significantly outperforms all baseline methods, and PoseRefine. further improves the accuracy on top of PoseEst.

### Ablation Study

| Configuration | Performance | Explanation |
|------|------|------|
| Full model (PoseEst.) | Best | Full shape guidance |
| w/o shape guidance | Significant drop in accuracy | Angle errors increase across all directions after removing 3D shape guidance |
| w/o configuration info | Accuracy drop | Guidance accuracy decreases after removing part labels |
| Replacing Shape with Depth | Worse accuracy | Depth predicted by DPT is noisy, causing severe shape distortion |
| PoseRefine on baselines | Improvement across all | The refinement module generalized well to other baseline methods |

### Key Findings

- Shape guidance is the largest performance contributor; removing it significantly increases the error across all pose parameters.
- 3D information based on depth prediction (e.g., DPT) cannot replace explicit shape priors, as monocular depth estimation suffers from severe noise in endoscopic scenes.
- PoseRefine. can not only improve its own initial predictions but also effectively refine the predictions of other baseline methods, demonstrating good generalization.
- The method scales smoothly to various robotic configurations with different arm thicknesses, lengths, and segment numbers.
- The method maintains reliable performance in challenging surgical scenarios such as abnormal brightness, occlusion, and motion blur.

## Highlights & Insights

- **Using the 3D shape prior as structured queries is highly ingenious.** Unlike the learnable queries in DETR, this approach uses physically meaningful 3D geometry as queries, which both introduces prior knowledge and maintains interpretability. This paradigm of "querying 2D images with 3D models" can be transferred to other tasks requiring 2D-3D alignment.
- **The two-stage "coarse estimation - deformation - refinement" iterative strategy** is intuitively designed. Deforming the prior model using the initial pose makes the second round of guidance more accurate, creating a positive feedback loop. This coarse-to-fine approach with geometric feedback is highly applicable to other pose estimation scenarios involving deformable objects.
- **The probabilistic pose model simultaneously outputs estimations and uncertainties**, where the uncertainty is highly correlated with the actual error, providing a reliability metric for clinical applications.

## Limitations & Future Work

- The current method requires a predefined 3D shape prior model of the robot, which demands remodeling for robots with unknown configurations.
- The constant curvature assumption may not be sufficiently accurate in complex, multi-segmented bending scenarios.
- Evaluated only on a single type of flexible surgical robot, its generalization to other flexible instruments such as catheters and endoscopes remains to be validated.
- Temporal information is not utilized; integrating temporal consistency constraints from video sequences might further improve accuracy.
- Real-time performance is not detailed in reports, and inference speed constraints must be considered for clinical deployment.

## Related Work & Insights

- **vs Keypoint-based methods**: Keypoint-based methods perform well on rigid objects, but the high degree-of-freedom deformation of flexible robots makes locating stable keypoints challenging. This work directly bypasses the difficulty of keypoint detection by utilizing part-level 3D priors.
- **vs SimPS (simulation-based Photometric Stereo)**: SimPS assists training with simulation rendering but lacks an effective mechanism to model shape deformations. The shape-guided scheme in this study provides stronger geometric constraints.
- **vs Depth-based 3D methods**: Relying on pretrained depth estimation models to obtain 3D information yields inferior performance compared to explicit shape priors, highlighting that domain-specific prior knowledge is more effective than general pretrained models under special imaging conditions (e.g., low-textured endoscopic scenes, uneven illumination).

## Rating

- Novelty: ⭐⭐⭐⭐ The framework designed with 3D shape prior guidance is novel, and both the part-level query and dynamic deformation mechanisms are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies, comprising various configuration variations, environmental challenges, and cross-validation.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and accompanied by a detailed project homepage.
- Value: ⭐⭐⭐⭐ Possesses practical value for surgical robot pose estimation, and the framework concepts are transferrable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EchoPOSE: 6D Pose Estimation of Sparse Echocardiograms for Left-Ventricular 3D Shape Reconstruction](../../CVPR2026/medical_imaging/echopose_6d_pose_estimation_of_sparse_echocardiograms_for_left-ventricular_3d_sh.md)
- [\[ECCV 2024\] A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images](a_rotation-invariant_texture_vit_for_fine-grained_recognition_of_esophageal_canc.md)
- [\[ECCV 2024\] Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)
- [\[CVPR 2026\] SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation](../../CVPR2026/medical_imaging/shape_structure-aware_hierarchical_unsupervised_domain_adaptation_with_plausibil.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)

</div>

<!-- RELATED:END -->
