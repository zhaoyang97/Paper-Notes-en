---
title: >-
  [Paper Note] 3D Hand Pose Estimation in Everyday Egocentric Images
description: >-
  [ECCV 2024][Human Understanding][3D Hand Pose] By systematically investigating four practices—cropped inputs, Intrinsics-aware Positional Encoding (KPE), auxiliary supervision (hand segmentation + grasp labels), and multi-dataset joint training—this work proposes the WildHands system. Under the constraint of using only a ResNet50 backbone and a small amount of data, WildHands achieves robust 3D hand pose estimation in in-the-wild egocentric images. Its zero-shot generalizatio…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "3D Hand Pose"
  - "Egocentric Vision"
  - "Perspective Distortion"
  - "Auxiliary Supervision"
  - "Zero-Shot Generalization"
date: 2026-05-08
content_hash: d94f22833fc2498f
---

# 3D Hand Pose Estimation in Everyday Egocentric Images

**Conference**: ECCV 2024  
**arXiv**: [2312.06583](https://arxiv.org/abs/2312.06583)  
**Code**: [https://github.com/ap229997/wild-hands](https://github.com/ap229997/wild-hands)  
**Area**: 3D Vision / Hand Pose Estimation  
**Keywords**: 3D Hand Pose, Egocentric Vision, Perspective Distortion, Auxiliary Supervision, Zero-Shot Generalization  

## TL;DR
By systematically investigating four practices—cropped inputs, Intrinsics-aware Positional Encoding (KPE), auxiliary supervision (hand segmentation + grasp labels), and multi-dataset joint training—this work proposes the WildHands system. Under the constraint of using only a ResNet50 backbone and a small amount of data, WildHands achieves robust 3D hand pose estimation in in-the-wild egocentric images. Its zero-shot generalization outperforms FrankMocap across all metrics and competes closely with HaMeR, which is $10\times$ larger.

## Background & Motivation
3D hand pose estimation is crucial in scenarios such as AR/VR and robotic manipulation. While existing methods perform reasonably well in controlled laboratory environments, transferring them to everyday egocentric images poses three major challenges: (1) poor visual signals—hand-object interactions lead to severe occlusions, compounded by low resolution and motion blur; (2) large perspective distortion—in egocentric views, the hand is close to the camera, resulting in significant deformation; (3) lack of in-the-wild 3D annotations—obtaining 3D hand annotations outside the lab is extremely difficult. Existing methods (such as FrankMocap and HaMeR) are primarily trained on third-person views or in laboratory settings, leading to poor performance when directly applied to egocentric scenarios. The unique characteristics of egocentric perspectives (where perspective deformation varies with the hand's placement in the field of view, and the hand's location itself conveys pose information) have not yet been systematically studied or utilized.

## Core Problem
How can a 3D hand pose estimation model be trained to handle large perspective distortions and severe occlusions in egocentric images when in-the-wild 3D hand annotations are absent? The key challenges are: (1) the projective deformation of the hand at different locations in egocentric images depends on camera intrinsics, necessitating explicit modeling of this relationship; (2) laboratory datasets lack diversity, whereas in-the-wild datasets lack 3D annotations, requiring the design of effective alternative supervision signals.

## Method

### Overall Architecture
WildHands is adapted from ArcticNet-SF. The input is a hand crop ($224 \times 224$) and the corresponding camera intrinsics. This input is encoded by a ResNet50 to obtain a $7 \times 7 \times 2048$ feature map. After fusing with Intrinsics-aware Positional Encoding (KPE), the feature map is compressed via convolutional layers and fed into an HMR-style iterative decoder to predict MANO parameters (shape $\beta$, local joint angles $\theta_{\text{local}}$, global pose $\theta_{\text{global}}$) and weak-perspective camera parameters. Through the MANO layer, the hand mesh and 3D keypoints are obtained. Differentiable rendering is used to generate the hand segmentation mask, and a grasp classification head is used to predict the grasp type. During training, 3D supervision is applied to laboratory datasets (ARCTIC, AssemblyHands), while auxiliary supervision with segmentation masks and grasp labels is applied to in-the-wild datasets (Epic-Kitchens, Ego4D).

### Key Designs
1. **KPE (Intrinsics-aware Positional Encoding)**: This is the core innovation. In egocentric views, the perspective deformation of the hand varies at different positions in the image, causing the same 3D hand pose to have different 2D appearances depending on its location. KPE converts pixel coordinates into camera field-of-view coordinates $(\theta_x, \theta_y)$, which are injected into the feature map via sinusoidal encoding. This informs the network of the crop area's location within the camera's field of view. A sparse version (encoding only the four corners and the center) yields the best results. It is concatenated with the $7 \times 7$ feature map and fused via three convolutional layers without using BatchNorm to preserve spatial information. KPE is more effective than alternatives like CamConv, perspective correction, or PCL.

2. **Auxiliary Supervision Strategy**: To address the lack of 3D annotations in wild data, two types of 2D supervision signals are utilized: (a) Hand segmentation mask: The predicted 3D hand is projected into a mask via differentiable rendering (SoftRasterizer) to compute a BCE loss against VISOR annotations or masks predicted by off-the-shelf models, enabling end-to-end gradient backpropagation for 3D parameters; (b) Grasp type labels: A 4-layer MLP classification head is trained to predict probabilities across 8 grasp classes from the predicted MANO parameters, utilizing the intrinsic correlation between grasp types and hand poses to provide additional constraints.

3. **Multi-Dataset Joint Training**: Training is performed by mixing laboratory and in-the-wild datasets. Each batch contains images from different datasets, and different loss weight combinations are applied to different datasets. Results show that incorporating more datasets consistently improves performance.

### Loss & Training
The total loss is a weighted sum of multiple terms: MANO parameter loss ($L_\theta, L_\beta$), 3D keypoint loss ($L_{\text{kp3d}}$), 2D projected keypoint loss ($L_{\text{kp2d}}$), camera parameter loss ($L_{\text{cam}}$), mask loss ($L_{\text{mask}}$), and grasp classification loss ($L_{\text{grasp}}$). Laboratory datasets use 3D-related losses ($\lambda_\theta = 10, \lambda_{\text{kp3d}} = 5, \lambda_{\text{kp2d}} = 5, \lambda_{\text{cam}} = 1$), whereas in-the-wild datasets rely solely on auxiliary losses ($\lambda_{\text{mask}} = 10, \lambda_{\text{grasp}} = 0.1$). The model is trained for 100 epochs with $\text{lr} = 1\text{e-}5$ using the Adam optimizer, a batch size of 144, and 2 $\times$ A40 GPUs.

## Key Experimental Results

| Dataset | Metric | WildHands | FrankMocap | HaMeR(ViT-H) | Gain |
|--------|------|-----------|------------|---------------|------|
| H2O | MPJPE(mm)↓ | 31.08 | 58.51 | 23.82 | vs FM -46.9% |
| H2O | MRRPE(mm)↓ | 49.49 | - | 147.87 | vs HaMeR -66.5% |
| AssemblyHands | MPJPE(mm)↓ | 80.40 | 97.59 | 45.49 | vs FM -17.6% |
| Epic-HandKps | L2 Error(px)↓ | 7.20 | 13.33 | 4.56 | vs FM -46.0% |
| ARCTIC(Leaderboard) | MPJPE(mm)↓ | 15.72 | - | - | Leaderboard #1 |

### Ablation Study
- **Cropping vs. Full Image**: Using hand crops improves MPJPE by 27.7% and MRRPE by 29.7% compared to full-image input, as cropping focuses on fine-grained visual details.
- **KPE yields the largest contribution**: Simply adding KPE improves MPJPE by 20.5%, MRRPE by 56.4%, and 2D metrics by 65.1%—proving that modeling perspective distortion is crucial.
- **KPE Variants**: Sparse KPE > Dense KPE > Intrinsics-free KPE > Fused with input layer > No KPE; KPE significantly outperforms CamConv, perspective correction, and PCL.
- **Auxiliary Supervision**: Mask supervision (MPJPE -8.5%, MRRPE -21.5%, 2D -55.5%) > Grasp supervision (MPJPE -2.5%, MRRPE -7.3%); the combination of both yields the best results.
- **Transformers also benefit**: KPE is equally effective on HandOccNet and HaMeR (ViT), bringing 21.5% and 21% MPJPE improvements, respectively.
- **Data Scaling**: Expanding training from a single dataset (ARCTIC) to ARCTIC+Assembly+Ego4D consistently improves both 3D and 2D metrics.

## Highlights & Insights
- **Ingenious KPE Design**: By converting image crop positions into camera field-of-view angles and representing them with sinusoidal positional encodings, perspective distortions in egocentric views are resolved at minimal computational cost. This design is backbone-agnostic (benefiting both CNNs and Transformers).
- **Practical Auxiliary Supervision**: Seamlessly linking 3D estimation with 2D mask supervision via differentiable rendering, alongside exploiting the correlation between grasp types and hand poses, provides an elegant way to extract 3D constraints from 2D annotations.
- **Efficient System Design**: Competing against HaMeR (which uses ViT-H and over 10 datasets) using only a ResNet50 and 3 datasets demonstrates that proper design choices are more efficient than brute-force scaling of models and data.
- **Zero-Shot Evaluation Strategy**: Evaluating on 4 unseen datasets thoroughly validates the generalization capability of the method.

## Limitations & Future Work
- **Reliance on Camera Intrinsics**: KPE requires known camera intrinsics, which may not always be available in in-the-wild scenarios. Although the authors note that EXIF metadata often contains intrinsic information, this dependency restricts generalizability.
- **Manual Loss Weights**: Loss weights for different datasets and supervisory signals are adjusted manually. Automated schemes such as uncertainty weighting could be explored.
- **Failure in Extreme Scenarios**: All models fail when fingers are mostly invisible (e.g., kneading dough) or under extreme grasp poses.
- **Lack of Temporal Information**: The current method relies on single-frame estimation. Temporal consistency and motion priors in egocentric videos are not utilized.
- **Expandable Auxiliary Supervision**: Richer auxiliary signals, such as depth foundation models (e.g., DepthAnything) or contact priors, could be incorporated.

## Related Work & Insights
- **vs. FrankMocap**: Utilizing the same ResNet50 backbone, WildHands outperforms FrankMocap across all metrics because the latter fails to account for perspective distortion and lacks in-the-wild auxiliary supervision.
- **vs. HaMeR**: HaMeR employs a ViT-H ($10\times$ larger) and over 10 datasets ($5\times$ more). While it exhibits stronger performance in MPJPE, WildHands significantly outperforms HaMeR in MRRPE (absolute pose, 49.49 vs. 147.87 on H2O), showing that KPE dramatically improves absolute pose estimation.
- **vs. EgoPoseFormer**: EgoPoseFormer relies on binocular input, whereas WildHands uses only monocular RGB, offering wider applicability and complementary advantages.

## Insights & Connections
- The concept of KPE can be transferred to other egocentric 3D understanding tasks (e.g., 3D object pose estimation, scene reconstruction). Any scenario involving 3D prediction from cropped images could benefit from this technique.
- The auxiliary supervision paradigm (differentiable rendering + 2D annotations $\to$ 3D constraints) can be extended to tasks like hand-object interaction reconstruction and body pose estimation.
- Automatic learning of loss weights for distinct supervisory signals in multi-dataset training is a promising direction for future research.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic study and introduction of KPE to the hand pose estimation field, though the core components are mostly combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations, zero-shot evaluation on 4 datasets, and controlled experiments for every design choice.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with fair and rigorous experimental setups, although some dense notation requires careful reading.
- Value: ⭐⭐⭐⭐ Tangible contribution to the egocentric hand pose estimation field; the concepts of KPE and auxiliary supervision are highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)

</div>

<!-- RELATED:END -->
