---
title: >-
  [Paper Note] 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model
description: >-
  [ECCV 2024][3D Vision][6D Pose Estimation] Proposes 6DGS, which inverts the 3DGS rendering workflow—casting rays uniformly from the surfaces of the ellipsoids (Ellicell), using an attention mechanism to bind rays to target image pixels, and then utilizing weighted least squares to solve for camera pose in closed form. Requiring no iterations or initial poses, it improves rotation accuracy by 12% and translation accuracy by 22% on real-world scenes…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "6D Pose Estimation"
  - "3D Gaussian Splatting"
  - "Ellicell Ray Casting"
  - "Closed-Form Solution"
  - "No Pose Initialization Required"
date: 2026-05-08
content_hash: c6da37d4edc4ba36
---

# 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model

**Conference**: ECCV 2024  
**arXiv**: [2407.15484](https://arxiv.org/abs/2407.15484)  
**Code**: [https://github.com/mbortolon97/6dgs](https://github.com/mbortolon97/6dgs)  
**Area**: 3D Vision / 6D Pose Estimation / 3D Gaussian Splatting  
**Keywords**: 6D Pose Estimation, 3D Gaussian Splatting, Ellicell Ray Casting, Closed-Form Solution, No Pose Initialization Required

## TL;DR
Proposes 6DGS, which inverts the 3DGS rendering workflow—casting rays uniformly from the surfaces of the ellipsoids (Ellicell), using an attention mechanism to bind rays to target image pixels, and then utilizing weighted least squares to solve for camera pose in closed form. Requiring no iterations or initial poses, it improves rotation accuracy by 12% and translation accuracy by 22% on real-world scenes, achieving near-real-time performance at 15fps.

## Background & Motivation
Novel view synthesis based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) is highly mature, but leveraging these representations for 6D pose estimation still faces critical bottlenecks. Existing methods like iNeRF adopt an iterative "analysis-by-synthesis" strategy: given an initial pose guess, a rendered image is compared with the target image, and the pose is iteratively optimized through photometric loss. This approach has three key limitations of prior work: (1) an initial pose close to the ground truth must be provided to ensure convergence; (2) it is prone to local optima; (3) rendering the image at each iteration is extremely slow (iNeRF is only 0.16fps, Parallel iNeRF is only 0.05fps). The emergence of 3DGS offers a new opportunity—it represents scenes using explicit geometric primitives (ellipsoids) which render extremely fast, and the geometric attributes of the ellipsoids (position, orientation, scale) can be directly utilized.

## Core Problem
**Can we bypass iterative optimization and directly solve for the 6DoF camera pose in closed form using a 3DGS model and a single image?** Existing NVS-based pose estimation methods rely on a "render-compare-update" iterative loop, which not only requires an initial pose prior but is also computationally expensive. 6DGS aims to answer: how to utilize the geometric structure of 3DGS ellipsoids to transform pose estimation from an iterative optimization problem into a one-shot ray-pixel matching problem.

## Method
The mechanism of 6DGS is highly elegant: standard 3DGS rendering casts rays from the camera optical center to ellipsoids to generate images; 6DGS does the reverse—casting rays outward in all directions from the ellipsoid surfaces, finding which rays best match the pixels of the target image, where the intersection of these optimal rays determines the camera's optical center.

### Overall Architecture
Input: A target RGB image + a pre-trained 3DGS model (a set of ellipsoids). Output: 6DoF camera pose. The workflow consists of four steps:
1. **Ellicell Ray Generation**: The surface of each 3DGS ellipsoid is divided uniformly into equal-area cells (Ellicell). Rays are cast from the ellipsoid center through each cell center, and the color values for each ray are computed using 3DGS rendering functions.
2. **Feature Extraction**: DINOv2 is used to extract pixel features from the target image; an MLP (with positional encoding) encodes ray information into feature vectors.
3. **Attention Binding**: An attention mechanism calculates the correlation scores between ray features and image features to select the Top-$K$ most correlated rays.
4. **Closed-Form Pose Solving**: The selected ray bundle is intersected using weighted least squares to compute the camera optical center; the camera rotation is then derived from the ray directions.

### Key Designs
1. **Radiant Ellicell**: This is the core innovation of the paper. It divides the surface of each 3DGS ellipsoid uniformly into equal-area cells, achieving deterministic and uniform ray sampling. Specifically, Ramanujan's approximation is first used to calculate the ellipsoid surface area, and then a "slicing-and-segmenting" method cuts along the principal axis into equal-width bands (ribbons), each of which is further divided into several cells. Inverse transform sampling of the ellipse CDF ensures that cell centers are uniformly distributed on the ellipsoid surface. Compared to Monte-Carlo sampling, Ellicell achieves higher precision with fewer rays.
2. **Ray Filtering and Color Computation**: Only rays facing the same hemisphere as the ellipsoid surface normal are retained (as backward-facing rays cannot reach the camera). The color of each ray is computed via the 3DGS volume rendering formula, providing photometric information for subsequent feature matching.
3. **Attention Binding Mechanism**: Ray features serve as queries and DINOv2 image features serve as keys to compute matching scores via an attention map. During training, supervision comes from training images with known poses—the distance between each ray and the known camera center is computed, and closer distances map to higher scores (via a tanh mapping) to generate self-supervised soft labels.
4. **Weighted Least Squares Pose Solving**: After selecting the Top-100 rays, due to discretization noise they usually do not intersect exactly at one point. Thus, the sum of squared perpendicular distances from all rays to the estimated optical center is minimized. Attention scores serve as weights, allowing high-confidence rays to contribute more.

### Loss & Training
- The training loss is an $L_2$ loss, minimizing the difference between predicted attention scores and ground-truth scores calculated based on distance.
- Uses the Adafactor optimizer with a weight decay of $1 \times 10^{-3}$.
- Uniformly samples 2000 ellipsoids per iteration to accelerate training.
- Training requires only 1500 iterations (approx. 45 minutes on an RTX 3090).
- Training data is taken directly from the images used to build the 3DGS model (approx. 100 images), requiring no extra annotations.

## Key Experimental Results

| Dataset | Method | Condition | MAE (°)↓ | MTE (u)↓ | Speed |
|--------|------|------|----------|----------|------|
| Mip-NeRF 360 | iNeRF | Fixed prior | ~39.5 (Bicycle) | ~0.116 | 0.16fps |
| Mip-NeRF 360 | Parallel iNeRF | Fixed prior | ~35.9 (Bicycle) | ~0.116 | 0.05fps |
| Mip-NeRF 360 | 6DGS (Ours) | **No prior** | Best overall | Best overall | **15fps** |
| Tanks&Temples | iNeRF | Random prior | 89.2 (Barn) | 0.682 | 0.16fps |
| Tanks&Temples | Parallel iNeRF | Fixed prior | 22.9 (Barn) | 0.131 | 0.05fps |
| Tanks&Temples | 6DGS (Ours) | **No prior** | Best overall | Best overall | **15fps** |

Key Conclusions:
- Under the **no prior pose** condition, 6DGS improves rotation accuracy by an average of 12% and translation accuracy by 22%.
- Speed is 300x faster than Parallel iNeRF (15fps vs 0.05fps) and around 94x faster than iNeRF.
- NeMo+VoGE performs worst under random initialization because it only uses approx. 5000 large ellipsoids, whereas 6DGS utilizes approx. 300,000 fine-grained ellipsoids from 3DGS.
- Even compared to baselines using fixed prior poses, 6DGS without any prior still performs better in most scenes.

### Ablation Study
- **Number of Selected Rays $N_{top}$**: 100 rays is the optimal balance point. Increasing the number of rays improves angular error but slightly increases translation error (introducing low-confidence rays causes the optical center estimate to lean too close to the object).
- **Number of Cast Rays per Ellipsoid**: 50 rays is optimal. Too many rays lead to network overfitting on the training set, degrading generalization during testing.
- **MLP Channel Size**: The default value is optimal. Increasing channels also causes generalization issues (since training contains only ~150 non-uniformly distributed images).
- Processing speed decreases as the number of rays and MLP channel sizes increase, falling to about 10-13fps once exceeding the default parameters.

## Highlights
- **Elegant Concept of Inverting Rendering**: Instead of doing iterative "rendering-and-comparing", it casts rays outward from the 3D surface to back-calculate camera position. This converts pose estimation into a ray-pixel matching and geometric solving problem, which is conceptually very clean.
- **Ingenious Ellicell Design**: The mathematical derivation for equal-area division on the ellipsoid surface is complete, utilizing Ramanujan's approximation + CDF inverse sampling to guarantee uniformity, which is far more efficient than random sampling.
- **Zero Need for Initial Pose**: This is a major advantage in practical deployment—existing methods assume an initial guess close to ground truth, but such a prior is often unavailable in actual applications.
- **Self-Supervised Training**: Ingeniously leverages training images with known poses to automatically generate supervision signals for ray-pixel matching without requiring manual annotations.
- **Closed-Form + Near-Real-Time**: 15fps achieved on consumer-grade hardware, which is highly valuable for real-time applications like VR/AR and robotic navigation.

## Limitations & Future Work
- **Static Scenes Only**: 6DGS is built on a static 3DGS model, unable to handle dynamic objects or scene changes, limiting its application in highly dynamic environments.
- **Ellicell Discretization Error**: Dividing the continuous ellipsoid surface into finite cells introduces quantization noise. Qualitative analysis shows that estimated poses often lean toward the object (meaning the camera-to-object distance is slightly underestimated).
- **Dependence on 3DGS Quality**: If the 3DGS reconstruction quality is poor, the inaccurate ray color information will cascade and impact pose estimation accuracy.
- **Training Data Requirement**: Although less than CamNet (500+ images), it still requires about 100 images with poses to train the attention module.
- **No Comparison with Learning-Based Methods**: It only compares against NVS-based analysis-by-synthesis methods, lacking comparisons with PnP-based or direct-learning regression methods.
- **Scalability to Large-Scale Scenes**: 300,000 ellipsoids each casting 50 rays equals 15 million candidate rays. The memory and computational costs still need to be evaluated.

## Related Work & Insights
- **vs iNeRF / Parallel iNeRF**: iNeRF is an iterative analysis-by-synthesis method relying on initial poses and NeRF rendering (which is extremely slow). Parallel iNeRF improves convergence by optimizing multiple candidate poses in parallel but is even slower. 6DGS completely avoids iteration through its closed-form solution, vastly outperforming in both speed and accuracy.
- **vs NeMo+VoGE**: Also uses ellipsoid representations, but NeMo+VoGE uses ~5000 large ellipsoids converted from meshes and relies on iterative feature alignment. 6DGS leverages 300,000 fine-grained ellipsoids from 3DGS, providing far more detail and coverage.
- **vs CROSSFIRE**: CROSSFIRE learns local features to ease local optima issues, but still requires iteration. The attention binding mechanism of 6DGS also performs feature matching, but does it in a single step.
- **vs IFFNeRF**: The authors' prior work, which inverts NeRF models but does not consider the special geometric properties of 3DGS ellipsoids (deformation, rotation, non-uniform distribution).

## Related Work & Insights
- **Generality of the Inversed Rendering Paradigm**: The concept of "casting rays outward from a 3D surface to find the camera" is not only applicable to pose estimation but can also be extended to tasks like establishing correspondence or relocalization in 3DGS scenes.
- **Dynamic Extensions**: Extending 6DGS to 4DGS / dynamic 3DGS is a natural progress step—performing Ellicell slicing on time-varying ellipsoids to simultaneously estimate timestamps and poses.
- **Ellicell to General Geometric Sampling**: The equal-area division method of Ellicell on ellipsoid surfaces might find utility in other tasks requiring uniform surface sampling on ellipsoids, such as collision detection or 3D shape analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of inverting 3DGS rendering is novel, and the Ellicell design is creative, though the core framework (ray matching + geometric solving) is relatively classic.
- Experimental Thoroughness: ⭐⭐⭐ Well covered by two datasets and multiple ablation studies, but lacks comparison with non-NVS methods and has not been validated on large-scale or outdoor scenes.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are complete and clear, diagrams are intuitive, and workflow descriptions are easy to understand.
- Value: ⭐⭐⭐⭐ The first work utilizing 3DGS for closed-form pose estimation. Its real-time performance is meaningful for real-world deployment, despite a somewhat limited scope of applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation](omni6d_large-vocabulary_3d_object_dataset_for_category-level_6d_object_pose_esti.md)
- [\[CVPR 2026\] PoseGaussian: 6D Pose Estimation for Unseen Objects via Sparse-View Object-Level 3D Gaussian Splatting](../../CVPR2026/3d_vision/posegaussian_6d_pose_estimation_for_unseen_objects_via_sparse-view_object-level_.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model](crm_single_image_to_3d_textured_mesh_with_convolutional_reconstruction_model.md)

</div>

<!-- RELATED:END -->
