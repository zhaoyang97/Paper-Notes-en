---
title: >-
  [Paper Note] Vision-Language Embodiment for Monocular Depth Estimation
description: >-
  [CVPR 2025][3D Vision][Monocular Depth Estimation] An embodied depth estimation framework is proposed, which embodies the physical characteristics of the camera model into a deep learning system to calculate Embodied Scene Depth as a geometric prior. Simultaneously, it leverages vision-language complementarity (depth text descriptions + textual VAE + conditional sampler) to fuse RGB image features and physical depth prior for monocular depth estimation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Camera Model Embodiment"
  - "Vision-Language Fusion"
  - "Scene Depth Prior"
  - "Variational Autoencoder"
date: 2026-05-08
content_hash: 0cbbf9d805b4bfde
---

# Vision-Language Embodiment for Monocular Depth Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.16535](https://arxiv.org/abs/2503.16535)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Monocular Depth Estimation, Camera Model Embodiment, Vision-Language Fusion, Scene Depth Prior, Variational Autoencoder

## TL;DR

An embodied depth estimation framework is proposed, which embodies the physical characteristics of the camera model into a deep learning system to calculate Embodied Scene Depth as a geometric prior. Simultaneously, it leverages vision-language complementarity (depth text descriptions + textual VAE + conditional sampler) to fuse RGB image features and physical depth prior for monocular depth estimation.

## Background & Motivation

- Monocular depth estimation is a core problem in computer vision, but the mapping from 3D to 2D is inherently ill-posed.
- Existing depth estimation models mainly rely on inter-image relationships for supervised training, ignoring the intrinsic information provided by the camera itself.
- Although geometric priors (normal constraints, planar constraints) can reduce uncertainty, their overall impact is limited.
- Existing CLIP-based depth estimation methods (such as DepthCLIP) use fixed discrete depth descriptions, lacking expressive capability.
- As two inherently ambiguous modalities, images and text have complementary advantages: images provide direct 3D observations, while text provides object scale priors.
- For flat areas such as roads, camera model parameters can directly calculate absolute depth with extremely high accuracy (>99% of pixels have an error of <10%).
- There is a lack of a depth estimation framework that uniformly integrates physical camera models, environmental semantics, and linguistic priors.
- Sparse LiDAR ground truth limits the density of available depth supervision signals.

## Method

### Overall Architecture

The system consists of three levels of embodiment: **Camera Embodiment**—utilizing camera intrinsic/extrinsic parameters and road segmentation to compute Embodied Scene Depth; **Language Embodiment**—using ExpansionNet-v2 to generate image descriptions, and generating depth text descriptions based on segmentation and Embodied Depth to combine into the input of a textual VAE; **Vision-Language Fusion**—an RGB encoder and a depth encoder fuse features through cross-attention, while a conditional sampler samples from the latent distribution of the textual VAE, and a depth decoder with shared weights outputs the final depth. An alternating optimization strategy is employed for training.

### Key Designs

**Design 1: Embodied Scene Depth**
- **Function**: Utilizes physical camera parameters to calculate dense scene depth priors in real time.
- **Mechanism**: Under the coplanarity assumption, using the camera intrinsic and extrinsic parameter matrix $A = [K|0][R,T;0,1]$, the depth $z_c$ of each pixel is solved given the camera height $h$. First, the road region is identified via semantic segmentation to obtain Embodied Road Depth (with 99%+ accuracy), which is then extended to the ground and vertical surfaces. Finally, Telea Inpainting is used to fill in blanks to obtain the complete Embodied Scene Depth.
- **Design Motivation**: Roads satisfy planar conditions, allowing direct analytical calculation of absolute depth—accurate like LiDAR but denser. Step-by-step extension to the entire scene, although reducing accuracy, provides valuable geometric constraints.

**Design 2: Depth-Guided Textual Variational Autoencoder**
- **Function**: Leverages linguistic descriptions to model the probability distribution of possible 3D scene layouts.
- **Mechanism**: For each object $O_i$, a depth text description $T_i$ is generated (containing depth value $d_i$ and ranking $r_i$). Together with the image caption, the combined text is passed through a CLIP text encoder and an MLP to estimate the mean $\hat{\mu}$ and standard deviation $\hat{\sigma}$ of the latent distribution. Employing the reparameterization trick $\hat{z} = \hat{\mu} + \epsilon \cdot \hat{\sigma}$ yields samples, which are then generated into a depth map by the depth decoder.
- **Design Motivation**: Object scale and spatial layout priors provided by text can constrain the solution space of depth estimation. The variational framework naturally models the diverse possibilities of scenes.

**Design 3: Embodiment-Driven Conditional Sampler**
- **Function**: Samples depth corresponding to a specific image from the latent distribution of the textual VAE conditioned on the image.
- **Mechanism**: Transformer blocks encode the fused features of RGB and Embodied Depth (fused via cross-attention $F_f^d = \text{softmax}(\frac{Q_r K_d}{d_k})V_d$) into $h \times w$ local samples $\tilde{\epsilon}$, which replace the standard Gaussian noise $\epsilon$ to generate $\tilde{z} = \hat{\mu} + \tilde{\epsilon} \cdot \hat{\sigma}$. The final depth is output through the shared-weight depth decoder.
- **Design Motivation**: The textual language manifold can only describe the distribution of possible 3D layouts, and image information is required to pinpoint the latent vector that best matches the current scene.

### Loss & Training

Alternating training: (1) Freeze the conditional sampler, train the textual VAE and the depth decoder using $\mathcal{L}_{KL}(\mu, \sigma) + \mathcal{L}_{SiLog}$; (2) Freeze the textual VAE, train the conditional sampler and the depth decoder using the SiLog loss. The KL divergence regularizes the latent distribution toward a standard Gaussian, and the SiLog loss enhances scale invariance.

## Key Experimental Results

### Main Results: KITTI Depth Estimation

| Method | AbsRel↓ | SqRel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|---------|--------|-------|----------------|
| BTS | 0.061 | 0.261 | 2.834 | 0.954 |
| Adabins | 0.058 | 0.190 | 2.360 | 0.964 |
| iDisc | 0.053 | 0.175 | 2.216 | 0.971 |
| ECoDepth | 0.054 | 0.171 | 2.173 | 0.970 |
| **Ours** | **0.050** | **0.159** | **2.054** | **0.974** |

### Ablation Study: Embodied Depth Accuracy (KITTI Dataset)

| Depth Type | ±5% Error Range | ±10% Error Range |
|---------|-----------|------------|
| Embodied Road Depth | 80.24% | 99.33% |
| Embodied Ground Depth | 60.30% | 74.89% |
| Embodied Scene Depth | 38.88% | 52.45% |

### Key Findings
- Embodied Road Depth has extremely high accuracy (99.33% of pixels have an error < 10%) and can serve as a substitute for LiDAR in road areas.
- Different semantic segmentation models have minimal impact on Embodied Depth (the gap compared with GT segmentation is < 1%), showing strong robustness.
- The complete framework achieves an AbsRel of 0.050 on KITTI, surpassing SOTA methods such as ECoDepth and iDisc.
- Although the accuracy of Embodied Scene Depth decreases in non-planar regions, it provides valuable dense geometric priors.
- The alternating training strategy (textual VAE and conditional sampler) is crucial to the overall performance.

## Highlights & Insights

1. **Direct Exploitation of Physical Camera Models**: Instead of learning depth priors, they are analytically calculated, achieving near-sensor-level accuracy in planar regions.
2. **Three-Level Embodiment Integration**: A unified framework of camera + environment + language, where each level complements the others.
3. **Innovative Depth Text Descriptions**: Object depth values and rankings are transformed into natural language, leveraging the semantic capacity of the CLIP text encoder.
4. **Low Dependency on Segmentation Models**: The accuracy differences of different segmentation models have minimal impact on the final depth estimation.

## Limitations & Future Work

- Primarily validated in driving scenarios like KITTI and DDAD; applicability to scenes without clear ground planes (such as indoors) is limited.
- Embodied Scene Depth relies on the coplanarity assumption, causing accuracy to degrade on non-planar surfaces like ramps and steps.
- Textual descriptions depend on the quality of automatic caption generation tools.
- Requires known camera intrinsic and extrinsic parameters, which limits zero-shot generalizability.
- Future work can extend the method to more scene types and integrate stronger VLMs.

## Related Work & Insights

- Standard depth estimation methods like DepthCLIP use fixed text templates, whereas this work dynamically generates text descriptions incorporating depth information.
- Shares a similar textual VAE structure with WordDepth but incorporates the fusion of Embodied Scene Depth.
- The approach of analytically computing depth from camera models can serve as an optional prior input for any depth estimation method.

## Rating

⭐⭐⭐⭐ — The idea of embodying camera models carries high practical value in driving scenarios, and the design of the vision-language fusion framework is sound; however, its application scenarios are constrained by the assumption of planar environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)
- [\[CVPR 2025\] DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)
- [\[CVPR 2026\] Iris: Integrating Language into Diffusion-based Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_integrating_language_into_diffusion-based_monocular_depth_estimation.md)
- [\[CVPR 2025\] Relative Pose Estimation through Affine Corrections of Monocular Depth Priors](relative_pose_estimation_through_affine_corrections_of_monocular_depth_priors.md)

</div>

<!-- RELATED:END -->
