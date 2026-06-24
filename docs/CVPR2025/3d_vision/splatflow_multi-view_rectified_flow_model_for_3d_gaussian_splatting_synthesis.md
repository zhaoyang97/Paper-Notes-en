---
title: >-
  [Paper Note] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] A SplatFlow framework is proposed, consisting of a multi-view rectified flow (RF) model and a Gaussian Splatting decoder (GSDecoder), which jointly generates multi-view images, depth, and camera poses in latent space, achieving unified 3DGS generation and editing via training-free inversion and inpainting techniques.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Rectified Flow Model"
  - "Text-to-3D Generation"
  - "3D Editing"
  - "Multi-View Generation"
date: 2026-05-08
content_hash: 8398c978ceee56c1
---

# SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2411.16443](https://arxiv.org/abs/2411.16443)  
**Code**: [Project Page](https://gohyojun15.github.io/SplatFlow/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Rectified Flow Model, Text-to-3D Generation, 3D Editing, Multi-View Generation  

## TL;DR

A SplatFlow framework is proposed, consisting of a multi-view rectified flow (RF) model and a Gaussian Splatting decoder (GSDecoder), which jointly generates multi-view images, depth, and camera poses in latent space, achieving unified 3DGS generation and editing via training-free inversion and inpainting techniques.

## Background & Motivation

- 3DGS has become the mainstream solution for high-fidelity real-time rendering, but existing 3DGS generation and editing methods are independent, lacking a unified framework.
- In terms of 3DGS generation: SDS-based methods require time-consuming scene-by-scene optimization; direct generation methods are mostly limited to synthetic object-level datasets and cannot handle the variable scene scales and camera trajectories of real-world scenes.
- In terms of 3DGS editing: Utilizing 2D diffusion models to guide editing requires extra stages (texture adjustment, refinement) or complex cross-view consistency modules.
- 2D diffusion models have demonstrated the capability of training-free editing via inversion, but this paradigm has not been extended to 3DGS.
- Real-world scenes vary in scale and camera trajectories, necessitating the joint learning of camera pose distributions under a generative model.
- Inspired by 2D diffusion models, generative models directly modeling 3DGS should also be capable of achieving training-free editing through inversion and inpainting techniques.

## Method

### Overall Architecture

SplatFlow consists of two main components: (1) a multi-view rectified flow (RF) model that jointly generates multi-view image latents, depth latents, and Plücker ray coordinates (representing camera poses) conditioned on text prompts in the latent space; (2) a GSDecoder that converts these latent representations into pixel-aligned 3DGS representations. By sharing the latent space using SD3's frozen encoder and combining training-free SDEdit inversion and RePaint inpainting techniques, it supports both 3DGS editing and various 3D tasks.

### Key Designs

**1. Multi-View Rectified Flow Model**

- **Function**: To jointly generate multi-view consistent images, depth, and camera poses from text prompts.
- **Mechanism**: The image latent $\mathcal{E}(\bm{I}_i)$, depth latent $\mathcal{E}(\bm{D}_i)$, and Plücker ray $\bm{r}_i$ of each view are concatenated along the channel dimension to form $\bm{X}_i \in \mathbb{R}^{(2n+6) \times h \times w}$, with $K$ views constituting the input $Y_0 \in \mathbb{R}^{K \times (2n+6) \times h \times w}$. A conditional flow matching objective is trained on this representation. During sampling, the predicted result at $t=0$ is projected back onto the ray manifold at each step to maintain camera pose accuracy. The vector field of SD3 can be fused to enhance single-view quality.
- **Design Motivation**: The key advantages of jointly modeling camera poses and images over modeling them separately are: (1) it allows the flexible handling of multiple tasks via inpainting techniques (where known parts constrain the prediction of unknown parts), and (2) real-world scenes require adaptive camera poses.

**2. Gaussian Splatting Decoder (GSDecoder)**

- **Function**: To efficiently convert multi-view latent representations into pixel-aligned 3DGS.
- **Mechanism**: Designed based on a feed-forward 3DGS reconstruction method, taking as input the image latents, depth latents, and camera poses of $K$ views. Depth latent integration is introduced to enhance 3D structural information, utilizing DepthAnythingV2 to extract depth maps. A vision-aided adversarial loss is incorporated in the late stages of convergence to improve visual quality without compromising training stability. The architecture is initialized based on the SD3 decoder, with the addition of cross-view attention.
- **Design Motivation**: Although the frozen encoder ensures compatibility with 2D generative models, it may lose fine-grained spatial details. Depth latents supplement 3D structural information, and the adversarial loss enhances perceptual quality.

**3. Training-Free Inversion and Inpainting Editing**

- **Function**: To achieve 3DGS editing and multiple 3D tasks with only the generative task trained.
- **Mechanism**: For 3DGS editing: SDEdit inversion is applied to the input multi-view latents up to $t_k$, which are then resampled under the target text prompt condition to generate edited latents. For 3D tasks: Leveraging the joint modeling property, known data (such as multi-view images + depth) can be used as constraints to infer unknown parts (such as camera poses) via RePaint inpainting, achieving camera pose estimation and novel view synthesis.
- **Design Motivation**: Since 2D diffusion models have proven that inversion and inpainting are powerful training-free editing tools, extending them to multi-view 3D models is a natural progression.

### Loss & Training

- RF model: Conditional Flow Matching loss $\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,Y_t,Y_1}[\|u_t(Y_t|Y_1) - u_\theta(Y_t,t)\|_2^2]$
- GSDecoder: LPIPS + MSE + vision-aided adversarial loss (delayed activation)
- $K=8$ views setup, fine-tuned based on SD3, with adapted input/output channels and cross-view attention implemented.
- Training data: MVImgNet + DL3DV-7K subset, with text descriptions generated using Llava-One Vision Qwen 7B.

## Key Experimental Results

### Main Results

Text-to-3DGS generation (MVImgNet / DL3DV):

| Method | MVImgNet FID↓ | MVImgNet CLIP↑ | DL3DV FID↓ | DL3DV CLIP↑ |
|------|-------------|---------------|------------|-------------|
| Director3D | 39.55 | 30.48 | 88.44 | 30.04 |
| Director3D+SDS++ | 41.80 | 31.00 | 95.88 | 31.68 |
| **SplatFlow** | **34.85** | **31.43** | **79.91** | 30.06 |
| SplatFlow+SDS++ | 35.46 | **32.30** | 85.31 | **31.90** |

### Ablation Study

GSDecoder component ablation:

| Configuration | PSNR↑ | LPIPS↓ |
|------|-------|--------|
| Image latents only | 20.3 | 0.32 |
| + Depth latents | 22.1 | 0.26 |
| + Adversarial loss | **23.5** | **0.21** |

### Key Findings

1. SplatFlow outperforms Director3D on a smaller training dataset (FID 34.85 vs 39.55), proving the superiority of joint modeling.
2. Depth latent integration significantly improves the convergence speed and reconstruction quality of GSDecoder.
3. Training-free editing works well in 3DGS scenes without requiring additional cross-view consistency modules.
4. Ray manifold constraint during the sampling process is crucial for the accuracy of camera pose estimation.
5. Novel view synthesis and camera pose estimation can be directly achieved via inpainting techniques.

## Highlights & Insights

- A unified framework for 3DGS generation and editing is realized for the first time, enabling training-free editing and various 3D tasks with only the generative model being trained.
- The design of jointly modeling images + depth + camera poses is elegant, allowing inpainting techniques to flexibly infer any missing modalities.
- Sharing the SD3 encoder establishes compatibility with 2D generative models, enabling the integration of SD3 knowledge during sampling.
- The ray manifold constraint provides a novel technical insight specifically for rectified flow models.

## Limitations & Future Work

- The scale of training data is limited (MVImgNet + DL3DV-7K subset); scaling up to larger datasets could further improve quality.
- The 8-view setup may be insufficient to cover complex, large-scale scenes.
- The performance of training-free editing is limited by the quality of the RF model's generative priors.
- It can be extended to the generation and editing of dynamic 3DGS scenes in the future.
- Introducing finer-grained editing controls (e.g., local editing, physical constraints).

## Related Work & Insights

- **Director3D**: Generates camera poses from text and then multi-view images, whereas SplatFlow learns the joint distribution for better performance.
- **LucidDreamer / DreamScene**: Scene generation methods based on single-view inpainting or SDS, which are unstable under large trajectory changes.
- **SDEdit / RePaint**: Training-free editing/inpainting techniques in 2D diffusion models, which are extended to rectified flows and 3D scenes in this work.
- Insight: Lifting the diverse training-free capabilities (editing, inpainting, inversion) of 2D generative models to 3D is a promising research direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified generation-editing framework and the joint modeling approach are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two real-world datasets, covering generation, editing, NVS, and pose estimation.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with sufficient technical details.
- **Value**: ⭐⭐⭐⭐ — Provides a concise and unified solution for 3DGS generation and editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting](rainygs_efficient_rain_synthesis_with_physically-based_gaussian_splatting.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality](pop-gs_next_best_view_in_3d-gaussian_splatting_with_p-optimality.md)

</div>

<!-- RELATED:END -->
