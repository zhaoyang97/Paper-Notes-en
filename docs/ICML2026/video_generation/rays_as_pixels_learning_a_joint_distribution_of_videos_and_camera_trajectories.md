---
title: >-
  [Paper Note] Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories
description: >-
  [ICML 2026][Video Generation][Video Diffusion] The authors represent per-pixel camera rays (origin + direction) as "raxel maps"—3-channel tensors with the same shape as RGB images. By processing these maps through a pre-…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Video Diffusion"
  - "Camera Pose"
  - "Joint Distribution"
  - "raxel"
  - "Decoupled Self-Cross Attention"
date: 2026-05-08
content_hash: 16821c2fbdfd847b
---

# Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories

**Conference**: ICML 2026  
**arXiv**: [2604.09429](https://arxiv.org/abs/2604.09429)  
**Code**: https://wbjang.github.io/raysaspixels/ (Project Page)  
**Area**: Video Generation  
**Keywords**: Video Diffusion, Camera Pose, Joint Distribution, raxel, Decoupled Self-Cross Attention

## TL;DR
The authors represent per-pixel camera rays (origin + direction) as "raxel maps"—3-channel tensors with the same shape as RGB images. By processing these maps through a pre-trained video VAE and using Decoupled Self-Cross Attention within a Flow Matching DiT, the model learns a joint distribution that simultaneously supports pose estimation, camera-controlled generation, and joint video-trajectory generation with a single set of weights.

## Background & Motivation
**Background**: In 3D vision, "recovering camera parameters from images" (inverse problems like SfM/COLMAP, DUSt3R, VGGT) and "rendering new views given camera parameters" (forward problems like NeRF, 3DGS, and camera-controlled video diffusion such as MotionCtrl, VD3D, or Wonderland) have historically existed as separate pipelines with independent training and evaluation.

**Limitations of Prior Work**: Existing camera-controlled video diffusion models treat poses as fixed inputs provided by external estimators (COLMAP, DUSt3R). However, these upstream estimators are prone to failure in scenarios with sparse inputs or ambiguous viewpoints, causing downstream generation to inherit this fragility. Conversely, pure estimators like VGGT only output geometry and lack the ability to "hallucinate" plausible pixels in occluded areas. Few joint works (e.g., Matrix3D) require 3DGS post-processing, meaning rendering is not end-to-end from the diffusion model.

**Key Challenge**: Pre-trained video diffusion models operate on dense spatial tensors ($H \times W \times 3$ RGB latents), whereas camera parameters are low-dimensional global matrices ($K, R, T$ matrices). Due to this structural mismatch, previous works either used adapters to project matrices into tokens (structural inadequacy) or used Plücker embeddings (6 channels, incompatible with VAEs, requiring MLP concatenation that bypasses pre-trained priors). Consequently, the camera remains an external condition rather than an integral part of the generative loop.

**Goal**: Enable a single pre-trained video DiT to learn the **joint distribution** $p(z, r)$ of video frames $z$ and camera trajectories $r$. This allows the same weights to sample $p(r|z)$ (pose estimation), $p(z|r)$ (trajectory-conditioned generation), and $p(z, r)$ (joint generation), while maintaining cycle self-consistency.

**Key Insight**: Since video VAEs already compress 3-channel dense images effectively, camera parameters are "disguised" as 3-channel dense maps. Instead of RGB, each pixel stores the vector sum $\mathbf{d} + \mathbf{o}$ of the corresponding ray in a canonical coordinate system. This allows the same VAE, DiT, and RoPE to encode cameras with zero structural modifications.

**Core Idea**: Use "rays as pixels" (raxel) to represent cameras as 3-channel latents isomorphic to video frames. Jointly denoise video and ray latents using Decoupled Self-Cross Attention, treating the camera as a dual modality rather than an external condition.

## Method

### Overall Architecture
During training, the model receives three types of frames: a target video $V$ of length $N$, $N_s$ clean source frames (conditions), and $N_t$ noisy sparse target frames (supervised), totaling a fixed number. Each frame $I_j$ is paired with extrinsic and intrinsic parameters $(K_j, P_j)$.

1.  **Normalization**: A source frame $s$ is randomly selected as the origin. All extrinsics are converted to relative poses $P_{\text{rel}}^{(j)} = P_s^{-1} P_j$ such that $s$ corresponds to the identity matrix. This ensures the model learns relative geometry rather than absolute world coordinates.
2.  **Raxel Map Construction**: For each pixel $\mathbf{u}$, it is back-projected using $K_j^{-1}$ into a unit direction in the camera frame, then transformed by $R_{\text{rel}}^{(j)}$ into the canonical world frame to obtain $\mathbf{d}$; the origin is $\mathbf{o} = T_{\text{rel}}^{(j)}$. Each raxel pixel stores $\mathbf{d} + \mathbf{o} \in \mathbb{R}^3$. The resulting map is $R_j \in \mathbb{R}^{H_r \times W_r \times 3}$ (using $H/2 \times W/2$ to save tokens).
3.  **Shared VAE Encoding**: Video frames use a spatio-temporal VAE $\mathcal{E}$ with 4x temporal compression to get $z_v$. Raxel maps are encoded without temporal compression to get $r_v, r_s, r_t$, aligning them with the video latent space.
4.  **Joint Denoising**: The latents $x = [z, r]$ are concatenated along the sequence dimension. A single velocity field $v_\theta$ is trained using Flow Matching with linear interpolation $x_t = (1-t)x_0 + t x_1$. Each DiT layer replaces standard self-attention with Decoupled Self-Cross Attention (DSCA).
5.  **Inference Modes** (asymmetric schedule): Fixing $z = z_s$ and denoising $r$ yields pose estimation; fixing $r$ and denoising $z$ yields trajectory-conditioned generation; denoising both yields joint generation.

The backbone is based on Wan 2.1 14B T2V, with an additional ray branch (independent LN, FFN, linear, 6B parameters) for a total of 20B parameters, all fine-tuned.

### Key Designs

1.  **Raxel Representation: Cameras as 3-Channel Images**:
    - **Function**: Encodes camera parameters into dense tensors with the same shape and channel count as video frames, compatible with pre-trained video VAEs.
    - **Mechanism**: Each pixel stores $\mathbf{d} + \mathbf{o}$, where $\mathbf{d} = R_{\text{rel}} K^{-1}\tilde{\mathbf{u}} / \|K^{-1}\tilde{\mathbf{u}}\|_2$ is the unit ray direction in world coordinates and $\mathbf{o} = T_{\text{rel}}$ is the camera origin. Compared to Plücker (6 channels, $[R\mathbf{d}, R\mathbf{d}\times T]$), raymaps (6 channels), or pointmaps (requires depth), raxel is the only solution that is spatially aligned, 3-channel compatible with VAEs, and requires no depth. Pose recovery is achieved via Orthogonal Procrustes between decoded $\hat{R}_k$ and reference $\hat{R}_s$, with focal length estimated via Median-of-Ratios $\hat{f}_x = \text{median}(u \cdot \hat{z} / \hat{x})$.
    - **Design Motivation**: This allows the camera to leverage pre-trained video priors directly rather than bypassing the VAE with an adapter. Ablations replacing raxel with Plücker embeddings (MLP projection without VAE) showed FID increasing from 7.33 to 21.97 and FVD from 68 to 333, proving the importance of a shared latent space.

2.  **Decoupled Self-Cross Attention (DSCA)**:
    - **Function**: Decouples attention for video latents $z$ and ray latents $r$ within each DiT block into intra-modal self-attention and inter-modal cross-attention.
    - **Mechanism**: The first step performs self-attention on $z$ and $r$ independently to ensure temporal smoothness in video and geometric coherence in trajectories. The second step uses symmetric cross-attention: $z \leftarrow r$ makes the video follow the trajectory, while $r \leftarrow z$ refines the rays based on visual cues. Since $z$ and $r$ are spatially aligned, RoPE is applied to both query/key in cross-attention to maintain the spatial correspondence. This reflects the probabilistic decomposition: $\log p(z, r) = \log p(r) + \log p(z|r) \equiv \log p(z) + \log p(r|z)$.
    - **Design Motivation**: A single global self-attention across the whole sequence often fails to couple the modalities deeply. DSCA significantly improves cycle consistency, reducing FID from 8.69 to 7.33 and $R_{\text{err}}$ from 0.048 to 0.020.

3.  **Flow Matching + Cosine Loss + Asymmetric Inferencing**:
    - **Function**: Enables a single velocity field $v_\theta(x_t, t)$ to learn two structurally different latents and allows different step counts for each during inference.
    - **Mechanism**: The training objective adds a cosine similarity term to the MSE: $\mathcal{L}(\theta) = \mathbb{E}[\|v_\theta - u_t\|^2 + \lambda (1 - v_\theta^\top u_t / (\|v_\theta\|\|u_t\|))]$, with $\lambda = 0.5$, to penalize directional deviations. During inference, cameras can be sampled in very few steps (optimal rotation accuracy is reached in just **2 steps**), while video frames require more steps.
    - **Design Motivation**: Ray latents are smooth and low-frequency, while video latents are high-frequency and information-dense. Without the cosine term, the direction of rays is often ignored due to the magnitude of video latent gradients.

### Loss & Training
- Flow Matching MSE combined with a cosine direction term ($\lambda = 0.5$).
- Datasets: Re10K and DL3DV, using ORB-SLAM and COLMAP poses aligned to a metric scale.
- Resolution: 480×832 center crop, maintaining original aspect ratios.
- Time-reversal augmentation: Both trajectories and their reversals are used to encourage scene-trajectory decoupling.
- Fixed source/target frame ratios to train the model to handle source frames at any temporal position.

## Key Experimental Results

### Main Results: Camera-Controlled Video Generation (Table 4)

| Dataset | Method | FID ↓ | FVD ↓ | $R_{\text{err}}$ ↓ | $T_{\text{err}}$ ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Re10K | MotionCtrl | 22.58 | 229.34 | 0.231 | 0.794 |
| Re10K | Wonderland | 16.16 | 153.48 | 0.046 | 0.093 |
| Re10K | Kaleido | 18.04 | 103.03 | 0.049 | 0.181 |
| Re10K | **Ours** | **15.76** | **98.72** | 0.056 | 0.115 |
| DL3DV-140 | Wonderland | 17.74 | 169.34 | 0.061 | 0.130 |
| DL3DV-140 | Kaleido | 41.18 | 458.60 | 0.011 | 0.026 |
| DL3DV-140 | **Ours** | **9.73** | **102.52** | 0.098 | 0.192 |
| T&T | Wonderland | 19.46 | 189.32 | 0.094 | 0.172 |
| T&T | Kaleido | 14.84 | 245.09 | 0.016 | 0.086 |
| T&T | **Ours** | **13.02** | **187.03** | 0.105 | 0.192 |

Ours achieves superior visual quality (FID/FVD) across all benchmarks without explicit 3D representations. While Kaleido shows better pose adherence ($R_{\text{err}}$), its visual quality drops significantly on DL3DV, suggesting it overfits to pose supervision at the expense of realism.

### Ablation Study: Cycle Self-Consistency on DL3DV-140 (Table 2)

| Configuration | FID ↓ | FVD ↓ | $R_{\text{err}}$ ↓ | $T_{\text{err}}$ ↓ |
| :--- | :--- | :--- | :--- | :--- |
| Ours (full) | 7.33 | 68.17 | 0.020 | 0.018 |
| w/o DSCA | 8.69 | 77.08 | 0.048 | 0.052 |
| w/o Cosine Sim. Loss | 9.48 | 97.84 | 0.058 | 0.094 |
| Plücker Embedding | 21.97 | 333.56 | 0.241 | 0.430 |

The cycle process involves sampling a trajectory $r' \sim p(r|z)$ from a video, then re-generating the video $z' \sim p(z|r', I_s)$. Single-conditional models cannot pass this test.

### Key Findings
- **Raxel is the Primary Driver**: Switching to Plücker embeddings caused a catastrophic drop in metrics, proving that shared VAE latent space represents a powerful, undervalued resource.
- **Rays Converge Faster**: Pose estimation reaches peak accuracy in just **2 steps**. Further steps (5 or 20) do not significantly improve or even slightly degrade rotation accuracy, allowing for high-speed inference of trajectories.
- **vs VGGT**: While the pure estimator VGGT is slightly more accurate in rotation (88.37 vs 91.86), ours provides video generation as a primary output with sufficient pose accuracy as a byproduct.
- **Shared Velocity Field Requirements**: Explicit direction modeling via the cosine loss is essential to prevent low-frequency ray signals from being overwhelmed by high-frequency video gradients.

## Highlights & Insights
- **Modality Masking Template**: The strategy of disguising non-visual modalities as images compatible with pre-trained latents is a generalizable pattern for adding modalities (segmentation, depth, etc.) to video diffusion models.
- **Probability Alignment with Architecture**: The probabilistic decomposition $\log p(z, r) = \log p(r) + \log p(z|r)$ is mapped directly to transformer operators (self-attn for marginals, cross-attn for conditionals).
- **Cycle Consistency as a Benchmark**: This provides a rigorous "sanity check" that single-conditional models physically cannot pass, serving as a new standard for unified 3D generative models.
- **Asymmetric Scheduling**: Using different step counts for different modalities within the same denoiser is a promising direction for accelerating multi-modal diffusion.

## Limitations & Future Work
- The training data focuses on static scenes and smooth trajectories; performance on dynamic objects (people, cars) and fast camera motion is currently unknown.
- The 4x temporal compression in the VAE may limit the reconstruction of high-frame-rate or extremely fast motion.
- Pose accuracy still trails specialized SfM models like VGGT because video latents do not explicitly encode depth.
- The 20B total parameter count is high, and the scalability of raxel+DSCA to smaller backbones remains to be verified.

## Related Work & Insights
- **vs Matrix3D**: Matrix3D uses RGB+pose+depth joint modeling but relies on a 3DGS optimization layer for rendering. Ours generates video frames directly, inheriting temporal priors from massive video data.
- **vs VD3D / MotionCtrl**: These models treat cameras as external conditions. By modeling the joint distribution, ours allows for the inverse $p(r|z)$.
- **vs Kaleido**: Kaleido injects cameras as positional embeddings in image diffusion. Ours stays in the video backbone domain, maintaining better temporal consistency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance](epic_efficient_video_camera_control_learning_with_precise_anchor-video_guidance.md)
- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](../../CVPR2026/video_generation/symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)
- [\[ICML 2026\] Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)

</div>

<!-- RELATED:END -->
