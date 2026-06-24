---
title: >-
  [Paper Note] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step
description: >-
  [CVPR 2025][Video Generation][Video Diffusion Model Distillation] VideoScene proposes a 3D-aware Leap Flow Distillation strategy that distills a video diffusion model into a one-step generator. It generates 3D-consistent videos from two sparse-view images. Coordinated with a Dynamic Denoising Policy Network (DDPNet) that adaptively selects the optimal starting noise level, it compresses the generation time from 2 minutes to 3 seconds while maintaining high quality.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Diffusion Model Distillation"
  - "3D Scene Generation"
  - "Consistency Distillation"
  - "Sparse View Reconstruction"
  - "Gaussian Splatting"
date: 2026-05-08
content_hash: 082c58b9b5780b1e
---

# VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step

**Conference**: CVPR 2025  
**arXiv**: [2504.01956](https://arxiv.org/abs/2504.01956)  
**Code**: [https://hanyang-21.github.io/VideoScene](https://hanyang-21.github.io/VideoScene)  
**Area**: 3D Vision / Video Generation  
**Keywords**: Video Diffusion Model Distillation, 3D Scene Generation, Consistency Distillation, Sparse View Reconstruction, Gaussian Splatting

## TL;DR

VideoScene proposes a 3D-aware Leap Flow Distillation strategy that distills a video diffusion model into a one-step generator. It generates 3D-consistent videos from two sparse-view images. Coordinated with a Dynamic Denoising Policy Network (DDPNet) that adaptively selects the optimal starting noise level, it compresses the generation time from 2 minutes to 3 seconds while maintaining high quality.

## Background & Motivation

**Background**: Reconstructing 3D scenes from sparse views is a crucial yet severely under-constrained problem. Traditional methods include geometric regularization and feed-forward 3D reconstruction models (e.g., pixelSplat, MVSplat). Recently, large-scale video diffusion models (e.g., CogVideoX, Stable Video Diffusion) have shown the capability to generate videos with plausible 3D structures, leading some works to leverage video generation priors to assist 3D reconstruction.

**Limitations of Prior Work**: (1) Video diffusion models suffer from extremely slow inference speeds, requiring 50 denoising steps, with CogVideoX taking over 2 minutes for a single inference. (2) Video diffusion models lack 3D constraints since they are trained on 2D video data, focusing on RGB spatial and temporal consistency rather than geometric consistency, which often leads to spatial instability and camera geometric distortion in generated videos. (3) Although feed-forward 3D models are fast, they suffer from poor generation quality in unobserved regions.

**Key Challenge**: Video diffusion models possess powerful generative prior but are too slow and lack 3D constraints; feed-forward 3D models ensure geometric consistency but have limited generation capability. There is a critical need for a method that simultaneously combines the advantages of both.

**Goal**: Distill a video diffusion model into an efficient model that generates 3D-consistent videos in a single step, building an efficient bridge from video to 3D.

**Key Insight**: The authors observe that the bottleneck of standard Consistency Distillation lies in training with noise added to clean data, while performing inference from pure noise. This distribution gap heavily impacts one-step generation. In particular, the early steps of the diffusion process (high noise levels) are the most difficult and contain the least information. Therefore, a feed-forward 3D model can be utilized to provide a "coarse but 3D-consistent" starting point, skipping these difficult early steps.

**Core Idea**: Use a feed-forward 3DGS model (MVSplat) to generate a coarse but 3D-consistent video as the starting point, add an appropriate amount of noise instead of starting from pure noise, and cooperate with consistency distillation and a policy network to achieve high-quality one-step 3D video generation.

## Method

### Overall Architecture

Input: Two sparse-view images + camera poses. Pipeline: (1) MVSplat generates a coarse 3DGS feed-forwardly $\rightarrow$ Render the coarse video along the interpolated camera trajectory $\rightarrow$ Encode into latent $\mathbf{x}_0^r$; (2) Add noise to $\mathbf{x}_0^r$ up to timestep $t$ ($t < T$) to obtain $\mathbf{x}_t^r$; (3) Perform consistency distillation on the student model and teacher model using $\mathbf{x}_t^r$; (4) DDPNet learns to select the optimal $t$ for each sample. During inference: MVSplat rendering (~0.5s) + noise addition + one-step denoising (~2.5s) = total of approximately 3s.

### Key Designs

1. **3D-Aware Leap Flow Distillation**:

    - **Function**: Skip the ineffective denoising steps of high noise levels and utilize 3D prior information to accelerate consistency distillation.
    - **Mechanism**: Standard consistency distillation is trained on $t \in [0, T]$ and inferred at $T$. However, the information content at high $t$ is extremely low (close to pure noise). Instead, this paper first uses MVSplat to generate a coarse 3DGS scene from two input images and renders the video, which is then encoded into $\mathbf{x}_0^r$. During training, noise is added to $\mathbf{x}_0^r$ up to a random $t \in [0, T']$ ($T' < T$), and consistency distillation is performed at this intermediate timestep: $\mathcal{L}_D(\theta, \theta^-; \Phi) = \mathbb{E}[d(\mathbf{f}_\theta(\mathbf{x}_{t_{n+1}}^r, t_{n+1}), \mathbf{f}_{\theta^-}(\hat{\mathbf{x}}_{t_n}^\phi(\mathbf{x}_{t_{n+1}}^r), t_n))]$. During inference, the generation also starts from $\mathbf{x}_0^r$ instead of pure noise.
    - **Design Motivation**: Although the coarse 3D video contains artifacts and blurriness, it maintains the correct 3D geometric structure. It provides a much stronger starting prior than pure noise, allowing high-quality results to be generated with only a small number of denoising steps at low-to-medium noise levels. This also solves the training-inference distribution mismatch.

2. **Dynamic Denoising Policy Network (DDPNet)**:

    - **Function**: Adaptively determine the optimal noise addition timestep $t$ for each input.
    - **Mechanism**: The timestep selection is modeled as a Contextual Bandit problem, where the environmental state is the input video latent $\mathbf{x}_0^r$, the action is selecting $t \in [0, T']$, and the reward is the generation quality (negative MSE loss). DDPNet is a lightweight CNN (4-layer 2D convolution) that outputs the policy distribution $\pi_\psi(t|\mathbf{x}_0^r)$, optimized via policy gradient: $\mathcal{L}_{DDP}(\psi) = \mathbb{E}_{t \sim \pi_\psi}[r(\mathbf{x}_0^r, t)]$. During training, DDPNet is decoupled from distillation—it does not propagate gradients to the student model and is only fully trained for the first 4000 steps, after which it is frozen to prevent overfitting.
    - **Design Motivation**: The optimal noise level depends on input quality. If the MVSplat rendering quality is high (clear structure), only a small amount of noise is needed for detail refinement. If the rendering quality is poor (severe artifacts), more noise is required to allow the model sufficient degrees of freedom to reconstruct. A fixed $t$ cannot adapt to such variations.

3. **Video Fine-Tuning and Distillation based on CogVideoX**:

    - **Function**: Adapt the general video generation model to 3D scene video generation.
    - **Mechanism**: CogVideoX-5B-I2V is selected as the backbone. The attention layers are first fine-tuned on the RealEstate10K dataset for 900 steps (warm-up). Then, the 3DGS model is frozen, and distillation training is conducted for 20k steps, updating only the attention layers in the transformer blocks. The first and last frames are used as conditioning inputs.
    - **Design Motivation**: The fine-tuning phase allows the model to adapt to 3D scene distributions (static scenes, camera motions), while the distillation phase learns one-step generation on top of this.

### Loss & Training

The distillation loss uses the standard consistency distillation loss (Eq. 6), with the distance metric $d(\cdot,\cdot)$ employing Huber loss. DDPNet is optimized using policy gradient. Training is conducted on 8×A100 (80G) for 2 days, using the AdamW optimizer with a learning rate of $3 \times 10^{-5}$, batch size of 2, and 49-frame sampling.

## Key Experimental Results

### Main Results

Comparison of video generation quality with different step numbers on RealEstate10K:

| Method | Steps | FVD↓ | Aesthetic↑ | Subject Consist.↑ | BG Consist.↑ |
|------|------|------|-----------|-------------------|-------------|
| SVD | 50 | 424.68 | 0.4906 | 0.9305 | 0.9287 |
| DynamiCrafter | 50 | 458.27 | 0.5336 | 0.8898 | 0.9349 |
| CogVideoX-5B | 50 | 521.04 | 0.5368 | 0.9179 | 0.9460 |
| CogVideoX-5B | 1 | 753.02 | 0.3987 | 0.7842 | 0.8976 |
| **VideoScene** | **1** | **103.42** | **0.5416** | **0.9259** | **0.9461** |
| VideoScene | 50 | 98.67 | 0.5570 | 0.9320 | 0.9407 |

Cross-dataset generalization (trained on RealEstate10K, tested on ACID):

| Method | Steps | FVD↓ | Aesthetic↑ |
|------|------|------|-----------|
| CogVideoX-5B | 50 | 114.04 | 0.5491 |
| CogVideoX-5B | 1 | 464.87 | 0.4492 |
| **VideoScene** | **1** | **121.93** | **0.5274** |

### Ablation Study

| Configuration | FVD↓ | Aesthetic↑ | Subject↑ | BG↑ |
|------|------|-----------|---------|-----|
| Base rendered video | 171.38 | 0.4769 | 0.8794 | 0.9240 |
| w/o 3D-aware leap flow | 543.53 | 0.4092 | 0.7842 | 0.9160 |
| w/o DDPNet | 106.28 | 0.4897 | 0.8850 | 0.9205 |
| Full model | **97.53** | **0.5306** | **0.9139** | **0.9440** |

### Key Findings

- VideoScene's one-step generation (FVD 103.42) significantly outperforms the 50-step results of all baselines (best at 424.68), showing a massive gap.
- VideoScene's 1-step and 50-step results are very close (103.42 vs 98.67), indicating that distillation is almost lossless, whereas other models suffer severe degradation at 1 step.
- Removing the 3D-aware leap flow causes the FVD to skyrocket from 97.53 to 543.53—the 3D prior is the most critical core component.
- While DDPNet shows limited FVD improvement ($106 \rightarrow 97$), it significantly improves aesthetic quality and consistency.
- Strong cross-dataset generalization capability—even on the unseen ACID dataset, the 1-step results are close to those of the fine-tuned 50-step baseline.
- Feature matching analysis (RANSAC) confirms that the geometric consistency of frames generated by VideoScene far outperforms other methods.

## Highlights & Insights

- **Perfect Complementarity of 3D Priors and Generative Priors**: MVSplat provides geometric structures but lacks details, while video diffusion models provide generative details but lack geometric constraints. "Denoising from coarse 3D video with added noise" elegantly merges the two—constraining structures with 3D information and replenishing textures and details with the diffusion model.
- **Skip Rather Than Accelerate**: Distinguishing itself from typical step reduction ($50 \rightarrow 1$), this work directly skips the high-noise phase—because with the 3D prior, there is no need to start from pure noise. This is an acceleration mindset derived directly from the problem itself.
- **Adaptive Mechanism of DDPNet**: Learning "how much noise to add" using a Contextual Bandit is a novel design. It transforms hyperparameter selection into a learnable decision-making process.

## Limitations & Future Work

- Dependence on MVSplat's quality—if the feed-forward 3D model fails (due to severe occlusions/extreme viewpoints), the starting point of the entire pipeline becomes unreliable.
- Currently handles setups with only two input images; extending to more viewpoints requires additional design.
- DDPNet is frozen after 4000 steps of training, which may fail to adapt to changes in the model's capabilities during the latter stages of distillation.
- Inference still takes ~3 seconds. Although 40 times faster than 2 minutes, there remains a gap for real-time applications.
- As a distillation approach, it requires a powerful teacher model (CogVideoX-5B), which still entails high deployment costs.

## Related Work & Insights

- **vs Standard Consistency Distillation (LCM, etc.)**: LCM distills from pure noise, suffering from a distribution gap between training and inference. VideoScene utilizes a 3D prior to provide a better starting point, avoiding this fundamental issue.
- **vs SVD/DynamiCrafter**: These general video models generate dynamic videos (human motion, object interaction), while VideoScene focuses on 3D-consistent videos of static scenes—filtering out undesirable dynamic changes through 3D prior constraints.
- **vs MVSplat (Feed-forward 3D)**: MVSplat performs well in visible areas but poorly in unobserved areas; VideoScene utilizes the generative capacity of the diffusion model to complete unobserved areas, creating a sequential and complementary relationship.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of the 3D-prior + leap flow distillation + policy network is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, containing multiple baselines, step-number comparisons, cross-dataset generalization, ablation analysis, and matching analysis.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, though slightly heavy on mathematical notations.
- Value: ⭐⭐⭐⭐⭐ Delivering a 40x speedup alongside quality improvements, this directly drives forward the practical application of video-to-3D.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OSV: One Step is Enough for High-Quality Image to Video Generation](osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](../../ICCV2025/video_generation/steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[AAAI 2026\] Phased One-Step Adversarial Equilibrium for Video Diffusion Models](../../AAAI2026/video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)
- [\[ICLR 2026\] Realtime Video Frame Interpolation Using One-Step Diffusion Sampling](../../ICLR2026/video_generation/realtime_video_frame_interpolation_using_one-step_diffusion_sampling.md)

</div>

<!-- RELATED:END -->
