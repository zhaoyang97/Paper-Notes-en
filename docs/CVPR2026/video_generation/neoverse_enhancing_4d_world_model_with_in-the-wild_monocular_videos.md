---
title: >-
  [Paper Note] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos
description: >-
  [CVPR 2026][Video Generation][4D world model] NeoVerse proposes a scalable 4D world model that enables the entire training pipeline to leverage large-scale in-the-wild monocular videos (millions of clips) via feed-forward pose-free 4DGS reconstruction and online monocular degradation simulation, achieving state-of-the-art performance on both 4D reconstruction and novel-trajectory video generation.
tags:
  - CVPR 2026
  - Video Generation
  - 4D world model
  - Gaussian splatting
  - monocular video
  - novel view synthesis
  - feed-forward reconstruction
date: 2026-05-08
content_hash: 6954b88ea7ff77c0
---

# NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos

**Conference**: CVPR 2026
**arXiv**: [2601.00393](https://arxiv.org/abs/2601.00393)
**Code**: [https://neoverse-4d.github.io](https://neoverse-4d.github.io) (coming soon)
**Area**: Video Generation
**Keywords**: 4D world model, Gaussian splatting, monocular video, novel view synthesis, feed-forward reconstruction

## TL;DR

NeoVerse proposes a scalable 4D world model that enables the entire training pipeline to leverage large-scale in-the-wild monocular videos (millions of clips) via feed-forward pose-free 4DGS reconstruction and online monocular degradation simulation, achieving state-of-the-art performance on both 4D reconstruction and novel-trajectory video generation.

## Background & Motivation

1. **State of the Field**: 4D world modeling — a hybrid paradigm combining reconstruction and generation — holds great promise for autonomous driving, digital content creation, and related domains. Existing methods typically first reconstruct a 3D/4D representation, then leverage geometric priors to guide video generation models, enabling spatiotemporal consistency and precise viewpoint control.

2. **Limitations of Prior Work**: The core bottleneck of current approaches lies in **insufficient scalability**, manifesting at two levels:
    - **Poor data scalability**: Methods such as ViewCrafter handle only static scenes, while SynCamMaster/ReCamMaster rely on expensive multi-view dynamic videos, resulting in high data acquisition costs and limited generalization.
    - **Poor training scalability**: Methods such as TrajectoryCrafter and FreeSim require offline preprocessing (depth estimation, offline Gaussian field reconstruction), incurring substantial computational overhead, high storage consumption, and inflexible training pipelines.

3. **Root Cause**: Abundant and inexpensive in-the-wild monocular videos cannot be directly utilized due to the lack of multi-view supervision signals and efficient online processing pipelines.

4. **Paper Goals**: To make the entire 4D world model training pipeline fully scalable to diverse in-the-wild monocular videos.

5. **Starting Point**: The authors observe that if (a) feed-forward 4D reconstruction without pose estimation and (b) efficient online degradation rendering simulation can be realized, arbitrary monocular videos can be converted into training data.

6. **Core Idea**: By combining feed-forward 4DGS reconstruction with online monocular degradation simulation, the full pipeline of the 4D world model is made scalable to millions of in-the-wild monocular videos.

## Method

### Overall Architecture

NeoVerse consists of two stages: (1) **Reconstruction Stage**: a feed-forward pose-free 4DGS reconstruction model built upon VGGT, which takes monocular video as input and outputs a 4D Gaussian field; (2) **Generation Stage**: degraded images rendered from the 4DGS under novel trajectories serve as conditions for a video generation model (Wan-T2V 14B with a control branch) to produce high-quality videos. During training, the generation stage employs online reconstruction and degradation simulation, using the original monocular video itself as the supervision target.

### Key Designs

1. **Bidirectional Motion Modeling**:

    - **Function**: Assigns forward and backward linear and angular velocities to 4D Gaussians, enabling interpolation of Gaussian states at arbitrary timestamps.
    - **Mechanism**: Frame features $\{F_t\}$ output by VGGT are split along the temporal dimension into two groups, which respectively perform cross-attention to encode forward motion features ($t \to t+1$) and backward motion features ($t \to t-1$). These features are used to predict bidirectional velocities $v^+, v^-$ and angular velocities $\omega^+, \omega^-$ for each Gaussian, enabling linear interpolation to propagate keyframe Gaussians to non-keyframe timestamps.
    - **Design Motivation**: Unlike the unidirectional motion in 4DGT, the bidirectional design supports efficient sparse-keyframe reconstruction (rendering $N$ frames from only $K$ input frames), substantially reducing online reconstruction overhead and enabling downstream applications requiring temporal control.

2. **Monocular Degradation Simulation**:

    - **Function**: Automatically generates degraded-rendering / original-video paired training data from monocular videos at training time.
    - **Mechanism**: Three complementary degradation modes are employed: (a) **Visibility-based Gaussian pruning** — a random transformation is applied to the camera trajectory to obtain a novel trajectory; occluded Gaussians are pruned using depth information and then re-rendered to the original viewpoint, simulating occlusion degradation; (b) **Average geometric filter** — mean filtering is applied to the novel-view depth map and Gaussian centers are adjusted accordingly, simulating flying pixels at depth-discontinuity edges; (c) A larger filter kernel variant of (b) is used to simulate depth-error distortions over a wider range.
    - **Design Motivation**: Multi-view datasets provide training pairs directly, but monocular videos do not. The three degradation modes are designed from first principles of geometric relationships, are simple yet effective, and enable arbitrary monocular videos to serve as training data.

3. **Efficient On-the-fly Reconstruction with Sparse Keyframes**:

    - **Function**: Avoids per-frame feed-forward inference on long videos, improving training efficiency.
    - **Mechanism**: Given an $N$-frame video, only $K$ keyframes are selected for feed-forward reconstruction; the bidirectional motion mechanism interpolates Gaussians to the remaining frames for efficient rendering. Frame-to-frame natural transitions are achieved via a time-varying opacity decay function $\alpha_i(t_q) = \alpha_i \exp(-\gamma \cdot d(t_q, t)^{1/(1-\tau_i)})$.
    - **Design Motivation**: Feed-forward network inference remains the training bottleneck, while rendering is extremely efficient. Using 11 keyframes to reconstruct an 81-frame video reduces reconstruction time to only 2 seconds.

### Loss & Training

- **Reconstruction Loss**: $\mathcal{L}_{recon} = \mathcal{L}_{rgb} + \lambda_1\mathcal{L}_{camera} + \lambda_2\mathcal{L}_{depth} + \lambda_3\mathcal{L}_{motion} + \lambda_4\mathcal{L}_{regular}$, comprising photometric loss (L2 + LPIPS), camera parameter loss, depth loss, bidirectional velocity supervision, and opacity regularization.
- **Generation Loss**: Rectified Flow is adopted; a control branch is trained on top of Wan-T2V 14B with the main generative model frozen (compatible with distillation LoRA for acceleration).
- **Two-stage Training**: Stage 1 trains the reconstruction model for 150K iterations; Stage 2 trains the generation model for 50K iterations, using 32 A800 GPUs.
- **Global Motion Tracking**: At inference time, dynamic and static Gaussians are separated by cross-frame visibility-weighted maximum velocity; different temporal aggregation strategies are applied to each category.

## Key Experimental Results

### Main Results

| Dataset | Metric | NeoVerse | AnySplat | NoPoSplat |
|--------|------|----------|----------|-----------|
| VRNeRF (static) | PSNR↑ | **20.73** | 18.02 | 11.27 |
| VRNeRF (static) | LPIPS↓ | **0.352** | 0.366 | 0.620 |
| Scannet++ (static) | PSNR↑ | **25.34** | 22.79 | 8.69 |
| ADT (dynamic) | PSNR↑ | **32.56** | - | - |
| DyCheck (dynamic) | PSNR↑ | **11.56** | - | 9.32 |

| Method | Total Inference Time (s) | Subj. Consist. | Back. Consist. | Imag. Quality |
|------|-------------|----------------|----------------|---------------|
| TrajectoryCrafter | 146 | 83.02 | 88.58 | 54.59 |
| ReCamMaster | 168 | 88.21 | 91.60 | 58.87 |
| NeoVerse (11 key) | **20** | 88.43 | 92.27 | 59.75 |
| NeoVerse (21 key) | **21** | **88.73** | **92.43** | 60.01 |

### Ablation Study

| Configuration | DyCheck PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------|-------|--------|
| w/o bidirectional motion | 11.27 | 0.285 | 0.570 |
| w/o opacity regularization | 10.86 | 0.244 | 0.576 |
| Full reconstruction model | 11.56 | 0.293 | 0.558 |
| Full pipeline (+ generation) | **14.59** | **0.323** | **0.501** |

### Key Findings

- Bidirectional motion modeling contributes significantly; removing it causes a 0.29 drop in DyCheck PSNR.
- The generation stage yields a substantial quality improvement (PSNR: 11.56 → 14.59), validating the effectiveness of the reconstruction-generation hybrid paradigm.
- Sparse keyframes (11 vs. all 81 frames) have minimal impact on generation quality, yet reduce inference time from 28s to 20s (7× faster than TrajectoryCrafter).
- On VBench evaluation, NeoVerse comprehensively outperforms TrajectoryCrafter and ReCamMaster on subject consistency, background consistency, and image quality.

## Highlights & Insights

- **Core Insight**: The bottleneck of 4D world models lies not in model architecture but in data and training scalability. Online degradation simulation elegantly converts monocular videos into multi-view training pairs, eliminating the dependency on expensive multi-view data.
- The **sparse keyframe reconstruction** design is particularly elegant — it exploits the fact that Gaussian rendering is far faster than network inference, reducing feed-forward inference cost by several times with negligible quality loss.
- The strategy of **freezing the generative model and training only a control branch** allows NeoVerse to directly integrate distillation LoRA, reducing generation inference time to just 18 seconds.
- Degradation simulation is grounded in first principles (geometric occlusion, depth averaging), requiring no additional learned noise models.

## Limitations & Future Work

- The current resolution is fixed at 336×560, falling short of the high-resolution requirements of practical applications.
- Global motion tracking relies on threshold-based separation of dynamic and static components, which may be insufficient for slowly moving objects.
- Although grounded in first principles, the degradation simulation remains an approximation; the actual degradation patterns in real novel-view rendering may be more complex.
- Training requires 32 A800 GPUs, representing a significant cost barrier for academic laboratories.

## Related Work & Insights

- **vs. TrajectoryCrafter**: Both are reconstruction-generation hybrid methods, but TrajectoryCrafter relies on offline preprocessing, limiting data scale; NeoVerse employs a fully online pipeline scalable to millions of videos and achieves 7× faster inference.
- **vs. ReCamMaster**: Pure generation methods yield strong visual quality but imprecise trajectory control; NeoVerse achieves both generation quality and precise trajectory control.
- **vs. AnySplat**: AnySplat targets pose-free reconstruction of static scenes; NeoVerse extends to 4D dynamic scenes with a 2.7 dB PSNR advantage.
- This work provides a viable path toward training world models on large-scale internet videos.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of bidirectional motion modeling and online degradation simulation is elegant, though individual modules are not entirely novel in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of static/dynamic reconstruction, generation quality, inference efficiency, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated problem formulation, and consistent notation system.
- Value: ⭐⭐⭐⭐⭐ Addresses the data bottleneck in 4D world model training with strong practical impact.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)
- [\[CVPR 2026\] The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)

<!-- RELATED:END -->
