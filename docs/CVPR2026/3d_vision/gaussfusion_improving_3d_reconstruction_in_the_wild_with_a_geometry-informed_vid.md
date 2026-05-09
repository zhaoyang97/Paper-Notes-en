---
title: >-
  [Paper Note] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes GaussFusion, a geometry-informed video-to-video generative model that conditions a video generator on a rendered Gaussian Primitives Buffer (GP-Buffer) — encoding depth, normals, opacity, and covariance — to effectively remove floaters, flickering, and blurring artifacts in 3DGS reconstructions. The framework is compatible with both optimization-based and feed-forward reconstruction paradigms, and its distilled variant achieves real-time inference at 16 FPS.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - video generative model
  - geometry prior
  - artifact removal
  - real-time inference
date: 2026-05-08
content_hash: e21c52cd32c8cd10
---

# GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator

**Conference**: CVPR 2026
**arXiv**: [2603.25053](https://arxiv.org/abs/2603.25053)
**Code**: N/A
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, video generative model, geometry prior, artifact removal, real-time inference

## TL;DR
This paper proposes GaussFusion, a geometry-informed video-to-video generative model that conditions a video generator on a rendered Gaussian Primitives Buffer (GP-Buffer) — encoding depth, normals, opacity, and covariance — to effectively remove floaters, flickering, and blurring artifacts in 3DGS reconstructions. The framework is compatible with both optimization-based and feed-forward reconstruction paradigms, and its distilled variant achieves real-time inference at 16 FPS.

## Background & Motivation
1. **Background**: 3D Gaussian Splatting (3DGS) has become the dominant 3D reconstruction representation, with two main technical paradigms: per-scene optimization and feed-forward prediction.
2. **Limitations of Prior Work**: Both paradigms still produce severe artifacts under sparse-view and under-covered scenarios — including floaters, flickering, blurring, and geometric errors. Existing repair methods (e.g., Difix3D, GenFusion, ExploreGS) condition solely on RGB renderings, making them unable to handle large-scale floaters and missing regions; moreover, they are typically trained for a specific reconstruction paradigm and fail to generalize across paradigms.
3. **Key Challenge**: Existing methods exploit only the color information of Gaussian primitives, ignoring rich geometric cues such as depth, opacity, normals, and covariance. Additionally, the lack of diverse artifact simulation in training data causes models to overfit to particular reconstruction pipelines.
4. **Goal**: How can a single model be trained to handle artifacts from both optimization-based and feed-forward 3DGS?
5. **Key Insight**: (1) Encode all primitive attributes of 3DGS as a pixel-aligned video representation (GP-Buffer), providing richer geometric cues than RGB alone; (2) design a comprehensive artifact simulation pipeline covering multiple degradation modes.
6. **Core Idea**: Condition a video generative model on a GP-Buffer that encodes complete Gaussian primitive geometry, combined with a cross-paradigm artifact simulation strategy, to achieve general-purpose 3DGS artifact removal.

## Method

### Overall Architecture
Given an existing 3DGS reconstruction $\mathcal{G}$, the method first renders a GP-Buffer (color, depth, normals, opacity, and covariance uncertainty) along novel-view trajectories. The GP-Buffer is then encoded and injected into a flow-matching video generator based on Wan-2.1, producing high-quality, artifact-free video frames. The generated frames are subsequently used to further refine the 3D reconstruction. The inputs are multi-view images and camera parameters; the output is a repaired 3DGS representation.

### Key Designs

1. **Gaussian Primitives Buffer (GP-Buffer)**

    - **Function**: Encodes the complete multi-modal information of 3DGS primitives as a pixel-aligned video representation.
    - **Mechanism**: Five channels are rendered — color $\mathbf{C}$, opacity $A$, depth $D$, normals $\mathbf{N}$, and geometric uncertainty $\mathbf{U}$. Normals are derived from camera-space position maps via finite differences: $\mathbf{N}(\mathbf{u}) = \text{normalize}(\partial_u \mathbf{P}_{\text{cam}} \times \partial_v \mathbf{P}_{\text{cam}})$. Geometric uncertainty is rendered by alpha-blending the unique elements of the inverse covariance matrix; low-texture regions represented by a few large Gaussians yield low values, while high-frequency regions yield higher values — providing a measure of local structural regularity.
    - **Design Motivation**: When conditioned only on RGB, the model struggles to distinguish correct renderings from artifacts, especially in the presence of large missing regions and geometric errors. The geometric channels in the GP-Buffer give the model the ability to "see through" artifacts. Ablation studies confirm that each additional geometric modality consistently improves performance.

2. **Geometry Adapter (GA)**

    - **Function**: Injects encoded GP-Buffer information into the DiT backbone of the video generator.
    - **Mechanism**: The five modalities of the GP-Buffer are each encoded into video latents via a VAE, concatenated, and aligned in spatial and channel dimensions via 3D convolution. The GA block operates as a parallel side network to the DiT, containing self-attention (for processing geometric features) and cross-attention (for fusing text descriptions), producing geometry-aware features $\mathbf{x}_g$ that are added to the main video latents: $\mathbf{x} \leftarrow \mathbf{x} + \mathbf{x}_g$. During training, the base model is frozen and only the GA layers are trained.
    - **Design Motivation**: Directly adding conditional latents to noisy latents (as in GenFusion and ExploreGS) is suboptimal. The GA achieves better geometric alignment through hierarchical geometric feature injection; ablations show PSNR improves from 20.90 to 22.55.

3. **Comprehensive Artifact Simulation Pipeline**

    - **Function**: Generates training data covering multiple reconstruction paradigms.
    - **Mechanism**: Four artifact source strategies are employed — (1) sparse-view simulation: randomly retaining 5% of frames (superior to uniform downsampling); (2) diverse initialization: SfM, random point clouds, and MapAnything dense point maps; (3) paired reconstruction: clean models use all views and full optimization, while degraded models use sparse subsets and reduced optimization steps; (4) feed-forward degradation: directly rendering predicted Gaussians from a feed-forward model (DepthSplat), introducing geometry inconsistencies and semi-transparent artifacts characteristic of feed-forward methods. In total, 75K+ paired video samples are generated.
    - **Design Motivation**: Prior methods simulate artifacts only through uniform downsampling and underfitting, causing the model to repair only optimization-based 3DGS artifacts. Mixing multiple degradation modes enables cross-paradigm generalization.

### Loss & Training
Training uses a flow-matching objective $\mathcal{L} = \mathbb{E}[\|u_\theta(x_t, c, t) - v_t\|^2]$. A two-stage fine-tuning strategy is adopted for efficient inference: in the first stage, Distribution Matching Distillation (DMD) distills the multi-step generator into a 4-step model; in the second stage, the distilled model is frozen and only the GA layers are fine-tuned. The base model is Wan-2.1-1.3B; the GA introduces an additional 0.6B parameters. Training runs for 100K steps on 8×H200 GPUs.

## Key Experimental Results

### Main Results (DL3DV dataset, optimization-based 3DGS repair)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Inference Speed |
|------|--------|--------|---------|-------|---------|
| Splatfacto (baseline) | 17.42 | 0.605 | 0.412 | 6.49 | 118.3 FPS |
| GenFusion | 18.36 | 0.690 | 0.391 | 9.98 | 1.1 FPS |
| Difix3D+ | 20.10 | 0.765 | 0.302 | 4.22 | 12.8 FPS |
| ExploreGS | 20.69 | 0.760 | 0.345 | 6.27 | 1.2 FPS |
| **Ours (Full)** | **22.55** | **0.832** | **0.278** | **3.93** | 4.3 FPS |
| **Ours (Few-step)** | **22.49** | **0.842** | **0.288** | 7.38 | **15.1 FPS** |

### Ablation Study (GP-Buffer modality ablation, DL3DV)

| RGB | Depth | Normal | Alpha | Cov. | PSNR ↑ | LPIPS ↓ | FID ↓ |
|-----|-------|--------|-------|------|--------|---------|-------|
| ✓ | | | | | 19.15 | 0.385 | 15.45 |
| ✓ | ✓ | | | | 19.29 | 0.361 | 10.54 |
| ✓ | ✓ | ✓ | | | 19.74 | 0.355 | 10.29 |
| ✓ | ✓ | ✓ | ✓ | | 19.96 | 0.344 | 8.61 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **20.75** | **0.329** | **6.72** |

### Key Findings
- Each geometric modality in the GP-Buffer contributes independently. The covariance uncertainty channel (Cov.), though often overlooked, yields the largest FID improvement (8.61→6.72).
- Joint training (mixing multiple datasets and degradation types) outperforms single-dataset training, confirming the importance of cross-paradigm artifact simulation.
- GaussFusion also improves performance on the feed-forward model DepthSplat (PSNR 21.77→22.80), whereas Difix3D+ and ExploreGS actually degrade PSNR on feed-forward models.
- The 4-step distilled model incurs negligible degradation in PSNR/SSIM/LPIPS, with only a slight FID increase (3.93→7.38), achieving real-time inference at 16 FPS.
- The Geometry Adapter outperforms direct conditional latent addition by 1.6 dB in PSNR.

## Highlights & Insights
- **The GP-Buffer design is highly insightful**: By rendering the complete attributes of Gaussian primitives — rather than color alone — the repair model gains an "X-ray"-like ability to perceive the scene. In particular, the covariance uncertainty channel enables the model to identify regions covered by a small number of large Gaussians (i.e., poorly reconstructed areas).
- **Paradigm-agnostic repair**: Through the comprehensive artifact simulation strategy, a single model handles artifacts from both optimization-based and feed-forward 3DGS — a capability no prior method has demonstrated. This is of significant practical importance for deployment.
- **Practical value of distillation**: Real-time inference at 16 FPS enables GaussFusion to repair frames "on-the-fly" during rendering, eliminating the need for offline post-processing.

## Limitations & Future Work
- As a video generative model, GaussFusion introduces an additional 0.6B parameters even after distillation, imposing relatively high memory and compute requirements.
- Generated frames may lose high-frequency details under extreme viewpoint changes (reflected in higher FID after distillation).
- After generating repaired frames, the current method still requires re-optimizing the 3DGS, making the overall pipeline non-end-to-end.
- The VAE encoder in the GP-Buffer was originally designed for RGB; while reconstruction error for other modalities is below 1%, dedicated multi-modal encoders may be beneficial in future work.

## Related Work & Insights
- **vs Difix3D+**: Processes each frame independently with an image diffusion model, lacking multi-view consistency and unable to remove large-scale floaters. GaussFusion ensures temporal consistency through a video generator.
- **vs MVSplat360**: Tailored specifically for the MVSplat feed-forward model and cannot generalize to optimization-based 3DGS. GaussFusion achieves cross-paradigm generality through mixed training.
- **vs ExploreGS / GenFusion**: Their conditioning strategy (RGB only) and limited training data diversity constrain repair capability and generalization. GaussFusion's GP-Buffer and comprehensive artifact simulation address both shortcomings.
- The central insight of this paper is that **fully exploiting the geometric information inherent in the reconstruction itself** is more effective than relying solely on external generative priors for 3D reconstruction repair.

## Rating
- Novelty: ⭐⭐⭐⭐ The GP-Buffer design and paradigm-agnostic training strategy represent meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset, multi-paradigm, comprehensive ablations, and speed comparisons — extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly articulated motivation.
- Value: ⭐⭐⭐⭐⭐ Real-time inference combined with cross-paradigm generalization confers high practical value for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](../../ICCV2025/3d_vision/vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2026\] Scene Grounding In the Wild](scene_grounding_in_the_wild.md)
- [\[CVPR 2026\] DROID-W: DROID-SLAM in the Wild](droid-slam_in_the_wild.md)

</div>

<!-- RELATED:END -->
