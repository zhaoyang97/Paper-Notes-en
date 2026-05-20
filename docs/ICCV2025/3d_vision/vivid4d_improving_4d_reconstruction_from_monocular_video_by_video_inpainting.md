---
title: >-
  [Paper Note] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting
description: >-
  [ICCV 2025][3D Vision][4D reconstruction] This paper proposes Vivid4D, which reformulates multi-view augmentation from monocular video as a video inpainting problem — warping the video to novel viewpoints using monocular…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D reconstruction"
  - "monocular video"
  - "video inpainting"
  - "diffusion models"
  - "view augmentation"
date: 2026-05-08
content_hash: be91632be8a21add
---

# Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting

**Conference**: ICCV 2025
**arXiv**: [2504.11092](https://arxiv.org/abs/2504.11092)  
**Code**: [https://xdimlab.github.io/Vivid4D/](https://xdimlab.github.io/Vivid4D/)  
**Area**: 3D Vision
**Keywords**: 4D reconstruction, monocular video, video inpainting, diffusion models, view augmentation

## TL;DR
This paper proposes Vivid4D, which reformulates multi-view augmentation from monocular video as a video inpainting problem — warping the video to novel viewpoints using monocular depth priors, then employing a video diffusion model to inpaint occluded regions. Through an iterative view expansion strategy and a robust reconstruction loss, Vivid4D significantly improves 4D dynamic scene reconstruction quality from monocular video.

## Background & Motivation

Reconstructing 4D dynamic scenes from casually captured monocular video is a core challenge in computer vision and graphics. With only a single viewpoint observation at each timestamp, the problem is severely underdetermined.

Two existing paradigms and their limitations:

**Geometric prior route** (optical flow, depth estimation, tracking, etc.): These auxiliary supervision signals can themselves be unreliable, and they do not necessarily correlate linearly with rendering performance (e.g., small depth errors may cause large color shifts). More critically, these priors are derived solely from the input viewpoint and cannot provide guidance for occluded or unobserved regions.

**Generative prior route** (video diffusion models): These can generate plausible RGB images for unseen viewpoints, but existing methods are either restricted to static scenes or require large amounts of data with camera poses for training — which are extremely difficult to obtain for dynamic scenes.

**Key Challenge**: Geometric priors cannot generate new content, while generative priors lack precise geometric constraints. How can the strengths of both be combined?

**Core Idea**: Vivid4D reframes view augmentation as a video inpainting task — leveraging geometric information from depth priors to warp existing views to novel viewpoints (preserving known regions), then using a video diffusion model to inpaint missing regions caused by occlusion (generating new content). Crucially, the training data requires only pose-free web videos.

## Method

### Overall Architecture

Vivid4D consists of three core stages: (1) training a video inpainting diffusion model; (2) iterative view augmentation to generate multi-view supervision; and (3) 4D reconstruction using both original and augmented videos.

Specifically, given a monocular video, COLMAP is first used to obtain camera parameters and sparse point clouds. Monocular depth estimation provides depth maps that are aligned to metric scale. The video is then iteratively warped to novel viewpoints, occluded regions are inpainted by the inpainting model, and all videos are used to supervise motion-field-based 3DGS for 4D reconstruction.

### Key Designs

1. **Anchor-Conditioned Video Inpainting Diffusion Model**:

    - **Function**: Fine-tuned on a pretrained video diffusion model, accepting a masked video, a binary mask, and an anchor video as inputs simultaneously.
    - **Mechanism**: The model extends the input channels of a standard video diffusion model by concatenating along the channel dimension: the VAE encoding of the masked video $\mathbf{z}_m$ (4 channels), the downsampled binary mask $\mathcal{M}'$ (1 channel), the noisy latent $\mathbf{z}_t$ (4 channels), and the VAE encoding of the anchor video $\mathbf{z}_a$ (4 channels). The training objective is:
    $\mathcal{L} = \mathbb{E}_{\mathbf{x},t,\epsilon \sim \mathcal{N}(0,1)} ||\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{z}_m, \mathcal{M}', \mathbf{z}_a)||_2^2$
    - **Design Motivation**: The anchor video (the original, unwarped video) provides complete spatiotemporal context, helping the model maintain consistency with the original scene when filling occluded regions.

2. **2D Tracking-Based Training Data Generation**:

    - **Function**: Automatically generates training data pairs from pose-free web videos.
    - **Mechanism**: A pretrained 2D tracking model samples $N$ points from the first frame and tracks them to subsequent frames. Pixel regions in each frame not covered by any tracked point constitute the mask region $\mathcal{M}_t$, representing areas newly exposed due to object or camera motion.
    - **Design Motivation**: These naturally occurring occlusion regions perfectly simulate the holes produced by warping, enabling training data acquisition without known camera poses and greatly expanding the range of usable training data.

3. **Iterative View Augmentation Strategy**:

    - **Function**: Progressively expands the warp angle from small to large, generating augmented views over $N$ iterations.
    - **Mechanism**: A data buffer $\mathcal{D}_j = \mathcal{D}_{j-1} \cup (\hat{\mathcal{V}}^j, D^j, \mathbf{T}^j)$ is maintained; at each iteration, the frame with the smallest warp angle in the buffer is selected for warping. The resulting video is passed to the inpainting model and added to the buffer. A supervision mask $S_t^j$ is introduced to prevent inconsistencies caused by repeated inpainting across iterations.
    - **Design Motivation**: Inaccuracies in depth priors (e.g., bleeding edges) introduce pronounced artifacts at large warp angles. The progressive strategy begins with small angles and uses existing high-quality views to generate increasingly offset viewpoints, minimizing distortion.

4. **Invariant Vicinity (IV) RGB Loss**:

    - **Function**: A robust pixel-level loss applied to augmented views.
    - **Mechanism**: For each rendered pixel, the L1 loss is computed against the nearest pixel within a $3\times3$ neighborhood of the corresponding supervision image, and gradients are back-propagated only through the pixel with minimum error.
    - **Design Motivation**: Depth estimation errors and diffusion model artifacts cause slight misalignments between augmented views and the ground truth. The IV loss reduces sensitivity to such misalignments.

### Loss & Training

The 4D reconstruction loss comprises two parts:
- **Original frames**: Standard L1 + SSIM + LPIPS full-image supervision.
- **Augmented frames**: IV RGB loss + SSIM + LPIPS, computed only in regions where the supervision mask $S_t^j = 1$:
$$\mathcal{L} = \sum_{j=1}^N \sum_{t=1}^T (\lambda_r \mathcal{L}_\text{IV}^{t,j} + \lambda_s \mathcal{L}_\text{ssim}^{t,j} + \lambda_l \mathcal{L}_\text{lpips}^{t,j})$$

**4D representation**: Motion-field-based 3DGS from Shape of Motion.
**Depth estimation**: Learning-based initialization aligned to metric scale using COLMAP sparse point clouds.
**Training data**: 5K videos processed from OpenVid-1M for training the inpainting model.

## Key Experimental Results

### Main Results (iPhone Dataset + HyperNeRF Dataset)

| Method | iPhone mPSNR↑ | iPhone mSSIM↑ | iPhone mLPIPS↓ | HyperNeRF mPSNR↑ | HyperNeRF mLPIPS↓ |
|------|-------------|-------------|--------------|----------------|----------------|
| 4D GS | 14.01 | 0.3877 | 0.5939 | 18.24 | 0.4450 |
| Shape of Motion | 14.56 | 0.4570 | 0.5292 | 18.82 | 0.4589 |
| CoCoCo | 14.99 | 0.4701 | 0.5280 | 19.00 | 0.4692 |
| StereoCrafter | 14.85 | 0.4945 | 0.5676 | 18.86 | 0.5181 |
| ViewCrafter | 14.94 | 0.4888 | 0.5772 | 18.91 | 0.4888 |
| **Vivid4D (Ours)** | **15.20** | **0.5004** | **0.4930** | **19.45** | **0.4449** |

### Ablation Study

| Configuration | mPSNR↑ | mSSIM↑ | mLPIPS↓ | Notes |
|------|--------|--------|---------|------|
| (a) No warp, no inpaint, no depth | 16.04 | — | — | Baseline (original frames only) |
| (b) No warp, no inpaint, with depth supervision | Marginal gain | — | — | Direct depth supervision has limited effect |
| (c) With warp, no inpaint | Larger gain | — | — | Geometric warp alone outperforms direct depth supervision |
| (d) With warp, with inpaint (Ours) | **Best** | — | — | Inpainting occluded regions yields further improvement |
| Without anchor video | 25.34 PSNR | 0.8053 SSIM | 0.1056 LPIPS | Evaluated on 5K inpainting videos |
| **With anchor video** | **27.22** | **0.8223** | **0.0801** | Anchor provides spatiotemporal context |
| Direct warp ($N=1$) | Sub-optimal | — | — | Large-angle warp introduces artifacts |
| **Iterative warp ($N=2$)** | **Best** | — | — | Progressive expansion reduces distortion |

### Key Findings

- Using depth priors for warping (indirect utilization) is more effective than using them directly as supervision signals, validating the non-linear relationship between geometric priors and rendering performance.
- Anchor video conditioning significantly improves inpainting quality (PSNR increases from 25.34 to 27.22), effectively reducing artifacts.
- The iterative expansion strategy outperforms single-step direct warping, as depth estimation errors have a smaller impact at smaller warp angles.
- Vivid4D outperforms all baselines on all metrics, including dedicated video inpainting methods (CoCoCo) and 3D-aware inpainting methods (ViewCrafter).
- The method effectively fills regions invisible in the input video (black holes/white regions in 4D GS and SoM) with inpainted content.

## Highlights & Insights

1. **Elegant reformulation of view augmentation as warp + inpaint**: Geometric priors (warp provides the layout of known content) and generative priors (diffusion model fills unknown regions) are seamlessly combined in a complementary rather than substitutive manner.
2. **Innovative training data acquisition**: By leveraging 2D tracking on pose-free web videos to automatically generate training pairs that simulate warp-induced occlusions, the method circumvents the difficulty of obtaining camera poses for dynamic scenes.
3. **Anchor video mechanism**: Simple channel concatenation effectively exploits the spatiotemporal information of the original video; the 3D U-Net naturally learns cross-view spatiotemporal correspondences.
4. **Practical utility of IV RGB loss**: Replacing strict per-pixel L1 with the minimum value within a $3\times3$ neighborhood elegantly handles sub-pixel misalignments caused by depth estimation errors.

## Limitations & Future Work

- Relies on COLMAP to obtain initial camera poses, which may fail for purely dynamic scenes (e.g., hand-held camera motion combined with object motion).
- The inpainting model is trained on a relatively small dataset (5K videos), potentially limiting the upper bound of inpainting quality.
- Iterative warping increases preprocessing time, as each iteration requires running the diffusion model and depth estimation.
- The geometric consistency of inpainted content depends on the capability of the diffusion model and may fail under extreme viewpoint changes.
- Evaluation is conducted on a small number of scenes (5 iPhone + 3 HyperNeRF), limiting the scope of assessment.

## Related Work & Insights

- Shape of Motion [Wang et al.] serves as the 4D representation backbone adopted in this work; Vivid4D achieves significant improvements over it through view augmentation.
- StereoCrafter [Zhao et al.] and ViewCrafter [Yu et al.] also leverage diffusion models for post-warp inpainting, but respectively require warped video or known-pose training data.
- **Insight**: For reconstruction tasks with limited viewpoints, **indirectly exploiting geometric priors** (by converting the problem into inpainting via warping) is more effective than directly incorporating them into the loss function, since errors introduced by the former are partially absorbed by the inpainting model.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The view augmentation = warp + inpaint formulation is natural and effective; generating training data via 2D tracking is a clever contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are well-designed, though the number of test scenes is limited (only 8 scenes).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figures are clear (especially the comparison between direct and iterative warping), and the overall logic is coherent.
- **Value**: ⭐⭐⭐⭐ Provides a practical augmentation framework for monocular 4D reconstruction, compatible with multiple 4D representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] Predict-Optimize-Distill: A Self-Improving Cycle for 4D Object Understanding](predict-optimize-distill_a_self-improving_cycle_for_4d_object_understanding.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2026/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)

</div>

<!-- RELATED:END -->
