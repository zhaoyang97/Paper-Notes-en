---
title: >-
  [Paper Note] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement
description: >-
  [CVPR 2025][3D Vision][3D Inpainting] This paper proposes IMFine, a 3D inpainting pipeline designed for unbounded scenes (including 360° captures). It generates multi-view consistent inpainted images through geometry-prior-guided warping and a test-time adaptation-based multi-view refinement network. Additionally, a novel inpainting mask detection technique is proposed to accurately distinguish the occluded regions that truly require inpainting…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Inpainting"
  - "Object Removal"
  - "Multi-view Consistency"
  - "3D Gaussian Splatting"
  - "Test-time Adaptation"
date: 2026-05-08
content_hash: 8f8d8827be3a3e69
---

# IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement

**Conference**: CVPR 2025  
**arXiv**: [2503.04501](https://arxiv.org/abs/2503.04501)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Inpainting, Object Removal, Multi-view Consistency, 3D Gaussian Splatting, Test-time Adaptation

## TL;DR
This paper proposes IMFine, a 3D inpainting pipeline designed for unbounded scenes (including 360° captures). It generates multi-view consistent inpainted images through geometry-prior-guided warping and a test-time adaptation-based multi-view refinement network. Additionally, a novel inpainting mask detection technique is proposed to accurately distinguish the occluded regions that truly require inpainting, significantly outperforming existing methods on diverse benchmarks.

## Background & Motivation

1. **Background**: 3D inpainting (object removal) is a key task in 3D editing. Existing methods mainly lift 2D inpainting priors to 3D. These approaches fall into two categories: implicit methods based on SDS distillation and explicit methods based on multi-view consistency.

2. **Limitations of Prior Work**: SDS-based methods suffer from over-saturation and over-smoothing issues. Explicit methods (e.g., SPInNeRF, MVIPNeRF) perform reasonably well in forward-facing scenes but produce floaters and blurry textures in unbounded scenarios with large viewpoint changes or 360° views. This is due to two reasons: (1) 3D geometry is difficult to recover perfectly; (2) perceptual losses only mitigate global inconsistency, leaving detail-level mismatches that cause blurry textures.

3. **Key Challenge**: Independently inpainting each viewpoint inevitably leads to inconsistency, while existing multi-view inpainting networks (e.g., MVInPainter) are constrained by dataset diversity, failing to generalize to unbounded scenes like 360° views.

4. **Goal**: To generate visually consistent and geometrically coherent 3D inpainting results in both forward-facing and unbounded scenarios.

5. **Key Insight**: To leverage single-reference view inpainting coupled with geometry-guided warping to establish coarse consistency, followed by per-scene test-time adaptation targeting the multi-view refinement network for precise detail enhancement.

6. **Core Idea**: To incorporate spatio-temporal attention layers into a pre-trained image inpainting model and adapt it into a multi-view refinement model via per-scene test-time fine-tuning, thereby generating high-fidelity and view-consistent inpainting results on warped images.

## Method

### Overall Architecture
Given a 3D Gaussian Splatting scene $\mathcal{G}$, training images $\{I_n\}$, camera poses $\{\Pi_n\}$, and object masks $\{M_n\}$, the pipeline consists of: (1) Object removal: learning the belonging probability of each Gaussian and pruning them; (2) Inpainting mask detection: distinguishing the object mask from the truly occluded NBS (Never-Before-Seen) regions; (3) Reference view inpainting: selecting a reference view and inpainting it using a 2D inpainting model; (4) Geometry reconstruction: aligning depth using a monocular depth estimator and filling in the depth values using gradient-guided optimization; (5) Warping: projecting the inpainted reference view onto other views; (6) Multi-view refinement: refining the warped images with a fine-tuned network; (7) GS fine-tuning: reconstructing the scene using the consistent, refined images.

### Key Designs

1. **Inpainting Mask Detection**:

    - **Function**: Accurately distinguishes regions in the object mask that truly require inpainting (Never-Before-Seen/NBS regions) from those already exposed in other viewpoints.
    - **Mechanism**: In 360° scenes, the object mask covers an area much larger than the region that actually needs inpainting (the latter accounts for only 50.78% on average). The method dilates the object mask to include adjacent boundary pixels, then maps the dilated mask onto the pruned GS scene (by optimizing a learnable attribute $p_m$ for each Gaussian). Through cross-view observations, the NBS regions are continuously marked while clean background regions are suppressed. Finally, Segment Anything (SAM) is utilized to extract the final precise mask.
    - **Design Motivation**: Using the entire object mask as the inpainting target unnecessarily doubles the inpainting area, increasing task difficulty and deteriorating quality. The proposed method achieves 81.12% IoU (vs. 42.55% of GaussianEditor).

2. **Test-time Adaptation-based Multi-view Refinement Network**:

    - **Function**: Refines artifacts in warped images (such as texture distortions and seam mismatches with neighboring areas) while preserving cross-view consistency.
    - **Mechanism**: Based on a pre-trained StableDiffusionInpainting model, the self-attention layer in each Transformer block is replaced with a sparse spatio-temporal attention layer (focusing only on adjacent frames and the reference frame) to reduce memory overhead. The key is to construct training data using the original multi-view images of the target scene itself: randomly selecting a reference view, adding random masks, applying geometric jitter, warping to other views, and pairing them with ground-truth images. During inference, frame order is randomly shuffled to enhance consistency. Only 1000 steps of fine-tuning are required.
    - **Design Motivation**: Large-scale multi-view datasets are scarce and suffer from domain gaps, whereas per-scene fine-tuning sidesteps generalization issues. The pre-trained model furnishes a strong image prior, meaning test-time adaptation only needs to learn "how to perform consistent refinement conditioned on the warped inputs."

3. **Geometry-guided Depth Inpainting**:

    - **Function**: Generates plausible 3D geometry for the inpainted region in the reference view.
    - **Mechanism**: Uses a monocular depth estimator to obtain the depth map $\bar{D_r}$ for the reference inpainted image, aligns it to the scene scale via a linear transformation, and solves an optimization problem incorporating gradient matching and Laplacian smoothing to fill depth details inside the mask. The constraints are: (1) preserving the trend of the estimated depth, and (2) making the boundaries of the mask seamlessly align with the known depth.
    - **Design Motivation**: Accurate depth is critical for subsequent warping quality. Simply using raw monocular depth estimates often leads to depth discontinuities along boundaries.

### Loss & Training
- Object Removal: $\mathcal{L}_1$ loss, 1000 optimization steps, pruning with threshold $\tau = 0.4$.
- Multi-view Refinement Training: Simplified variational lower bound objective (standard diffusion loss), 1000 fine-tuning steps, lr = $3 \times 10^{-5}$, batch = 1.
- GS Fine-tuning: $L_1$ loss + SSIM loss, 7000 steps.

## Key Experimental Results

### Main Results
On a self-collected dataset (20 scenes covering indoor/outdoor settings with 90°/180°/360° views) and the SPINeRF dataset:

| Method | PSNR↑ (Ours Dataset) | LPIPS↓ (Ours Dataset) | FID↓ (Ours Dataset) | PSNR↑ (SPINeRF) |
|------|-------------|--------------|------------|-----------------|
| GaussianEditor | 15.71 | 0.6163 | 375.03 | 14.41 |
| GScream | 17.18 | 0.4431 | 290.63 | 16.96 |
| SPInNeRF | 18.75 | 0.3519 | 206.43 | 17.47 |
| MVIPNeRF | 18.63 | 0.4332 | 278.99 | 17.67 |
| **Ours** | **19.67** | **0.2685** | **149.52** | **17.58** |

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | FID↓ | Description |
|------|-------|--------|------|------|
| w/o Warping | 17.85 | 0.3215 | 198.24 | Direct inpainting, inconsistent under large viewpoint changes |
| w/o Refinement | 18.90 | 0.3069 | 206.96 | Warped images used directly, containing artifacts |
| General Refinement | 19.08 | 0.2719 | 165.80 | Trained on a general dataset, suffering from domain gaps |
| Single-View Refinement | 19.46 | 0.2725 | 154.33 | Refined independently per view, lacking view consistency |
| **Multi-View Refinement (Full)** | **19.67** | **0.2685** | **149.52** | Complete method |

### Key Findings
- Warping is key to ensuring coarse consistency (without warping, PSNR drops by ~2 dB), while refinement further boosts quality (by ~0.8 dB).
- Single-view refinement fails to ensure cross-view consistency, whereas the multi-view version shows a significant advantage.
- Test-time adaptation outperforms the general-purpose model by around 0.6 dB PSNR, highlighting the necessity of per-scene fine-tuning.
- Inpainting mask detection reduces the target area by around 50% on average, and the 81.12% IoU far exceeds baseline methods.
- The advantage is particularly pronounced in unbounded scenes, while the performance gap is smaller in forward-facing scenes.

## Highlights & Insights
- **Test-time Adaptation Strategy**: Rather than relying on large-scale multi-view dataset training, it fine-tunes on target-scene data, elegantly bypassing data scarcity and domain gap issues. This per-scene adaptation philosophy is universally applicable to the broader 3D editing field.
- **Discerning Inpainting Mask vs. Object Mask**: This work is the first to explicitly point out that the object mask $\neq$ the inpainting mask in 360° scenarios, introducing an effective detection method. This seemingly simple observation has a substantial impact in practice.
- **Sparse Spatio-Temporal Attention**: Employed to control the memory footprint, with frame ordering randomly shuffled during inference to improve consistency. This design is simple yet effective.

## Limitations & Future Work
- Each scene requires about 1 hour of fine-tuning, which limits practical deployment efficiency.
- Performance remains dependent on the accuracy of the monocular depth estimator and the image segmentation modules.
- The self-collected dataset, though diverse, is limited in scale (20 scenes).
- Future work can explore more efficient parameter-efficient fine-tuning (PEFT) methods (e.g., LoRA) to accelerate test-time adaptation.
- Dynamic scenes and lighting variations are not currently addressed.

## Related Work & Insights
- **vs. SPInNeRF/InNeRF360**: These methods inpaint multi-view images independently and rely only on perceptual loss to mitigate inconsistencies. In contrast, ours fundamentally guarantees consistency through warping and refinement.
- **vs. MVInPainter**: MVInPainter trains a general-purpose multi-view inpainting network but is limited by the diversity of the training dataset. This work bypasses this bottleneck via test-time adaptation.
- **vs. GaussianEditor**: GE employs SDS loss for 3D editing but results in over-saturation and over-smoothing, yielding an FID of up to 375. The explicit method in this work exhibits clear advantages in unbounded scenes.
- The warping-refinement paradigm can be extended to other 3D editing tasks, such as texture replacement and style transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the test-time adaptation-based multi-view refinement and the inpainting mask detection are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation on a self-collected benchmark along with robust ablation studies; the video results are highly convincing.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline diagrams, detailed methodological descriptions.
- Value: ⭐⭐⭐⭐ 3D inpainting geared towards unbounded scenes is a highly practical demand, and the proposed method is robust and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [\[CVPR 2025\] MVBoost: Boost 3D Reconstruction with Multi-View Refinement](mvboost_boost_3d_reconstruction_with_multi-view_refinement.md)
- [\[CVPR 2025\] MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D](mvpaint_synchronized_multi-view_diffusion_for_painting_anything_3d.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
