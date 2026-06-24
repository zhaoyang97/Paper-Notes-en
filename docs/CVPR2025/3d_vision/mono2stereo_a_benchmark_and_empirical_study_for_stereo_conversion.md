---
title: >-
  [Paper Note] Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion
description: >-
  [CVPR 2025][3D Vision][stereo conversion] Constructs the first large-scale stereo conversion benchmark Mono2Stereo (2.4 million pairs), proposes the stereo quality metric SIoU (0.84 Spearman correlation with human judgment), and introduces a dual-condition diffusion model with Edge Consistency loss, resolving the conflict between the weak stereo effect of single-stage methods and the poor image quality of two-stage methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "stereo conversion"
  - "dual-condition diffusion"
  - "SIoU metric"
  - "Edge Consistency loss"
  - "benchmark"
date: 2026-05-08
content_hash: b8abd1e853091e22
---

# Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion

**Conference**: CVPR 2025  
**arXiv**: [2503.22262](https://arxiv.org/abs/2503.22262)  
**Code**: [mono2stereo-bench.github.io](https://mono2stereo-bench.github.io)  
**Area**: Image Generation  
**Keywords**: stereo conversion, dual-condition diffusion, SIoU metric, Edge Consistency loss, benchmark

## TL;DR

Constructs the first large-scale stereo conversion benchmark Mono2Stereo (2.4 million pairs), proposes the stereo quality metric SIoU (0.84 Spearman correlation with human judgment), and introduces a dual-condition diffusion model with Edge Consistency loss, resolving the conflict between the weak stereo effect of single-stage methods and the poor image quality of two-stage methods.

## Background & Motivation

**Background**: With the popularity of AR/VR devices, the demand for converting 2D to 3D stereo content has surged. Existing methods are divided into two categories:
- **Two-stage methods** (e.g., StereoDiffusion): First estimate a disparity map to warp the left image, then use an inpainting model to fill disoccluded regions—good stereo effect but poor image quality.
- **Single-stage methods**: Directly generate the right image from the left image—good image quality but prone to degenerating into an identity mapping (output $\approx$ input), leading to weak stereo effects.

**Core Problem**: Two key problems remain unsolved:
1. Lack of large-scale open training datasets and standardized benchmarks, preventing systematic comparison of methods.
2. Traditional metrics (RMSE, SSIM) measure global pixel differences, but the differences between left and right perspectives are concentrated in extremely small areas around object edges, rendering traditional metrics incapable of reflecting stereo perceptual quality.

**Key Insights**:
- An identity mapping model (output = input) even outperforms the commercial stereo conversion model OWL3D in RMSE, PSNR, and SSIM, yet has absolutely no stereo effect—proving that traditional metrics fail to evaluate stereo quality.
- The differences between left and right views are mainly concentrated in the high-frequency details of object edges.

## Method

### Overall Architecture

Three complementary contributions:
1. **Mono2Stereo Dataset**: 2.4 million stereo pairs + 5 types of test sets.
2. **SIoU Evaluation Metric**: Assesses stereo quality from both disparity consistency and edge structures.
3. **Dual-Condition Diffusion Baseline Model**: Takes both geometric condition (left image) and viewpoint condition (warped image) as inputs, combined with the Edge Consistency loss.

### Key Designs

### Key Design 1: Mono2Stereo Dataset Construction

- **Data Source**: Collected 200 stereo movies/videos (20 million frames) from the internet, manually filtered to remove low-quality and sensitive content.
- **Format**: Side-by-side 3D format to preserve maximum information.
- **Processing**: Sampled 1 frame every 8 frames to reduce redundancy, uniformly resized to 960×540, resulting in 2.4 million pairs.
- **Test Set Partition**: 500 pairs for each of 5 dimensions—Indoor (disparity range), Outdoor, Simple (scene complexity), Complex, and Animation (color distribution).
- **OOD Evaluation**: Additionally uses the Inria 3DMovie dataset (2727 pairs) to assess out-of-distribution generalization.

### Key Design 2: SIoU Evaluation Metric

$$\text{SIoU}(\mathbf{L}, \mathbf{R}, \mathbf{G}) = \alpha \cdot \frac{\mathbf{E}_g \cap \mathbf{E}_r}{\mathbf{E}_g \cup \mathbf{E}_r} + (1-\alpha) \cdot \frac{|\mathbf{G}-\mathbf{L}| \cap |\mathbf{R}-\mathbf{L}|}{|\mathbf{G}-\mathbf{L}| \cup |\mathbf{R}-\mathbf{L}|}$$

- **First term (weight $\alpha=0.75$)**: Edge structure IoU—the IoU between the Canny edge maps of the generated image $\mathbf{G}$ and the right image $\mathbf{R}$, measuring if the edge displacement is correct.
- **Second term (weight $1-\alpha=0.25$)**: Disparity consistency IoU—the IoU between the difference map of the generated and left image, and the difference map of the right and left image, measuring if disparity regions match.
- **Human Evaluation Validation**: Spearman correlation 0.84, Kendall 0.73; whereas RMSE is only 0.26 and SSIM is only 0.21.

### Key Design 3: Dual-Condition Diffusion Model + Edge Consistency Loss

**Dual-Condition Input**:
- **Geometric Condition $\mathbf{Z}_l$**: VAE encoding of the left image, providing full structural and textural information (advantage of single-stage methods).
- **Viewpoint Condition $\mathbf{Z}_w$**: VAE encoding of the warped left image based on disparity, providing explicit viewpoint cues (advantage of two-stage methods).
- Both are concatenated with the noisy sample $\mathbf{Z}_r^t$ and fed into the U-Net, adapting the additional channels by replicating the first convolutional layer.

**Edge Consistency Loss**:

$$\ell = \mathbb{E}_{t,\mathbf{Z}_r,\epsilon}\|\epsilon - \hat{\epsilon}\|^2 + \alpha \cdot \mathbb{E}_{t,\mathbf{Z}_r,\epsilon}\|S(\epsilon) - S(\hat{\epsilon})\|^2$$

- $S$: Sobel edge detection operator.
- **Intuition**: Since the primary differences between left and right views reside at object edges, the model is prone to degenerating into an identity mapping; applying extra constraints to the edges of the velocity prediction forces the model to focus on edge displacements.
- **No Decoding Required**: Leverages the spatial correlation between the velocity representation and the image to directly constrain edges in the latent space, avoiding the forward pass overhead through the decoder during training.

### Loss & Training

Total loss = standard diffusion denoising loss + $\alpha$ $\times$ Edge Consistency loss ($\alpha=1$).

## Key Experimental Results

### Main Results

Comparison on the Mono2Stereo test set (Table 3):

| Method | SIoU↑ | RMSE↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|---|
| Identity Mapping | 0.164 | 5.07 | 34.76 | 0.788 |
| 3D Photography | 0.210 | 8.89 | 29.30 | 0.285 |
| StereoDiffusion | 0.238 | 7.42 | 30.96 | 0.621 |
| OWL3D (Commercial) | 0.278 | 5.81 | 33.25 | 0.732 |
| Geometric Cond. | 0.259 | 5.40 | 33.48 | **0.829** |
| Viewpoint Cond. | 0.281 | 6.12 | 32.53 | 0.762 |
| **Dual Condition** | **0.284** | **5.34** | **34.09** | 0.795 |

- The dual-condition model achieves the overall best performance on SIoU and PSNR.
- Inria 3DMovie OOD test: Dual-cond. SIoU 0.340 vs OWL3D 0.286 vs Geometric 0.314.

### Ablation Study

**Different Condition Combinations (Table 6)**:
- Geometric + depth/edge map: No improvement or even degradation, because depth/edge maps differ significantly from the natural image distribution.
- Geometric + Viewpoint (dual condition): SIoU 0.262 > Geometric 0.253 > Viewpoint 0.260.

**Effect of EC Loss (Table 7)**:
- Geometric condition: SIoU 0.242 $\rightarrow$ 0.253 (+4.5%)
- Viewpoint condition: SIoU 0.257 $\rightarrow$ 0.260
- Dual condition: SIoU 0.259 $\rightarrow$ 0.262

### Key Findings

1. **Identity Mapping Trap**: Under traditional metrics, identity mapping performs "best," revealing a fundamental flaw in the evaluation methodology.
2. The geometric condition excels at image quality (SSIM 0.829), while the viewpoint condition excels at stereo effects (higher SIoU). The dual-condition design achieves the best of both worlds.
3. Additional conditions such as depth maps and edge maps are counterproductive because of their large distribution differences from natural images.
4. The EC loss operates edge constraints directly in the latent space, avoiding decoder overhead during training while effectively mitigating degeneration.
5. SIoU aligns highly with human judgment (0.84 Spearman), serving as a potential standard evaluation metric for the stereo conversion field.

## Highlights & Insights

1. **Evaluation metric contribution might outlast the model itself**: SIoU fills the gap in stereo quality assessment, carrying infrastructure-level value for the entire field.
2. **Revealing the Identity Mapping Paradox**: A "do-nothing" model performing better under traditional metrics stands as a powerful critique of the field's evaluation methodology.
3. **Clear intuition for dual-condition design**: Rather than a simple stacking of features, it explicitly assigns different roles to the two conditions (structure vs. perspective), tackling the problem directly.
4. **Clever EC loss design**: Leverages spatial correlation in the latent space to avoid decoding during training, balancing both effectiveness and efficiency.

## Limitations & Future Work

1. The study only utilizes an image diffusion model (SD V2) and does not explore video diffusion models (e.g., StereoCrafter). Temporal consistency is not addressed.
2. The dataset originates from internet movies, leaving room for improvement in scene diversity (e.g., lack of professional domains such as medical or remote sensing scenes).
3. The training resolution of 640×480 is relatively low, and generalization to 4K/8K high-resolution scenarios remains to be validated.
4. Disparity estimation depends on DepthAnything, whose accuracy directly influences the quality of the viewpoint condition.

## Related Work & Insights

- **StereoDiffusion**: A training-free stereo conversion method that relies on parameter tuning and exhibits severe artifacts in complex scenes; ours addresses this through large-scale training data.
- **Marigold**: A paradigm utilizing pre-trained diffusion models for depth estimation; ours borrows its latent space encoding/decoding strategy.
- **StereoCrafter**: A concurrent method using video diffusion models, which is complementary to our work (video vs. image).
- **Insight**: In tasks where "differences between left and right views are extremely subtle", model degradation into an identity mapping is a common issue (also observed in tasks like low-light enhancement). The "edge constraint" concept of EC loss can be extended to other generation tasks with minor but critical differences.

## Rating

⭐⭐⭐⭐ — A solid and comprehensive trinity of contributions (dataset + metric + model). The SIoU metric holds promise to become an industry standard, while the dual-condition design and EC loss resolve real conflicts. One star is deducted because it only covers images without video, and the training resolution is relatively low.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](../../ICCV2025/3d_vision/from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] Consistency-aware Self-Training for Iterative-based Stereo Matching](consistency-aware_self-training_for_iterative-based_stereo_matching.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)

</div>

<!-- RELATED:END -->
