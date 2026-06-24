---
title: >-
  [Paper Note] Free-viewpoint Human Animation with Pose-correlated Reference Selection
description: >-
  [CVPR 2025][Image Generation][Free-viewpoint Human Animation] Proposes a pose-correlated reference selection diffusion network that computes target-reference pose correlation maps via a pose-correlation module to adaptively select the most relevant reference features. It supports high-quality human animation generation under dramatic viewpoint changes (including camera zoom) and introduces the MSTed multi-camera TED video dataset.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Free-viewpoint Human Animation"
  - "Multi-reference Images"
  - "Pose Correlation"
  - "Adaptive Reference Selection"
  - "Diffusion Models"
date: 2026-05-08
content_hash: b1c1b1a206cec714
---

# Free-viewpoint Human Animation with Pose-correlated Reference Selection

**Conference**: CVPR 2025  
**arXiv**: [2412.17290](https://arxiv.org/abs/2412.17290)  
**Code**: None (Project page provided, dataset to be released)  
**Area**: Diffusion Models / Image Generation / Video Generation  
**Keywords**: Free-viewpoint Human Animation, Multi-reference Images, Pose Correlation, Adaptive Reference Selection, Diffusion Models

## TL;DR
Proposes a pose-correlated reference selection diffusion network that computes target-reference pose correlation maps via a pose-correlation module to adaptively select the most relevant reference features. It supports high-quality human animation generation under dramatic viewpoint changes (including camera zoom) and introduces the MSTed multi-camera TED video dataset.

## Background & Motivation

**Background**: Human animation based on diffusion models (e.g., AnimateAnyone, Champ) has achieved significant progress under fixed viewpoints—generating high-fidelity animation videos given a reference image and a driving pose sequence.

**Limitations of Prior Work**: Existing methods are strictly limited to the same viewpoint and camera distance as the reference image—failing to achieve viewpoint transitions such as telephoto to close-up (zoom-in) or close-up to full-body (zoom-out). There are three main reasons: (1) The visual information provided by a single reference image is incomplete, where close-up references lack lower-body information and wide references lack facial details; (2) Self-occlusion becomes more severe during large viewpoint changes; (3) Target and reference poses are severely misaligned in space, causing traditional feature matching mechanisms to fail.

**Key Challenge**: The contradiction between the increased camera viewpoint degrees of freedom and the fixed amount of information in a single reference image—large viewpoint changes force the diffusion model to "hallucinate" more appearance information from scratch, putting excessive demands on its generation capability.

**Goal**: Achieve human animation under dramatic viewpoint changes—supporting zooming, camera switching, and multi-shot composition, while maintaining character appearance consistency.

**Key Insight**: Leverage multiple reference images to provide more comprehensive visual coverage. However, directly increasing the number of reference images linearly scales the computational cost. Thus, a pose correlation module is designed to identify which reference regions are most relevant to the target frame, selecting only the most crucial features for generation.

**Core Idea**: Find "key reference regions" via the spatial attention correlation mapping between poses to achieve efficient utilization of multiple reference images and enable free-viewpoint human animation.

## Method

### Overall Architecture
Based on a double UNet architecture (AnimateAnyone/Champ style): Reference UNet extracts features from multiple reference images, Pose Guider encodes the target pose, and Denoising UNet performs generation. The core innovation lies in adding a Pose Correlation Module (PCM) and an adaptive reference selection strategy after the Reference UNet. Training consists of two steps: an image phase and a temporal phase.

### Key Designs

1. **Pose Correlation Module (PCM)**:

    - **Function**: Computes the spatial correlation map between each reference pose and the current target pose, indicating which regions in the reference image contain useful information for the current target frame.
    - **Mechanism**: Uses two independent pose encoders with separate weights to extract reference pose features $\mathbf{F}^i_{\text{ref}}$ and target pose features $\mathbf{F}^j_{\text{tgt}}$ respectively. These are fed into a transformer cross-attention layer (references as queries, target as keys/values). A correlation attention map $\mathbf{R}^{i,j}$ is generated via zero-initialized convolutions. The correlation map is interpolated to the spatial size of reference features at each layer and enhances the feature response of relevant areas through pixel-wise multiplication.
    - **Design Motivation**: Unlike image-level similarity metrics, PCM models spatial correlation directly in pose space. Even if the target frame pose is completely misaligned with the reference frame (e.g., close-up vs. full body), it can identify the corresponding body parts. Zero-initialization guarantees that the pre-trained weights are not disrupted at the beginning of training.

2. **Adaptive Reference Selection**:

    - **Function**: Controls the number of reference features sent to the Denoising UNet to a fixed value $K_l$, preventing a linear growth in computational cost with multiple reference images.
    - **Mechanism**: Flatten and concatenate the correlation maps and features of all reference memories, sort them by correlation value, and select the top-$K_l$ tokens as the core reference features. During training, $K_l$ compensatory tokens are additionally sampled uniformly (to prevent local optima caused by the non-differentiability of argsort), while only the top-$K_l$ tokens are selected during inference. The selected features are weighted and spatially concatenated with the intermediate latents of the Denoising UNet, then fed into the spatial self-attention layer.
    - **Design Motivation**: Many regions across different reference images are redundant (e.g., multiple images depicting the arms). The selection strategy decouples the computational complexity from the number of reference images. Compensatory sampling introduces "exploration" gradients, ensuring the correlation module receives sufficient training signals.

3. **MSTed Multi-Camera TED Video Dataset**:

    - **Function**: Provides human video training/evaluation data with dramatic real-world camera viewpoint and distance variations.
    - **Mechanism**: Extracts 1,084 unique identities and 15,260 video segments (approx. 30 hours) from public TED talk videos, covering various shot scales such as close-up, medium, and wide shots. DINOv2 is used to segment shots based on frame similarity, and YOLO filters out multi-person or discontinuous character segments.
    - **Design Motivation**: Existing multi-view datasets (such as DyMVHumans) are captured in controlled studios with fixed camera-to-subject distances, which fail to reflect realistic zooming and camera switching scenarios. MSTed fills this gap.

### Loss & Training
Standard diffusion noise prediction MSE loss. Training consists of two stages: image training and temporal training, with the number of references randomly sampled between 1 and M. During inference, multiple 12-frame clips are concatenated via temporal aggregation to generate long videos.

## Key Experimental Results

### Main Results (MSTed Dataset)

| Method | L1↓ | PSNR↑ | LPIPS↓ | MOVIE↓ | FVD↓ |
|------|-----|-------|--------|--------|------|
| MagicAnimate | 154.02 | 27.92 | 0.5984 | 119.33 | 35.08 |
| AnimateAnyone | 113.69 | 29.38 | 0.5458 | 94.93 | 33.10 |
| Champ | 81.69 | 30.87 | 0.4618 | 67.84 | 25.68 |
| **Ours (R=1)** | **78.91** | **32.18** | **0.2045** | **56.53** | **20.88** |
| **Ours (R=2)** | **74.20** | **32.49** | **0.1869** | **55.60** | **7.04** |

### Ablation Study (MSTed)

| Configuration | PSNR↑ | LPIPS↓ | FVD↓ | Explanation |
|------|-------|--------|------|------|
| Baseline (Single Ref) | 30.80 | 0.2377 | 26.32 | No PCM and selection |
| + 2 ref | 31.94 | 0.2180 | 9.82 | Multi-ref significantly improves FVD |
| + 2 ref + PCM | 32.20 | 0.2070 | 7.60 | Correlation map yields further improvements |
| Full (+ selection) | **32.49** | **0.1869** | **7.04** | Selection strategy brings continuous gains |

### Key Findings
- Even when using only 1 reference image, models trained on multi-reference data outperform all single-reference baselines—indicating that multi-reference training enables the model to learn better cross-viewpoint correlation.
- Changing from 1 reference to 2 references drops the FVD from 20.88 to 7.04 (a 66% improvement), proving the critical value of multi-reference inputs for dramatic viewpoint-changing scenarios.
- On DyMVHumans, performance continues to improve as the number of references increases from 1 to 10 (FVD: 9.047 $\rightarrow$ 5.459) with no signs of saturation.
- Visualized correlation maps indeed highlight the most critical areas for generation, such as the head and hands, validating the interpretative ability of the PCM.
- The reference selection strategy reduces inference time while maintaining identical generation quality.

## Highlights & Insights
- **Association Modeling in Pose Space**: Modeling reference-target associations in pose space rather than image space is a key innovation. Even if the image appearance is completely different (close-up vs. full body), the pose structure provides a semantic bridge across shot scales. This concept can be transferred to cross-domain human generation and facial animation.
- **top-K Selection + Uniform Compensatory Sampling**: The solution to the non-differentiable argsort problem is simple and effective. Adding random exploration during training ensures gradient coverage, while using only top-K during inference guarantees efficiency. This paradigm can be generalized to other scenarios requiring token selection/pruning.
- **MSTed Dataset**: The public multi-shot human video dataset fills a significant gap and provides foundational value for camera control and free-viewpoint animation research.

## Limitations & Future Work
- The backbone network is still UNet and does not use the more advanced DiT architecture, which the authors acknowledge will be upgraded in the future.
- MSTed mainly features "presentation" scenes with limited motion diversity (mostly upper-body gestures), which may not generalize well to large-scale movements like dancing or sports.
- Reference images require manual selection or pre-extraction; automatic reference image acquisition strategies are not yet explored.
- Future improvements: Incorporate 3D human representations (such as SMPL, when available) to provide more accurate structural priors; introduce facial ID consistency loss to enhance close-up quality.

## Related Work & Insights
- **vs AnimateAnyone/Champ**: These methods only support fixed viewpoints, whereas this work extends to free-viewpoints via multi-reference + pose correlation. Even when restricted to a single reference, this method is stronger—thanks to the generalization gains from multi-reference training.
- **vs Human4DiT**: H4DiT controls viewpoint using a 4D transformer and camera parameters, but requires precise camera calibration. This method does not require camera parameters and implicitly encodes viewpoint changes through spatial pose information, offering greater flexibility.
- **vs 3D Human Reconstruction Methods (HumanNeRF/HUGS)**: 3D methods rely on dense viewpoints or long video optimization, and their rendering quality is limited. This work leverages the generative capability of diffusion models to "hallucinate" missing information, yielding better performance with sparse references.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of pose correlation and reference selection is clever; the problem definition (free-viewpoint animation) is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison on two datasets with detailed ablation studies, though lacking user studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, but contains a relatively large number of formulas; the readability can be further optimized.
- Value: ⭐⭐⭐⭐ Directly addresses key limitations of human animation (viewpoint degrees of freedom), with direct value for film/TV and virtual avatar applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)
- [\[ICCV 2025\] CompleteMe: Reference-based Human Image Completion](../../ICCV2025/image_generation/completeme_reference-based_human_image_completion.md)
- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[CVPR 2025\] MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)

</div>

<!-- RELATED:END -->
