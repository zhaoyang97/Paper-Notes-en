---
title: >-
  [Paper Note] Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer
description: >-
  [ICLR 2026][3D Vision][3D colorization] Color3D introduces a paradigm of "colorize one key view → fine-tune a personalized colorizer → propagate colors to all views and timesteps…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3D colorization"
  - "Gaussian splatting"
  - "personalized fine-tuning"
  - "Lab color space"
  - "visual consistency"
date: 2026-05-08
content_hash: 46171f3b81aef6c7
---

# Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer

**Conference**: ICLR 2026
**arXiv**: [2510.10152](https://arxiv.org/abs/2510.10152)
**Code**: [https://yecongwan.github.io/Color3D/](https://yecongwan.github.io/Color3D/) (Project Page)
**Area**: 3D Vision / Image Generation
**Keywords**: 3D colorization, Gaussian splatting, personalized fine-tuning, Lab color space, visual consistency

## TL;DR
Color3D introduces a paradigm of "colorize one key view → fine-tune a personalized colorizer → propagate colors to all views and timesteps," reducing the complex 3D colorization problem to single-image colorization plus color propagation. It achieves rich colorization, cross-view consistency, and user controllability simultaneously on both static and dynamic 3D scenes.

## Background & Motivation

**Background**: 3DGS/NeRF enables high-quality novel view synthesis, yet reconstructing colorful 3D scenes from grayscale inputs remains challenging. 2D image colorization is mature (supporting language-guided, reference-guided, and automatic modes), but directly colorizing multiple views leads to severe cross-view color inconsistency.

**Limitations of Prior Work**:
   - Existing 3D colorization methods (ChromaDistill, ColorNeRF) enforce consistency by averaging cross-view color variations, which dilutes palette richness and produces desaturated, tonally flat results.
   - Smoothing color variations makes results unpredictable, sacrificing user controllability.
   - Existing methods handle only static scenes; controllable colorization of dynamic scenes remains entirely unexplored.

**Key Challenge**: A fundamental trade-off among cross-view consistency, color richness, and controllability — averaging strategies guarantee consistency at the cost of the latter two.

**Goal**:
   - Unify controllable colorization of static and dynamic 3D scenes.
   - Maintain color richness while ensuring cross-view and cross-temporal consistency.
   - Support plug-and-play integration of arbitrary 2D colorization models.

**Key Insight**: The core insight is that only one key view needs to be colorized; a scene-specific colorizer is then fine-tuned to learn a deterministic color mapping for that view. Through the inductive bias of the colorizer, identical content across different viewpoints is mapped to identical colors.

**Core Idea**: Reduce 3D colorization to "single-image colorization + personalized colorizer color propagation," naturally guaranteeing cross-view and cross-temporal consistency by learning a scene-specific one-to-one color mapping.

## Method

### Overall Architecture
A two-stage pipeline. **Stage 1** (personalized colorizer training): select a key view → colorize it with any 2D colorization model → apply single-view data augmentation → fine-tune the colorizer to learn a deterministic color mapping. **Stage 2** (3D scene colorization): use the personalized colorizer to infer colors for all other views/frames → reconstruct the colorful 3D scene using Lab-space 3DGS/4DGS.

### Key Designs

1. **Key View Selection**:

    - Function: Select the most informative view among all grayscale inputs as the sole colorization target.
    - Mechanism: CLIP features are extracted for each view; a pairwise cosine similarity matrix is computed; the information entropy $H(I_i) = -\sum_j P_{ij} \log P_{ij}$ is computed per view. The view with maximum entropy is selected: $I^* = \arg\max H(I_i)$ — higher entropy indicates more uniform association with all other views, implying broader and more diverse visual coverage.
    - Design Motivation: If the selected key view covers only a small portion of the scene, the personalized colorizer cannot generalize to unseen content.

2. **Single-View Data Augmentation**:

    - Function: Generate diverse training samples from a single colorized key view.
    - Mechanism: Combines generative and traditional augmentation — (1) Outpainting: a 2×2 grid partition followed by SD-based region extension; (2) Image-to-Video: SVD generates continuous video frames simulating motion and object appearance; (3) Novel View: Stable Virtual Camera generates orbital viewpoints. Traditional augmentation includes rotation, flipping, grid shuffling, and elastic transformation.
    - Design Motivation: Training a colorizer on a single image causes severe overfitting; generative augmentation expands the sample space without requiring generated content to perfectly match the scene — only a consistent color style is needed.

3. **Personalized Colorizer Architecture & Training**:

    - Function: Learn a scene-specific deterministic color mapping.
    - Mechanism: The DDColor encoder is frozen (preserving high-level semantic color feature extraction), with trainable adapters added for scene adaptation, and a lightweight CNN decoder initialized from scratch (to avoid the built-in color priors of a pretrained decoder causing cross-view inconsistency). Trained with a simple L1 loss: $\mathcal{L} = \|P^{ab} - G^{ab}\|_1$.
    - Design Motivation: The pretrained decoder is the primary source of cross-view color inconsistency; a from-scratch decoder is a "clean slate" that learns only the color mapping of the current scene.

4. **Lab Gaussian Representation**:

    - Function: Perform 3DGS rendering and optimization in the CIE Lab color space.
    - Mechanism: The three groups of SH coefficients in 3DGS are replaced from RGB with $\{SH_L, SH_a, SH_b\}$; the L and ab channels are optimized separately. An edge loss $\mathcal{L}_{edge}$ is applied to the L channel to preserve structural details, while the ab channel uses only L1+D-SSIM. During the first half of training, all three SH coefficient groups represent the L channel for warm-up (learning geometry first); during the second half, two groups are allocated to the ab channel for color learning.
    - Design Motivation: Lab space decouples luminance from chrominance; the known L channel provides a stable structural constraint signal, reducing the impact of chrominance prediction noise on optimization.

### Loss & Training
- **Colorizer**: $\mathcal{L}_{colorizer} = \|P^{ab} - G^{ab}\|_1$
- **L channel**: $\mathcal{L}_l = (1-\beta)\mathcal{L}_1 + \beta\mathcal{L}_{D-SSIM} + \mathcal{L}_{edge}$
- **ab channel**: $\mathcal{L}_{ab} = (1-\beta)\mathcal{L}_1 + \beta\mathcal{L}_{D-SSIM}$, $\beta=0.2$
- **Warm-up**: The first 50% of iterations optimize only the L channel for structural learning; the ab channel is introduced in the latter 50% for color learning.

## Key Experimental Results

### Main Results (DL3DV-140 Static Scenes, Automatic Colorization)

| Method | FID↓ | Colorful↑ | ME↓ | TC↓ |
|--------|------|-----------|-----|-----|
| 3DGS+ImageColorizer | 63.56 | 28.15 | 0.146 | 0.038 |
| 3DGS+VideoColorizer | 77.89 | 22.38 | 0.128 | 0.031 |
| **Color3D (Ours)** | **37.48** | **32.65** | **0.084** | **0.017** |

### Ablation Study (Contribution of Key Components)

| Configuration | FID↓ | Colorful↑ | ME↓ | TC↓ |
|---------------|------|-----------|-----|-----|
| Full Color3D | 37.48 | 32.65 | 0.084 | 0.017 |
| w/o personalized fine-tuning | ~63 | ~28 | ~0.15 | ~0.04 |
| w/o data augmentation | ~45 | ~30 | ~0.10 | ~0.02 |
| w/o Lab Gaussian | ~42 | ~31 | ~0.09 | ~0.02 |

### Key Findings
- **Substantial FID reduction**: Color3D achieves an FID of 37.48 on DL3DV-140, more than 40% lower than direct image colorization (63.56).
- **Higher color richness**: Colorful score of 32.65 vs. 28.15/22.38, demonstrating that the personalized colorizer does not dilute colors.
- **Significantly improved consistency**: ME (multi-view error) and TC (temporal consistency) substantially outperform all baselines.
- **Multiple control modes supported**: Language-guided, automatic, and reference-guided colorization all function effectively.

## Highlights & Insights
- **The paradigm shift of reducing 3D colorization to a single-image problem is highly elegant**: It avoids the complexity of handling color consistency directly in 3D space. As long as one image is correctly colorized, the personalized colorizer naturally propagates colors to all viewpoints — enabling any 2D colorization method to be used for 3D colorization in a plug-and-play manner.
- **The "frozen encoder + from-scratch decoder" design philosophy is worth adopting broadly**: The pretrained encoder provides semantic understanding, while the from-scratch decoder avoids introducing inconsistent color priors. This "preserve capability, remove bias" strategy is applicable across many transfer learning scenarios.
- **Lab space decoupling + warm-up strategy is practically effective**: Learning structure before color prevents chrominance noise from interfering with geometry optimization.

## Limitations & Future Work
- Each scene requires fine-tuning a colorizer (~30 min), precluding zero-shot generalization to new scenes.
- Key view selection relies on a CLIP-feature entropy heuristic, which may not be globally optimal.
- Colors introduced by generative augmentation (outpainting, SVD) may not fully align with the scene.
- For extreme viewpoint variation (e.g., 360° panoramic scenes), a single key view may provide insufficient coverage.
- Motion blur caused by fast motion in dynamic scenes may degrade colorization quality.

## Related Work & Insights
- **vs. ColorNeRF**: ColorNeRF injects color into NeRF and enforces consistency through averaging → color fades; Color3D maintains vibrancy by learning a deterministic mapping.
- **vs. ChromaDistill**: The distillation strategy similarly dilutes color diversity; Color3D relies solely on the color information of a single view.
- **vs. 3DGS+ImageColorizer (naive baseline)**: Direct per-frame colorization leads to severe inconsistency; Color3D's personalized propagation reduces FID by 40%+.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Paradigm shift — reducing 3D colorization to single-image colorization plus color propagation is an exceptionally elegant formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 140 static scenes plus dynamic scenes, three control modes, extensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, well-designed method figures, complete logical chain.
- Value: ⭐⭐⭐⭐ Strong practicality; unifies controllable colorization of static and dynamic scenes with broad application prospects including cultural heritage preservation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)
- [\[ICLR 2026\] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering](radiogs_radiometric_gaussian_surfels.md)
- [\[AAAI 2026\] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting](../../AAAI2026/3d_vision/fantasystyle_controllable_stylized_distillation_for_3d_gaussian_splatting.md)
- [\[CVPR 2026\] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds](../../CVPR2026/3d_vision/s2am3d_scale-controllable_part_segmentation_of_3d_point_cloud.md)
- [\[ICCV 2025\] LACONIC: A 3D Layout Adapter for Controllable Image Creation](../../ICCV2025/3d_vision/laconic_a_3d_layout_adapter_for_controllable_image_creation.md)

</div>

<!-- RELATED:END -->
