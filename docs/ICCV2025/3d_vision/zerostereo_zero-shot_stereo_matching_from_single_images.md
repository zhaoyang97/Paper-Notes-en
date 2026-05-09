---
title: >-
  [Paper Note] ZeroStereo: Zero-shot Stereo Matching from Single Images
description: >-
  [ICCV2025][3D Vision][stereo matching] This paper proposes ZeroStereo, a pipeline that starts from an arbitrary single image, uses monocular depth estimation to generate pseudo disparity, and synthesizes high-quality right-view images via a fine-tuned diffusion inpainting model. The approach achieves state-of-the-art zero-shot stereo matching generalization using only 35K synthetic training samples.
tags:
  - ICCV2025
  - 3D Vision
  - stereo matching
  - zero-shot generalization
  - diffusion inpainting
  - monocular depth estimation
  - view synthesis
date: 2026-05-08
content_hash: a3096b7548121fc5
---

# ZeroStereo: Zero-shot Stereo Matching from Single Images

**Conference**: ICCV2025  
**arXiv**: [2501.08654](https://arxiv.org/abs/2501.08654)  
**Code**: [GitHub](https://github.com/Windsrain/ZeroStereo)  
**Area**: 3D Vision  
**Keywords**: stereo matching, zero-shot generalization, diffusion inpainting, monocular depth estimation, view synthesis

## TL;DR

This paper proposes ZeroStereo, a pipeline that starts from an arbitrary single image, uses monocular depth estimation to generate pseudo disparity, and synthesizes high-quality right-view images via a fine-tuned diffusion inpainting model. The approach achieves state-of-the-art zero-shot stereo matching generalization using only 35K synthetic training samples.

## Background & Motivation

- **Core Problem**: Supervised stereo matching models perform well on standard benchmarks, but generalization to real-world scenes degrades significantly due to the extreme scarcity of annotated real-world stereo data.
- **Limitations of Prior Work**:
    1. **Domain-invariant feature learning** (DSMNet, GraftNet, ITSA, etc.): A non-trivial domain gap still exists between synthetic and real data.
    2. **Self-supervised learning** (photometric loss): Severely affected by occlusions, ghosting artifacts, and ill-posed regions; large-scale collection of high-quality stereo image pairs is itself non-trivial.
    3. **View synthesis**:
        - Early methods (Luo et al., MfS-Stereo) generate right-view images via monocular depth and forward warping, but occluded regions are filled with neighboring pixels or random backgrounds, causing structural inconsistency.
        - NeRF-Stereo requires multi-view inputs for scene reconstruction, offering poor flexibility; NeRF also performs poorly for distant scene reconstruction, making it unsuitable for large-scale outdoor settings.
- **Motivation**: Can a single image combined with a pre-trained diffusion model be used to complete occluded regions with high quality, while assessing pseudo-label reliability without any additional training?

## Method

### Overall Pipeline

Given a single left image $\mathbf{I}_l$:

1. **Monocular Depth Estimation**: Depth Anything V2 (DAv2) produces a normalized inverse depth map $\mathbf{D}$.
2. **Adaptive Disparity Selection (ADS)**: $\mathbf{D}$ is multiplied by a scaling factor $s \cdot w$ to obtain the pseudo disparity map $\mathbf{d}$, where $s$ is sampled from a piecewise uniform distribution.
3. **Forward Warping**: Following MfS-Stereo, a warped right image $\tilde{\mathbf{I}}_r$, a non-occlusion mask $\mathbf{M}_{noc}$, and an inpainting mask $\mathbf{M}_{inp}$ are generated.
4. **Diffusion Inpainting**: A fine-tuned SDv2I model performs semantically coherent completion of occluded regions, yielding the final right image $\mathbf{I}_r$.
5. **Training-Free Confidence Generation (TCG)**: The confidence map $\mathbf{C}$ of the depth estimation is computed via horizontal flip symmetry.
6. **Stereo Training**: RAFT-Stereo / IGEV-Stereo is trained using the ZeroStereo Loss.

### Diffusion Inpainting

- **Base Model**: Stable Diffusion V2 Inpainting (SDv2I).
- **Motivation for Fine-tuning**:
    - Standard text-guided inpainting lacks text prompts suited to stereo occlusion regions.
    - Stereo occlusion masks are diverse and irregular; directly applying the pre-trained model yields suboptimal results.
- **Fine-tuning Protocol**:
    - Trained on synthetic datasets with accurate dense disparity ground truth: Scene Flow, Tartan Air, CREStereo Dataset, and VKITTI 2.
    - VAE is frozen; only the U-Net is fine-tuned; text conditioning is disabled; DDPM noise scheduler with 1000 steps.
    - 50K steps, batch size 32 (gradient accumulation over 4 steps), AdamW with One-cycle LR of 2e-5, cropped to 512×512.
- **Inference Protocol**: DDIM scheduler with 50 sampling steps. Final output is blended via the mask: $\mathbf{I}_r = \mathbf{M}_{inp} \odot \mathbf{I}_d + (1-\mathbf{M}_{inp}) \odot \tilde{\mathbf{I}}_r$.
- **Advantages**: Compared to StereoDiffusion (which causes structural distortion via latent-space warping), ZeroStereo requires only 5.8G memory vs. 14.6G, and runs in 1.9s vs. 31.2s.

### Training-Free Confidence Generation (TCG)

- **Core Idea**: Modern monocular depth models predict relative depth (inverse depth); the relative depth between pixels should remain consistent after horizontal flipping.
- **Procedure**:
    1. The left image is horizontally flipped and fed into DAv2 alongside the original to obtain $\mathbf{D}$ and $\mathbf{D}'$.
    2. The flipped depth is un-flipped for alignment, and the per-pixel discrepancy is computed: $\mathbf{u} = 1 - |\mathbf{D} - \mathbf{H}^{-1}(\mathbf{D}')|$.
    3. Normalization yields the confidence map $\mathbf{C}$.
- **Effect**: Low-confidence regions concentrate on edges, textureless areas, and fine structures — precisely the most ambiguous locations in stereo matching.

### Adaptive Disparity Selection (ADS)

- **Problem**: MfS-Stereo uniformly samples the maximum disparity over a fixed range, leading to foreground distortion and excessive occlusion when the disparity-to-width ratio is too large at low resolutions, and insufficient left-right differences when the ratio is too small at high resolutions.
- **Solution**: Disparity is computed as $\mathbf{d} = \mathbf{D} \cdot s \cdot w$, where $s$ is sampled from a three-segment distribution:
    - Central segment $(c-r, c+r)$, probability $p_c = 0.8$ (primary operating range).
    - Small-disparity segment $(c-2r, c-r)$, probability $p_s = 0.1$.
    - Large-disparity segment $(c+r, c+2r)$, probability $p_l = 0.1$.
    - Default: $c=0.1, r=0.05$.
- **Effect**: Adaptively adjusts the disparity range according to image width, ensuring diversity while avoiding degenerate cases.

### ZeroStereo Loss

Three loss terms are combined:

1. **Disparity Loss**: $\mathcal{L}_d = \|\tilde{\mathbf{d}} - \mathbf{d}\|_1$.
2. **Non-occluded Photometric Loss** $\mathcal{L}_{np}$: The estimated disparity is used to backward-warp the right image and compared against the left image using an SSIM + L1 combination ($\beta=0.85$); unreliable pixels are excluded using the non-occlusion mask and the backward-warped inpainting mask.
3. **ZeroStereo Loss**: $\mathcal{L}_{Zero} = \mathbf{C} \odot \mathcal{L}_d + \mu \cdot (1-\mathbf{C}) \odot \mathcal{L}_{np}$, with $\mu=0.1$. **High-confidence regions are supervised primarily by pseudo ground truth; low-confidence regions rely on photometric consistency self-supervision.**

## Key Experimental Results

### Ablation Study (IGEV-Stereo, Tab. 1)

| Module Combination | KITTI-15 EPE | KITTI-15 >3px | Midd-T EPE | ETH3D >1px |
|---|---|---|---|---|
| Baseline | 1.52 | 4.89 | 2.71 | 2.38 |
| +ADS | 1.24 | 4.84 | 2.28 | 2.27 |
| +Inpainting | 1.44 | 4.85 | 2.34 | 1.92 |
| +ADS+Inpainting | 1.06 | 4.74 | 2.26 | 2.05 |
| +ADS+Inp+TCG | 1.05 | 4.71 | 2.18 | 2.01 |
| **All + ZeroStereo Loss** | **1.04** | **4.73** | **2.09** | **1.90** |

### Zero-Shot Generalization Benchmark (Tab. 8, Zero-RAFT-Stereo)

| Dataset | KITTI-15 >3px (All) | Midd-T H >2px (Noc) | ETH3D >1px (Noc) |
|---|---|---|---|
| Zero-RAFT-Stereo | **4.53** | **4.45** | **2.13** |
| NS-RAFT-Stereo (NeRF-Stereo official) | 5.41 | 6.45 | 2.55 |
| RAFT-Stereo (SceneFlow GT) | 5.47 | 8.66 | 2.29 |

### Dataset Scale Comparison (Tab. 5, Zero-RAFT-Stereo)

Using only 35K synthetic samples (MfS35K), the method outperforms FoundationStereo trained on 1106K samples, demonstrating that data diversity matters more than absolute scale.

### Synthesis Efficiency (Tab. 4)

| Method | Resolution | VRAM | Time per Image |
|---|---|---|---|
| RePaint | 256×256 | 2.7G | 156.5s |
| StereoDiffusion | 512×512 | 14.6G | 31.2s |
| **ZeroStereo (Ours)** | 512×512 | **5.8G** | **1.9s** |

## Highlights & Insights

1. **Elegant paradigm**: The task of stereo data generation is decomposed into three modular stages — monocular depth estimation, forward warping, and diffusion inpainting — each leveraging the strongest pre-trained models in its respective domain, avoiding end-to-end training from scratch.
2. **TCG requires no additional training**: The flip symmetry of monocular depth models serves as a cost-free signal for confidence estimation, making the approach both concise and transferable to other pseudo-label settings.
3. **ADS adapts to resolution**: This directly addresses the practical engineering issue of fixed disparity ranges being mismatched to varying image resolutions.
4. **35K samples outperform millions**: This demonstrates that training data synthesized from real single images possesses inherently superior scene diversity compared to larger purely synthetic datasets.
5. **Boundary error analysis** (Tab. 7): Models trained on MfS35K exhibit the lowest errors in boundary regions, confirming that diffusion inpainting yields genuinely higher-quality completion at occlusion boundaries than competing approaches.

## Limitations & Future Work

1. **The fine-tuned SDv2I still fails on complex scenes**: Ill-posed regions such as transparent objects and mesh-like structures cannot be handled correctly even at the forward warping stage.
2. **Occasional color inconsistency**: Because fine-tuning uses synthetic datasets, a color distribution gap exists between synthetic and real images.
3. **Dependence on monocular depth model quality**: The performance ceiling of the entire pipeline is determined by DAv2's accuracy; failures in certain scenes propagate to all downstream components.
4. **Potential improvements**:
    - Replacing SDv2I with more advanced diffusion models (e.g., SDXL Inpainting or Flux) to enhance inpainting quality.
    - Introducing multi-scale disparity generation strategies.
    - Incorporating video diffusion models for temporally consistent stereo video data generation.
    - The flip-invariance assumption in TCG may not hold for scenes with text or strong directional content; additional geometric transformations such as rotation could be considered.

## Related Work & Insights

- **MfS-Stereo** [Watson et al.]: The direct predecessor of this work, generating stereo pairs via monocular depth and forward warping but filling occluded regions with random backgrounds.
- **NeRF-Stereo** [Tosi et al.]: Reconstructs 3D scenes via NeRF to render stereo pairs and introduces Ambient Occlusion for confidence estimation, but requires multi-view inputs and performs poorly on distant scenes.
- **Marigold** [Ke et al.]: Demonstrates that diffusion models can be fine-tuned from synthetic data for monocular depth estimation, inspiring the analogous approach adopted here for stereo image synthesis.
- **Depth Anything V2**: The monocular depth model used in this work to provide the basis for pseudo disparity.
- **Stable Diffusion V2 Inpainting**: The base inpainting model fine-tuned in this work.
- **Insight**: The paradigm of using strong pre-trained models from one domain to automatically generate training data for another domain holds promise for extension to optical flow estimation, scene flow estimation, and beyond.

## Rating

- Novelty: ⭐⭐⭐⭐ — Individual modules are not entirely novel, but the overall pipeline design and the flip-symmetry idea in TCG are notably elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablation study, multi-dataset and multi-model evaluation, fair comparison with NeRF-Stereo, and additional boundary error and synthesis efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐ — The paper is clearly structured with information-dense figures and tables and consistent notation.
- Value: ⭐⭐⭐⭐ — Highly practical; achieving state-of-the-art performance with only 35K samples is particularly significant for resource-constrained settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](../../CVPR2026/3d_vision/what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [\[ICCV 2025\] Stereo Any Video: Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)
- [\[ICCV 2025\] BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)

</div>

<!-- RELATED:END -->
