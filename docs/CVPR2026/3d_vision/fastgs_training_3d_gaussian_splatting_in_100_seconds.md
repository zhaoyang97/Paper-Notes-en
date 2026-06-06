---
title: >-
  [Paper Note] FastGS: Training 3D Gaussian Splatting in 100 Seconds
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] FastGS is a multi-view consistency-based acceleration framework for 3DGS that precisely controls Gaussian count via View-Consistent Densification (VCD) and View-Consistent Pru…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "training acceleration"
  - "multi-view consistency"
  - "Gaussian density control"
  - "pruning strategy"
date: 2026-05-08
content_hash: 88f14530e404b106
---

# FastGS: Training 3D Gaussian Splatting in 100 Seconds

**Conference**: CVPR2026
**arXiv**: [2511.04283](https://arxiv.org/abs/2511.04283)  
**Code**: [fastgs.github.io](https://fastgs.github.io)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, training acceleration, multi-view consistency, Gaussian density control, pruning strategy

## TL;DR

FastGS is a multi-view consistency-based acceleration framework for 3DGS that precisely controls Gaussian count via View-Consistent Densification (VCD) and View-Consistent Pruning (VCP). It achieves scene training in approximately 100 seconds on datasets such as Mip-NeRF 360, delivering over 15× speedup over vanilla 3DGS with comparable rendering quality.

## Background & Motivation

1. **Training time bottleneck in 3DGS**: Vanilla 3DGS typically requires tens of minutes to train a single scene. Its Adaptive Density Control (ADC) generates a large number of redundant Gaussians, resulting in persistently high computational overhead and limiting practical deployment experience.

2. **Insufficient densification strategies**: Although Taming-3DGS incorporates multi-view information, its scoring approach based on Gaussian-associated attributes (opacity, scale, gradient) fails to strictly enforce multi-view consistency, still producing millions of redundant Gaussians.

3. **Limited effectiveness of existing pruning strategies**: Speedy-Splat performs pruning via Hessian approximation accumulation, which only indirectly leverages multi-view information and leads to significant degradation in rendering quality. Other methods relying on simple opacity- or scale-based thresholds are similarly ineffective at eliminating redundancy.

4. **Lack of strict multi-view consistency constraints**: A large number of Gaussians contribute to rendering quality in only a few views while being nearly useless in others—that is, they do not satisfy a bundle-adjustment-style multi-view consistency constraint.

5. **Limitations of budget mechanisms**: Methods such as DashGaussian limit Gaussian count via budget mechanisms, yet scenes still require millions of Gaussians to maintain quality, leaving practical speedup limited.

6. **Remaining inefficiency in the rasterization stage**: The 3-sigma rule in vanilla 3DGS generates numerous redundant Gaussian–tile pairs. Even the precise tile-intersection strategy of Speedy-Splat does not fully resolve invalid coverage caused by marginal Gaussians.

## Method

### View-Consistent Densification (VCD)

The core idea is to evaluate whether each Gaussian requires densification from the perspective of multi-view reconstruction quality. The procedure is as follows:

1. Randomly sample $K$ training views, render images, and compute per-pixel L1 error maps.
2. Apply min-max normalization to the error maps and mark high-error pixels using threshold $\tau$.
3. Project each Gaussian into 2D image space to obtain its footprint region $\Omega_i$.
4. Compute the mean number of high-error pixels within the footprint as the importance score $s_d^i$.
5. A Gaussian is permitted to densify only when $s_d^i$ exceeds threshold $\tau_d$ (set to 5 in experiments).

This approach ensures that newly added Gaussians genuinely serve under-reconstructed regions across multiple views, preventing redundant growth that benefits only a small subset of views, without requiring a budget mechanism.

### View-Consistent Pruning (VCP)

Adopting a scoring strategy analogous to VCD, VCP additionally incorporates the overall photometric loss to assess each Gaussian's contribution to rendering quality degradation:

1. Compute the overall photometric loss $E_{\text{photo}}$ (a combination of L1 and SSIM) for each sampled view.
2. The pruning score $s_p^i$ is a normalized weighted product of the high-error pixel count and photometric loss across views.
3. A Gaussian is removed when $s_p^i$ exceeds threshold $\tau_p$ (set to 0.9).

This strategy more directly and effectively identifies Gaussians with the lowest contribution to multi-view rendering quality compared to Hessian approximation-based methods.

### Compact Bounding Box (CB)

Building upon the precise tile-intersection approach of Speedy-Splat, CB further tightens the bounding region:

- A stricter effective-region threshold is set based on Mahalanobis distance.
- A scaling factor $\beta$ controls the effective support range of each 2D Gaussian.
- Smaller $\beta$ produces more compact ellipses, reducing invalid Gaussian–tile pairs at the margins.

### Training Pipeline

- Built upon 3DGS-accel (vanilla 3DGS + per-splat backpropagation from Taming + SH optimization acceleration).
- Densification is performed every 500 iterations and stops at 15K iterations.
- Pruning is applied every 500 iterations before 15K and every 3,000 iterations thereafter.
- Total training runs for 30K iterations using the Adam optimizer.

## Key Experimental Results

### Table 1: Training Speed and Quality Comparison on Static Scenes (RTX 4090)

| Method | Mip-NeRF 360 Time (min) | PSNR↑ | SSIM↑ | #Gaussian↓ | FPS↑ |
|---|---|---|---|---|---|
| 3DGS | 20.93 | 27.53 | 0.812 | 2.63M | 146 |
| Taming-3DGS | 5.36 | 27.48 | 0.794 | 0.68M | 221 |
| DashGaussian | 6.35 | 27.73 | 0.817 | 2.40M | 155 |
| Speedy-Splat | 13.38 | 26.91 | 0.781 | 0.30M | 552 |
| **FastGS** | **1.93** | 27.56 | 0.797 | **0.40M** | **579** |
| FastGS-Big | 3.58 | **27.93** | **0.820** | 1.15M | 469 |

### Table 2: Ablation Study (Mip-NeRF 360)

| Method | Time (min)↓ | PSNR↑ | #Gaussian↓ | FPS↑ |
|---|---|---|---|---|
| 3DGS-accel (baseline) | 7.10 | 27.46 | 2.64M | 182 |
| +VCD | 3.53 | 27.69 | 0.53M | 222 |
| +VCP | 5.32 | 27.70 | 1.96M | 285 |
| +CB | 6.13 | 27.44 | 2.78M | 303 |
| Full (VCD+VCP+CB) | **1.93** | 27.56 | **0.40M** | **579** |

VCD is the largest single contributor, reducing Gaussian count from 2.64M to 0.53M (an 80% reduction) and providing over 2× training speedup.

## Highlights & Insights

1. **Extreme training speed**: Scene training completes in as little as 77 seconds (Tanks & Temples), averaging approximately 100 seconds—far exceeding existing state-of-the-art methods.
2. **Simplicity and generality**: VCD and VCP require no budget mechanism and can be directly applied to dynamic reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and SLAM, achieving 2–6× speedup across all tasks.
3. **Principled multi-view consistency**: Analogous to bundle adjustment, the framework requires each Gaussian to make a positive contribution to multi-view rendering, rather than serving only individual views.
4. **Compatibility with multiple backbones**: FastGS achieves 8.8× speedup on Mip-Splatting and 3.6× on Scaffold-GS while maintaining rendering quality.
5. **FastGS-Big surpasses DashGaussian**: The larger variant achieves 0.2 dB higher PSNR, reduces training time by 43.6%, and uses half the number of Gaussians.

## Limitations & Future Work

1. **Incompatibility with post-training of feed-forward 3DGS**: Such methods output extremely dense Gaussians, making it difficult for VCP to effectively prune large numbers of points within a few thousand iterations; even 3K-iteration post-training still requires approximately 20 seconds.
2. **Rendering quality not optimal**: The default FastGS configuration achieves slightly inferior LPIPS scores compared to DashGaussian and vanilla 3DGS under maximum speedup settings.
3. **Hyperparameter sensitivity**: Hyperparameters such as $\tau_d$, $\tau_p$, and $\beta$ require per-scene tuning, and the paper does not thoroughly discuss their robustness.
4. **Evaluation limited to RTX 4090**: The transferability of speedup gains to other GPU hardware is not demonstrated.
5. **Persistent quality–speed trade-off**: FastGS-Big achieves higher quality at the cost of roughly halved speed, indicating that the Pareto frontier between the two remains open for further exploration.

## Related Work & Insights

- **vs. Taming-3DGS**: Both leverage multi-view information, but Taming indirectly scores Gaussians via their attributes (opacity/scale/gradient), imposing insufficient constraints and still requiring 0.68M Gaussians. FastGS directly evaluates contribution to reconstruction quality and achieves comparable quality with only 0.40M Gaussians.
- **vs. Speedy-Splat**: Its pruning relies on Hessian approximation gradients, which only indirectly exploit multi-view consistency and lead to severe quality degradation (PSNR 26.91 vs. FastGS 27.56). FastGS's VCP is more precise while preserving quality.
- **vs. DashGaussian**: The current state of the art uses resolution scheduling to maintain quality but still requires 2.40M Gaussians. FastGS-Big surpasses its quality with half the Gaussian count.
- **vs. Mini-Splatting**: Uses an intersection-preserving simplification strategy; although Gaussian count is low (0.53M), training time (17.69 min) is far slower than FastGS.

The multi-view consistency scoring concept can be generalized to any 3D representation learning task requiring control over point cloud or primitive count. The error-map-based scoring of VCD/VCP is compatible with importance sampling for NeRF acceleration. The Mahalanobis distance-based pruning in CB is transferable to other tile-based rasterization methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The VCD/VCP designs are concise and effective; individual components are not complex, but their combination yields strong results.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Six task categories, multiple backbones, multiple datasets, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, visual comparisons are intuitive, and supplementary materials are thorough.
- **Value**: ⭐⭐⭐⭐⭐ — Training 3DGS in 100 seconds has significant practical value and strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](../../ICCV2025/3d_vision/faster_and_better_3d_splatting_via_group_training.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] ReLaGS: Relational Language Gaussian Splatting](relags_relational_language_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
