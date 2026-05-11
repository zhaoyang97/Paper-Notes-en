---
title: >-
  [Paper Note] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs
description: >-
  [CVPR 2026][3D Vision][blur-robust SLAM] Rather than naively inserting a deblurring network into the SLAM front-end, Unblur-SLAM is designed around a central decision: which blurry frames can be deblurred prior to tracki…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "blur-robust SLAM"
  - "3DGS"
  - "single-image deblurring"
  - "sub-frame modeling"
  - "hybrid bundle adjustment"
date: 2026-05-08
content_hash: 21ea01bc18b25e55
---

# Unblur-SLAM: Dense Neural SLAM for Blurry Inputs

**Conference**: CVPR 2026
**arXiv**: [2603.26810](https://arxiv.org/abs/2603.26810)
**Code**: [https://github.com/SlamMate/Unblur-SLAM.git](https://github.com/SlamMate/Unblur-SLAM.git)
**Area**: 3D Vision / Neural SLAM
**Keywords**: blur-robust SLAM, 3DGS, single-image deblurring, sub-frame modeling, hybrid bundle adjustment

## TL;DR

Rather than naively inserting a deblurring network into the SLAM front-end, Unblur-SLAM is designed around a central decision: which blurry frames can be deblurred prior to tracking, and which must be modeled directly in 3D space. This insight drives a complete pipeline comprising blur detection, physically constrained deblurring, 3D Gaussian blur refinement, and a severe-blur fallback, enabling the system to handle both motion blur and defocus blur while substantially improving tracking and reconstruction quality.

## Background & Motivation

Most SLAM systems assume sufficiently sharp input frames. Both classical feature-based methods and modern dense/neural SLAM approaches fundamentally rely on establishing reliable correspondences across adjacent views. Once images are degraded by motion blur or defocus blur, front-end tracking deteriorates and back-end reconstruction is consequently impaired.

Existing blur-aware SLAM works suffer from two main problems. First, many methods assume blur originates exclusively from camera motion and therefore focus on modeling motion blur while neglecting defocus blur. Real-world data does not conform to this assumption, as both types of blur frequently co-occur in smartphone, handheld, and indoor low-light capture scenarios. Second, many blur-aware SLAM approaches treat every frame as blurry, which significantly increases computational cost and is inconsistent with the reality that blurry frames constitute only a fraction of typical sequences.

The authors therefore propose a more refined problem formulation:

- Not every frame requires expensive blur optimization.
- Not every blurry frame can be recovered by a single-image deblurring network.
- When single-image deblurring fails, the system should switch to a more robust 3D modeling strategy rather than collapse.

In other words, what SLAM needs is not a universal deblurrer, but a system capable of routing frames of different blur levels and types through appropriate processing branches. Unblur-SLAM is designed under this premise.

## Method

### Overall Architecture

The system first evaluates the blur magnitude of each input frame and then routes it into one of three categories:

- **Sharp frames**: Passed directly to the Droid-SLAM front-end and 3DGS back-end without additional expensive processing.
- **Successfully deblurred blurry frames**: Blurry frames that can be recovered by the single-image deblurring network; these are deblurred before tracking and mapping, with residual blur further refined in the back-end.
- **Failed blurry frames**: Frames where blur is too severe or of too complex a type for single-image deblurring to succeed. These frames are not forced through the tracker; instead, their imaging process is explained directly in 3DGS space via multi-sub-frame rendering and a blur network.

This three-way routing is the central contribution of the paper. It avoids both extremes: it does not assume every frame requires heavy blur optimization, nor does it discard information when deblurring fails.

### Key Designs

1. **Blur Detection and Frame Routing**

    - **Function**: Determine whether a given frame should follow the standard SLAM pipeline, the deblur-then-SLAM pipeline, or the 3D blur fallback.
    - **Mechanism**: The authors construct a blur detection benchmark comprising real and semi-synthetic data, evaluate 39 image quality/blur metrics, and select ARNIQA as the default blur detector. At runtime, frames with blur scores below a threshold are treated as sharp; otherwise they enter the deblurring branch, where additional criteria such as the Laplacian ratio are used to assess whether deblurring succeeds.
    - **Design Motivation**: The system should not treat all frames uniformly. Identifying blur magnitude first allows computational budget to be allocated where it is genuinely needed.

2. **Physically Constrained Single-Image Deblurring Network**

    - **Function**: Provide the front-end tracker and depth estimator with images as close as possible to the mid-exposure sharp frame.
    - **Mechanism**: A two-stage training strategy is adopted. The first stage trains on semi-synthetic motion blur data (RED, GoPro, ReplicaBlurry) to teach the network to recover the mid-exposure frame rather than merely sharpen the image. The second stage fine-tunes on the DPDD defocus dataset to improve robustness to defocus blur. Physics-based mid-frame constraints are applied during training to prevent the network from learning spurious sharp textures inconsistent with 3D geometry.
    - **Design Motivation**: Methods such as I2-SLAM have demonstrated that conventional 2D deblurring networks, without geometric consistency constraints, tend to produce textures that appear sharp but are detrimental to multi-view matching. The authors therefore emphasize recovering the mid-exposure ground truth rather than pursuing purely visual sharpening.

3. **3DGS Residual Blur Modeling and Severe-Blur Fallback**

    - **Function**: Continue modeling residual blur for imperfectly deblurred frames; provide a robust alternative path for frames where deblurring fails.
    - **Mechanism**: The system uses Splat-SLAM-style 3D Gaussian Splatting as the map representation. For successfully deblurred frames, a Blur Proposal Network estimates per-pixel convolution kernels and masks to apply depth-conditioned residual deblurring and detail enhancement to rendered images. For failed frames, multiple sub-frames are rendered along a virtual camera trajectory, each processed by the blur proposal, and then averaged to form an observation that directly models the blurry imaging process.
    - **Design Motivation**: Mildly to moderately blurry frames are better suited to a deblur-then-track strategy, while severely blurry frames are better handled by explaining their blur in 3D space. This is more robust than forcing all frames through a single pipeline.

### Loss & Training

Back-end optimization employs three categories of losses.

- **Sharp frame loss**: Higher weight is assigned to sharp frames, which serve as strong geometric and appearance anchors.
- **Deblur frame loss**: Multi-scale RGB and depth consistency optimization is applied to successfully deblurred frames, regularized by a sparse mask term.
- **Fail frame loss**: RGB and depth errors computed from multi-sub-frame synthesis are used to optimize failed frames.

The total loss is a weighted sum over all frames, with $w_\text{sharp} > w_\text{deblur} = w_\text{fail}$. Sliding-window BA, loop closure, and global BA are retained, and a Gaussian scale regularizer is incorporated in global optimization to prevent excessive stretching.

## Key Experimental Results

### Main Results

The method is evaluated on extreme-blur synthetic scenes, an offline deblurring standard benchmark, and real-world SLAM datasets, measuring both tracking ATE and reconstruction PSNR/SSIM/LPIPS.

| Dataset / Method | Key Metric 1 | Key Metric 2 | Result |
|---|---|---|---|
| ArchViz-1 MBA-SLAM | ATE | PSNR | 0.0075 / 28.45 |
| ArchViz-1 Ours | ATE | PSNR | **0.0075 / 28.76** |
| ArchViz-2 MBA-SLAM | ATE | PSNR | 0.0036 / 30.16 |
| ArchViz-2 Ours | ATE | PSNR | **0.0027 / 32.71** |
| ArchViz-3 MBA-SLAM | ATE | PSNR | 0.0141 / 27.85 |
| ArchViz-3 Ours | ATE | PSNR | **0.0067 / 30.09** |
| Deblur-NeRF offline Prev. SOTA (CoMoGaussian) | PSNR / SSIM / LPIPS | - | 27.85 / 0.8431 / 0.0822 |
| **Ours** | PSNR / SSIM / LPIPS | - | **29.49 / 0.9213 / 0.0728** |

Two observations stand out. First, Unblur-SLAM outperforms MBA-SLAM on the ArchViz synthetic scenes, which consist almost entirely of extreme blur, with particularly notable gains in ATE and PSNR on the third sequence. Second, and more strikingly, the online method surpasses a range of offline methods on the Deblur-NeRF benchmark, demonstrating the effectiveness of its 3D blur modeling.

### Real-World Data and Runtime Efficiency

The authors further evaluate trajectory error on TUM RGB-D and IndoorMCD, and report mapping quality and runtime speed on TUM.

| Experiment | Comparison | Metric | Result |
|---|---|---|---:|
| TUM Tracking | Droid-SLAM | ATE [m] | 0.380 |
| TUM Tracking | Ours* | ATE [m] | 0.352 |
| TUM Tracking | **Ours** | ATE [m] | **0.336** |
| MCD Tracking | Droid-SLAM | ATE [m] | 0.138 |
| MCD Tracking | **Ours** | ATE [m] | **0.128** |
| TUM Mapping fr1_desk | I2-SLAM / Ours | PSNR | 27.23 / **28.03** |
| TUM Mapping fr2_xyz | I2-SLAM / Ours | PSNR | **32.06** / 31.14 |
| TUM Mapping fr3_office | I2-SLAM / Ours | PSNR | 28.91 / **29.22** |
| Runtime Speed | Splat-SLAM | FPS | 1.24 |
| Runtime Speed | I2-SLAM | FPS | 0.095 |
| Runtime Speed | Ours w/o ref. | FPS | 0.85 |
| Runtime Speed | **Ours** | FPS | **0.74** |

### Key Findings

- Frame routing is critical. Routing all frames through heavy blur refinement slows the system considerably, while feeding all blurry frames directly to the tracker leads to tracking failures. Unblur-SLAM achieves a balance between stability and speed precisely through its routing mechanism.
- Physically constrained mid-frame deblurring training is necessary. The authors specifically note that conventional single-image deblurring without 3D consistency constraints can actually harm SLAM performance.
- The severe-blur fallback is not a rare edge case. Whenever the camera undergoes rapid motion or significant defocus is present, this branch prevents frame dropping or system failure.
- Although the final speed remains below that of pure Splat-SLAM, it is substantially faster than I2-SLAM, indicating that the system retains a degree of practical online usability.

## Highlights & Insights

- The most commendable aspect of this paper is its systems perspective. Much related work addresses only one of two problems — sharper deblurring or more robust SLAM — whereas this paper designs distinct branches around the failure modes of the entire pipeline.
- The authors do not overestimate single-image deblurring. It is treated as a preprocessing corrector, and upon failure, the system immediately switches to blur modeling in 3D space — a mature engineering approach.
- The use of distinct losses and update strategies for successfully deblurred frames and failed frames reflects the authors' understanding that blur is not a scalar intensity but represents different perturbations to the imaging process.
- The online method's ability to surpass offline methods on Deblur-NeRF is notable, indicating that the 3DGS + blur network combination possesses strong deblurring capability beyond its role in SLAM.

## Limitations & Future Work

- Although faster than many blur-aware SLAM systems, 0.74 FPS remains far from real-time, particularly on mobile platforms.
- The system currently targets static scenes. In the presence of prominent dynamic objects, blur sources and geometric changes become coupled, which the existing model may not handle stably.
- The severe-blur fallback relies on virtual sub-frames and the blur proposal network, resulting in a long and parameter-heavy pipeline; more compact representations warrant future exploration.
- The current blur detector uses a fixed threshold strategy, and its generalization to different devices and exposure settings requires more extensive external evaluation.
- A natural future direction is to incorporate spatial priors from 3D foundation models into blur-aware SLAM, enabling the system not only to see more clearly but to reason more deeply about geometry.

## Related Work & Insights

- **vs. MBA-SLAM / Deblur-SLAM**: These methods focus primarily on motion blur and typically assume all frames are blurry. Unblur-SLAM covers both motion blur and defocus blur and explicitly handles three frame categories: sharp, blurry, and failed.
- **vs. I2-SLAM**: I2-SLAM recognizes the importance of imaging process modeling but does not fully exploit the geometric consistency of single-image deblurring. This paper addresses that gap through physically constrained training and 3DGS refinement.
- **vs. offline 3DGS deblurring methods**: Methods such as BAGS and CoMoGaussian target high-quality offline reconstruction; this paper brings those ideas into online SLAM and adds a fallback for severely blurry frames.
- A broader insight for the research community is that perceptual preprocessing modules should not be designed in isolation. What most determines system-level performance is often the handling of preprocessing failures. The routing philosophy of Unblur-SLAM is equally applicable to other challenging imaging conditions such as low light, noise, and haze.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Jointly addresses motion blur and defocus blur, and systematically incorporates a deblurring-failure branch into SLAM; the combined design is strong.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic, real-world, offline benchmark, and runtime evaluations are all included, though additional long real-world sequences and dynamic scenes would strengthen the paper.
- **Writing Quality**: ⭐⭐⭐⭐ — The pipeline is lengthy but well-organized, with clearly delineated responsibilities for each module.
- **Value**: ⭐⭐⭐⭐⭐ — Highly relevant for real-world SLAM deployment, particularly in handheld and mobile device scenarios where blur is frequent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM](sgad-slam_splatting_gaussians_at_adjusted_depth_for_better_radiance_fields_in_rg.md)
- [\[CVPR 2026\] VGGT-SLAM++: Visual SLAM with DEM-Based Covisibility and Local Bundle Adjustment](vggt-slam.md)
- [\[CVPR 2026\] DROID-W: DROID-SLAM in the Wild](droid-slam_in_the_wild.md)
- [\[AAAI 2026\] FoundationSLAM: Unleashing the Potential of Deep Foundation Models in End-to-End Dense Visual SLAM](../../AAAI2026/3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)

</div>

<!-- RELATED:END -->
