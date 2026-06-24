---
title: >-
  [Paper Note] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution
description: >-
  [CVPR 2025][Video Generation][Video Super-Resolution] The BF-STVSR framework is proposed to model temporal motion interpolation using a B-spline Mapper and capture spatial high-frequency details using a Fourier Mapper, achieving SOTA performance in continuous spatial-temporal video super-resolution without relying on external optical flow networks.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Super-Resolution"
  - "B-spline"
  - "Fourier"
  - "Continuous Spatial-Temporal Super-Resolution"
  - "Motion Interpolation"
date: 2026-05-08
content_hash: a0c52a5c06ed6419
---

# BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2501.11043](https://arxiv.org/abs/2501.11043)  
**Code**: Yes (mentioned in paper)  
**Area**: Video Generation  
**Keywords**: Video Super-Resolution, B-spline, Fourier, Continuous Spatial-Temporal Super-Resolution, Motion Interpolation

## TL;DR

The BF-STVSR framework is proposed to model temporal motion interpolation using a B-spline Mapper and capture spatial high-frequency details using a Fourier Mapper, achieving SOTA performance in continuous spatial-temporal video super-resolution without relying on external optical flow networks.

## Background & Motivation

1. **Background**: Continuous spatial-temporal video super-resolution (C-STVSR) aims to simultaneously upscale videos to arbitrary temporal and spatial resolutions. Existing methods such as VideoINR and MoTIF employ Implicit Neural Representations (INR) to map spatial-temporal coordinates to pixel values, but they exhibit insufficient capacity in modeling the complexity of video data.

2. **Limitations of Prior Work**: The INR components of VideoINR and MoTIF only utilize simple coordinate concatenation without introducing effective positional encodings, leading to the **spectral bias** problem—making it difficult to capture high-frequency spatial details. MoTIF also relies on a pre-trained optical flow network (RAFT) to provide motion guidance, which increases computational overhead and restricts model flexibility.

3. **Key Challenge**: The authors discovered a counterintuitive phenomenon—directly adding positional encodings (such as Fourier features) in C-STVSR not only fails to improve performance but actually **degrades** it. This stands in stark contrast to the widespread success of positional encodings in image super-resolution. This issue is particularly severe when a pre-trained optical flow network is present, as the optical flow network might restrict the flexibility of the model to exploit diverse video information.

4. **Goal**: (1) How to effectively model video motion without depending on external optical flow; (2) How to overcome spectral bias to capture spatial high-frequency details; (3) How to design a continuous representation suited to the spatial-temporal characteristics of videos.

5. **Key Insight**: The temporal axis (motion) and spatial axis (details) of video exhibit distinctly different characteristics—motion is smooth and continuous, whereas spatial details are dominated by frequency information. Therefore, dedicated modules should be designed separately for each axis, rather than using a unified MLP to handle both.

6. **Core Idea**: Smooth temporal motion trajectories are modeled using B-spline basis functions, and spatial frequency details are captured using Fourier basis functions, replacing the unified MLP + external optical flow scheme.

## Method

### Overall Architecture

The overall pipeline of BF-STVSR is as follows: Given two low-resolution input frames ^L, I_1^L \in \mathbb{R}^{3 \times H \times W}$, the goal is to generate a high-resolution intermediate frame ^H \in \mathbb{R}^{3 \times sH \times sW}$ at an arbitrary time \in [0,1]$ and arbitrary spatial scale factor $. An encoder extracts three feature maps ^L, F_{(0,1)}^L, F_1^L$, where {(0,1)}^L$ is a template feature that fuses information from both frames. The B-spline Mapper predicts motion vectors to the target time, and the Fourier Mapper predicts high-resolution spatial features. Finally, the intermediate frame is generated via forward warping using softmax splatting.

### Key Designs

1. **B-spline Mapper (Temporal Motion Modeling)**:
    - Function: Predicts high-resolution motion vectors {0 \to t}^H, M_{1 \to t}^H$ and reliability maps.
    - Mechanism: Instead of directly predicting motion to the target time $, it predicts B-spline coefficients $ and knots $, and smoothly interpolates along the temporal axis via B-spline basis functions \psi(z_r, \delta_r, \hat{t}) = c_r \odot eta^n\left(\frac{\hat{t} - k_r}{d}\right)$. The coefficients/knots are estimated from the encoded features by a three-layer SIREN network, and the dilation factor is predicted from the frame interval.
    - Design Motivation: B-splines are naturally suited for modeling continuous, smooth signals, and object motion in video is inherently smooth and continuous, making it more elegant than direct prediction of motion vectors by MLPs. Meanwhile, estimating motion directly from encoded features rather than an external optical flow network eliminates the dependency on RAFT.

2. **Fourier Mapper (Spatial Frequency Modeling)**:
    - Function: Predicts high-resolution spatial features ^H, F_1^H$ from low-resolution features.
    - Mechanism: Estimates the dominant frequency $ and amplitude $ for each query coordinate to construct Fourier bases: \phi(z_r, \delta_r) = A_r \odot [\cos(\pi F_r \delta_r); \sin(\pi F_r \delta_r)]$. The frequency and amplitude are estimated by SIREN networks, respectively, followed by a linear projection to obtain the final features.
    - Design Motivation: The spectral bias of INR leads to the loss of high-frequency details. Explicitly predicting dominant frequency information effectively captures spatial details, inspired by the LTE method in image super-resolution. Unlike LTE, it does not include a phase estimator.

3. **Forward Warping and Decoding**:
    - Function: Propagates spatial features to the target time and generates the final high-resolution frame.
    - Mechanism: Uses softmax splatting to forward-warp and fuse ^H, F_1^H$ according to the motion vectors {0 \to t}^H, M_{1 \to t}^H$, obtaining the intermediate feature ^H$. The warped features are concatenated with time $ and the template feature {(0,1)}^H$ before being decoded.
    - Design Motivation: Compared to backward warping, forward warping can more naturally handle occlusions and many-to-one mappings.

### Loss & Training

- **Simplified Loss Function**: Uses only the Charbonnier loss \mathcal{L} = \mathcal{L}_{char}(\hat{I}_t^H, I_t^H)$, removing the optical flow supervision \mathcal{L}_{RAFT} used in MoTIF.
- **Two-Stage Training**: The spatial scale factor is fixed at 4× for the first 450K iterations, and uniformly sampled from [2,4] for the remaining 150K iterations.
- Adam optimizer with cosine annealing learning rate (^{-4} \to 10^{-7}$), batch size of 32.
- Training Stability: Replaces predicted optical flow with ground-truth optical flow with a certain probability (gradually decaying from 1.0 to 0).

## Key Experimental Results

### Main Results (Fixed-scale STVSR, 4× Spatial, 8× Temporal)

| Method | Vid4 PSNR/SSIM | GoPro-Center | GoPro-Avg | Adobe-Center | Adobe-Avg | Params |
|------|---------------|-------------|-----------|-------------|-----------|--------|
| VideoINR | 25.61/0.7709 | 30.26/0.8792 | 29.41/0.8669 | 29.92/0.8746 | 29.27/0.8651 | 11.31M |
| MoTIF | 25.79/0.7745 | 31.04/0.8877 | 30.04/0.8773 | 30.63/0.8839 | 29.82/0.8750 | 12.55M |
| BF-STVSR+\mathcal{L}_{RAFT} | 25.80/0.7754 | 31.14/0.8893 | 30.20/0.8799 | **30.84/0.8877** | **30.14/0.8808** | 13.47M |
| **BF-STVSR** | **25.85/0.7772** | **31.17/0.8898** | **30.22/0.8802** | 30.83/0.8880 | 30.12/0.8808 | 13.47M |

Compared to MoTIF, it achieves a gain of +0.18 dB PSNR on GoPro-Avg and +0.30 dB on Adobe-Avg. The version without optical flow supervision performs slightly better.

### Ablation Study (Impact of Optical Flow and Positional Encoding)

| Configuration | Optical Flow Net | B-spline | Fourier | \mathcal{L}_{RAFT} | GoPro-Avg | Adobe-Avg |
|------|---------|----------|---------|--------|-----------|-----------|
| MoTIF Baseline | ✓ | ✗ | ✗ | ✓ | 30.04/0.8773 | 29.82/0.8750 |
| +Flow+Fourier | ✓ | ✗ | ✓ | ✓ | 29.94/0.8764 | 29.73/0.8741 |
| +Flow+B-spline | ✓ | ✓ | ✗ | ✓ | 30.03/0.8774 | 29.81/0.8756 |
| B-spline Only | ✗ | ✓ | ✗ | ✗ | 30.12/0.8783 | 30.02/0.8784 |
| Fourier Only | ✗ | ✗ | ✓ | ✓ | 30.16/0.8792 | 30.11/0.8801 |
| B+F+\mathcal{L}_{RAFT} | ✗ | ✓ | ✓ | ✓ | 30.20/0.8799 | 30.14/0.8808 |
| **B+F (Full)** | ✗ | ✓ | ✓ | ✗ | **30.22/0.8802** | 30.12/0.8808 |

### Key Findings

- **Optical Flow Networks are a Burden**: Combining B-spline/Fourier with a pre-trained optical flow network actually degrades performance (lines 2 and 3). Removing the optical flow network comprehensively improves performance.
- **Both Modules are Indispensable**: Using either B-spline or Fourier alone is inferior to combining them; B-spline primarily contributes temporal consistency, while Fourier primarily contributes spatial details.
- **No Need for Optical Flow Supervision**: Removing \mathcal{L}_{RAFT} improves performance instead of degrading it, indicating that the model has successfully learned motion autonomously from the encoded features.
- **Higher Computational Efficiency**: Removing the optical flow network yields the lowest FLOPs and inference times, and custom CUDA kernels are implemented to accelerate B-spline computation.

## Highlights & Insights

- **Innovation Driven by Counterintuitive Findings**: The discovery of positional encodings failing in C-STVSR is highly valuable, revealing the conflict between pre-trained optical flow networks and positional encodings—the hard constraints provided by optical flow limit the optimization space of positional encodings.
- **Axis-Separated Design Paradigm**: Handling distinct temporal and spatial characteristics with different mathematical tools—B-splines for smooth motion continuity and Fourier bases for spatial frequency details—is a highly transferable concept that can be applied to other multi-axis signal processing tasks.
- **Custom CUDA Kernels**: Implementing dedicated CUDA kernels for B-spline basis functions provides engineering optimizations that make the method highly viable for practical deployment.

## Limitations & Future Work

- **Difficulties in Large-Motion Scenes**: When object motion between frames is extremely large, all C-STVSR methods (including this work) still produce blur and artifacts.
- **High Training Cost**: The two-stage training with 450K + 150K iterations is time-consuming.
- Generalization to other domains (such as surgical videos, surveillance, etc.) has not been verified since training was restricted to Adobe240.
- Performance on Vid4 is surpassed by TMNet, as TMNet was trained on Vimeo90K which shares similar characteristics with Vid4—domain matching of training data remains crucial.

## Related Work & Insights

- **vs MoTIF**: MoTIF relies on an external RAFT optical flow network for motion guidance; this paper proves that learning motion directly from encoded features is more efficient and yields better results.
- **vs VideoINR**: VideoINR relies on simple MLPs + coordinate concatenation for continuous mapping; this paper replaces them with targeted B-splines and Fourier basis functions.
- **vs LTE (Image SR)**: The Fourier Mapper in this work borrows the dominant-frequency estimation concept from LTE but adapts it to the video domain, omitting the phase estimator.

## Rating

- Novelty: ⭐⭐⭐⭐ The counterintuitive findings are inspiring, and the axis-separated design is clean, though individual components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across multiple datasets and metrics, with detailed ablation studies and computational efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Logical deduction of motivation; well-designed figures and tables.
- Value: ⭐⭐⭐⭐ Substantially advances the C-STVSR field, improving both efficiency and effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] PatchVSR: Breaking Video Diffusion Resolution Limits with Patch-Wise Video Super-Resolution](patchvsr_breaking_video_diffusion_resolution_limits_with_patch-wise_video_super-.md)
- [\[CVPR 2026\] Thermal Diffusion Matters: Infrared Spatial-Temporal Video Super-Resolution through Heat Conduction Priors](../../CVPR2026/video_generation/thermal_diffusion_matters_infrared_spatial-temporal_video_super-resolution_throu.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](../../ICCV2025/video_generation/vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[CVPR 2025\] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)

</div>

<!-- RELATED:END -->
