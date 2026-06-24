---
title: >-
  [Paper Note] Flying with Photons: Rendering Novel Views of Propagating Light
description: >-
  [ECCV 2024][3D Vision][Transient Imaging] Proposed the Transient Field representation, combined with a first-of-its-kind multi-view ultrafast imaging dataset, achieving the first novel-view rendering of light propagation videos in real-world scenes from dynamic perspectives, capable of processing complex light transport effects such as scattering, reflection, refraction, and diffraction.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Transient Imaging"
  - "Novel View Synthesis"
  - "Light Propagation"
  - "Neural Radiance Fields"
  - "SPAD"
date: 2026-05-08
content_hash: cbbd698cb3baa0ce
---

# Flying with Photons: Rendering Novel Views of Propagating Light

**Conference**: ECCV 2024  
**arXiv**: [2404.06493](https://arxiv.org/abs/2404.06493)  
**Code**: [https://anaghmalik.com/FlyingWithPhotons](https://anaghmalik.com/FlyingWithPhotons) (Project page, including code and datasets)  
**Area**: 3D Vision  
**Keywords**: Transient Imaging, Novel View Synthesis, Light Propagation, Neural Radiance Fields, SPAD

## TL;DR

Proposed the Transient Field representation, combined with a first-of-its-kind multi-view ultrafast imaging dataset, achieving the first novel-view rendering of light propagation videos in real-world scenes from dynamic perspectives, capable of processing complex light transport effects such as scattering, reflection, refraction, and diffraction.

## Background & Motivation

**Background**: Ultrafast cameras (such as SPADs) can record light propagation videos (transient videos) at trillions of frames per second, which is of great value for understanding light transport mechanisms. Meanwhile, novel view synthesis methods like NeRF have achieved immense success in regular imaging.

**Limitations of Prior Work**: Existing ultrafast imaging systems are primarily designed for single-view acquisition, lacking multi-view transient datasets. Existing novel view synthesis methods (NeRF, K-Planes, etc.) are designed for time scales where the speed of light is negligible and cannot handle view-dependent propagation delay effects at picosecond-level time resolutions.

**Key Challenge**: On ultrafast time scales, the camera-to-scene distance directly affects measurements (due to the speed of light delay), which is completely opposite to the assumption in existing video synthesis methods that "the time scale is much larger than the light propagation time." How to correctly model the finite speed of light effects within a volume rendering framework is the core challenge.

**Goal**: Learn 3D representations from multi-view captured transient videos and render light propagation videos from arbitrary (including dynamic) novel views.

**Key Insight**: Define a new transient field representation—mapping 3D points and viewing directions to a high-dimensional discrete-time signal and introducing a light propagation delay term into the volume rendering equation. Concurrently, build a physical multi-view SPAD acquisition system to provide training data.

**Core Idea**: Replace the radiance field with a transient field within the NeRF framework, modeling the speed of light delay using time-shifting convolution to achieve novel view synthesis of light propagation.

## Method

### Overall Architecture

The system consists of two parts: (1) **Hardware acquisition system**: A scanning SPAD system based on a rotation stage and an elevation arm, combined with a 532nm pulsed laser, to capture picosecond-scale transient videos from a hemispherical multi-view setup; (2) **Rendering framework**: An Instant-NGP-based neural network that learns the density field and transient field, synthesizing novel-view transient videos through a modified volume rendering equation (incorporating propagation delay terms).

### Key Designs

1. **SPAD Measurement Model**:

    - **Function**: Establish a forward model from the scene's impulse response to SPAD photon counts.
    - **Mechanism**: The detector integrates photons within a time window of width $W$: $\lambda_{\mathbf{r}}[n] \propto \int_{nW}^{(n+1)W} h(\mathbf{r}, t) \, dt$. Photon arrival follows an inhomogeneous Poisson process: $\tilde{\tau}_{\mathbf{r}}[n] \sim \text{Poisson}(P\eta\lambda_{\mathbf{r}}[n] + B)$, where $P$ is the number of laser pulses, $\eta$ is the detection efficiency, and $B$ is the background noise.
    - **Design Motivation**: A physically accurate measurement model ensures that the learned representation is consistent with the real sensor. Under low flux conditions (<5% detection rate), second-order effects like dead-time can be neglected.

2. **Transient Field**:

    - **Function**: Define a new neural field representation that outputs a high-dimensional discrete-time signal instead of a single radiance value.
    - **Mechanism**: The transient field $\boldsymbol{\tau}_\theta: (\mathbf{r}(s), \mathbf{d}) \mapsto \mathbb{R}_+^N$ maps 3D points and directions to an $N$-dimensional transient vector. The density field $\sigma(\mathbf{r}(s))$ remains a scalar. The key innovation is the modified volume rendering equation:
    $\boldsymbol{\tau}_{\mathbf{r}} = \int_{s_n}^{s_f} T(s)\sigma(\mathbf{r}(s)) \left[\boldsymbol{\tau}_\theta(\mathbf{r}(s), \mathbf{d}) * \delta\left[n - \|\mathbf{r}(s) - \mathbf{o}\|/(cW)\right]\right] ds$
   where $*$ denotes convolution, and $\delta[\cdot]$ is the Kronecker delta function realizing time shift. The transient signal of each sample point is delayed by the speed of light based on its distance to the camera center $\|\mathbf{r}(s) - \mathbf{o}\|$ (where $c$ is the speed of light).
    - **Design Motivation**: Without modeling propagation delay, the transient of the same 3D point viewed from different angles would have different time offsets, leading to mapping ambiguity. Explicitly modeling the delay allows learning a canonical spatiotemporal representation.

3. **Time Warping**:

    - **Function**: Realize different ways of visualization by adding or removing propagation delay.
    - **Mechanism**: Depth-based time warping removes the propagation delay from scene points to the camera, using the expected ray termination distance to calculate the delay amount. The warped visualization possesses a consistent appearance across different views.
    - **Design Motivation**: In dynamic camera rendering, view-dependent propagation delays make visualization difficult to comprehend. Time warping provides a more intuitive visualization of light propagation.

4. **Relativistic Rendering Extension**:

    - **Function**: Simulate visual effects when the camera moves at relativistic speeds.
    - **Mechanism**: Model four relativistic effects: (1) time dilation; (2) focal length deformation due to Lorentz contraction; (3) light aberration (rays compressing and bending toward the direction of motion); (4) searchlight effect (increased photon flux when moving toward the light source).
    - **Design Motivation**: Relativistic effects can be observed under camera motion on ultrafast time scales, enhancing the physical completeness and demonstration quality of the method.

### Loss & Training

- **Loss Function**: L2 loss + gamma compression: $\mathcal{L} = \sum_{v,\mathbf{r},n} \|g(\tilde{\tau}_{\mathbf{r}}^{(v)}[n]) - \tau_{\mathbf{r}}^{(v)}[n]\|_2^2$, where $g(x) = x^{1/\gamma}$ is used to compress the high dynamic range
- **Gamma Parameter**: $\gamma=5$ for simulated data, $\gamma=2$ for real data
- **Optimizer**: Adam, learning rate is annealed by multiplying by 0.33 at 50%, 75%, and 90% training progress
- **Training Budget**: 500K iterations/approx. 10 hours for simulated scenes, 1M iterations/approx. 20 hours for real scenes (A40 GPU)
- Instant-NGP implementation based on NerfAcc

## Key Experimental Results

### Main Results

| Method | Parameters | Rendering Time | PSNR(dB)↑ | LPIPS↓ | SSIM↑ | T-IOU↑ |
|------|--------|---------|-----------|--------|-------|--------|
| T-NeRF | 15M | 7.1s | 26.35 | 0.338 | 0.887 | 0.729 |
| K-Planes | 37M | 320.7s | 20.55 | 0.431 | 0.666 | 0.358 |
| W/o propagation delay | 15M | 11.9s | 27.79 | 0.334 | 0.861 | 0.334 |
| **Ours** | **15M** | **12.8s** | **32.97** | **0.247** | **0.965** | **0.830** |

### Real Data

| Method | Parameters | Rendering Time | PSNR↑ | LPIPS↓ | SSIM↑ | T-IOU↑ |
|------|--------|---------|-------|--------|-------|--------|
| K-Planes | 43M | 37min | 24.12 | 0.516 | 0.594 | 0.395 |
| W/o propagation delay | 15M | 5.78s | 17.12 | 0.529 | 0.346 | 0.174 |
| **Ours** | **15M** | **28.0s** | **24.95** | **0.431** | **0.666** | **0.468** |

### Key Findings

- **Propagation delay modeling is crucial**: Removing propagation delay drops PSNR from 32.97 to 27.79 (simulation) and from 24.95 to 17.12 (real-world), with T-IOU decreasing substantially. This demonstrates that explicitly modeling the speed of light delay is key to novel view synthesis at ultrafast time scales.
- **Limitations of T-NeRF**: T-NeRF can only recover direct light components and fails to model indirect light transport effects (multiple scattering, refraction, etc.).
- **Inefficiency of K-Planes**: K-Planes requires rendering each time frame individually, resulting in rendering times 25$\times$ (simulation) to 80$\times$ (real-world) longer than our method.
- **Direct-Global Separation**: By preprocessing transient data using Gaussian mixture models, models for direct and global light transport components can be trained separately, achieving 3D visualization of direct-global separation.

## Highlights & Insights

- **Pioneering Work**: Achieved multi-view novel view synthesis of light propagation in real scenes for the first time, bridging the gap between ultrafast imaging and neural rendering.
- **Elegant Integration of Physical Modeling**: Gracefully introduced speed of light delay in the volume rendering equation via convolution + Kronecker delta—the core contribution, mathematically concise and physically correct.
- **Complete System Contribution**: Provided not only algorithms but also built a hardware acquisition system and the first multi-view transient video dataset, open-sourcing the code and data.
- **New Evaluation Metric**: Proposed Transient IoU (T-IOU) for evaluating the temporal accuracy of synthesized transients.
- **Rich Extended Applications**: Time warping, relativistic rendering, and direct-global separation demonstrate the flexibility of the framework.

## Limitations & Future Work

- Long acquisition time (20-30 minutes per transient video), restricted to static scenes.
- Real data resolution is 512$\times$512 with a temporal resolution of about 70ps, still short of theoretical limits.
- Currently only handles grayscale transients for real captures; color is only implemented in simulation.
- Emerging SPAD arrays could be leveraged for multi-view synchronized acquisition to support dynamic scenes.
- Joint inference of scene geometry, albedo, and material properties can be explored.

## Related Work & Insights

- **vs TransientNeRF**: T-NeRF only handles direct components of co-axial LiDAR and does not support indirect light transport. The proposed method supports non-coaxial light sources and full global light transport effects.
- **vs K-Planes**: A video novel view synthesis method, but it does not model propagation delays, needing to render each frame individually, which is extremely slow.
- **vs Jarabo et al.**: Also explored novel view rendering of transient videos, but relied on known geometry and single-view data. The proposed method jointly optimizes geometry and appearance from multiple views, making it more general.
- **vs Velten et al.**: Original light-in-flight visualization work. The proposed method extends their time warping technique to volume rendering frameworks and dynamic cameras.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Achieves multi-view novel view synthesis of light propagation for the first time; both the transient field representation and propagation delay modeling are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively validated with simulation and real captures, showing multiple light transport effects, but the number of real-world scenes is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear writing, rigorous physical model derivation, and excellent visualization design (peak-time visualization).
- Value: ⭐⭐⭐⭐ Paved a new direction for ultrafast imaging $\times$ neural rendering with high academic value, although short-term application scenarios are narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GAURA: Generalizable Approach for Unified Restoration and Rendering of Arbitrary Views](gaura_generalizable_approach_for_unified_restoration_and_rendering_of_arbitrary_.md)
- [\[ECCV 2024\] MaRINeR: Enhancing Novel Views by Matching Rendered Images with Nearby References](mariner_enhancing_novel_views_by_matching_rendered_images_with_nearby_references.md)
- [\[ECCV 2024\] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks](tracknerf_bundle_adjusting_nerf_from_sparse_and_noisy_views_via_feature_tracks.md)
- [\[CVPR 2026\] Guardians of the Hair: Rescuing Soft Boundaries in Depth, Stereo, and Novel Views](../../CVPR2026/3d_vision/guardians_of_the_hair_rescuing_soft_boundaries_in_depth_stereo_and_novel_views.md)
- [\[CVPR 2026\] PRIMU: Uncertainty Estimation for Novel Views in Gaussian Splatting from Primitive-Based Representations of Error and Coverage](../../CVPR2026/3d_vision/primu_uncertainty_estimation_for_novel_views_in_gaussian_splatting_from_primitiv.md)

</div>

<!-- RELATED:END -->
