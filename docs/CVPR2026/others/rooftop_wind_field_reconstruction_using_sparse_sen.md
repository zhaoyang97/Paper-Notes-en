---
title: >-
  [Paper Note] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods
description: >-
  [CVPR 2026][wind field reconstruction] This work establishes a learning-observation framework based on PIV wind tunnel experimental data…
tags:
  - "CVPR 2026"
  - "wind field reconstruction"
  - "sparse sensors"
  - "UNet"
  - "ViTAE"
  - "CWGAN"
  - "PIV experimental data"
  - "sensor optimization"
date: 2026-05-08
content_hash: bc66afa5a3ed122c
---

# Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods

**Conference**: CVPR 2026
**arXiv**: [2603.13077](https://arxiv.org/abs/2603.13077)
**Code**: [github.com/Yng314/windreconstruction](https://github.com/Yng314/windreconstruction)
**Area**: Other / Fluid Reconstruction
**Keywords**: wind field reconstruction, sparse sensors, UNet, ViTAE, CWGAN, PIV experimental data, sensor optimization

## TL;DR

This work establishes a learning-observation framework based on PIV wind tunnel experimental data, systematically comparing Kriging interpolation with three deep learning models (UNet/ViTAE/CWGAN) for rooftop wind field reconstruction under 5–30 sparse sensors. It demonstrates that under multi-direction training (MDT), deep learning consistently outperforms Kriging (SSIM improvement of 18–34%), and sensor placement robustness is enhanced by up to 27.8% via QR decomposition-based sensor layout optimization.

## Background & Motivation

**Background**: Rooftop spaces increasingly host UAV/UAM landing pads, solar panels, HVAC systems, and other functions, making real-time full-field wind information critical for safe operation. Rooftop flow fields are highly complex—dominated by building geometry effects, they exhibit flow separation, conical vortices, and cross-directional variability. Traditional CFD is computationally expensive and lacks real-time capability, while practical sensor deployments remain extremely sparse.

**Limitations of Prior Work**:
- Existing studies predominantly rely on CFD simulation data for training, failing to capture real measurement noise and turbulence characteristics → models may fail in real-world deployment
- Most studies evaluate only a single network architecture, lacking systematic comparison across different DL methods
- Direction-specific training (training and testing on a single wind direction) limits cross-directional generalization
- Sensor layouts typically follow uniform grids, lacking data-driven optimization

**Key Insight**: This paper is the first to use real PIV wind tunnel experimental data for a comprehensive benchmark, systematically comparing traditional Kriging with three representative DL architectures (deterministic UNet, hybrid ViTAE, generative CWGAN) across two training strategies (single-direction SDT / multi-direction MDT), six sensor densities (5–30), sensor position perturbations, and QR-optimized layouts.

## Method

### Overall Architecture

PIV wind tunnel experimental data (15×15 grid, u/v velocity components) → sparse sensor sampling (Voronoi uniform layout or QR-optimized layout) → four reconstruction methods → evaluation via four metrics: SSIM/NMSE/FAC2/MG. Input dimensions are 15×15×3 (u velocity, v velocity, sensor mask); output is 15×15×2 (reconstructed u/v velocity fields).

Experimental data were acquired in the boundary layer wind tunnel at the Institute of Industrial Science, University of Tokyo. The test object is a 1:200 scale rectangular building model (height:width:length = 1:1:2). Instantaneous velocity fields on the rooftop plane (z/H = 1.05) were measured via PIV under three inflow wind directions (0°, 22.5°, 45°), with a temporal resolution of 0.001 s and spatial resolution of 0.035H. Each 8-second acquisition produced 7,999 temporal snapshots.

### Key Designs

1. **Complementary Design of Four Comparison Methods**:
   - Kriging interpolation: traditional geostatistical method with Gaussian variogram (correlation length 0.5–10.0 grid units), zero nugget effect enforcing exact interpolation, reliant on spatial stationarity assumptions; serves as baseline
   - UNet (472K parameters / 0.03 GFLOPs): encoder-decoder with skip connections, deterministic mapping, 3-layer downsampling (16→8→4→2), filters scaling from 32 to 128 channels, 1×1 convolution output
   - ViTAE (467K parameters / 0.02 GFLOPs): Transformer–CNN hybrid architecture, 3×3 patch partitioning producing 25 patches, linearly projected to 64 dimensions, 8-layer Transformer encoder (8-head attention), CNN decoder restoring spatial resolution
   - CWGAN (8.77M parameters / 1.3 GFLOPs): conditional Wasserstein GAN; generator follows UNet architecture (64→128→256 channels); discriminator uses strided convolutions + LeakyReLU, with sigmoid removed to accommodate the Wasserstein distance
   - **Design Motivation**: the three DL architectures represent three distinct modeling philosophies—deterministic mapping, hybrid attention, and generative adversarial modeling

2. **SDT vs. MDT Training Strategies**:
   - SDT (single-direction training): trained exclusively on three experimental runs at 0°, tested on 22.5° and 45° for cross-directional generalization
   - MDT (multi-direction training): one experimental run per wind direction ($\mathcal{D}_{0°}^{(1)}, \mathcal{D}_{22.5°}^{(1)}, \mathcal{D}_{45°}^{(1)}$) used for training; remaining independent runs used for testing
   - Data splits are performed by independent experimental runs rather than random snapshot sampling; no temporal continuity exists across runs → prevents temporal leakage
   - **Design Motivation**: wind direction varies in real-world deployment; MDT evaluates model generalization under realistic conditions

3. **QR Decomposition-Based Sensor Placement Optimization**:
   - A wind field data matrix $\mathbf{Y} \in \mathbb{R}^{N \times 450}$ is constructed from training data; after centering, SVD extracts POD modes, retaining the top $r=40$ modes (covering 84.6% of total energy) to form the reduced basis matrix $\boldsymbol{\Psi}_r \in \mathbb{R}^{450 \times r}$
   - Column-pivoted QR decomposition is applied to $\boldsymbol{\Psi}_r^T$: $\boldsymbol{\Psi}_r^T \mathbf{P} = \mathbf{Q}\mathbf{R}$; the column ordering of the permutation matrix $\mathbf{P}$ yields the sensor importance ranking
   - **Design Motivation**: maximizes the linear independence of the measurement matrix $\mathbf{H}\boldsymbol{\Psi}_r$, ensuring selected sensors provide maximum information about dominant flow structures

### Loss & Training

- UNet/ViTAE: MSE loss, Adam optimizer, adaptive learning rate decay + early stopping (patience = 20), 80-20 train/validation split
- CWGAN: Wasserstein distance + L1 reconstruction loss (weight ratio 1:100), 5 discriminator updates per generator update, Adam optimizer (lr = 0.0001) + early stopping

## Key Experimental Results

### Main Results: Performance of Each Method Under MDT at Varying Sensor Densities

| # Sensors | Method | SSIM↑ | FAC2↑ | Inference Time (ms) |
|:---:|------|:---:|:---:|:---:|
| 5 | Kriging | 0.415 | — | ~1.493 |
| 5 | UNet | 0.531 | — | ~0.109 |
| 5 | CWGAN | **0.550** | — | ~0.164 |
| 20 | UNet | ~0.78 | >0.80 | ~0.109 |
| 20 | CWGAN | **~0.80** | >0.80 | ~0.164 |
| 30 | Kriging | ~0.78 | ~0.778 | ~1.493 |
| 30 | UNet | ~0.82 | ~0.808 | ~0.109 |
| 30 | CWGAN | **~0.85** | ~0.811 | ~0.164 |

Under MDT, DL vs. Kriging: SSIM +18.2–33.5%, FAC2 +3.5–24.2%, NMSE −10.2–27.8%.

### Computational Efficiency and Robustness Comparison

| Model | Parameters | GFLOPs | Size (MB) | SSIM Drop (Perturbation) | QR Optimization Gain |
|------|:---:|:---:|:---:|:---:|:---:|
| UNet | 471,586 | 0.0285 | 1.80 | 6.5–14.8% | −0.7% (SDT) / +0.4% (MDT) |
| ViTAE | 467,491 | 0.0210 | 1.78 | 6.7–16.8% | +2.6% / +4.8% |
| CWGAN | 8,770,000 | 1.301 | 33.46 | **3.3–8.2%** | +6.5% / +1.8% |
| Kriging | — | — | — | 5.4–13.9% | +4.1% / +7.9% |

### Key Findings

- **Kriging outperforms DL under SDT**: with only 5 sensors and single-direction training, Kriging achieves SSIM = 0.502, far exceeding DL's 0.194–0.237 (a 52–61% gap) → under extreme sparsity and low training diversity, spatial correlation assumptions prove more effective
- **MDT is the critical turning point for DL**: switching to MDT yields 131–146% SSIM improvement for DL at 5 sensors, while Kriging degrades as its spatial stationarity assumption is violated by multi-directional flow fields
- **20 sensors marks the performance crossover**: under SDT, DL begins to comprehensively surpass Kriging on NMSE at this density
- **CWGAN's "deterministic" behavior**: the 100:1 L1 weighting causes CWGAN to behave effectively as a deterministic mapping, with virtually no variation across multiple samples
- **0° wind direction is the most difficult to reconstruct**: the largest boundary-center discrepancy and velocity class imbalance make it the primary cause of Kriging degradation under MDT
- **QR optimization is more effective under MDT**: 90% of cases show positive improvement vs. 60% under SDT

## Highlights & Insights

- **First systematic DL wind field reconstruction benchmark using real PIV data**—eliminating CFD simulation bias and directly targeting real deployment conditions
- The SDT vs. MDT comparison clearly reveals the decisive influence of training data diversity on DL methods—a conclusion that generalizes to other flow field reconstruction tasks
- QR sensor optimization combines POD dimensionality reduction with information-theoretic principles, providing a theoretically grounded approach to data-driven sensor placement
- The practical guidance on method selection by scenario is valuable: single-direction, few sensors → Kriging; multi-direction, more sensors → UNet (balanced and stable); highest accuracy required → CWGAN (high computational cost); resource-constrained → ViTAE

## Limitations & Future Work

- Only 2D planar wind fields (15×15 grid), limited to a single height z/H = 1.05, with 3D structure absent
- Only three wind direction angles (0°/22.5°/45°); generalization beyond this range requires additional experimental data or transfer learning
- CWGAN has 18.6× more parameters than UNet (8.77M vs. 471K) but achieves only 5–9% SSIM improvement, indicating a poor efficiency ratio
- Evaluated on a single isolated rectangular building; applicability to complex multi-building configurations is unverified
- Each snapshot is reconstructed independently, without exploiting temporal dynamics for multi-step prediction

## Related Work & Insights

- **vs. traditional POD-LSE methods**: linear dimensionality reduction methods underperform in the presence of nonlinear turbulent features → DL shows clear advantages at medium-to-high sensor densities
- **vs. CFD-trained studies**: systematic biases in CFD (turbulence closure models, discretization errors) may introduce training-deployment domain gaps; PIV experimental data directly eliminates this issue
- **Insights**: the sparse-to-dense reconstruction framework is transferable to flow field monitoring in meteorology, oceanography, and indoor environments; QR sensor optimization is conceptually linked to compressed sensing theory

## Rating

⭐⭐⭐⭐ (4/5)

Overall assessment: no novel architectural contributions at the methodological level, but the experimental design is exceptionally comprehensive (4 methods × 2 strategies × 6 sensor configurations × perturbation analysis × QR optimization × temporal averaging strategies). The systematic benchmark on real PIV data offers high practical value for the built environment engineering community. Code is open-sourced with good reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[CVPR 2026\] SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)

</div>

<!-- RELATED:END -->
