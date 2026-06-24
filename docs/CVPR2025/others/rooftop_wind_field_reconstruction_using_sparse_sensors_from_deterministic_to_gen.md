---
title: >-
  [Paper Note] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods
description: >-
  [CVPR 2025][wind field reconstruction] A framework is established for rooftop wind field reconstruction based on wind tunnel PIV experimental data (rather than CFD simulations). The reconstruction performance of Kriging interpolation and three deep learning models (UNet, ViTAE, and CWGAN) is systematically compared using 5-30 sparse sensors. Results show that Multi-Directional Training (MDT) enables deep learning methods to comprehensively outperform Kriging (with SSIM improv…
tags:
  - "CVPR 2025"
  - "wind field reconstruction"
  - "sparse sensors"
  - "PIV"
  - "Kriging"
  - "UNet"
  - "CWGAN"
  - "ViTAE"
  - "sensor optimization"
  - "QR decomposition"
date: 2026-05-08
content_hash: 4404e82df42ab8c3
---

# Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods

**Conference**: CVPR 2025  
**arXiv**: [2603.13077](https://arxiv.org/abs/2603.13077)  
**Code**: [GitHub](https://github.com/Yng314/windreconstruction)  
**Area**: Wind field reconstruction / Fluid mechanics and deep learning  
**Keywords**: wind field reconstruction, sparse sensors, PIV, Kriging, UNet, CWGAN, ViTAE, sensor optimization, QR decomposition  

## TL;DR

A framework is established for rooftop wind field reconstruction based on wind tunnel PIV experimental data (rather than CFD simulations). The reconstruction performance of Kriging interpolation and three deep learning models (UNet, ViTAE, and CWGAN) is systematically compared using 5-30 sparse sensors. Results show that Multi-Directional Training (MDT) enables deep learning methods to comprehensively outperform Kriging (with SSIM improvements of up to 32.7%), and optimizing sensor layouts via QR decomposition enhances robustness by up to 27.8%.

## Background & Motivation

**Background**: Rooftop spaces accommodate multi-functional demands such as photovoltaics, gardens, HVAC equipment, and UAV vertical takeoff and landing. However, rooftop wind fields exhibit highly complex spatiotemporal variations due to building geometric effects and aerodynamic interactions (such as flow separation and conical vortices). Accurate real-time wind field information is critical for UAV operations and wind control system adjustments.

**Limitations of Prior Work**: (1) **Measurement limitations**—Limited sensor count and spatial coverage make it difficult to obtain full-field information; (2) **CFD limitations**—High computational cost, sensitivity to boundary conditions, lack of real-time capability, and inaccurate restoration of turbulence characteristics; (3) **Data dependency**—Most existing deep learning methods are trained on CFD simulation data rather than experimental observations, which introduces systematic biases; (4) **Single architecture**—Lack of a systematic comparison across different deep learning architectures; (5) **Sensor layout**—Most studies rely on predefined grid layouts without validating data-driven optimization.

**Key Challenge**: The need to reconstruct complex non-linear wind fields in real-time over a 15×15 grid domain using extremely sparse sensors (5-30), while ensuring the method remains robust to perturbations in sensor positions.

**Goal**: Establish a wind field reconstruction framework based on real experimental data, systematically evaluate the performance of different methods under various sensor densities, training strategies, and perturbation conditions, and provide practical guidance on method selection and sensor layout.

**Key Insight**: Utilizing wind tunnel PIV experimental data from the University of Tokyo, comparing deterministic (Kriging/UNet/ViTAE) to generative (CWGAN) methods, and introducing POD-QR sensor placement optimization.

**Core Idea**: Training on real PIV experimental data (instead of CFD simulations), where the multi-directional training strategy enables deep learning methods to learn cross-directional flow patterns, thereby comprehensively outperforming Kriging's spatial stationarity assumption under sparse sensor conditions.

## Method

### Overall Architecture

Wind tunnel PIV experiment (three wind directions of 0°/22.5°/45° × multiple acquisitions) → Data partitioning (SDT single-direction / MDT multi-directional) → Sensor sampling (uniform / QR-optimized / perturbed) → Four reconstruction methods (Kriging / UNet / ViTAE / CWGAN) → Four-dimensional evaluation (SSIM / MG / NMSE / FAC2)

### Key Designs

1. **Based on PIV experimental data learning-from-observation framework**
    - Wind tunnel experiment: Rectangular building model (height-to-width-to-length ratio of 1:1:2), geometric scale of 1:200, model height $H=0.2\text{ m}$.
    - PIV measurement: Instantaneous horizontal velocity field at the $z/H=1.05$ plane, temporal resolution of $0.001\text{ s}$, spatial resolution of $0.035H$.
    - Three wind directions: 0° (leading-edge separation), 22.5° (asymmetric conical vortices), 45° (diagonally symmetric double conical vortices).
    - Each wind direction has 2-3 independent acquisitions, each with 7999 temporal snapshots.
    - Design Motivation: Compared to CFD dependency, experimental data naturally contains real-world noise and turbulence variability, rendering the trained models more robust for real-world deployment.

2. **Systematic comparison of four reconstruction methods**
    - **Kriging interpolation**: Gaussian variogram model, correlation length optimized between 0.5 and 10.0 grid units, nugget=0 (exact interpolation), serving as the traditional baseline.
    - **UNet**: Encoder-decoder + skip connections, input of 15×15×3 (u/v velocities + mask), 3 downsampling stages (32 → 64 → 128 channels).
    - **CWGAN**: Conditional Wasserstein GAN, generator uses a UNet structure, discriminator uses strided convolutions + LeakyReLU, generator loss = adversarial loss + 100 × $L_1$ reconstruction loss, 5 discriminator updates per 1 generator update.
    - **ViTAE**: Vision Transformer Autoencoder, 3×3 patch embedding → 25 patches → 64-dimensional projection → 8 Transformer blocks (8-head attention) → CNN decoder.
    - Design Motivation: The three architectures represent three distinct modeling philosophies—deterministic mapping (UNet), generative adversarial learning (CWGAN), and attention-based global features (ViTAE).

3. **POD-QR sensor position optimization**
    - Perform SVD on the wind field data matrix to obtain POD spatial modes, retaining the first $r=40$ modes (~84.6% energy).
    - Perform column-pivoted QR decomposition on the transposed POD basis: $\Psi_r^T \mathbf{P} = \mathbf{QR}$, where the permutation vector directly provides the sensor importance ranking.
    - Ensure that the selected sensor locations maximize the linear independence of the measurement matrix $\mathbf{H}\Psi_r$.
    - Design Motivation: Data-driven identification of optimal sensor locations, outperforming uniform grid layouts.

### Loss & Training

- **SDT (Single-Direction Training)**: Trained solely on 0° data, evaluating on 22.5° and 45° (testing cross-directional generalization).
- **MDT (Multi-Directional Training)**: Trained on one acquisition dataset from each wind direction, evaluating on the remaining independent acquisitions.
- Data partitioning is based on independent experimental acquisitions (rather than random snapshot sampling) to prevent temporal leakage.

## Key Experimental Results

### Key Comparison: SDT vs MDT (Deep Learning vs Kriging)

| Strategy | Method | 5-Sensor SSIM | 30-Sensor SSIM | 5-Sensor FAC2 | 30-Sensor FAC2 |
|------|------|-----------|-----------|-----------|-----------|
| SDT | Kriging | 0.502 | 0.804 | 0.749 | 0.858 |
| SDT | UNet | 0.237 | 0.756 | 0.714 | 0.888 |
| SDT | CWGAN | 0.194 | 0.773 | 0.660 | 0.882 |
| MDT | Kriging | 0.415 | 0.661 | 0.647 | 0.778 |
| MDT | UNet | 0.539 | 0.784 | 0.735 | 0.806 |
| MDT | CWGAN | **0.550** | **0.816** | 0.735 | 0.803 |
| MDT | ViTAE | 0.531 | 0.772 | 0.738 | 0.811 |

### Computational Complexity

| Model | Parameters | GFLOPs | Inference Time (relative to UNet) |
|------|--------|--------|-------------------|
| UNet | 471K | 0.0285 | 1× (~0.109ms) |
| ViTAE | 467K | 0.0210 | 2.1× (~0.229ms) |
| CWGAN | 8.77M | 1.301 | 1.5× (~0.164ms) |
| Kriging | - | - | 13.7× (~1.493ms) |

### Robustness Gain from QR Optimization

| Strategy | Method | SSIM Gain | NMSE Gain | Overall Gain |
|------|------|----------|----------|------------|
| SDT | CWGAN | +0.2% | +27.8% | +6.5% |
| SDT | Kriging | -0.0% | +18.1% | +4.1% |
| MDT | Kriging | +12.5% | +9.9% | +7.9% |
| MDT | ViTAE | +1.1% | +9.9% | +4.8% |

### Key Findings

- Under SDT, Kriging outperforms deep learning (with a 52.7-61.4% higher SSIM for 5 sensors), because the spatial stationarity assumption is relatively valid under a single wind direction.
- Under MDT, deep learning reverses the advantage—achieving an 18.2-33.5% gain in SSIM and a 3.5-24.2% gain in FAC2.
- MDT is critical for deep learning: yielding up to a 146.0% SSIM improvement (for 5 sensors), with CWGAN achieving the highest absolute SSIM (0.816).
- Reconstructing the 0° wind direction is the most challenging: it exhibits the largest boundary-center discrepancy (0.772 vs. 0.210 for 22.5°), the sharpest spatial gradients, and sample imbalance in low-velocity regions.
- Robustness to sensor perturbations: Medium density (15-25 sensors) is the most fragile, whereas extremely sparse and high-density layouts remain relatively stable.
- Deep learning inference speed is an order of magnitude faster than Kriging (0.109 ms vs. 1.493 ms).

## Highlights & Insights

- This is the first study to systematically train and evaluate wind field reconstruction methods using real PIV experimental data rather than CFD simulations, yielding results of higher practical and operational value.
- Revealed the critical finding that "the training strategy is more important than the model architecture"—the introduction of MDT allows all deep learning methods to outperform Kriging.
- The analysis thoroughly explains Kriging's advantage under SDT (where the spatial stationarity assumption holds for a single wind direction) and its disadvantage under MDT (where the strong non-stationarity of the 0° wind direction violates Kriging's assumptions).
- The asymmetric sensor layouts from QR-decomposition optimization reveal the spatial heterogeneity of the experimental data.
- Provides a clear decision tree for method selection: MDT+UNet is recommended as the default, CWGAN for maximizing accuracy, ViTAE for edge/real-time scenarios, and Kriging for SDT with very few sensors.

## Limitations & Future Work

- The study only considers a single rectangular building and a single measurement height ($z/H=1.05$); generalization to other building geometries and measurement heights requires retraining.
- Only three wind directions (0°, 22.5°, and 45°) are tested; generalization capabilities to other wind directions remain unvalidated.
- POD-QR sensor optimization is data-driven, requiring recalculation if the building shape or wind conditions change.
- Deep learning performs poorly under extremely sparse conditions (5-sensor SDT), indicating that data amount and training strategy represent performance bottlenecks.
- Physics-informed loss functions (e.g., Navier-Stokes constraints) have not been explored to enhance physical consistency.
- While submitted to CVPR, the work is fundamentally a computational fluid dynamics and sensor engineering problem, showing limited alignment with the core focus of the computer vision community.

## Related Work & Insights

- **vs. CFD-based wind field reconstruction**: This work emphasizes that CFD data carries biases from turbulence closure models and discretization, whereas PIV experimental data naturally includes real turbulence and measurement noise, making it better suited for actual deployment.
- **vs. POD-LSE**: Traditional dimensionality reduction methods require massive training data and struggle to handle non-linearity—this work replaces linear modal decomposition with deep learning.
- **vs. Single-architecture studies** (GAN-only or CNN-only): This study conducts a horizontal comparison of three representative architectures, revealing dependencies between architectural choices and application constraints (accuracy, efficiency, and robustness).
- Practical reference for urban environmental fluid reconstruction: Sensor configuration, optimization methods, and training strategies must be considered jointly.

## Rating

- Novelty: ⭐⭐⭐ The methods are direct applications of existing architectures, with no new model or loss designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering four methods × two training strategies × six sensor densities × perturbation/optimization experiments.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and deeply analyzed, but the length is somewhat excessive and some analyses are redundant.
- Value: ⭐⭐⭐⭐ Possesses practical guidance value for engineering applications, though the methodological contribution is limited; domain relevance to CVPR is also a concern.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[ICML 2025\] GPU-friendly and Linearly Convergent First-order Methods for Certifying Optimal $k$-sparse GLMs](../../ICML2025/others/gpu-friendly_and_linearly_convergent_first-order_methods_for_certifying_optimal_.md)
- [\[CVPR 2025\] Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)
- [\[ICML 2025\] Modern Methods in Associative Memory](../../ICML2025/others/modern_methods_in_associative_memory.md)
- [\[CVPR 2025\] Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays](image_reconstruction_from_readout-multiplexed_single-photon_detector_arrays.md)

</div>

<!-- RELATED:END -->
