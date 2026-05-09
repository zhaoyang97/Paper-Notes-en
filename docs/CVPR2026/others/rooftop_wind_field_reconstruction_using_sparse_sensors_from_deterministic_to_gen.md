---
title: >-
  [Paper Note] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods
description: >-
  [CVPR 2026][wind field reconstruction] Based on wind tunnel PIV experimental data, this paper systematically compares Kriging interpolation and three deep learning methods (UNet, ViTAE, CWGAN) for rooftop wind field reconstruction under sparse sensor conditions, and proposes QR decomposition-based sensor placement optimization to enhance robustness.
tags:
  - CVPR 2026
  - wind field reconstruction
  - sparse sensors
  - deep learning
  - PIV measurement
  - sensor optimization
  - UNet
  - Vision Transformer
  - GAN
date: 2026-05-08
content_hash: 4ff98bb26d408781
---

# Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods

**Conference**: CVPR 2026
**arXiv**: [2603.13077](https://arxiv.org/abs/2603.13077)
**Code**: [Available](https://github.com/Yng314/windreconstruction)
**Area**: Other (Fluid Mechanics)
**Keywords**: wind field reconstruction, sparse sensors, deep learning, PIV measurement, sensor optimization, UNet, Vision Transformer, GAN

## TL;DR

Based on wind tunnel PIV experimental data, this paper systematically compares Kriging interpolation and three deep learning methods (UNet, ViTAE, CWGAN) for rooftop wind field reconstruction under sparse sensor conditions, and proposes QR decomposition-based sensor placement optimization to enhance robustness.

## Background & Motivation

**Multi-functional rooftop demands**: Rooftops are increasingly used for photovoltaics, gardens, HVAC equipment, and UAV vertical take-off and landing (VTOL), requiring real-time and accurate wind field information to ensure safe operation.

**Highly complex rooftop wind environments**: Influenced by building geometry and aerodynamic effects, nonlinear flow patterns such as separation flows and conical vortices emerge, with significant spatial variation across different wind directions.

**Limitations of traditional methods**: In-situ measurements are constrained by sensor count and spatial coverage; CFD simulations are computationally expensive, lack real-time capability, and struggle to accurately capture real turbulence characteristics.

**Existing studies rely on simulation data**: Most existing deep learning approaches are trained on CFD-generated data, introducing systematic bias and failing to reflect real measurement noise and turbulence variability.

**Lack of systematic multi-architecture comparison**: Prior studies typically evaluate only a single network architecture, without comprehensive comparisons of mainstream deep learning frameworks on experimental data.

**Sensor placement optimization overlooked**: Most studies adopt predefined sensor positions (e.g., uniform grids) without data-driven optimal sensor layout design.

## Method

### Overall Architecture

A **learning-from-observation** framework is proposed: high-resolution rooftop flow field data are acquired via wind tunnel PIV experiments (15×15 grid, three wind directions: 0°/22.5°/45°), and four reconstruction methods are systematically compared under sparse sensor conditions (5–30 sensors):

- **Kriging interpolation** (traditional baseline): geostatistical interpolation based on spatial correlation
- **UNet** (deterministic): deterministic mapping via encoder–decoder with skip connections
- **CWGAN** (generative): conditional Wasserstein GAN with adversarial training to capture nonlinear turbulence features
- **ViTAE** (hybrid): Transformer attention mechanism combined with a CNN decoder to extract both global and local features

Two training strategies are evaluated: single-direction training (SDT) and mixed-direction training (MDT).

### Key Designs

**UNet**: Input is 15×15×3 (u/v velocity components + sensor mask), zero-padded to 16×16; encoder filters 32→128 with three downsampling stages to a 2×2 bottleneck; decoder upsampling with skip connections restores spatial resolution; a final 1×1 convolution outputs the two-channel velocity field. Parameters: 471K, 0.0285 GFLOPs.

**CWGAN**: The generator follows a UNet structure, processing sensor measurements and noise inputs with encoder channels 64→128→256; the discriminator uses strided convolution + LeakyReLU with channels 64→512, without sigmoid to accommodate the Wasserstein distance. Generator loss = adversarial loss + 100×L1 reconstruction loss; the discriminator is updated 5 times per generator update. Parameters: 8.77M, 1.301 GFLOPs.

**ViTAE**: 3×3 patch partitioning yields 25 patches, linearly projected to 64-dimensional vectors with learnable 2D sinusoidal positional encodings; 8 Transformer blocks (8-head attention, MLP expansion ratio 4.0); a CNN decoder progressively reconstructs the spatial velocity field. Parameters: 467K, 0.021 GFLOPs.

**QR sensor optimization**: SVD is applied to training data to extract the top 40 POD modes (~84.6% energy coverage); column-pivoted QR decomposition is performed on the transposed POD basis, with the permutation vector directly providing sensor importance ranking and selecting positions that maximize linear independence of the measurement matrix.

### Loss & Training

- **UNet / ViTAE**: MSE loss + Adam optimizer + adaptive learning rate decay + early stopping (patience=20)
- **CWGAN**: Wasserstein distance + L1 reconstruction loss (weight ratio 100:1), Adam (lr=0.0001), 5:1 discriminator/generator update ratio
- **Kriging**: Gaussian variogram model, correlation length 0.5–10 grid units, nugget=0 (exact interpolation baseline)

## Key Experimental Results

### Data & Setup

Boundary layer wind tunnel at the University of Tokyo, 1:200 scale, model height H=0.2 m, reference wind speed $U_H=0.70$ m/s. PIV measurements at the $z/H=1.05$ plane, temporal resolution 0.001 s, spatial resolution 0.035H. Eight experimental sessions across three wind directions (0°×3, 22.5°×2, 45°×3), each with 7,999 time snapshots.

### Main Results

| Condition | Comparison | SSIM Gain | FAC2 Gain | NMSE Gain |
|-----------|-----------|-----------|-----------|-----------|
| MDT vs. Kriging | DL overall | up to 32.7% | up to 24.2% | up to 27.8% |
| MDT vs. SDT | DL, 5 sensors | 131–146% | — | — |
| MDT vs. SDT | DL, 30 sensors | 4.0–7.4% | — | — |

| Model | Parameters | GFLOPs | Inference Time (relative) | Characteristics |
|-------|-----------|--------|--------------------------|-----------------|
| UNet | 471K | 0.0285 | 1× (~0.109 ms) | Best geometric accuracy (MG closest to 1.0) |
| ViTAE | 467K | 0.0210 | 2.1× | Highest computational efficiency, comparable accuracy |
| CWGAN | 8.77M | 1.301 | 1.5× | Highest SSIM, best robustness |
| Kriging | — | — | 13.7× | Best under low-sensor SDT |

### Ablation Study

- **SDT vs. MDT**: Mixed-direction training is critical for deep learning, with SSIM gains up to 173.7%; Kriging shows the opposite trend, with performance degradation of 20.9–51.7% under MDT due to violated spatial stationarity assumptions.
- **Sensor perturbation robustness**: Under ±1 grid perturbation, CWGAN is the most stable (SSIM degradation 3.3–8.2%), while ViTAE exhibits the strongest dependency on precise spatial relationships (NMSE degradation up to 81.5%).
- **QR optimization**: Under MDT, 90% of metrics show positive improvement (60% under SDT), with average robustness gains of 3.7% (MDT) vs. 3.1% (SDT), at the cost of a slight SSIM decrease.
- **Temporal averaging strategy**: Post-averaging (reconstruct then average) generally outperforms pre-averaging (average then reconstruct), though the difference is small; pre-averaging serves as a viable alternative under data-limited conditions.
- **L1 weight ablation (CWGAN)**: Reducing the L1 weight increases generative diversity but leads to significant performance degradation; strong L1 regularization is necessary for safety-critical applications.

### Key Findings

- Under 5-sensor SDT, Kriging shows a clear advantage (SSIM 0.502 vs. DL 0.194–0.237); 20 sensors marks the performance inflection point.
- The 0° wind direction is the most challenging to reconstruct: largest boundary-to-center difference (0.772), highest spatial gradient (0.143), and imbalanced low-speed samples.
- Intermediate sensor densities (15–25 sensors) correspond to peak perturbation vulnerability: sparse configurations have fewer interpolation assumptions to violate, while dense configurations benefit from redundancy.

## Highlights & Insights

- **First systematic multi-architecture comparison on real wind tunnel experimental data**: Training and evaluation are conducted directly on PIV data without reliance on CFD simulations, better reflecting actual deployment conditions.
- **Comprehensive method selection guidance**: Clear recommendations are provided for optimal methods under different scenarios (number of sensors, wind direction diversity, robustness requirements).
- **QR sensor optimization framework**: A mathematically rigorous POD+QR approach achieves data-driven sensor placement, with robustness gains up to 27.8%.
- **Rigorous experimental design**: Independent experimental sessions are used to split training/test sets, avoiding temporal leakage; four complementary evaluation metrics provide comprehensive assessment.

## Limitations & Future Work

- The study targets only a single isolated rectangular building at a single measurement height ($z/H=1.05$); geometric generalizability remains unvalidated.
- Only three wind directions (0°, 22.5°, 45°) are tested; generalization to more wind directions requires additional experimental data.
- The POD-QR sensor optimization is coupled to specific building geometry and wind conditions, requiring recomputation when the scenario changes.
- The relatively coarse 15×15 spatial resolution may limit the capture of fine-scale flow structures.
- Field validation under real atmospheric conditions is absent.

## Related Work & Insights

- **Traditional methods**: POD-LSE (linear dimensionality reduction + linear stochastic estimation), Kriging interpolation (spatial correlation assumptions).
- **Deep learning for wind field reconstruction**: GANs applied to urban wind environments (mostly CFD-trained), CNN/Transformer approaches for flow field prediction.
- **Sensor optimization**: Greedy algorithms, information-theoretic methods; the QR decomposition adopted here offers mathematical rigor and computational efficiency.

## Rating

- **Novelty**: ⭐⭐⭐ — The methods themselves are not new (UNet/ViTAE/CWGAN are all existing architectures); the contribution lies in the first systematic comparison on experimental data combined with sensor optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive: multiple methods × multiple sensor counts × multiple training strategies × perturbation robustness × sensor optimization × temporal averaging × ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, in-depth analysis, rich figures and tables; however, the paper is lengthy and some discussions could be streamlined.
- **Value**: ⭐⭐⭐⭐ — Provides a systematic method selection guide for practical deployment of rooftop wind field reconstruction; strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_a.md)
- [\[CVPR 2026\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[CVPR 2026\] SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embedding-based_approach_for_ab-initio_reconstruction_of_helica.md)

</div>

<!-- RELATED:END -->
