---
title: >-
  [Paper Note] Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis
description: >-
  [CVPR 2026][Physics & Scientific Computing][Atmospheric turbulence synthesis] This paper proposes an exposure-time-dependent modulation transfer function (ET-MTF) that treats exposure time as a continuous variable…
tags:
  - "CVPR 2026"
  - "Physics & Scientific Computing"
  - "Atmospheric turbulence synthesis"
  - "exposure-time modeling"
  - "modulation transfer function (MTF)"
  - "point spread function (PSF)"
  - "turbulence image restoration"
date: 2026-05-08
content_hash: 7af29c8f5eb004dd
---

# Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.01398](https://arxiv.org/abs/2603.01398)  
**Code**: [Available](https://github.com/Jun-Wei-Zeng/ET-Turb)  
**Area**: Scientific Computing
**Keywords**: Atmospheric turbulence synthesis, exposure-time modeling, modulation transfer function (MTF), point spread function (PSF), turbulence image restoration

## TL;DR

This paper proposes an exposure-time-dependent modulation transfer function (ET-MTF) that treats exposure time as a continuous variable, and constructs a large-scale synthetic turbulence dataset ET-Turb (5,083 videos, 2 million frames), significantly improving the generalization of turbulence restoration models on real-world data.

## Background & Motivation

Atmospheric turbulence introduces geometric distortions (tilt) and exposure-time-dependent blur into long-range imaging through random fluctuations in the refractive index, severely affecting applications such as remote sensing, video surveillance, and astronomical observation. The performance of learning-based methods is highly dependent on the realism of training data, and collecting large-scale paired real turbulence data is extremely costly, making synthetic datasets essential.

The core limitation of existing synthesis methods lies in overly coarse treatment of exposure time:

- **Fixed-exposure methods**: Many methods apply a single exposure setting to all samples, resulting in uniform blur statistics that fail to reflect the temporal variability of real imaging.
- **Binary-exposure methods**: Some methods distinguish only between "short exposure" and "long exposure" modes, using $\text{MTF}_{\text{SE}}$ and $\text{MTF}_{\text{LE}}$ respectively, ignoring the smooth transition produced by intermediate exposure times.
- **Physical simulation methods**: Physical setups such as hot-air sources are limited by short optical paths, and multi-layer phase screen methods incur prohibitive computational costs.

These limitations result in a significant domain gap between synthetic and real turbulence data, restricting the generalization of trained models.

## Method

### Overall Architecture

The paper models turbulence degradation as $I(\mathbf{x}) = \mathcal{B}_\tau(\mathcal{T}(J(\mathbf{x})))$, where $\mathcal{T}$ is a geometric distortion operator (independent of exposure time) and $\mathcal{B}_\tau$ is an exposure-time-dependent blur operator. The synthesis pipeline consists of three steps:

1. **ET-MTF derivation**: Starting from Azoulay's theory, derive a continuous exposure-time-dependent modulation transfer function.
2. **PSF derivation**: Obtain a pure blur point spread function with tilt effects removed from the ET-MTF.
3. **Blur width field**: Extend the scalar blur width to a spatially varying field under optical turbulence statistical constraints.

### Key Designs

#### 1. Exposure-Time-Dependent MTF (ET-MTF)

**Function**: Establish a continuous and smooth MTF model spanning from short to long exposure.

**Mechanism**: Based on Azoulay's finite-exposure MTF theory, the effective coherence length $\rho_p(\tau)$ is introduced. Under short exposure, turbulence is frozen within the physical aperture $D$; under long exposure, the sensor integrates multiple turbulence realizations, equivalent to a larger effective aperture $D + v_w \tau$:

$$\text{MTF}_{\text{ET}}(\boldsymbol{\xi}, \tau) = e^{-\left(\frac{\lambda \|\boldsymbol{\xi}\|}{\rho_p(\tau)}\right)^{5/3}}$$

$$\rho_p(\tau) = 1 + 0.35 \left(\frac{r_0}{D + v_w \tau}\right)^{1/3}$$

where $r_0$ is the Fried parameter and $v_w$ is the wind speed. As $\tau$ increases, $\rho_p(\tau)$ decreases smoothly, accelerating high-frequency attenuation in the MTF and naturally producing a continuous transition from weak to strong blur.

**Design Motivation**: The existing $\text{MTF}_{\text{SE}}$ and $\text{MTF}_{\text{LE}}$ define only two extreme states, with no physical model for the intermediate transition. Direct empirical interpolation lacks physical interpretability.

#### 2. Blur Width Reparameterization

**Function**: Extend the ET-MTF from dependence solely on exposure time to joint dependence on local blur width.

**Mechanism**: The blur width $\omega$ is defined via the full width at half maximum (FWHM) of the PSF as $\omega \approx \frac{0.49 \lambda f}{r_0}$, from which $r_0$ is back-substituted into the effective coherence length:

$$\rho_p(\omega, \tau) = 1 + 0.28 \left(\frac{\lambda f}{\omega(D + v_w \tau)}\right)^{1/3}$$

The final ET-MTF is jointly determined by spatial location (through $\omega$) and time (through $\tau$).

**Design Motivation**: The original $\rho_p(\tau)$ is spatially uniform in the image plane, whereas real turbulence exhibits spatially varying blur patterns due to local refractive index fluctuations.

#### 3. Spatially Varying Blur Width Field

**Function**: Assign distinct blur widths to each spatial location, enabling spatially non-uniform blur modeling.

**Mechanism**: The blur width is modeled as a spatially correlated random field $\mathcal{W}(\mathbf{x}, \tau)$, with its mean and standard deviation constrained by optical turbulence theory:

$$\mathcal{W}(\mathbf{x}, \tau) = \max(\epsilon, \bar{\omega}(\tau) + \sigma_\omega(\tau) \mathcal{R}(\mathbf{x}))$$

where $\bar{\omega}(\tau)$ and $\sigma_\omega(\tau)$ are both functions of $\tau$ (given by detailed physical expressions), $\mathcal{R}(\mathbf{x})$ is a low-pass-filtered zero-mean unit-variance Gaussian random field, and $\epsilon > 0$ ensures non-negativity.

The final spatially varying blur operation is:

$$\mathcal{B}_\tau(I_T(\mathbf{x})) = \text{PSF}_{\text{ET}}(\mathbf{x}, \mathcal{W}(\mathbf{x}, \tau), \tau) * I_T(\mathbf{x})$$

#### 4. Inter-Frame Correlation Modeling

**Function**: Extend single-frame synthesis to video sequences, modeling the temporal evolution of turbulence degradation.

**Mechanism**: Under the Taylor frozen-flow hypothesis, turbulence is treated as a quasi-static refractive index field advected by the mean wind:

$$\mathcal{H}(J_t(\mathbf{x})) = \mathcal{H}\left(J_0\left(\mathbf{x} - \frac{f \mathbf{v}_w t}{L}\right)\right)$$

Temporally correlated video frames are generated by translating the extended degradation field along the wind direction.

### Loss & Training

The core contribution of this paper lies in dataset construction rather than network training. The ET-Turb dataset is designed with 12 turbulence configurations to systematically cover diverse optical and atmospheric conditions:

- **Parameter space**: Propagation distance 30–1000 m, focal length 0.1–1 m, f-number 2.8–24, $C_n^2$ ranging from $0.5 \times 10^{-14}$ to $300 \times 10^{-14}$ m$^{-2/3}$, wind speed 1–10 m/s, exposure time 0.5–40 ms.
- **Data scale**: 5,083 videos, 2,005,835 frames, split into 3,988 training / 1,095 testing.
- **Real dataset**: ET-Turb-Real contains 74 videos captured by 3 different imaging devices.

## Key Experimental Results

### Main Results

Evaluation of models trained on different synthetic datasets on real turbulence data (no-reference metrics, lower is better):

| Training Dataset | TSR-WGAN NIQE↓ | TSR-WGAN BRISQUE↓ | TMT NIQE↓ | TMT BRISQUE↓ | DATUM NIQE↓ | DATUM BRISQUE↓ | MambaTM NIQE↓ | MambaTM BRISQUE↓ |
|---|---|---|---|---|---|---|---|---|
| TMT-dynamic | 4.231 | 52.502 | 4.361 | 58.581 | 4.219 | 54.921 | 4.217 | 55.062 |
| ATSyn-dynamic | 4.224 | 54.462 | 4.483 | 59.707 | 4.308 | 59.126 | 4.247 | 56.876 |
| **ET-Turb** | **4.190** | **50.981** | **4.221** | **56.691** | **4.204** | **54.070** | **4.212** | **55.050** |

ET-Turb achieves best performance on 7 out of 8 evaluation items (4 models × 2 metrics).

### Ablation Study

Comparison of different exposure modeling strategies (MambaTM model):

| Exposure Strategy | NIQE↓ | BRISQUE↓ |
|---|---|---|
| Fixed exposure τ=1ms | 4.355 | 55.457 |
| Binary MTF_SE/LE | 4.297 | 55.123 |
| **Continuous ET-MTF** | **4.212** | **55.050** |

### Key Findings

1. Models trained with **fixed exposure** struggle to restore strong blur, as training data lacks exposure variation.
2. **Binary MTF models** show improvement but still exhibit residual blur, indicating insufficient coverage of intermediate exposure ranges.
3. **Continuous ET-MTF** produces the most natural and visually consistent restoration results, demonstrating the critical role of continuous modeling.
4. Models trained on ET-Turb, when transferred zero-shot to real data, avoid common artifacts seen in models trained on other datasets, such as distorted architectural text and deformed distant utility poles.

## Highlights & Insights

1. **Elegant physical modeling**: The intuitive concept of "effective aperture = physical aperture + wind speed × exposure time" naturally bridges the short/long exposure MTF with clear physical interpretation.
2. **Reparameterization technique**: Replacing the Fried parameter $r_0$ with blur width $\omega$ cleverly introduces spatial variability.
3. **Dataset design philosophy**: Systematic sampling over 12 configurations × 7 physical parameters provides better coverage of real-world scenario diversity than random sampling.
4. **Sound evaluation design**: Using no-reference metrics on real data avoids the circular reasoning of evaluating synthetic data on synthetic benchmarks.

## Limitations & Future Work

1. The validity of the **Taylor frozen-flow assumption** is limited to short exposure timescales and may break down under extreme conditions.
2. Only isotropic turbulence models are considered; real atmospheric turbulence, especially near the ground, may be anisotropic.
3. The synthetic data models only blur and geometric distortion, without accounting for other atmospheric effects such as scattering and dispersion.
4. Exposure time is restricted to 0.5–40 ms; very long exposure scenarios (e.g., astronomical observation) may require different modeling.
5. Future work could incorporate learnable exposure-time scheduling strategies for end-to-end degradation-aware training.

## Rating

⭐⭐⭐⭐ 4/5

This paper makes a solid contribution to physical modeling within the relatively narrow domain of turbulence synthesis. The derivation of ET-MTF has clear physical grounding, the dataset design is thorough, and the experimental evaluation is comprehensive (4 SOTA models × 3 datasets in cross-validation). Points are deducted because this is primarily a dataset/simulation tool paper lacking architectural innovation; furthermore, the metric gains in the ablation study are modest (NIQE from 4.297→4.212), though visual differences are more pronounced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[CVPR 2026\] EHETM: High-Quality and Efficient Turbulence Mitigation with Events](high-quality_and_efficient_turbulence_mitigation_with_events.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](../../NeurIPS2025/physics/gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](../../NeurIPS2025/physics/eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)

</div>

<!-- RELATED:END -->
