---
title: >-
  [Paper Note] Dual-Band Thermal Videography: Separating Time-Varying Reflection and Emission Near Ambient Conditions
description: >-
  [CVPR 2026][thermal imaging] This paper proposes a dual-band long-wave infrared (LWIR) video analysis framework that jointly leverages spectral cues (constant emissivity ratio across dual bands) and temporal cues (smooth object radiance variation vs. abrupt background radiance changes) to achieve, for the first time, pixel-wise separation of reflected and emitted components in dynamic scenes near ambient temperature, along with recovery of per-pixel emissivity and temperature fields.
tags:
  - CVPR 2026
  - thermal imaging
  - dual-band
  - reflection-emission separation
  - thermal radiation
  - emissivity estimation
date: 2026-05-08
content_hash: a5f137ecbb80765e
---

# Dual-Band Thermal Videography: Separating Time-Varying Reflection and Emission Near Ambient Conditions

**Conference**: CVPR 2026  
**arXiv**: [2509.11334](https://arxiv.org/abs/2509.11334)  
**Code**: [dual-band-thermal.github.io](https://dual-band-thermal.github.io)  
**Area**: Others (Computational Imaging)  
**Keywords**: thermal imaging, dual-band, reflection-emission separation, thermal radiation, emissivity estimation

## TL;DR

This paper proposes a dual-band long-wave infrared (LWIR) video analysis framework that jointly leverages spectral cues (constant emissivity ratio across dual bands) and temporal cues (smooth object radiance variation vs. abrupt background radiance changes) to achieve, for the first time, pixel-wise separation of reflected and emitted components in dynamic scenes near ambient temperature, along with recovery of per-pixel emissivity and temperature fields.

## Background & Motivation

Thermal cameras capture LWIR (8–14 µm) radiation comprising two components:

**Emitted component**: thermal radiation determined by the object's own temperature and emissivity

**Reflected component**: reflection of background radiation from the surrounding environment

Separating these two components is a long-standing challenge in thermal imaging. The core difficulties are:

- **Under-constrained problem**: even with multi-band measurements, the problem remains indeterminate without emissivity priors
- **Near-ambient conditions**: when object temperature is close to ambient, emission and reflection signals are of comparable magnitude, making separation more difficult
- **Overly strong prior assumptions**: industrial settings assume negligible background (objects far above ambient temperature); controlled environments assume spatiotemporally uniform background; the graybody assumption (constant emissivity across LWIR) is often violated

Two key insights in this paper break these limitations:

**Spectral cue**: Object emissivity may vary with wavelength within LWIR sub-bands (violating the graybody assumption), but the emissivity ratio between two bands $k_1 = \epsilon_2/\epsilon_1$ remains a fixed constant.

**Temporal cue**: Object temperature evolves smoothly under thermal conduction, whereas background reflections can change abruptly due to human or object motion.

## Method

### Overall Architecture

Given dual-band thermal video sequences $\{I_m^1, \ldots, I_m^N\}$ ($m \in \{1,2\}$), the method estimates three quantities:
- Per-band emissivity $\epsilon_m$
- Time-varying object temperature $T_o(t)$
- Time-varying effective background temperature $T_b(t)$

Two operating modes are considered: **calibrated** and **uncalibrated**.

### Key Designs

1. **Thermal imaging model**:

    - Per-pixel radiance is governed by Kirchhoff's law: $\epsilon(\lambda) + \tau(\lambda) + r(\lambda) = 1$
    - After gain/offset correction, pixel intensity is expressed as: $I_m(t) = \epsilon_m U_m(T_o(t)) + (1-\epsilon_m) U_m(T_b(t))$
    - where $U_m(T)$ follows the Sakuma–Hattori model, which can be linearly approximated near ambient temperature as $U_m(T) = a_m T + b_m$

2. **Calibrated mode**:

    - Thermocouple measurements of surface temperature and a blackbody reference background with known temperature are used
    - Emissivity is computed analytically from known quantities: $\epsilon_m = \frac{I_m - U_m(T_b)}{U_m(T_o) - U_m(T_b)}$
    - In new scenes, $T_o$ and $T_b$ are recovered in closed form via a dual-band linear system

3. **Uncalibrated mode (core contribution)**:

    - **Problem analysis**: In the dual-band system $\mathbf{I} = \mathbf{E}\mathbf{T}$, the matrix $\mathbf{E}$ is 2×2 but nearly degenerate (Sakuma–Hattori functions are approximately linear near ambient temperature), requiring additional constraints
    - **Static background constraint**: At pixels with constant background radiance, the ratio of dual-band temporal derivatives yields the constant $k_1 = \epsilon_2/\epsilon_1$
    - **Dynamic background constraint**: The signal is decomposed into a smooth component $\tilde{I}_m(t)$ (emission-dominated) and a residual (reflection). The ratio of dual-band residuals yields the constant $k_2 = (1-\epsilon_2)/(1-\epsilon_1)$
    - **Joint recovery**: Emissivity is analytically recovered from $k_1$ and $k_2$: $\epsilon_1 = \frac{k_2-1}{k_2-k_1}$, $\epsilon_2 = k_1 \cdot \frac{k_2-1}{k_2-k_1}$

4. **Optimization framework**: Four variable groups are jointly optimized: $\tilde{I}_1(t)$ (smooth signal), $\tilde{I}_2(0)$ (offset), $\epsilon_m$ (emissivity), $I_m^\varepsilon(t)$ (noise model)

    - **Smooth signal estimation**: Using the $k_1$ constraint, $\tilde{I}_2(t)$ is constructed recursively as $\tilde{I}_2(t) = \tilde{I}_2(t-1) + k_1 \frac{a_2}{a_1}(\tilde{I}_1(t) - \tilde{I}_1(t-1))$
    - **Reconstruction constraint**: Using $k_2$, the reconstructed signal is $\hat{I}_2(t) = \tilde{I}_1(t) + k_2 \frac{a_2}{a_1}(\ddot{I}_1(t) - \tilde{I}_1(t))$

### Loss & Training

This is a non-learning method based on classical optimization (no neural networks):

$$\arg\min \; \gamma_1 \mathcal{L}_{smooth} + \gamma_2 \mathcal{L}_{Huber} + \gamma_3 \mathcal{L}_{MSE} + \gamma_4 \mathcal{L}_{noise}^{L2} + \gamma_5 \mathcal{L}_{noise}^{M}$$

- $\mathcal{L}_{smooth}$: second-order temporal smoothness prior $\|\tilde{I}_m(t-1) - 2\tilde{I}_m(t) + \tilde{I}_m(t+1)\|^2$
- $\mathcal{L}_{Huber}$: normalized Huber loss (robust to abrupt background changes)
- $\mathcal{L}_{MSE}$: MSE between reconstruction and observation
- $\mathcal{L}_{noise}$: noise model regularization (L2 + zero-mean constraint)

**Hardware setup**:
- Two FLIR Boson thermal cameras (≤40 mK / ≤20 mK NETD), 640×512 resolution
- Spectral filters: 8.5/9.5/10.6/12.1 µm, mounted on FW103H/M motorized filter wheels
- Ground truth: TC-08 data logger + Type-T thermocouples

## Key Experimental Results

### Main Results

**Simulation experiments (temperature recovery error)**:

| Method | Description | Temperature Error |
|--------|-------------|------------------|
| BCP | Blackbody Channel Prior | Large (fails for cold objects) |
| Two-wavelength pyrometry | Ignores background radiation | Degrades under high noise |
| Naive LS | Least squares (best of 5 random initializations) | Unstable |
| **Ours** | Dual-band + temporal constraints | **Significantly outperforms all baselines under moderate noise** |

**Real-world temperature estimation**:

| Scene | Ours (uncalib) | Ours (calib) | BCP | Naive | Notes |
|-------|---------------|-------------|-----|-------|-------|
| Wineglass | **1.72%** | 5.04% | 14.6% | 31.68% | Peak 63.6°C |
| Coffee Pot | 5.34% | **0.36%** | 6.62% | 45.5% | Peak 63.1°C |

**Emissivity calibration comparison**:

| Material | Reference | Mirror method | Ours |
|----------|-----------|--------------|------|
| Chrome Ball | 0.10 | 0.43 | **0.12** |
| Al. Cup | 0.05 | 0.16 | **0.10** |
| Blue Paint | 0.87 | 0.88 | **0.88** |
| Glass Jar | 0.95 | 0.93 | **0.93** |
| Wineglass | 0.95 | 0.97 | **0.96** |

The proposed method substantially outperforms the FLIR-recommended reflector method for low-emissivity materials.

### Ablation Study

| Loss term | Temperature error increase upon removal | Notes |
|-----------|----------------------------------------|-------|
| Reconstruction term $\mathcal{L}_{MSE}$ | 90.10% | Most critical |
| Smoothness term $\mathcal{L}_{smooth}$ | 56.73% | Temporal prior is important |
| Huber loss | 11.31% | Robustness to abrupt changes |
| Noise term | 11.64% | More useful under high noise |

### Key Findings

1. **The uncalibrated method achieves only 1.72% temperature error on the wineglass scene**, approaching practical utility
2. **Low-emissivity materials** (e.g., chrome ball, aluminum cup) show the largest advantage — existing methods fail substantially
3. **Spectral filter selection**: the 9.5 µm filter yields the highest condition number and is best suited for pairing with the full band
4. **Visually invisible thermal signals successfully separated**: e.g., fingerprint heat dissipation on glass vs. finger reflections, slow coffee pot cooling vs. moving human body reflections

## Highlights & Insights

1. **Elegant physics-driven modeling**: rather than relying on neural networks, the method derives constraints from thermal radiation physics, leveraging dual-band and temporal priors to regularize the under-constrained problem
2. **Dual calibrated/uncalibrated modes**: the calibrated mode achieves higher accuracy at the cost of setup overhead; the uncalibrated mode is fully automatic with slightly lower accuracy, covering diverse use cases
3. **First separation in dynamic near-ambient scenes**: no prior method could operate under conditions where reflected and emitted signals are of comparable magnitude
4. **Reveals rich information in thermal imagery**: scene information is uncovered in dimensions entirely imperceptible to the human eye and visible-light cameras
5. **Explicit noise modeling**: spectral filtering reduces SNR; per-pixel per-frame noise modeling improves robustness

## Limitations & Future Work

1. **Assumption of uncorrelated background and object temperature variations**: the assumption breaks down when the entire room heats uniformly
2. **Sensitivity limitations of low-cost microbolometers with filters**: small temperature differences on low-emissivity objects are difficult to detect
3. **Requires two cameras**: a beamsplitter was not used (due to SNR loss), but dual cameras introduce parallax
4. **LWIR only**: extension to MWIR or additional bands may further improve accuracy
5. **Non-real-time**: computational efficiency of the optimization framework is not discussed

## Related Work & Insights

- **BCP (Blackbody Channel Prior)**: a recent reflection removal method from the CV community, but assumes the locally brightest pixel approximates a blackbody — fails for cold objects
- **Two-wavelength pyrometry**: a classical industrial method assuming graybody objects and neglecting background radiation
- **Shape from Heat Conduction (ECCV 2024)**: recovers shape from thermal conduction; this paper complements it by handling the reflected component
- Inspiration: reflection/emission separation in thermal imaging is analogous to diffuse/specular separation in visible-light imaging, but additionally involves thermal conduction — a cross-domain problem at the intersection of light transport and heat transfer

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of dual-band and temporal constraints is a fundamentally new approach that elegantly resolves a long-standing under-constrained problem
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulation + real experiments + calibrated/uncalibrated comparisons + ablation; real-world scenes are somewhat limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Physical derivations are rigorous and clear, flowing seamlessly from imaging model to constraint derivation to optimization
- Value: ⭐⭐⭐⭐ — Opens new analytical dimensions in thermal imaging with implications for computational thermal imaging, non-contact thermometry, and thermal NLOS sensing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Thermal Polarimetric Multi-view Stereo](../../ICCV2025/others/thermal_polarimetric_multi-view_stereo.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](neural_collapse_in_test-time_adaptation.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)
- [\[AAAI 2026\] Depth-Synergized Mamba Meets Memory Experts for All-Day Image Reflection Separation](../../AAAI2026/others/depth-synergized_mamba_meets_memory_experts_for_all-day_image_reflection_separat.md)
- [\[AAAI 2026\] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](../../AAAI2026/others/lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)

</div>

<!-- RELATED:END -->
