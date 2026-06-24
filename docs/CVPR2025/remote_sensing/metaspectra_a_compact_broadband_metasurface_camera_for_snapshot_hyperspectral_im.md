---
title: >-
  [Paper Note] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging
description: >-
  [CVPR2025][Remote Sensing][Metasurface] Proposed MetaSpectra+, a compact and multi-functional camera based on hybrid metasurface-refractive optics. By utilizing double-layer metasurfaces to independently control dispersion, exposure, and polarization for each channel, it achieves snapshot hyperspectral+HDR or hyperspectral+polarization joint imaging within an ~250nm visible bandwidth, achieving SOTA reconstruction accuracy on benchmark datasets.
tags:
  - "CVPR2025"
  - "Remote Sensing"
  - "Metasurface"
  - "Hyperspectral imaging"
  - "Snapshot imaging"
  - "HDR"
  - "Polarization imaging"
  - "Computational optics"
date: 2026-05-08
content_hash: 24a86fe2ede30fbc
---

# MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging

**Conference**: CVPR2025  
**arXiv**: [2603.09116](https://arxiv.org/abs/2603.09116)  
**Code**: [meta-imaging.qiguo.org](https://meta-imaging.qiguo.org)  
**Area**: Remote Sensing  
**Keywords**: Metasurface, Hyperspectral imaging, Snapshot imaging, HDR, Polarization imaging, Computational optics

## TL;DR

Proposed MetaSpectra+, a compact and multi-functional camera based on hybrid metasurface-refractive optics. By utilizing double-layer metasurfaces to independently control dispersion, exposure, and polarization for each channel, it achieves snapshot hyperspectral+HDR or hyperspectral+polarization joint imaging within an ~250nm visible bandwidth, achieving SOTA reconstruction accuracy on benchmark datasets.

## Background & Motivation

### Background

**Background**: Multi-functional metasurface imaging can simultaneously acquire various modalities (spectrum, polarization, HDR, etc.) in a compact monocular system, but it is limited by **severe chromatic aberration, with the operating bandwidth restricted to only 10-100nm**.

### Limitations of Prior Work

**Limitations of Prior Work**: Snapshot hyperspectral imaging requires recovering a 3D hyperspectral data cube from a single exposure, where the key challenges lie in optical coding design and computational reconstruction.

### Key Challenge

**Key Challenge**: Existing schemes:

### Mechanism

**Mechanism**: Sampling-based (coded aperture, filter arrays, etc.): Requires relay optical elements, leading to a large system volume.

### Additional Note

**Additional Note**: Dispersion/diffractive coding types (DOE/grating): Relatively compact but high manufacturing cost.

### Additional Note

**Additional Note**: Multi-functional metasurfaces: Compact but narrow band.

### Additional Note

**Additional Note**: Core challenge: Broaden the usable bandwidth of multi-functional metasurface imaging to cover the entire visible spectrum.

## Method

### Optical Design: Double-layer Metasurface + Refractive Optics

**First layer: Beam-splitting metasurface M₀**
- Splits the incident collimated beam into $V=4$ optical channels ($2\times2$ grid), each deflected by ~33°.
- Employs a randomized interleaving strategy (vs. regular mosaic) to synthesize multi-channel phase distribution, suppressing high-order diffraction artifacts.
- To ensure full visible-light coverage, the 4 channels use different design wavelengths: $\lambda_c = \{450, 550, 600, 750\}\text{nm}$.

**Second layer: Dispersion control metasurfaces M₁₋₄**
- Key formula: $\Delta x_i(\lambda) = \frac{\lambda f}{\lambda_c}(\alpha_i + \beta_i)$
    - When $\alpha_i + \beta_i = 0$: Achromatic (no dispersion shift) $\rightarrow$ sub-images $I_3$, $I_4$
    - When $\alpha_i + \beta_i \neq 0$: Controlled dispersion $\rightarrow$ sub-images $I_1$, $I_2$ (orthogonal dispersion directions)
- Decoupled design: Refractive optics handle imaging, while the metasurface handles beam splitting and dispersion control, achieving a lower F-number.

### Two Operating Modes

**Mode 1: HDR + Hyperspectral**
- $I_3$, $I_4$ (achromatic channels) insert different ND filters to form exposure bracketing.
- Power ratio is around 4, providing an extra dynamic range of ~12dB.
- Reconstructs HDR using the Debevec-Malik method, then jointly reconstructs the hyperspectral image with $I_1$, $I_2$.

**Mode 2: Polarization + Hyperspectral**
- $0^{\circ}/90^{\circ}$ linear polarizers are placed in front of $I_3$, $I_4$, respectively.
- $I_1$, $I_2$ have no filters, and $I_3 + I_4$ combined are used for hyperspectral reconstruction.
- Degree of Linear Polarization: $\text{DoLP}_{\text{HV}} = |I_3 - I_4| / |I_3 + I_4|$

### Post-Processing Algorithms
- **DWDN**: Wiener deconvolution + multi-scale convolutional network.
- **DDPM**: Diffusion model reconstruction with normalization factors and bias correction.

### Prototype System
- The beam-splitting metasurface has a diameter of 2mm, the dispersion-control metasurface has a diameter of 4mm, with a spacing of 4mm.
- The total track length (TTL) is only 17mm, which is the shortest among all compared methods.
- Operating band: 450-700nm (covering most of the visible spectrum).

## Key Experimental Results

### Hyperspectral Reconstruction on KAUST Dataset (Comparison with Snapshot Methods)

### Main Results

| Method | PSNR(dB) | SSIM | SAM |
|------|----------|------|-----|
| Ours (DWDN) | **32.92**| **0.94**| **0.17**|
| Ours (DDPM) | 33.31 | 0.92 | 0.23 |
| 2-in-1 Cam | 31.14 | 0.86 | 0.24 |
| Array-HSI | 27.44 | 0.89 | 0.20 |
| SCCD | 26.78 | 0.81 | 0.36 |

- PSNR outperforms the strongest snapshot method by ~1.8dB, with SSIM reaching 0.94.
- TTL is only 17mm, which is far smaller than other methods (20-140mm).

### Real-world Scenes
- High-quality hyperspectral reconstruction achieved across 5 validation scenes.
- HDR mode: Dynamic range improved by approximately 11dB.
- Polarization mode: Successfully distinguishes between $0^{\circ}$ and $90^{\circ}$ polarization components.

## Highlights & Insights

1. **Bandwidth Breakthrough**: The operating bandwidth is 250nm, surpassing existing multi-functional metasurface systems by an order of magnitude (10-100nm $\rightarrow$ 250nm).
2. **Decoupling Design Philosophy**: Refractive optics handle imaging while metasurfaces handle beam splitting and dispersion control, with each executing distinct functions to overcome the performance bottleneck of single-layer metasurfaces.
3. **Flexible Multi-functionality**: Switching between HDR and polarization modes is achieved merely by swapping filters, without altering the primary optical assembly.
4. **Shortest TTL (17mm)**: Achieves the most compact package design while maintaining state-of-the-art reconstruction quality.
5. **Complete End-to-End System**: Outlines a fully reproducible solution spanning nanofabrication, calibration, to algorithmic reconstruction.

## Limitations & Future Work

1. **Limited Depth of Field (DoF)** (0.2-0.7m), making it suitable only for close-range imaging scenarios.
2. The randomized interleaving beam-splitting strategy **trades optical efficiency for artifact suppression**, leading to some energy loss.
3. The 4-channel design limits the upper bound of spectral and spatial resolution.
4. While DDPM reconstruction offers higher PSNR, its SAM score is worse (0.23 vs 0.17), indicating a trade-off between spectral fidelity and spatial structural preservation.
5. Currently, only a fixed-channel scheme of 2 achromatic + 2 dispersive modes is supported, presenting limited scalability.
6. The calibration process requires scanning wavelength by wavelength with a narrow-band light source, which is tedious and time-consuming.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The metasurface-refractive hybrid optical paradigm is highly novel, with a significant breakthrough in bandwidth)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Exceptionally solid validation encompassing simulations, real-world prototype, and multi-modal verifications)
- Writing Quality: ⭐⭐⭐⭐⭐ (Beautiful illustrations with clear mathematical and physical derivations)
- Value: ⭐⭐⭐⭐⭐ (A major advance in computational imaging that is poised to accelerate the widespread adoption of compact multi-modal imaging)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes](dense_dispersed_structured_light_for_hyperspectral_3d_imaging_of_dynamic_scenes.md)
- [\[CVPR 2025\] Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning](hierarchical_dual-change_collaborative_learning_for_uav_scene_change_captioning.md)
- [\[CVPR 2025\] Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing](think_and_answer_me_benchmarking_and_exploring_multi-entity_reasoning_grounding_.md)
- [\[CVPR 2025\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)
- [\[CVPR 2025\] DiSciPLE: Learning Interpretable Programs for Scientific Visual Discovery](disciple_learning_interpretable_programs_for_scientific_visual_discovery.md)

</div>

<!-- RELATED:END -->
