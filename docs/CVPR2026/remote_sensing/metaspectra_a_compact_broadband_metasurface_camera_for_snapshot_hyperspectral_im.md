---
title: >-
  [Paper Note] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging
description: >-
  [CVPR 2026][Remote Sensing][Metasurface Imaging] MetaSpectra+ proposes a metasurface–refractive hybrid optical paradigm that employs a dual-layer metasurface to independently control the dispersion, exposure, and polarization of four channels, enabling snapshot hyperspectral+HDR/polarization multi-functional imaging over a 250 nm bandwidth with a minimum total track length (TTL) of 17 mm. On the KAUST benchmark, it achieves a PSNR of 33.31 dB, comprehensively surpassing existing snapshot hyperspectral systems.
tags:
  - CVPR 2026
  - Remote Sensing
  - Metasurface Imaging
  - Hyperspectral Reconstruction
  - Snapshot Imaging
  - HDR
  - Polarization Imaging
date: 2026-05-08
content_hash: 4515b40fb12f223b
---

# MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging

**Conference**: CVPR 2026
**arXiv**: [2603.09116](https://arxiv.org/abs/2603.09116)
**Code**: [https://meta-imaging.qiguo.org](https://meta-imaging.qiguo.org)
**Area**: Remote Sensing / Computational Hyperspectral Imaging
**Keywords**: Metasurface Imaging, Hyperspectral Reconstruction, Snapshot Imaging, HDR, Polarization Imaging

## TL;DR

MetaSpectra+ proposes a metasurface–refractive hybrid optical paradigm that employs a dual-layer metasurface to independently control the dispersion, exposure, and polarization of four channels, enabling snapshot hyperspectral+HDR/polarization multi-functional imaging over a 250 nm bandwidth with a minimum total track length (TTL) of 17 mm. On the KAUST benchmark, it achieves a PSNR of 33.31 dB, comprehensively surpassing existing snapshot hyperspectral systems.

## Background & Motivation

**Background**: Snapshot hyperspectral imaging (Snapshot HSI) aims to recover a 3D hyperspectral data cube from a single 2D sensor measurement. Existing approaches include sampling-based methods (coded aperture, lens arrays, spectral filter arrays) and coding-based methods (embedding spectral information into the spatial domain via wavelength-dependent PSFs using DOEs, gratings, or prisms). Meanwhile, multifunctional metasurfaces have attracted attention for their ability to simultaneously acquire depth, polarization, and spectral information in a monocular form factor.

**Limitations of Prior Work**: Metasurface optical elements suffer from severe chromatic aberration, and the vast majority of multifunctional metasurface systems operate only within an extremely narrow band of 10–100 nm, far from covering the full visible spectrum. Furthermore, existing approaches couple beam-splitting and imaging functions into a single metasurface, resulting in a large F-number and insufficient compactness.

**Key Challenge**: The strong dispersion of metasurfaces is a double-edged sword—it is the physical basis for spectral modulation, yet it strictly limits the usable bandwidth. In multifunctional imaging, one must simultaneously exploit dispersion to encode spectral information and eliminate it when needed (e.g., HDR/polarization channels require achromatic behavior). These two requirements are mutually exclusive in conventional single-layer metasurface designs.

**Goal**: (1) How to extend the operating bandwidth of multifunctional metasurfaces from tens of nanometers to 250 nm to cover the entire visible spectrum? (2) How to independently control the dispersion of each channel within the same system—some channels retaining controllable dispersion for spectral encoding, others being achromatic for HDR/polarization? (3) How to reduce the F-number while maintaining compactness?

**Key Insight**: The authors observe that dispersion is fundamentally the algebraic sum of the deflection vectors of two optical elements ($\Delta \mathbf{x}_i(\lambda) = \frac{\lambda f}{\lambda_c}(\boldsymbol{\alpha}_i + \boldsymbol{\beta}_i)$). Therefore, by distributing beam-splitting and dispersion control across two metasurface layers, the dispersion of each channel can be independently controlled by adjusting the second-layer deflection vector $\boldsymbol{\beta}_i$. When $\boldsymbol{\alpha}_i + \boldsymbol{\beta}_i = 0$, full achromatization is achieved; otherwise, controllable dispersion is retained. Imaging functionality is delegated to a refractive lens, achieving functional decoupling.

**Core Idea**: A dual-layer metasurface is used, with one layer responsible for beam splitting and the other for dispersion control, while a refractive lens handles imaging. The additivity of deflection vectors enables independent dispersion tuning per channel, thereby realizing broadband multifunctional hyperspectral imaging in a compact form factor.

## Method

### Overall Architecture

The optical system of MetaSpectra+ comprises five components: objective lens (achromatic doublet with field stop, focal length 400 mm) → beam-splitting metasurface M0 (splits collimated light into 2×2 = 4 channels) → dispersion-control metasurfaces M1–M4 (independently modulating dispersion/achromatization per channel) → eyepiece lenses (four achromatic doublets, 12 mm focal length each) → optical filters + sensor (7.1 mm × 7.1 mm global-shutter image sensor). The total track length (TTL) is only 17 mm. Among the four channels, I1/I2 carry orthogonal dispersion for spectral encoding (CTIS configuration), while I3/I4 are achromatic channels for HDR or polarization imaging. Post-processing algorithms based on DWDN or DDPM reconstruct the hyperspectral data cube from the four sub-images.

### Key Designs

1. **Beam-Splitting Metasurface M0 (Randomly Interleaved Multi-Channel Beam Splitting)**:

    - Function: Splits and deflects incident collimated light into four independent optical channels at deflection angles of approximately 33°.
    - Mechanism: For each channel $i$, a linear phase delay is applied: $M_{0,i}(\mathbf{x}, \lambda_c) = \exp(j\frac{2\pi}{\lambda_c} \boldsymbol{\alpha}_i \cdot \mathbf{x})$. The overall phase profile is constructed by randomly interleaving four sub-profiles with equal probability: $M_0(\mathbf{x}, \lambda_c) = M_{0,k}(\mathbf{x}, \lambda_c), k \sim \text{Multinomial}(1/4)$. Dispersion causes multi-order diffraction at non-design wavelengths, but empirically only the 0th and 1st orders are significant; a downstream field stop blocks the 0th order, so the effective modulation approximates $M_{0,i}(\mathbf{x}, \lambda) \approx a_1(\lambda) M_{0,i}(\mathbf{x}, \lambda_c)$.
    - Design Motivation: Random interleaving effectively suppresses high-order diffraction artifacts compared to a regular 2×2 mosaic (which produces strong high-order diffraction at large deflection angles), at the cost of a minor loss in light efficiency. Additionally, the four channels use different design wavelengths $\lambda_{c,1:4} = \{450, 550, 600, 750\}$ nm, ensuring that at least one high-efficiency channel covers any given portion of the full visible spectrum.

2. **Dispersion-Control Metasurfaces Mi (Deflection-Vector Additivity for Achromatization/Dispersion Control)**:

    - Function: Applies an additional deflection to each channel to achieve achromatization or retain controllable dispersion.
    - Mechanism: The $i$-th dispersion-control metasurface imparts deflection $\boldsymbol{\beta}_i$, producing a wavelength-dependent PSF shift of $\Delta \mathbf{x}_i(\lambda) = \frac{\lambda f}{\lambda_c}(\boldsymbol{\alpha}_i + \boldsymbol{\beta}_i)$. The key insight is that this shift is determined by the sum of deflection vectors from both layers: when $\boldsymbol{\alpha}_i + \boldsymbol{\beta}_i = 0$, the PSF does not shift with wavelength, achieving achromatic focusing (channels I3/I4); when the sum is nonzero, controllable wavelength-dependent dispersion is retained (channels I1/I2, with orthogonally directed dispersion for CTIS encoding).
    - Design Motivation: This deflection-vector additivity principle is the core innovation of MetaSpectra+. It reduces dispersion control to an algebraic design problem rather than a complex wave-optics optimization, enabling each channel to be independently assigned a desired dispersion magnitude and direction with great flexibility.

3. **Multifunctional Imaging Configurations (Zero-Cost Modality Extension)**:

    - Function: Inserts different filters into the achromatic channels to extend HDR or polarization imaging capability with zero additional optical complexity.
    - HDR Mode: ND filters with OD = 0.3 are inserted into channels I1–I3, and OD = 0.9 into channel I4, forming an approximately 4:1 exposure bracket. I3 and I4 are fused via the Debevec–Malik method to produce an HDR image, extending dynamic range by approximately 11 dB compared to a single exposure.
    - Polarization Mode: A 0° linear polarizer is placed before channel I3 and a 90° linear polarizer before channel I4. The horizontal–vertical degree of linear polarization is computed as $\text{DoLP}_{HV} = |I_3 - I_4| / |I_3 + I_4|$. Channels I1 and I2 are unaffected by polarization and are used for spectral encoding.
    - Design Motivation: The achromatic channels I3/I4 are naturally suited for these extensions—since they carry no dispersion, their image quality most closely resembles that of a conventional camera, making them ideal for HDR bracketing or polarization contrast measurement.

### Loss & Training

Two reconstruction network options are provided: (1) **DWDN**: performs Wiener deconvolution in the feature domain followed by a multi-scale feedforward convolutional network for refinement; (2) **DDPM**: partitions sub-images into patches and reconstructs the hyperspectral cube patch-by-patch via a diffusion model, estimating a normalization factor $a^{k,t}$ and bias $b^{k,t}$ at each step to maintain spatial consistency across patches (an improvement over the method of Hazineh et al.). Training data are drawn from the Harvard and ICVL hyperspectral datasets; sub-images are synthesized using the D-Flat simulator based on the optical design, with noise level $\sigma$ sampled uniformly from $[0.001, 0.01]$.

## Key Experimental Results

### Main Results

Comparison with existing snapshot hyperspectral imaging systems on the KAUST benchmark dataset (450–700 nm band):

| Method | Conference | Optical Type | # Sub-images | TTL (mm) | PSNR (dB)↑ | SSIM↑ | SAM↓ |
|--------|------------|-------------|-------------|---------|------------|-------|------|
| **Ours (DDPM)** | – | MS+Lens | 4 | **17** | **33.31** | 0.92 | 0.23 |
| **Ours (DWDN)** | – | MS+Lens | 4 | **17** | 32.92 | **0.94** | **0.17** |
| 2-in-1 Cam | SIG'24 | DOE+Lens | 2 | 50 | 31.14 | 0.86 | 0.24 |
| SfD | arXiv'25 | Lens | 5 | 44.5 | 27.54 | 0.82 | 0.40 |
| Array-HSI | SIG Asia'24 | DOE+CFA | 4 | 20 | 27.44 | 0.89 | 0.20 |
| SCCD | Optica'21 | DOE+CCA | 1 | 50 | 26.78 | 0.81 | 0.36 |
| Baek et al. | ICCV'21 | DOE | 1 | 50 | 26.68 | 0.74 | 0.39 |
| HRNet | CVPRW'20 | Lens | 1 | – | 23.03 | 0.76 | 0.31 |
| MST++ | CVPRW'22 | Lens | 1 | – | 21.85 | 0.68 | 0.32 |

### Ablation Study

| Configuration | PSNR (dB) | SSIM | SAM | Notes |
|--------------|-----------|------|-----|-------|
| Full system (DDPM) | 33.31 | 0.92 | 0.23 | Diffusion-based recovery; best PSNR |
| Full system (DWDN) | 32.92 | 0.94 | 0.17 | Non-diffusion recovery; better SSIM and SAM |
| Achromatic channels only (RGB→HSI) | 21–23 | ~0.7 | >0.3 | No dispersion encoding; degrades to RGB upsampling |
| Regular 2×2 interleaved M0 | – | – | – | Strong high-order diffraction artifacts at large deflection angles |
| HDR mode (I3+I4 fusion) | – | – | – | Dynamic range increased by ~11 dB |

### Key Findings

- MetaSpectra+ **comprehensively surpasses** all existing snapshot hyperspectral systems on the KAUST benchmark across all metrics: PSNR exceeds the second-best (2-in-1 Cam) by 2.17 dB, while achieving a TTL of only 17 mm (second-best Array-HSI: 20 mm; all others ≥ 44.5 mm).
- DWDN and DDPM each have advantages: DDPM achieves higher PSNR (33.31 vs. 32.92), while DWDN yields better SSIM (0.94 vs. 0.92) and SAM (0.17 vs. 0.23), indicating that DDPM offers sharper reconstruction but slightly lower spectral fidelity.
- Achromatization is critical: removing dispersion encoding and relying solely on RGB-to-hyperspectral upsampling causes a PSNR drop of approximately 10 dB, demonstrating that controllable dispersion is essential for high-accuracy reconstruction.
- The complementary coverage strategy using different design wavelengths is effective: the four channels with $\lambda_c = \{450, 550, 600, 750\}$ nm ensure efficient coverage of the entire 450–700 nm band.
- Real-world experiments validate an 11 dB dynamic range gain in HDR mode and accurate DoLP measurements in polarization mode, both achieved without sacrificing hyperspectral reconstruction quality.

## Highlights & Insights

- **Fundamental innovation of the metasurface–refractive hybrid paradigm**: Decoupling beam-splitting and imaging functions between the metasurface and the refractive lens breaks the bandwidth and F-number limitations of single-layer metasurface designs. This paradigm is generalizable to other diffractive/metasurface optical systems.
- **Elegant exploitation of deflection-vector additivity**: The compact mathematical relation $\Delta \mathbf{x} \propto (\boldsymbol{\alpha} + \boldsymbol{\beta})$ is the cornerstone of the entire system, reducing complex wave-optics dispersion control to vector algebra and making the transition between achromatization and controllable dispersion trivial.
- **Zero-cost multifunctional extension**: Achromatic channels are naturally suited for HDR and polarization extensions, requiring only the insertion of filters without any modification to the optical design, reflecting a well-structured modular philosophy.

## Limitations & Future Work

- **Limited depth of field**: The prototype system has a depth of field of only 0.2–0.7 m, constrained by the 400 mm objective focal length; long-range applications require optical element replacement.
- **High metasurface fabrication barrier**: The SiN nanopillar arrays (300 nm wide, 775 nm tall) rely on specialized nanofabrication; mass-production cost and consistency remain bottlenecks for commercialization.
- **Random interleaving sacrifices light efficiency**: Although high-order artifacts are suppressed, random sampling means each channel receives only approximately 1/4 of the incident light, potentially limiting performance in low-light scenarios.
- **Slow DDPM inference**: The diffusion model reconstructs patch-by-patch with multiple denoising steps, making real-time deployment impractical.
- **Validation limited to 450–700 nm**: Despite claiming broadband operation, the system has not been demonstrated in the near-infrared (700–1000 nm), limiting applicability in agriculture, phenotyping, and remote sensing applications that require NIR.

## Related Work & Insights

- **vs. 2-in-1 Cam (SIGGRAPH'24)**: The closest prior work, also employing a DOE+Lens hybrid scheme, but with only 2 sub-images, 50 mm TTL, and PSNR of 31.14 dB. MetaSpectra+ achieves 4 channels, a more compact form factor, and higher accuracy via metasurfaces, outperforming it across all dimensions.
- **vs. Array-HSI (SIGGRAPH Asia'24)**: Both use 4 sub-images, but Array-HSI employs DOE+CFA with 20 mm TTL and PSNR of 27.44 dB. MetaSpectra+ achieves a 5.5 dB PSNR gain at an even shorter TTL, demonstrating the superiority of metasurface-based dispersion control over DOE+CFA approaches.
- **vs. SCCD/Baek (Optica'21/ICCV'21)**: Single sub-image DOE methods achieving only 26–27 dB PSNR; the multi-channel broadband strategy of MetaSpectra+ demonstrates a clear advantage.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The metasurface–refractive hybrid paradigm combined with deflection-vector additivity for dispersion control represents a fundamental innovation at the optical design level.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive simulation comparisons, real prototype validation, and HDR/polarization demonstrations are provided, though outdoor and dynamic scene evaluation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The optical modeling derivations are complete and rigorous, with a clear logical flow from physical principles to system design.
- Value: ⭐⭐⭐⭐⭐ Simultaneously achieving the most compact form factor and highest reconstruction accuracy sets a new benchmark for snapshot multifunctional imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_vision-language_instruction_from_offline_teacher_to_prompt-tuned_ne.md)
- [\[CVPR 2026\] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)

</div>

<!-- RELATED:END -->
