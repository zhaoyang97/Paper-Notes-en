---
title: >-
  [Paper Note] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging
description: >-
  [CVPR 2026][Remote Sensing][Metasurface] This paper presents MetaSpectra+, the first multifunctional metasurface imaging system operating across the full visible spectrum (250 nm bandwidth). Through a dual-layer metasurface design enabling beam splitting and precise dispersion control, the system acquires a hyperspectral data cube together with HDR/polarization images in a single snapshot, achieving 33.31 dB PSNR on benchmark datasets with a total track length (TTL) of only 17 mm.
tags:
  - CVPR 2026
  - Remote Sensing
  - Metasurface
  - Hyperspectral Imaging
  - HDR
  - Snapshot Imaging
  - Dispersion Control
  - Multifunctional Optics
date: 2026-05-08
content_hash: 881206fd014535c4
---

# MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging

**Conference**: CVPR 2026
**arXiv**: [2603.09116](https://arxiv.org/abs/2603.09116)
**Code**: [meta-imaging.qiguo.org](https://meta-imaging.qiguo.org)
**Area**: Computational Imaging / Hyperspectral Imaging
**Keywords**: Metasurface, Hyperspectral Imaging, HDR, Snapshot Imaging, Dispersion Control, Multifunctional Optics

## TL;DR
This paper presents MetaSpectra+, the first multifunctional metasurface imaging system operating across the full visible spectrum (250 nm bandwidth). Through a dual-layer metasurface design enabling beam splitting and precise dispersion control, the system acquires a hyperspectral data cube together with HDR/polarization images in a single snapshot, achieving 33.31 dB PSNR on benchmark datasets with a total track length (TTL) of only 17 mm.

## Background & Motivation

**Background**: Multifunctional metasurfaces have demonstrated the ability to simultaneously capture multiple imaging modalities (e.g., different focal lengths, PSFs, dynamic ranges) in a compact monocular form factor. However, they are severely limited by chromatic aberration, with operational bandwidths of only 10–100 nm. Conventional snapshot hyperspectral systems are either bulky (requiring relay optics) or costly to fabricate (spectral filter arrays).

**Limitations of Prior Work**: (1) Existing multifunctional metasurface systems operate over extremely narrow bandwidths (10–100 nm), far from covering the full visible spectrum. (2) Conventional snapshot hyperspectral systems lack compactness, with TTL typically exceeding 20 mm. (3) No existing system can simultaneously capture hyperspectral, HDR, and polarization data in a single shot—yet applications such as agricultural phenotyping and forensic analysis demand precisely co-registered multimodal data.

**Key Challenge**: The intrinsic dispersion of metasurfaces is the fundamental barrier to broadband achromatic imaging. However, the goals of "eliminating dispersion" and "exploiting dispersion to encode spectral information" are conceptually contradictory.

**Goal**: To break the bandwidth limitation of metasurface systems while preserving their compactness, and to realize multifunctional snapshot imaging that simultaneously acquires hyperspectral, HDR, and polarization data.

**Key Insight**: Reframing dispersion from a defect into a controllable function—some channels retain dispersion for spectral encoding while others suppress it for HDR/polarization, with both coexisting in the same system.

**Core Idea**: A dual-layer architecture comprising a beam-splitting metasurface and a dispersion-control metasurface decouples the imaging and beam-splitting functions. Precise control or elimination of dispersion in each optical channel is achieved through joint modulation of deflection vectors.

## Method

### Overall Architecture
MetaSpectra+ adopts a hybrid optical architecture: an objective lens (refractive, $f = 400$ mm achromatic doublet) handles image formation, while the dual-layer metasurface performs beam splitting and functional control. The beam-splitting metasurface $M_0$ splits the collimated beam into $V = 4$ channels arranged in a $2\times2$ grid, each deflected by approximately 33°. Each channel passes through a dispersion-control metasurface $M_i$, an eyepiece lens $L_i$ with a 12 mm focal length, and an optional filter $F_i$, projecting four sub-images simultaneously onto a shared RGB sensor ($7.1\,\text{mm} \times 7.1\,\text{mm}$). Channels 1 and 2 retain orthogonal dispersion for spectral encoding (CTIS configuration), while channels 3 and 4 suppress dispersion to produce achromatic images for HDR (exposure bracketing) or polarization. A post-processing algorithm jointly reconstructs a 31-channel hyperspectral data cube and HDR/polarization images from the four sub-images.

### Key Designs

1. **Dual-Layer Dispersion-Control Metasurface**:
   - *Function*: Precisely control or eliminate dispersion in each optical channel.
   - *Mechanism*: $M_0$ applies a deflection $\alpha_i$ to channel $i$, and $M_i$ applies a compensating deflection $\beta_i$. The wavelength-dependent PSF shift is $\Delta x_i(\lambda) = \frac{\lambda f}{\lambda_c}(\alpha_i + \beta_i)$. When $\alpha_i + \beta_i = 0$, dispersion is fully suppressed (achromatic channel); when $\alpha_i + \beta_i \neq 0$, controlled dispersion is retained (spectral encoding channel).
   - *Design Motivation*: This transforms dispersion from an unavoidable defect into a controllable asset—a "turning a liability into a tool" design philosophy. Decoupling image formation (refractive) from beam splitting (metasurface) allows the system to operate at a significantly lower F-number while maintaining compactness.

2. **Random Interleaved Beam-Splitting Strategy**:
   - *Function*: Achieve wide-angle multi-channel beam splitting.
   - *Mechanism*: The phase pattern of $M_0$ is formed by randomly interleaving four deflection sub-patterns with equal-weight multinomial distribution: $M_0(x,\lambda_c) = M_{0,k}(x,\lambda_c),\ k \sim \text{Multinomial}(1/V)$. Each channel uses a distinct design wavelength $\lambda_{c,1:4} = \{450, 550, 600, 750\}$ nm to ensure full visible-spectrum coverage.
   - *Design Motivation*: Regular $2\times2$ mosaic interleaving produces strong high-order diffraction artifacts at large deflection angles. Random interleaving effectively suppresses these artifacts at the cost of a slight reduction in optical efficiency.

3. **Dual Post-Processing Algorithms (DWDN + DDPM)**:
   - *Function*: Reconstruct the hyperspectral data cube from the four sub-images.
   - *Mechanism*: The DWDN scheme applies feature-domain Wiener deconvolution followed by multi-scale convolutional refinement. The DDPM scheme employs a denoising diffusion probabilistic model for patch-wise reconstruction, with per-step normalization factors (including bias terms and scheduled learning rates) to ensure cross-patch spectral consistency.
   - *Design Motivation*: Provides two accuracy/speed trade-off options—DWDN is faster, while DDPM achieves higher PSNR.

### Loss & Training
Both DWDN and DDPM are trained on synthetic data derived from the Harvard and ICVL datasets. Sub-images are generated by rendering PSFs through the D-Flat simulator, with noise levels $\sigma$ sampled from $U(0.001, 0.01)$. DDPM uses an L1 noise loss with a U-Net of channel dimensions [64, 128, 256, 512, 1024], optimized with AdamW for 15,000 epochs. For real-world experiments, fine-tuning on three parallel scenes is performed to bridge the simulation-to-measurement gap.

## Key Experimental Results

### Main Results

| System | PSNR (dB)↑ | SSIM↑ | SAM↓ | TTL (mm)↓ | Sub-images |
|---|---|---|---|---|---|
| **MetaSpectra+ (DDPM)** | **33.31** | 0.92 | 0.23 | **17** | 4 (+2 achromatic) |
| **MetaSpectra+ (DWDN)** | 32.92 | **0.94** | **0.17** | **17** | 4 (+2 achromatic) |
| 2-in-1 Cam (SIGGRAPH24) | 31.14 | 0.88 | 0.22 | 50 | 5.8 |
| Array-HSI (SGA24) | 27.44 | 0.89 | 0.20 | 20 | 4 |
| SRD (OE24) | 26.39 | 0.81 | 0.26 | — | 1 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Random vs. regular interleaving | Artifacts↓↓, slight efficiency loss | Random interleaving shows clear advantage at large deflection angles |
| DDPM vs. DWDN | PSNR +0.39 dB / SSIM −0.02 | Diffusion model is more accurate but may introduce detail discrepancies |
| HDR mode | Dynamic range +11 dB | Achromatic channel power ratio ~4:1 for exposure bracketing |
| Different $\lambda_c$ coverage strategies | Full-spectrum coverage | Four channels with distinct design wavelengths provide complementary coverage |

### Key Findings
- MetaSpectra+ simultaneously achieves the highest PSNR and shortest TTL among all compared snapshot hyperspectral systems.
- The hybrid optical architecture enables a significantly lower F-number than purely metasurface-based systems while maintaining compactness.
- Switching between HDR and polarization modes requires only a filter swap, with no modification to the core architecture.
- A physical prototype validates the complete pipeline from design → fabrication → calibration → real-world scenes.

## Highlights & Insights
- The design philosophy of "transforming a dispersion defect into a controlled function" is particularly elegant—channels that require dispersion and channels that suppress it coexist within the same system.
- The refractive + metasurface hybrid architecture achieves a clear division of responsibilities, decoupling imaging quality from beam-splitting functionality and avoiding the inherent conflicts of a single metasurface performing both tasks.
- The system demonstrates excellent configurability—switching between HDR and polarization modes requires only a filter change.
- A complete physical prototype is presented, including SEM-verified nanopillar arrays, going beyond pure simulation.

## Limitations & Future Work
- Random interleaved beam splitting reduces diffraction efficiency; the current prototype achieves only ~10 FPS, limiting high-speed video applications.
- The depth of field is limited to 0.2–0.7 m, requiring an objective lens change for adjustment.
- Metasurface fabrication relies on electron-beam lithography, which is relatively costly, though commercial foundry services are available.
- DDPM inference is slow (50 denoising steps + 20 normalization iterations), limiting real-time applications.
- The authors suggest that adopting higher-refractive-index materials (GaN/TiO₂) to improve diffraction efficiency could increase frame rates.

## Related Work & Insights
- **vs. 2-in-1 Cam (SIGGRAPH24)**: Uses a DOE + lens with TTL of 50 mm (3× that of this work) and PSNR 1.78 dB lower. The proposed metasurface approach is more compact and more precise.
- **vs. Array-HSI (SGA24)**: A DOE + CFA scheme achieving only 27.44 dB PSNR (5.48 dB lower). MetaSpectra+ additionally supports HDR and polarization.
- **vs. MetaHDR and other existing multifunctional metasurfaces**: Prior systems operate over bandwidths of only 10–100 nm. MetaSpectra+ is the first multifunctional metasurface system covering the full visible spectrum.
- The co-design paradigm of optical hardware and computational reconstruction merits attention: hardware encodes information while algorithms decode it, and neither can function without the other.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First multifunctional metasurface imaging system spanning the full visible spectrum; the dispersion control design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Simulation comparisons are comprehensive; a physical prototype with real-scene validation is provided; ablation studies could be more in-depth.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Physical modeling is rigorous, figures are clear, and hardware details are thorough.
- **Value**: ⭐⭐⭐⭐ — Represents an important contribution to computational imaging and metasurface optics, though somewhat distant from mainstream computer vision research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)
- [\[CVPR 2026\] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)
- [\[CVPR 2026\] Conflated Inverse Modeling for Urban Vegetation Patterns](conflated_inverse_urban_vegetation.md)

<!-- RELATED:END -->
