---
title: >-
  [Paper Note] Visual Surface Wave Elastography: Revealing Subsurface Physical Properties via Visible Surface Waves
description: >-
  [ICCV 2025][Medical Imaging][Elastography] This paper proposes VSWE (Visual Surface Wave Elastography), a method that extracts the dispersion relation from a video of surface wave propagation and combines it with physics-based finite element optimization to infer subsurface layer thickness and stiffness. High-accuracy parameter recovery is demonstrated in both simulated and real gelatin experiments, providing a proof-of-concept for at-home health monitoring.
tags:
  - ICCV 2025
  - Medical Imaging
  - Elastography
  - surface waves
  - dispersion relation
  - finite element method
  - video analysis
date: 2026-05-08
content_hash: 0bc0d1722ceef64d
---

# Visual Surface Wave Elastography: Revealing Subsurface Physical Properties via Visible Surface Waves

**Conference**: ICCV 2025
**arXiv**: [2507.09207](https://arxiv.org/abs/2507.09207)
**Code**: Coming soon
**Area**: Medical Imaging / Material Characterization
**Keywords**: Elastography, surface waves, dispersion relation, finite element method, video analysis

## TL;DR
This paper proposes VSWE (Visual Surface Wave Elastography), a method that extracts the dispersion relation from a video of surface wave propagation and combines it with physics-based finite element optimization to infer subsurface layer thickness and stiffness. High-accuracy parameter recovery is demonstrated in both simulated and real gelatin experiments, providing a proof-of-concept for at-home health monitoring.

## Background & Motivation

**Background**: Elastography is an important technique for measuring tissue stiffness, applicable to detecting tumors, musculoskeletal degeneration, liver disease, and more. Existing elastography modalities—transient elastography, shear wave elastography, and MR elastography—all rely on expensive specialized equipment and trained operators.

**Limitations of Prior Work**: (a) Ultrasound and MRI elastography systems are cost-prohibitive and unsuitable for home use; (b) trained medical experts are required, limiting the feasibility of routine screening; (c) existing video-based material characterization methods (e.g., Visual Vibration Tomography) require modeling the global vibration modes of the entire object, which is impractical for complex geometries such as the human body.

**Key Challenge**: External excitation induces visible surface waves on the human body (e.g., skin ripples caused by a massage gun) that carry information about subsurface physical properties, yet no method currently exists to extract this information from video.

**Goal**: Infer subsurface layer thickness $T$ and stiffness $E$ from a video of surface wave propagation.

**Key Insight**: The approach draws inspiration from seismology, where surface waves are used to infer subsurface structure, but replaces sparse sensors with dense visual data (video). The key insight is that the dispersion relation (wavenumber–frequency relationship) is fully determined by thickness and stiffness under common biomechanical assumptions; therefore, one need only extract the dispersion relation from video and solve for the matching physical parameters.

**Core Idea**: Extract the dispersion relation from a surface wave video, then identify the best-matching thickness and stiffness via FEM simulation combined with SSIM-based optimization, enabling purely video-based inference of subsurface physical properties.

## Method

### Overall Architecture
The input is a video capturing surface wave propagation on a medium. The pipeline consists of two stages: (1) extracting the dispersion relation from the video; and (2) solving for the thickness and stiffness parameters that best align the theoretical dispersion relation with the observed one.

### Key Designs

1. **Video Motion Extraction**

    - **Function**: Extract sub-pixel-level surface displacement fields from video.
    - **Mechanism**: Phase-based motion processing is applied by computing local phase shifts in the complex steerable pyramid and converting them to pixel displacements, yielding horizontal and vertical displacements $\tilde{u}(\tilde{x}, \tilde{y}, t)$ and $\tilde{v}(\tilde{x}, \tilde{y}, t)$ for each pixel $(x, y)$ at each frame $t$.
    - **Design Motivation**: Surface waves typically produce sub-pixel motion; conventional optical flow methods lack sufficient precision, whereas phase-based methods are inherently sensitive to sub-pixel displacements.

2. **Dispersion Relation Extraction**

    - **Function**: Transform the spatiotemporal displacement signal into the frequency–wavenumber domain to obtain the dispersion relation.
    - **Mechanism**: A 2D FFT is applied along each row of the displacement video (spatial $x$ → wavenumber $\gamma$; temporal $t$ → frequency $\omega$), and the magnitude is taken to form a dispersion image. Averaging over all rows and both displacement directions improves the signal-to-noise ratio: $\mathbf{D}_{\text{obs}} = \frac{1}{2H}\sum_{i=1}^{H}(|\hat{\tilde{u}}(i)| + |\hat{\tilde{v}}(i)|)$
    - **Design Motivation**: The dispersion relation is a compact representation of the wave propagation behavior of a medium, fully determined by its physical parameters. The FFT is the standard and efficient tool for extracting frequency-domain information.

3. **Physical Model and Parameter Optimization**

    - **Function**: Identify thickness $T$ and stiffness $E$ such that the theoretical dispersion relation best matches the observed one.
    - **Mechanism**: A homogeneous, isotropic, linear-elastic medium is assumed, consisting of a soft tissue layer (unknown $T$ and $E$; known $\rho = 1$ g/cm³ and $\nu = 0.45$) overlying a rigid bone layer. For a given $(T, E)$, the elastic wave equation is solved via finite element method (FEM) with Bloch–Floquet periodic boundary conditions to obtain the theoretical dispersion relation $\mathfrak{D}(T, E)$. The theoretical dispersion curve is converted to an image $\mathbf{D}_{\text{hyp}}(T, E)$ via a Gaussian kernel, and the objective is to maximize $\text{SSIM}(\mathbf{D}_{\text{hyp}}(T, E), \mathbf{D}_{\text{obs}})$. Grid search is currently used for optimization.
    - **Design Motivation**: SSIM's measure of structural similarity is more robust than MSE or PSNR; experiments confirm that SSIM yields the sharpest optimization landscape and the most accurate parameter estimates.

### Loss & Training
The method is entirely physics-driven and requires no machine learning training. The core optimization objective is $\text{argmax}_{T,E}\ \text{SSIM}(\mathbf{D}_{\text{hyp}}, \mathbf{D}_{\text{obs}})$. The authors also introduce six dimensionless feature numbers $\pi_1 \sim \pi_6$ to guide experimental design across different parameter regimes (e.g., adjusting the observation window, frame rate, and frequency range to maintain performance).

## Key Experimental Results

### Main Results: Real Gelatin Samples
Three gelatin samples of different volumes (1000/1100/1500 mL) are tested, with ground-truth thickness measured by calipers and ground-truth stiffness measured by a rheometer:

| Sample | True Thickness (mm) | VSWE Est. Thickness | True Stiffness (kPa) | VSWE Est. Stiffness | Stiffness Error |
|--------|--------------------|--------------------|---------------------|--------------------|----|
| 1000 mL | ~42 | Within confidence interval | ~12–16 | Consistent with rheometer | <1.2% |
| 1100 mL | ~46 | Within confidence interval | ~12–16 | Consistent with rheometer | <1.2% |
| 1500 mL | ~55 | Within confidence interval | ~12–16 | Consistent with rheometer | <1.2% |

Approximately 60 videos are collected per sample at varying temperatures (warming after removal from refrigerator). The VSWE-estimated stiffness decreases with temperature consistently with the rheometer measurements.

### Ablation Study: Sensitivity Analysis

| Parameter Variation | VSWE Detection Capability |
|--------------------|--------------------------|
| ±5% thickness change | Detectable |
| ±10% thickness change | Detectable |
| ±5% stiffness change | Detectable |
| ±10% stiffness change | Detectable |

Validated on 180 simulated samples (5 thicknesses × 4 stiffness values × 9 perturbations); VSWE remains sensitive to parameter variations at the 5% level.

### 3D Human Leg Simulation
Using realistic anatomical geometry from the Visible Human Project with full 3D COMSOL physics simulation:
- VSWE successfully recovers the spatial variation trend of thickness across sliding windows on the upper calf.
- Correct thickness estimates are obtained in three anatomically distinct regions near the ankle.
- The constant tissue stiffness is successfully recovered.

### Key Findings
- **SSIM outperforms alternative objective functions**: Compared to curve matching, MSE, and PSNR, SSIM is the only objective that yields the correct optimal parameters on the simulated leg data.
- **Incomplete dispersion relations remain tractable**: Only partial wave modes may be observed in a video, yet SSIM's structural matching property enables correct parameter inference from partial matches.
- **Temperature–stiffness relationship**: Gelatin stiffness is negatively correlated with temperature in the experiments, and VSWE quantitatively tracks this variation.

## Highlights & Insights
- **Purely physics-driven with purely visual input**: No ML model training is required, nor any specialized equipment such as ultrasound or MRI. Only a high-speed camera and a vibration source are needed to measure tissue properties. This zero-shot physics-based approach offers far greater interpretability than data-driven methods.
- **Dispersion relation as an intermediate representation**: Spatiotemporal information in the video is compressed into a 2D dispersion image, providing an elegant bridge from video to physical parameters. This paradigm is transferable to other inverse problems involving wave propagation.
- **Dimensionless number guidelines**: Six dimensionless feature numbers $\pi_1 \sim \pi_6$ are introduced to guide experimental design, enabling the method to generalize across scenarios with vastly different parameter scales (e.g., larger or smaller objects, harder or softer materials) and ensuring that VSWE performance is not tied to specific parameter magnitudes.
- **Advantage of local analysis**: Unlike Visual Vibration Tomography, VSWE does not require modeling the full-body geometry; local surface wave analysis suffices to infer local tissue properties.

## Limitations & Future Work
- **High-speed camera required**: Current experiments use a 600 FPS high-speed camera; standard smartphone cameras may not provide sufficient temporal resolution.
- **Simplified physical model**: The method assumes isotropic, linear-elastic, homogeneous stiffness with known density and Poisson's ratio; real human tissue is considerably more complex (viscoelastic, anisotropic, multi-layered).
- **Only 1D wave propagation analyzed**: Current analysis considers wave propagation in a single direction; 2D wavefield analysis could provide richer information.
- **Not yet validated on real human subjects**: The 3D leg experiment remains simulation-based; real human subjects introduce additional motion artifacts from breathing and muscle contraction.
- **Grid search is inefficient**: Gradient-based or Bayesian optimization could accelerate the parameter search.
- **Future directions**: Incorporating viscoelastic models; validating with smartphone slow-motion (240 FPS); exploring multi-frequency sequential excitation to improve SNR.

## Related Work & Insights
- **vs. Visual Vibration Tomography [CVPR'22]**: VVT infers spatially varying stiffness and density from global vibration modes and requires known global geometry. VSWE requires only local surface wave analysis, making it more suitable for complex geometries such as the human body.
- **vs. conventional elastography (ultrasound/MRI)**: Conventional methods offer higher accuracy but require expensive equipment and expert operation; VSWE trades some accuracy for drastically lower cost, making it suitable for at-home preliminary screening.
- **vs. seismological surface wave methods**: Seismology uses sparse sensors, whereas VSWE uses dense visual data, enabling recovery of a more complete dispersion relation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Cross-disciplinary innovation combining seismological surface wave theory with computer vision; highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated through both simulation and real experiments with thorough dimensionless number analysis; real human subject experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The progression from intuition to mathematical formulation is smooth, and physical concepts are explained clearly.
- Value: ⭐⭐⭐⭐ Conceptually novel with strong potential for at-home healthcare, though practical deployment remains some distance away.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](../../NeurIPS2025/medical_imaging/surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)
- [\[ICCV 2025\] NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction](neurons_emulating_the_human_visual_cortex_improves_fidelity_and_interpretability.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[CVPR 2026\] Decoupling Vision and Language: Codebook Anchored Visual Adaptation](../../CVPR2026/medical_imaging/decoupling_vision_and_language_codebook_anchored_visual_adaptation.md)

<!-- RELATED:END -->
