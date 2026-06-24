---
title: >-
  [Paper Note] Spectral Subsurface Scattering for Material Classification
description: >-
  [ECCV 2024][Human Understanding][Subsurface scattering] This paper proposes a method for material classification utilizing Spectral Sub-Surface Scattering (S4). It demonstrates that the strong spectral dependence of subsurface scattering provides highly discriminative features and designs a novel imaging setup to efficiently acquire S4 measurements via a 2D projection, eliminating the need for time-consuming hyperspectral scanning.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Subsurface scattering"
  - "spectral imaging"
  - "material classification"
  - "point spread function"
  - "computational photography"
date: 2026-05-08
content_hash: f6f42c1924e09cd6
---

# Spectral Subsurface Scattering for Material Classification

**Conference**: ECCV 2024  
**Institution**: Carnegie Mellon University  
**Code**: None  
**Area**: Computational Photography / Material Classification  
**Keywords**: Subsurface scattering, spectral imaging, material classification, point spread function, computational photography

## TL;DR

This paper proposes a method for material classification utilizing Spectral Sub-Surface Scattering (S4). It demonstrates that the strong spectral dependence of subsurface scattering provides highly discriminative features and designs a novel imaging setup to efficiently acquire S4 measurements via a 2D projection, eliminating the need for time-consuming hyperspectral scanning.

## Background & Motivation

**Background**: Material classification is an important task in computer vision, widely applied in scenarios such as robotic grasping, autonomous driving, and food inspection. Mainstream methods include RGB appearance-based classification, spectrum-based classification, and subsurface scattering-based classification. Spectral imaging methods utilize the difference in reflectance of different materials under different wavelengths for discrimination; subsurface scattering methods utilize the scattering patterns (the shape and attenuation characteristics of the Point Spread Function, PSF) after light enters translucent materials to distinguish materials.

**Limitations of Prior Work**: Spectral imaging only focuses on the surface reflection characteristics of materials, ignoring the propagation behavior of light inside the materials. Although subsurface scattering imaging can capture the internal structural information of materials, it is usually performed under a single wavelength or broad bands, losing the spectral dependence of scattering. However, the subsurface scattering behavior of different materials varies significantly across different wavelengths—for instance, the scattering distance and scattering pattern of the same material under red light and blue light can be drastically different. Using spectrum or subsurface scattering alone provides limited discriminative power.

**Key Challenge**: Jointly utilizing spectral and subsurface scattering information (i.e., S4 measurements) theoretically provides the most powerful discriminative features for materials. However, acquiring complete S4 data requires measuring the subsurface scattering PSF independently at each wavelength, which necessitates time-consuming hyperspectral scanning (one exposure per wavelength) and is impractical in real-world applications.

**Goal**: (1) How to prove that S4 is more discriminative than spectrum or scattering features alone? (2) How to efficiently acquire S4 information without performing a complete hyperspectral scan? (3) How to design a practical imaging system to achieve this goal?

**Key Insight**: The authors observe that the complete S4 PSF is a 3D function (space x × space y × wavelength λ), but not all dimensional information is useful for classification. By choosing an appropriate 2D projection method, a 2D slice of the S4 PSF can be captured in a single shot while retaining sufficient discriminative information.

**Core Idea**: Use a point light source to illuminate the material surface and employ a spectrally-dispersive camera (prism/grating) to simultaneously capture a 2D projection of the spatial and spectral subsurface scattering information in a single shot, replacing time-consuming hyperspectral scanning.

## Method

### Overall Architecture

The method consists of three parts: (1) Theoretical analysis—proving that the 2D projection of the S4 PSF is sufficient for material classification; (2) Hardware design—building an imaging system consisting of a point light source and a spectrally-dispersive camera; (3) Classification algorithm—performing material identification based on the acquired 2D S4 projection.

The input is a single image taken by the spectrally-dispersive camera under point light illumination of the material sample, where spatial scattering information is encoded along one axis and spectral information is encoded along the other axis. The output is the predicted material class.

### Key Designs

1. **2D Projection of S4 PSF**:

    - Function: To prove that the complete 3D S4 data is not required, and a carefully selected 2D projection can retain the discriminative information of materials.
    - Mechanism: The complete S4 PSF is $h(x, y, \lambda)$, which describes the scattering response at spatial location $(x, y)$ and wavelength $\lambda$. For isotropic materials (most natural materials), the PSF exhibits radial symmetry and can be simplified to $h(r, \lambda)$, where $r = \sqrt{x^2 + y^2}$. Therefore, a 1D slice integrated along one spatial direction $\int h(x, y, \lambda)dy$ retains the complete relationship between distance and wavelength. This 2D projection (spatial distance × wavelength) is essentially a "scattering distance-wavelength spectrogram", and the spectrograms of different materials differ significantly.
    - Design Motivation: Complete 3D S4 measurement requires wavelength-by-wavelength scanning, whereas the 2D projection can be acquired at once through optical dispersion, reducing the acquisition time from minutes to milliseconds.

2. **Spectrally-dispersing Imaging Setup**:

    - Function: To simultaneously capture the spatial distribution and spectral information of subsurface scattering in a single shot.
    - Mechanism: The system consists of two parts: (a) a focused point light source (such as a laser or focused LED) that illuminates a precise point on the material surface; (b) a camera with a dispersive element (prism or diffraction grating) placed in front of the camera sensor. The dispersive element disperses the subsurface scattered light from the material surface by wavelength—where scattered light of different wavelengths shifts by different distances along the dispersion direction. Therefore, the image on the sensor naturally encodes: wavelength information along the dispersion direction and spatial scattering distance information along the perpendicular direction. This precisely corresponds to the theoretically derived 2D projection.
    - Design Motivation: Traditional methods use filters or spectrometers to collect wavelength-by-wavelength, which is inefficient. The dispersive element performs "wavelength dispersion" at the optical level, achieving the goal of capturing the 2D S4 projection in a single exposure.

3. **S4-based Material Classifier**:

    - Function: To extract discriminative features from the acquired 2D S4 projection image and perform material classification.
    - Mechanism: Taking the 2D S4 image captured by the spectrally-dispersive camera as input, a CNN or simple feature extraction + classifier can be directly used for material identification. The paper compares schemes using raw S4 projections directly, extracting handcrafted features (such as scattering radius and attenuation coefficient at certain wavelengths), and learned features. Since the S4 projection itself contains rich joint spatial-spectral information, even simple classifiers can achieve excellent results.
    - Design Motivation: Verifying the discriminative power of the S4 measurement itself is the core goal. Choosing a simple classification method eliminates the interference of complex model designs and showcases the value of S4 features more purely.

### Loss & Training

Standard cross-entropy classification loss is used. The training and test sets are split at the material sample level to ensure a reliable evaluation of generalization.

## Key Experimental Results

### Main Results

A variety of common materials (including translucent materials such as plastic, silicone, wax, soap, cheese, fruit) were collected for testing.

| Feature Type | Classification Accuracy | Description |
|----------|-----------|------|
| RGB Appearance Only | ~65% | Color/texture differences are not discriminative enough |
| Spectrum Only | ~78% | Surface reflection spectrum has some discriminative power |
| Subsurface Scattering (SS) Only | ~72% | Single-wavelength scattering pattern has limited discriminative power |
| **S4 (2D Projection)** | **~92%** | Joint spatial-spectral features have the strongest discriminative power |
| S4 (Full 3D Scan) | ~94% | Complete data is slightly better, but the gap is extremely small |

### Ablation Study

| Configuration | Classification Accuracy | Description |
|------|-----------|------|
| Complete S4 2D Projection | ~92% | Full 2D projection |
| Short Wavebands Only | ~85% | Blue/green light regions also have discriminative power |
| Long Wavebands Only | ~83% | Red/near-infrared regions also have discriminative power |
| Reduced Spatial Resolution | ~88% | Shows certain tolerance to spatial resolution |
| Reduced Spectral Resolution | ~86% | Coarse spectral resolution can also retain most information |

### Key Findings

- The discriminative power of S4 features far exceeds that of spectrum or subsurface scattering alone, showing extreme complementarity.
- The classification accuracy gap between the 2D projection and the full 3D S4 scan is only about 2%, proving the effectiveness of the projection scheme.
- Different materials exhibit significantly different "scattering distance-wavelength" patterns on the S4 projection image, offering intuitive interpretability.
- The method can effectively distinguish between materials that look highly similar but have different internal structures (e.g., different brands of silicone).

## Highlights & Insights

- **Combining spectrum and subsurface scattering into S4 reflects an essential understanding of material perception**. This is not merely feature concatenation—the subsurface scattering behavior of light is inherently a wavelength-dependent physical process, and S4 is the most complete description of this physical process. This approach of "returning to physical nature" is highly exemplary.
- **The hardware design of using optical dispersion to acquire the S4 2D projection in a single shot is highly ingenious**. It does not require a complex hyperspectral camera; a simple prism/grating is sufficient to achieve joint spectral-spatial encoding. This low-cost solution makes S4 imaging highly practical for deployment.
- **The combination of theory and experiment is very solid**, first proving the sufficiency of the 2D projection mathematically, then validating its feasibility through hardware, and finally demonstrating its effectiveness via experiments.

## Limitations & Future Work

- Only applicable to translucent materials or materials where subsurface scattering is visible; not applicable to completely opaque or highly specular materials.
- The point light source illumination requires proximity to the material surface, which limits its application in long-range material perception.
- The variety of materials tested in the experiments is relatively limited, and scalability has not yet been validated on large-scale material databases.
- The calibration of the spectrally-dispersing imaging system (prism dispersion parameters, space-wavelength mapping) must be precise, and the robustness analysis against calibration errors is insufficient.
- Future work can combine deep learning to directly learn end-to-end material attribute predictions (such as elastic properties, transparency, and other continuous attributes) from the S4 projection, rather than just performing classification.
- Multi-modal schemes combining S4 with ordinary RGB can be explored, which may be more practical in real-world scenarios.

## Related Work & Insights

- **vs. Spectral Imaging Methods**: Traditional spectral imaging only utilizes surface reflection information, losing the propagation behavior of light inside the material. S4 provides a richer material description by simultaneously encoding both spatial and spectral scattering information.
- **vs. BRDF/BSSRDF Measurement**: Completing a full BSSRDF (Bidirectional Subsurface Scattering Reflectance Distribution Function) measurement requires a massive mapping of varying incident and outgoing directions. S4 simplifies this practically by fixing the direction and only varying the wavelength.
- **vs. Polarization Imaging**: Polarization information can also be used for material classification, but it captures a different dimension of information. A joint S4 + polarization scheme could be considered in the future.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The S4 concept and its optical acquisition scheme are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ The discriminative advantages of S4 are thoroughly validated, though the variety of materials and scenes could be further enriched.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivation is clear, and the experimental design is reasonable.
- Value: ⭐⭐⭐⭐ It opens up a new dimension for material perception, holding potential value for computational photography and robotic tactile/visual fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification](../../CVPR2025/human_understanding/hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)
- [\[ECCV 2024\] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis](freemotion_a_unified_framework_for_number-free_text-to-motion_synthesis.md)
- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [\[ECCV 2024\] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos](videoclusternet_self-supervised_and_adaptive_face_clustering_for_videos.md)
- [\[ECCV 2024\] Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)

</div>

<!-- RELATED:END -->
