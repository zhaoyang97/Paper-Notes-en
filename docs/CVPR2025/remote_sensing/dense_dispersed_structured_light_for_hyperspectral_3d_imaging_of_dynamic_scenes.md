---
title: >-
  [Paper Note] Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes
description: >-
  [CVPR 2025][Remote Sensing][hyperspectral 3D imaging] This paper proposes the Dense Dispersed Structured Light (DDSL) method, which utilizes an inexpensive diffraction grating film (<\$20), a stereo RGB camera, and an RGB projector. By designing spectrally multiplexed DDSL patterns, the required number of projection frames is significantly reduced, achieving real-time hyperspectral 3D imaging at 6.6 fps with a spectral resolution of 15.5 nm FWHM and a depth error of 4 mm.
tags:
  - "CVPR 2025"
  - "Remote Sensing"
  - "hyperspectral 3D imaging"
  - "structured light"
  - "diffraction grating"
  - "stereo cameras"
  - "dynamic scenes"
date: 2026-05-08
content_hash: eb47dce56fd2b23b
---

# Dense Dispersed Structured Light for Hyperspectral 3D Imaging of Dynamic Scenes

**Conference**: CVPR 2025  
**arXiv**: [2412.01140](https://arxiv.org/abs/2412.01140)  
**Code**: —  
**Area**: Remote Sensing / Computational Imaging  
**Keywords**: hyperspectral 3D imaging, structured light, diffraction grating, stereo cameras, dynamic scenes

## TL;DR
This paper proposes the Dense Dispersed Structured Light (DDSL) method, which utilizes an inexpensive diffraction grating film (<\$20), a stereo RGB camera, and an RGB projector. By designing spectrally multiplexed DDSL patterns, the required number of projection frames is significantly reduced, achieving real-time hyperspectral 3D imaging at 6.6 fps with a spectral resolution of 15.5 nm FWHM and a depth error of 4 mm.

## Background & Motivation

**Background**: Hyperspectral 3D imaging simultaneously captures depth maps and hyperspectral images, enabling comprehensive analysis of geometry and materials. Recent methods have progressed in spectral and depth accuracy.

**Limitations of Prior Work**:
   - **Long acquisition time**: Existing methods typically require several minutes and can only capture static scenes.
   - **Expensive and bulky hardware**: They rely on professional hyperspectral cameras or complex optical systems.
   - **Inability to handle dynamic scenes**: Slow acquisition leads to motion blur and temporal inconsistencies.

**Key Challenge**: The hyperspectral dimension itself contains a massive volume of information, and traditional methods require numerous projections/acquisitions at different wavelengths; however, dynamic scenes demand extremely fast acquisition speeds.

**Key Insight**: Utilize a diffraction grating to disperse the broadband light from an RGB projector into spectral components, encoding multiple spectral channels with a few projection frames through meticulously designed multiplexed patterns.

**Core Idea**: Diffraction grating + spectrally multiplexed patterns + image formation model = low-cost, real-time hyperspectral 3D imaging.

## Method

### Hardware Configuration
- **RGB projector**: A standard DLP projector projecting designed DDSL patterns.
- **Diffraction grating film**: Placed in front of the projector lens, with a cost of <\$20.
    - Function: Disperses the projected broadband structured light into different wavelength components.
    - Each projected pixel forms a "rainbow" stripe on the scene.
- **Stereo RGB camera**: Two standard RGB cameras capturing the scene from different viewpoints.

### Key Designs

1. **Spectrally Multiplexed DDSL Pattern Design**

    - Function: Significantly reduces the required number of projection frames.
    - Mechanism:
        - Traditional methods require independent projections for each wavelength, needing 20-30 frames for hyperspectral imaging.
        - DDSL leverages the spatial dispersion property of the diffraction grating to encode multiple spectral channels in a single frame.
        - It designs orthogonal or quasi-orthogonal spatial patterns so that projections of different wavelengths do not interfere with each other on the scene.
    - Key constraints: Ensure that dispersed light of different wavelengths does not spatially overlap (using offset encoding).
    - Effect: Requires only ~3-5 projection frames to cover all spectral channels.

2. **Image Formation Model**

    - Function: Establishes the physical model relating projected patterns, the diffraction grating, scene reflectance, and camera observations.
    - Core formula:
        - The light emitted from the projector is spatially dispersed by wavelength after passing through the diffraction grating.
        - The dispersion angle is determined by the grating equation: $d \sin\theta = m\lambda$
        - The light reflected from each scene point contains its spectral reflectance information.
        - The RGB camera observation is the integration of the spectral reflectance and the camera response function.
    - Function: Inverts the observed RGB images into hyperspectral images.

3. **Hyperspectral and Depth Reconstruction Algorithm**

    - Function: Recovers hyperspectral images and depth maps from stereo RGB observations.
    - Depth estimation: Uses disparity from the stereo RGB cameras for traditional stereo matching.
    - Spectral reflectance recovery:
        - Establishes a system of linear equations based on the image formation model.
        - Solves for spectral reflectance using regularized optimization.
        - Leverages redundant information from multi-frame projections to improve robustness.
    - Joint optimization: Alternately optimizes depth and spectral reflectance until convergence.

4. **Fast Acquisition Pipeline**

    - The projector projects the designed DDSL pattern sequence at a high frame rate (~20fps).
    - Two RGB cameras capture synchronously.
    - Every 3-5 frames constitute a complete hyperspectral sampling cycle.
    - Effective frame rate: 6.6 fps.

### Spectral Multiplexing Principle (Geometric Intuition)
- The diffraction grating disperses white light into a rainbow: red light has a larger deflection angle, and blue light has a smaller deflection angle.
- Projecting pixels at specific spatial locations causes different wavelengths to fall on different locations in the scene after passing through the grating.
- By precisely designing the spatial distribution of the projection patterns, the illumination position of each wavelength is controlled.
- The camera side decodes spectral information based on the known geometric relationships.

## Key Experimental Results

### System Performance Metrics

| Metric | Value |
|------|------|
| Spectral Resolution (FWHM) | 15.5 nm |
| Depth Error | 4 mm |
| Frame Rate | 6.6 fps |
| Grating Cost | <\$20 |
| Spectral Range | Visible light (~400-700nm) |

### Comparison with Existing Methods

| Method | Frame Rate | Acquisition Time | Cost | Dynamic Scene |
|------|------|---------|------|---------|
| Traditional Hyperspectral + SL | Extremely low | Several minutes | High | ✗ |
| Filter wheel methods | Low | Tens of seconds | Medium | ✗ |
| Coded aperture | Medium | Several seconds | High | Limited |
| **DDSL (Ours)** | **6.6 fps** | **Real-time** | **Low** | **✓** |

### Spectral Reconstruction Accuracy

| Evaluation | Metric |
|------|------|
| RMSE (spectral reflectance) | Low |
| SAM (spectral angle mapper) | Low |
| Consistency with reference spectrometer | High |

### Key Findings
- The loss of light efficiency introduced by the diffraction grating film is acceptable.
- The spectrally multiplexed patterns reduce the number of projection frames by approximately 5 to 6 times.
- The depth accuracy is comparable to traditional stereo matching methods (4mm).
- The 15.5nm spectral resolution is sufficient to distinguish common materials.

## Highlights & Insights
- **Extremely low-cost**: Only requires a diffraction grating film costing <\$20 and standard RGB equipment.
- **Real-time and dynamic**: The first practical hyperspectral 3D imaging method for dynamic scenes.
- **Physics-model-driven**: Based on a precise imaging model of diffractive optics, rather than being purely data-driven.
- **Simple hardware**: Can be built using off-the-shelf projectors and cameras.

## Limitations & Future Work
- The spectral resolution (15.5nm) is lower than that of professional hyperspectral instruments (<5nm).
- The working range is limited to visible light; near-infrared would require a different grating.
- Projection distance and ambient light affect system performance.
- The depth accuracy is restricted by the stereo baseline and resolution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of diffraction grating and structured light is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Real system implementation and multi-scene testing.
- Writing Quality: ⭐⭐⭐⭐ Detailed and clear derivation of the physical model.
- Value: ⭐⭐⭐⭐⭐ Low-cost real-time hyperspectral 3D has broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)
- [\[CVPR 2025\] SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion](sgformer_satellite-ground_fusion_for_3d_semantic_scene_completion.md)
- [\[CVPR 2026\] Semantic-Adaptive Diffusion for Dynamic Spatiotemporal Fusion](../../CVPR2026/remote_sensing/semantic-adaptive_diffusion_for_dynamic_spatiotemporal_fusion.md)
- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](../../NeurIPS2025/remote_sensing/greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)
- [\[CVPR 2026\] UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes](../../CVPR2026/remote_sensing/unigeoseg_towards_unified_open-world_segmentation_for_geospatial_scenes.md)

</div>

<!-- RELATED:END -->
