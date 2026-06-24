---
title: >-
  [Paper Note] SpectraM-PS: Spectrally Multiplexed Photometric Stereo Under Unknown Spectral Composition
description: >-
  [ECCV 2024 (Oral)][3D Vision][Photometric Stereo] A spectrally multiplexed photometric stereo method (SpectraM-PS) is proposed that eliminates the need for physical model constraints. Under conditions where the spectral composition of the light source is completely unknown, it recovers surface normals from a single RGB image in a data-driven manner, achieving a breakthrough from traditional multi-shot photometric stereo to single-shot.
tags:
  - "ECCV 2024 (Oral)"
  - "3D Vision"
  - "Photometric Stereo"
  - "Spectral Multiplexing"
  - "Normal Estimation"
  - "Unknown Spectrum"
  - "Single-shot"
date: 2026-05-08
content_hash: 0da0bac05cae2bc8
---

# SpectraM-PS: Spectrally Multiplexed Photometric Stereo Under Unknown Spectral Composition

**Conference**: ECCV 2024 (Oral)  
**Institution**: National Institute of Informatics (NII), Japan
**Code**: None  
**Area**: Others  
**Keywords**: Photometric Stereo, Spectral Multiplexing, Normal Estimation, Unknown Spectrum, Single-shot

## TL;DR

A spectrally multiplexed photometric stereo method (SpectraM-PS) is proposed that eliminates the need for physical model constraints. Under conditions where the spectral composition of the light source is completely unknown, it recovers surface normals from a single RGB image in a data-driven manner, achieving a breakthrough from traditional multi-shot photometric stereo to single-shot.

## Background & Motivation

**Background**: Photometric Stereo (PS) is a classic 3D reconstruction method that recovers surface normals by capturing multiple images under different lighting directions. Traditional PS requires at least 3 images under different lighting directions and requires known light directions and intensities. In recent years, deep learning-based PS methods (such as PS-Transformer, SDM-UniPS, etc.) have significantly reduced the requirements for lighting calibration but still require multiple input images. Spectrally Multiplexed PS is a method that sets multiple light sources from different directions to different colors (e.g., red, green, blue), turns on all three light sources simultaneously, and captures a single image using an RGB camera. Theoretically, this reduces the number of acquisitions required for PS from multiple times to a single shot.

**Limitations of Prior Work**: Existing spectrally multiplexed PS methods rely on a strict "narrowband assumption" — assuming that the three RGB channels are completely separated spectrally and each channel only responds to the light source of the corresponding color. However, in real-world scenarios, the spectral distribution of LED light sources is broadband, the spectral response functions of RGB cameras overlap, and the spectral reflectance of materials is wavelength-dependent. These factors lead to severe "spectral crosstalk" between channels, meaning that each color channel actually blends information from multiple light source directions. Traditional methods correct the crosstalk matrix by calibrating the light source spectrum and camera response, but this requires precise hardware calibration, which is difficult in practice and generalizes poorly.

**Key Challenge**: The core advantage of spectrally multiplexed PS is single-shot acquisition, but accurate demultiplexing requires precise knowledge of the spectral composition (light source spectrum $\times$ camera response $\times$ material reflectance), which is difficult to obtain in real scenarios. The core challenge is how to perform accurate normal estimation without knowing the spectral composition.

**Goal**: (1) How to perform spectrally multiplexed PS under completely unknown light source spectral compositions? (2) How to design a normal estimation method robust to spectral crosstalk? (3) How to handle complex reflections from non-Lambertian surfaces?

**Key Insight**: The author is a senior researcher in the field of photometric stereo (author of CNN-PS, SDM-UniPS, etc.). He observed that instead of trying to precisely model and correct the physical process of spectral crosstalk, a data-driven approach can be used to let the network automatically learn the mapping from spectrally multiplexed images to normals, completely bypassing the dependence on spectral composition. This is the core concept of being "Physics-Free".

**Core Idea**: Directly predict surface normals from a single spectrally multiplexed RGB image using a data-driven deep learning method without requiring knowledge of the light source spectra, camera response, or material spectral reflectance.

## Method

### Overall Architecture

During the capture stage of the system, three light sources of different colors (color LEDs) and different directions are used to simultaneously illuminate the target object, and a single image is captured using a standard RGB camera. The three channels of this RGB image each encode mixed lighting information from different light source directions. During the inference stage, this RGB image is fed into an end-to-end CNN network to directly output the surface normal vector for each pixel.

The key difference from traditional methods is that it does not require calibrating the light source spectra, calibrating the camera's spectral response, or knowing the spectral reflectance of the object. The network is trained on large-scale synthetic data to learn the direct mapping from multiplexed images to normals.

### Key Designs

1. **Physics-Free End-to-End Network**:

    - **Function**: Directly predict pixel-wise surface normals from a single spectrally multiplexed RGB image.
    - **Mechanism**: Uses a U-Net-like encoder-decoder architecture. The encoder extracts multi-scale features, and the decoder generates pixel-wise normal predictions. The key innovation is that the network is trained using a large amount of synthetic data with varying spectral compositions, including different light source spectral distributions, different camera response functions, and different material spectral reflectance combinations. Through this diverse training data, the network implicitly learns robustness to spectral crosstalk—it no longer relies on specific spectral calibration parameters but instead learns a feature representation invariant to spectral composition changes.
    - **Design Motivation**: Traditional physical modeling methods require precise spectral parameter settings; any deviation between the actual hardware and calibration leads to significant errors. Data-driven methods naturally generalize by covering these variations with sufficiently diverse training data.

2. **Spectral Diversity Augmentation**:

    - **Function**: Generate training data with rich spectral variations to enhance the network's generalization ability to unknown spectra.
    - **Mechanism**: When synthesizing training data, different light source LED spectral distribution functions are randomly sampled (including Gaussian distributions with different peak wavelengths and bandwidths), different RGB camera spectral response functions are randomly sampled (simulating different camera models), and the spectral reflectance of materials is randomly sampled (taking spectral dimension data from BRDF databases like MERL). During rendering, the radiant power received by each channel is calculated according to the physical process: $I_c = \int S_c(\lambda) \cdot L_i(\lambda) \cdot \rho(\lambda) \cdot f(\mathbf{n}, \mathbf{l}_i) d\lambda$, where $S_c$ is the spectral response of camera channel $c$, $L_i$ is the spectral distribution of light source $i$, $\rho$ is the material spectral reflectance, and $f$ is the BRDF term. By generating a large number of such training samples with different spectral combinations, the network is forced to learn the normal recovery capability independent of specific spectral configurations.
    - **Design Motivation**: The network needs to "see" enough spectral variations to generalize to unknown spectral compositions at test time. This domain randomization strategy at the data level is simpler and more effective than designing spectral-invariant features at the network level.

3. **Non-Lambertian Surface Handling**:

    - **Function**: Enable the network to process real surfaces with non-Lambertian effects such as highlights and shadows.
    - **Mechanism**: Multiple BRDF models (not limited to Lambertian diffuse reflection), including Cook-Torrance specular reflection and rough surface models, are incorporated into the training data rendering. By training under diverse BRDF conditions, the network learns to automatically identify and handle specular regions (where color information reflects more of the light source color rather than the material color) and self-shadowed regions (where light sources from certain directions are occluded). Additionally, the network's multi-scale feature extraction capability allows it to utilize local contextual information to compensate for the lack of single-point spectral information.
    - **Design Motivation**: Almost all real object surfaces are non-Lambertian, and highlights and shadows severely interfere with the interpretation of spectrally multiplexed signals. Traditional methods require explicit detection and exclusion of these anomalous regions, whereas data-driven methods can learn to handle them end-to-end.

### Loss & Training

The cosine similarity loss $\mathcal{L} = 1 - \frac{\mathbf{n}_{pred} \cdot \mathbf{n}_{gt}}{|\mathbf{n}_{pred}||\mathbf{n}_{gt}|}$ is used as the primary loss to directly optimize the angular error between the predicted normals and ground truth normals. Training employs synthetic data, rendering a large number of surface normal-image pairs under various spectral configurations.

## Key Experimental Results

### Main Results

Evaluated on standard photometric stereo benchmark datasets such as DiLiGenT using real images captured with colored LED light sources.

| Method | Input | Mean Angular Error (MAE°) | Spectral Calibration | Description |
|------|------|-------------------|----------|------|
| Classic PS (3 images) | 3 grayscale images | ~8-12° | Required | Requires 3 shots |
| Calibrated Spectrally Multiplexed PS | 1 RGB | ~10-15° | Required | Requires precise calibration |
| Uncalibrated Spectrally Multiplexed PS (Prior) | 1 RGB | ~18-25° | Not required but poor accuracy | Assumptions too strong |
| **SpectraM-PS (Ours)** | **1 RGB** | **~7-10°** | **Not required** | Best single-shot performance |
| SDM-UniPS (10 images) | 10 grayscale images | ~5-7° | Not required | Requires multi-shot |

### Ablation Study

| Configuration | MAE(°) | Description |
|------|--------|------|
| Full model (Diverse Spectra + Non-Lambertian) | Best | Complete training strategy |
| Trained with fixed spectrum only | Increases by 3-5° | Lacks spectral generalization ability |
| Trained with Lambertian surfaces only | Increases by 2-3° | Cannot handle highlights |
| Reduced training spectral diversity | Gradually increases | Lower diversity yields worse accuracy |
| Different LED color configurations | Minimal difference | Insensitive to LED selection |

### Key Findings

- SpectraM-PS achieves accuracy close to or even exceeding traditional PS methods requiring 3 images using only a single image, while eliminating any need for spectral calibration.
- Spectral diversity data augmentation is the key factor for performance; training with only a fixed spectrum leads to a significant drop in generalization capability.
- The method is insensitive to the choice of LED colors actually used (as long as the three colors are not completely identical), demonstrating good practicality.
- As an ECCV 2024 Oral paper, it has received high recognition from the academic community.

## Highlights & Insights

- **The "Physics-Free" concept is very bold and effective**. In the field of photometric stereo, which traditionally relies heavily on physical modeling, completely letting go of physical model constraints and turning to a data-driven approach to absorb physical complexity represents a paradigm shift. This imposes very high demands on training data diversity, which the authors elegantly address through spectral domain randomization.
- **The spectrally multiplexed approach, which translates a multi-shot problem into a single-shot problem**, shares the same philosophy as "replacing multiple samplings with coded measurements" in compressive sensing. This idea of joint space-spectral coding can be transferred to other computational photography tasks that require multiple measurements.
- The author Ikehata has been deeply involved in the PS field for many years (CNN-PS @ ECCV 2018, SDM-UniPS @ CVPR 2023), and this work is a natural extension of his research trajectory — from generalizing multi-image PS to implementing single-image PS.

## Limitations & Future Work

- Ultimately, the amount of information provided by a single image is limited, and accuracy may be insufficient near complex shapes and material boundaries.
- The capability to handle interreflections and global illumination effects remains to be verified.
- It requires three color light sources to illuminate from different directions simultaneously, which is less flexible in hardware setup than methods using ambient light.
- The current method assumes that light source directions are roughly known (although spectra are unknown); relaxing the light source direction constraint is the next step.
- This method can be combined with multi-frame information (e.g., continuous frames in video) to further improve accuracy, achieving a temporal extension of spectrally multiplexed PS.
- Future work could explore combining this method with NeRF/3DGS to achieve single-shot 3D reconstruction and material estimation.

## Related Work & Insights

- **vs CNN-PS (Same author, ECCV 2018)**: CNN-PS was the first work to use CNNs for general PS but still required multiple input images. SpectraM-PS reduces the input from multiple images to one through spectral multiplexing, making it a major methodological advance in PS.
- **vs SDM-UniPS (Same author, CVPR 2023)**: SDM-UniPS achieved general PS for arbitrary light source numbers and directions but still required multiple photos. SpectraM-PS approaches the accuracy of multi-image methods under a single-shot constraint.
- **vs Traditional Spectrally Multiplexed PS**: Traditional methods (such as the pioneering work by Hernández et al.) require precise spectral calibration, whereas SpectraM-PS completely removes this constraint, greatly enhancing practicality.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pushes PS from multi-shot to single-shot without spectral calibration, offering a strong paradigm breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated effectiveness on standard benchmarks, with relatively complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problem and solid motivation, ensuring the quality expected of an Oral paper.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the fields of photometric stereo and computational photography, with high practical potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PS-EIP: Robust Photometric Stereo Based on Event Interval Profile](../../CVPR2025/3d_vision/ps-eip_robust_photometric_stereo_based_on_event_interval_profile.md)
- [\[AAAI 2026\] Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues](../../AAAI2026/3d_vision/geometry_meets_light_leveraging_geometric_priors_for_universal_photometric_stere.md)
- [\[ICLR 2026\] Light of Normals: Unified Feature Representation for Universal Photometric Stereo](../../ICLR2026/3d_vision/light_of_normals_unified_feature_representation_for_universal_photometric_stereo.md)
- [\[ECCV 2024\] TC-Stereo: Temporally Consistent Stereo Matching](temporally_consistent_stereo_matching.md)
- [\[ECCV 2024\] Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](deblur_e-nerf_nerf_from_motion-blurred_events_under_high-speed_or_low-light_cond.md)

</div>

<!-- RELATED:END -->
