---
title: >-
  [Paper Note] GPS as a Control Signal for Image Generation
description: >-
  [CVPR 2025][Image Generation][GPS conditioning] By using GPS coordinates from photo EXIF metadata as a new control signal for diffusion models, a joint GPS+text conditioned image generation model is trained. It can capture fine-grained architectural and appearance variations across different neighborhoods or landmarks within a city, and perform 3D landmark reconstruction extracted from 2D models via angle-conditioned SDS.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "GPS conditioning"
  - "diffusion model"
  - "geotagged photos"
  - "NeRF"
  - "score distillation sampling"
  - "compositional generation"
date: 2026-05-08
content_hash: a348e732530c5299
---

# GPS as a Control Signal for Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2501.12390](https://arxiv.org/abs/2501.12390)  
**Code**: [Project Page](https://cfeng16.github.io/gps-gen/)  
**Area**: Image Generation  
**Keywords**: GPS conditioning, diffusion model, geotagged photos, NeRF, score distillation sampling, compositional generation

## TL;DR

By using GPS coordinates from photo EXIF metadata as a new control signal for diffusion models, a joint GPS+text conditioned image generation model is trained. It can capture fine-grained architectural and appearance variations across different neighborhoods or landmarks within a city, and perform 3D landmark reconstruction extracted from 2D models via angle-conditioned SDS.

## Background & Motivation

**Background**: Diffusion models have widely used conditions such as text, depth maps, semantic masks, and camera poses for image, video, or 3D generation. Geotags (GPS) are highly abundant but overlooked signals in photo metadata.

**Limitations of Prior Work**:
- Text conditioning cannot precisely control geotagged features of a scene (e.g., architectural styles in specific neighborhoods).
- Traditional SfM $\rightarrow$ NeRF pipelines easily collapse due to pose estimation failures on unstructured tourist photos.
- Existing GPS-to-image works are limited to satellite images and require calibrated training data.

**Key Challenge**: GPS coordinates contain rich implicit visual priors (landmark locations, architectural styles, perspective information), yet current generative models fail to utilize this signal.

**Goal**: To demonstrate that GPS coordinates serve as a useful control signal for image generation, and showcase their application in compositional generation and 3D reconstruction.

**Key Insight**: Fine-tuning pre-trained Stable Diffusion by encoding GPS coordinates and appending them to CLIP text embeddings as additional conditional tokens.

**Core Idea**: GPS coordinates provide geographic priors complementary to text, not only enabling generative models to capture fine-grained geographic variations within a city but also providing implicit perspective supervision for 3D reconstruction.

## Method

### Overall Architecture

1. **Data Collection**: Collect geotagged tourist photos from Flickr (500k for Manhattan, 310k for Paris).
2. **GPS-to-Image Diffusion Model**: Fine-tune on SD v1.4 to jointly condition on GPS and text.
3. **Angle-to-Image Diffusion Model**: For specific landmarks, use azimuth instead of GPS as conditioning.
4. **GPS-guided 3D Reconstruction**: Extract NeRF from the angle-to-image model via SDS.

### Key Designs

#### 1. GPS Conditioning Encoding

- Normalize GPS coordinates $(x, y)$ (latitude and longitude) to $[-1, 1]$.
- Use positional encoding with a frequency of 10 and a two-layer MLP to encode as $\mathbf{g} = [f(x), f(y)] \in \mathbb{R}^{2 \times D}$.
- Concatenate the GPS embedding **to the end of the CLIP text token sequence** as the "GPS" CLIP text condition.
- Randomly drop conditions during training: 5% text-only, 5% GPS-only, 5% unconditional.

#### 2. Dual-conditioned Classifier-Free Guidance

Inference utilizes an InstructPix2Pix-style dual-conditioned CFG:

$$\tilde{\boldsymbol{\epsilon}}_\phi = \boldsymbol{\epsilon}_\phi(\varnothing, \varnothing) + \omega_{\mathbf{p}}(\boldsymbol{\epsilon}_\phi(\mathbf{p}, \varnothing) - \boldsymbol{\epsilon}_\phi(\varnothing, \varnothing)) + \omega_{\mathbf{g}}(\boldsymbol{\epsilon}_\phi(\mathbf{p}, \mathbf{g}) - \boldsymbol{\epsilon}_\phi(\mathbf{p}, \varnothing))$$

Three forward passes are performed for unconditional, text-only, and text+GPS, where weights $\omega_\mathbf{p}$ and $\omega_\mathbf{g}$ respectively control semantic and geographic guidance strength.

#### 3. GPS-guided 3D Landmark Reconstruction

- Parameterize the GPS as the **azimuth** $\alpha = \arctan\frac{x-x_o}{y-y_o}$ relative to the landmark center.
- Train an angle-to-image diffusion model (individually for each landmark).
- Incorporate a DreamBooth-style **prior preservation loss** to prevent overfitting during fine-tuning.
- Optimize NeRF driven by SDS loss: render a random view at each step $\rightarrow$ calculate the azimuth $\rightarrow$ generate GPS-conditioned image $\rightarrow$ backpropagate SDS gradients.
- GPS conditioning replaces traditional view-dependent prompting, providing more accurate viewpoint priors to avoid the Janus problem.

### Loss & Training

- GPS-to-image training: $\mathcal{L}_{recon} = \mathbb{E}[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\phi(\mathbf{z}_t; \mathbf{p}, \mathbf{g}, t)\|_2^2]$
- 3D landmark reconstruction: $\mathcal{L} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{preservation}$, where $\lambda = 1.0$.

## Key Experimental Results

### Main Results

| Method | CLIP Score ↑ | GPS Score ↑ | Avg ↑ |
|------|-------------|-------------|-------|
| GPS NN | 18.77 | **13.66** | 16.22 |
| SD (Text+Address) | **26.65** | 4.25 | 15.45 |
| SD (Text) | 29.13 | 1.21 | 15.17 |
| **Ours** | 27.88 | 8.15 | **18.02** |
| Ours (w/o text) | – | 13.71 | – |

Our method achieves the best overall performance in terms of CLIP Score and GPS Score.

### 3D Landmark Reconstruction Comparison

| Method | CLIP Score ↑ | PQ ↑ | Tourist Score ↑ |
|------|-------------|------|----------------|
| NeRF (SfM-based) | 20.57 | 1.32 | 1.36 |
| DreamFusion | 29.49 | 2.21 | 2.09 |
| **Ours** | **31.87** | **3.31** | **3.45** |

### Ablation Study

- **Angle-to-Image Azimuth Accuracy**: Ours achieves 22.36% vs. SD (3.06%) vs. Random (2.78%).
- **GPS vs. Text Address**: GPS Score of 8.15 vs. 4.25; continuous GPS coordinates significantly outperform textual addresses.
- **Prior preservation loss**: 3D reconstruction quality degrades significantly upon removal.

### Key Findings

1. GPS and text conditions are highly complementary: text controls semantics, while GPS controls geographic appearance (validated by attention map visualization).
2. Continuous GPS coordinates are significantly superior to discretized address names as conditioning.
3. GPS conditioning effectively mitigates the Janus (multi-faced) problem of DreamFusion.
4. SfM-based methods fail completely on 3 out of 6 landmarks, whereas the GPS-guided SDS succeeds on all of them.
5. Average images (compositional generation) can capture the architectural style of specific neighborhoods.

## Highlights & Insights

1. **Brand-New Control Signal**: Systematically introduces GPS as a conditioning signal for image generation for the first time, opening up a new direction.
2. **Two Birds with One Stone**: The exact same GPS conditioning serves both controllable generation and provides viewpoint priors for 3D reconstruction.
3. **Easily Accessible Data**: Leverages existing GPS tags from photo EXIF metadata without requiring extra annotations, yielding a rich and free signal.
4. **Compositional Generation Power**: Demonstrates compositional semantics of GPS+text, such as generating a "superman" sculpture at MoMA and a cosplay character in Times Square.
5. **Average Image**: Generates a "representative image" of an area by averaging noise predictions across multiple GPS coordinates in that region, capturing local architectural styles.

## Limitations & Future Work

1. Relies heavily on collection of photos with rich GPS annotations, which limits performance in areas with sparse GPS data.
2. 3D models generated via SDS suffer from color over-saturation (an inherent limitation of SDS).
3. The semantic information embedded in GPS tags is difficult to decouple completely from text.
4. Evaluated only on Manhattan and Paris; generalization to other geographic regions remains unverified.
5. Based on SD v1.4, where resolution and quality are limited by the base model; upgrading to more recent base models could yield better results.

## Related Work & Insights

1. **DreamFusion** (Poole et al., 2022): The source of the SDS framework, which this work extends to GPS-guided SDS.
2. **InstructPix2Pix** (Brooks et al., 2023): The source of the dual-conditioned CFG inference strategy.
3. **Snavely et al. (2006)**: Photo Tourism, reconstructed 3D from geotagged photo collections, a classic pioneering work.
4. **DreamBooth** (Ruiz et al., 2023): Prior preservation loss to prevent overfitting during fine-tuning.

**Insights**: Photo metadata (not only GPS, but also timestamps, camera intrinsic/extrinsic parameters, etc.) may contain more unexploited control signals. The GPS-to-image concept can be extended to street view generation, autonomous driving scene synthesis, and other applications.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — A completely brand-new control signal, where defining the problem itself is a contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers two tasks (generation and 3D reconstruction) with qualitative, quantitative evaluation and user studies, though the number of validated cities is small.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear illustrations with highly impressive visual results.
- **Value**: ⭐⭐⭐ — Well-defined but relatively niche application scenarios (tourist photos/landmark reconstruction).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)
- [\[AAAI 2026\] SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](../../AAAI2026/image_generation/saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](mca_ctrl_attention_control_customization.md)

</div>

<!-- RELATED:END -->
