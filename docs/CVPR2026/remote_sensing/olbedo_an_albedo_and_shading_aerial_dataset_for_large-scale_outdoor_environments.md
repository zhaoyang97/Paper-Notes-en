---
title: >-
  [Paper Note] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments
description: >-
  [CVPR 2026][Remote Sensing][Intrinsic image decomposition] Olbedo introduces the first large-scale real-world aerial albedo–shading decomposition dataset (5,664 UAV images, 4 terrain types…
tags:
  - "CVPR 2026"
  - "Remote Sensing"
  - "Intrinsic image decomposition"
  - "albedo"
  - "aerial dataset"
  - "inverse rendering"
  - "urban digital twin"
date: 2026-05-08
content_hash: 3ed64f94efc3a17d
---

# Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments

**Conference**: CVPR 2026
**arXiv**: [2602.22025](https://arxiv.org/abs/2602.22025)  
**Code**: [Project](https://gdaosu.github.io/olbedo/)  
**Area**: Remote Sensing
**Keywords**: Intrinsic image decomposition, albedo, aerial dataset, inverse rendering, urban digital twin

## TL;DR

Olbedo introduces the first large-scale real-world aerial albedo–shading decomposition dataset (5,664 UAV images, 4 terrain types, multi-year multi-illumination conditions). A physics-based inverse rendering pipeline generates multi-view-consistent pseudo-ground-truth annotations. Results demonstrate that synthetic pre-training combined with Olbedo LoRA fine-tuning substantially improves outdoor albedo prediction and supports downstream applications including relighting, material editing, and scene change analysis.

## Background & Motivation

**Background**: Intrinsic image decomposition (IID) aims to separate an image into albedo $R$ and shading $S$ such that $I = R \cdot S$. This decomposition underpins relighting, material editing, and 3D content creation. Deep learning has significantly advanced IID, progressing from CNNs (NIID-Net) to diffusion models (Intrinsic Image Diffusion, RGB↔X, Marigold-IID); however, these advances have been almost entirely confined to **indoor or synthetic environments**.

**Limitations of Prior Work**: The primary obstacle to real-world outdoor IID is the **absence of large-scale densely annotated ground-truth datasets**. Existing datasets fall into three categories:
   - **Synthetic datasets** (MPI Sintel, CGIntrinsics, InteriorVerse, Hypersim): provide perfect ground truth but suffer from a sim-to-real gap, and are predominantly indoor or object-level.
   - **Controlled real-world datasets** (MIT Intrinsics, MID, DIR): compute pseudo-ground-truth under multiple illumination conditions, but are limited to small-scale indoor scenes.
   - **In-the-wild real datasets** (IIW): contain real images but only sparse relative judgments, without dense albedo maps.

   **Critical Gap**: No large-scale, real-world, densely annotated outdoor aerial IID dataset exists.

**Key Challenge**: Outdoor scenes are subject to dynamic and complex illumination (sun + sky), making it impossible to control lighting as in indoor settings to obtain ground truth. An alternative strategy is required to generate reliable outdoor albedo supervision signals.

**Goal**: Construct the first large-scale real-world aerial albedo–shading dataset providing dense pseudo-ground-truth supervision, enabling existing IID models to adapt to outdoor aerial scenes.

**Key Insight**: Photogrammetric reconstruction provides accurate 3D geometry (normals, depth, shadow maps); combined with astronomical ephemeris for solar position and a physics-based sun–sky illumination model, albedo is estimated from lit–shadow pixel pairs near shadow boundaries—yielding physically consistent albedo annotations without controlled illumination.

**Core Idea**: A physics-based inverse rendering pipeline generates dense albedo pseudo-ground-truth from multi-view aerial RAW images, forming the first outdoor aerial IID dataset. Models pre-trained on synthetic data can then adapt to real outdoor scenes via lightweight fine-tuning.

## Method

### Overall Architecture

RAW-format UAV acquisition → Photogrammetric reconstruction (Bentley iTwin Capture, ~3 cm accuracy) → Normal/depth/shadow map extraction → HDR full-sky radiance capture (2025 flights, fisheye camera, 11-bracket exposure) → Inverse rendering optimization under a sun–sky illumination model → Per-image albedo/shading maps + confidence masks

### Key Designs

1. **Physics-Based Inverse Rendering Albedo Decomposition**:

    - **Function**: Separate albedo and shading from multi-view aerial images captured at a single point in time.
    - **Mechanism**: Image formation model $I = R \odot (\phi \odot S_{sun} + S_{sky})$, where $S_{sun} = V_{sun} \cdot \langle \mathbf{n}, \theta_{sun} \rangle^+$ is sun shading (dependent on surface normal and sun visibility), $S_{sky}$ is sky shading (hemispherical integral), and $\phi = \psi_{sun}/\psi_{sky}$ is the sun-to-sky irradiance ratio. The key assumption is that lit–shadow pixel pairs near shadow boundaries share the same albedo, from which $\phi$ is derived: $\frac{I_{lit} - I_{shadow}}{I_{shadow}} = \frac{\phi \odot S_{sun}}{S_{sky}}$. A Gaussian mixture model (two components: signal + noise) estimates the optimal $\phi$ over all lit–shadow pairs across the image; albedo is then recovered as $R = I / (\phi \odot S_{sun} + S_{sky})$.
    - **Design Motivation**: Exploiting the physical constraints at shadow boundaries eliminates the need for multi-illumination acquisition. The Lambertian diffuse reflectance assumption, while a simplification, is sufficiently valid for aerial scenes dominated by roads, rooftops, and vegetation.

2. **Data Quality Assurance**:

    - **Function**: Ensure reliability of training supervision signals.
    - **Mechanism**: (1) **RAW acquisition**: DJI Phantom 4 Pro captures DNG format, preserving linear radiometric space and avoiding the tonal curve/gamma distortion introduced by JPEG compression. (2) **Confidence masks**: Binary masks are automatically generated from geometric boundaries (reconstruction holes/depth discontinuities) and shadow projection boundaries; supervision is applied only within high-confidence regions during training, preventing unreliable pseudo-ground-truth from overriding pre-trained priors. (3) **Manual filtering**: A cleaner subset of 2.49K images is retained.
    - **Design Motivation**: Inverse rendering annotations fail at geometric holes, shadow edges, specular reflections, and glass transmittance; these regions must be explicitly excluded.

3. **LoRA Lightweight Fine-Tuning Strategy**:

    - **Function**: Fine-tune existing pre-trained IID models using Olbedo.
    - **Mechanism**: LoRA is applied to the Q/K/V/O projections of the U-Net attention modules in diffusion-based models (Intrinsic Image Diffusion, Marigold-IID, RGB↔X), with only ~1.66M trainable parameters (~0.19% of the U-Net). CNN-based models (NIID-Net) are fine-tuned directly. Diffusion models are fine-tuned on the albedo channel only; the text prompt for RGB↔X is fixed to "albedo".
    - **Design Motivation**: Given the relatively small scale of Olbedo (5,664 images), LoRA prevents overfitting while retaining the generalization capability of pre-trained models. Olbedo is positioned as a **complement to synthetic data**, not a replacement.

### Loss & Training

Each model follows its original training pipeline with only hyperparameter adjustments. Diffusion model LoRA: 3 epochs, lr = 5e-6 (IID, Marigold) or 5e-7 (RGB↔X). Confidence masks constrain the supervised regions during training. All images are downsampled 8× to 683×455 for processing. Albedo is normalized by 98th-percentile intensity to maintain chromaticity consistency.

## Key Experimental Results

### Main Results

Albedo prediction accuracy before and after fine-tuning is evaluated on the MatrixCity synthetic outdoor benchmark (520 images at 1920×1080):

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-------|-------|--------|
| Pretrained IntrinsicAnything | 12.155 | 0.436 | 0.564 |
| Pretrained Colorful Diffuse | 10.437 | 0.449 | 0.590 |
| Pretrained NIID-Net | 12.782 | 0.549 | 0.793 |
| **Fine-tuned NIID-Net** | **16.152** | **0.594** | **0.769** |
| Pretrained Intrinsic Image Diffusion | 15.598 | 0.493 | 0.554 |
| **Fine-tuned IID** | **17.249** | **0.531** | **0.485** |
| Pretrained Marigold-IID | 10.152 | 0.508 | 0.591 |
| **Fine-tuned Marigold** | **17.118** | **0.570** | **0.461** |
| Pretrained RGB↔X | 15.054 | 0.559 | 0.472 |
| **Fine-tuned RGB↔X** | **17.735** | **0.611** | **0.413** |

### Ablation Study

| Model | PSNR Gain | SSIM Gain | LPIPS Gain | Notes |
|-------|-----------|-----------|------------|-------|
| NIID-Net | +3.37 (26.4%) | +0.045 (8.2%) | −0.024 (3.0%) | CNN direct fine-tuning |
| Intrinsic Image Diffusion | +1.65 (10.6%) | +0.038 (7.7%) | −0.069 (12.5%) | LoRA + latent encoder |
| Marigold-IID | **+6.97 (68.6%)** | +0.062 (12.2%) | −0.130 (22.0%) | Largest gain (pre-training biased to indoor) |
| RGB↔X | +2.68 (17.8%) | +0.052 (9.3%) | −0.059 (12.5%) | Best overall performance |

### Key Findings

- **Consistent improvement across all fine-tunable models**: Regardless of architecture (CNN vs. diffusion) or pre-training data (InteriorVerse vs. Hypersim), Olbedo fine-tuning yields significant gains, confirming that the dataset effectively bridges the indoor-to-outdoor domain gap.
- **Marigold-IID achieves the largest gain** (PSNR +68.6%): The original pre-training is heavily biased toward indoor scenes, resulting in a low baseline; after fine-tuning, performance approaches the best-performing model. This highlights the high value of Olbedo's outdoor albedo supervision.
- **RGB↔X achieves the best overall performance** (PSNR = 17.735, SSIM = 0.611, LPIPS = 0.413): attributed to richer synthetic pre-training data and text-conditioned modality selection.
- **Fine-tuning only 0.19% of parameters is effective**: The parameter efficiency of LoRA demonstrates that Olbedo's role is to inject outdoor illumination diversity rather than replace pre-trained priors.

## Highlights & Insights

- **Filling a critical data gap**: As the first real-world aerial IID dataset, Olbedo directly addresses the fundamental bottleneck of lacking dense outdoor albedo ground truth. It is positioned as a training resource rather than an evaluation benchmark (MatrixCity serves as an external evaluation set to avoid overfitting to pseudo-ground-truth accuracy).
- **Practicality of the physics-based inverse rendering pipeline**: The approach leverages lit–shadow pixel pairs and sun/sky decomposition without requiring multi-illumination conditions, enabling albedo extraction from a single flight. This pipeline generalizes to other outdoor remote sensing scenarios.
- **Rich downstream application value**:
    - **Relighting**: Replacing RGB textures with albedo textures in rendering yields more consistent shadows and shading.
    - **Segmentation assistance**: SAM produces more stable segmentation on albedo maps than on RGB images, as shadow-induced spurious edges are eliminated.
    - **Material editing**: Modifying albedo and recovering the original illumination via inverse Retinex ($S = I/R$) preserves lighting and texture details better than alpha blending and AI-based editing methods (FlowEdit, Nano Banana, Qwen Image Edit).
    - **Scene change detection**: Albedo differencing is more robust to illumination variation than RGB differencing, reducing false positives caused by shadows.

## Limitations & Future Work

- **Lambertian assumption**: All surfaces are assumed to be diffuse; specular rooftops and glass produce artifacts. Incorporating BRDF models (e.g., Cook-Torrance) could further improve quality, but would require additional illumination observations.
- **Only 4 scenes**: The four locations—Office, Arena, Residential, and Park—limit diversity analysis. Extension to more urban, rural, and industrial scenes would improve generalization.
- **Pseudo-ground-truth is less accurate than synthetic ground truth**: Annotation quality is limited at geometric holes, shadow boundaries, and vegetation regions; although confidence masks partially mitigate this, accuracy remains below that of synthetic data.
- **Degradation under overcast conditions**: Under heavy overcast, effective lit–shadow pairs are insufficient, $\phi$ approaches zero, and the method falls back to a sky-only model with reduced albedo estimation accuracy.
- **Resolution constraints**: Images are downsampled 8× to 683×455 for processing, potentially losing fine-grained detail.

## Related Work & Insights

- **vs. InteriorVerse / Hypersim**: These synthetic indoor datasets provide perfect ground truth but suffer from a sim-to-real gap and are limited to indoor scenes. Olbedo provides real outdoor supervision and is complementary to synthetic pre-training—synthetic data supplies indoor priors while Olbedo injects outdoor illumination diversity.
- **vs. IIW**: IIW is the first large-scale real-world IID dataset but provides only sparse relative judgments (A is brighter than B), without dense albedo maps. Olbedo provides dense per-pixel albedo/shading maps, making it suitable for training rather than evaluation only.
- **vs. RGB↔X**: RGB↔X is among the strongest current IID methods, employing text conditioning for modality selection. After fine-tuning on Olbedo, it achieves the best overall performance, indicating that its architecture is well-suited for cross-domain adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first outdoor aerial IID dataset represents a significant contribution in its own right; while the inverse rendering pipeline builds on existing methods, its systematic integration and dataset construction constitute substantive innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six baseline models, four fine-tuned comparisons, external evaluation, and four downstream applications provide comprehensive coverage; however, the use of only 4 scenes limits diversity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, with transparent discussion of the dataset construction process and limitations; supplementary material is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a major gap in outdoor IID datasets; the open-sourced dataset, baselines, and evaluation protocol offer substantial community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[ICCV 2025\] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation](../../ICCV2025/remote_sensing/citynav_a_large-scale_dataset_for_real-world_aerial_navigation.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)

</div>

<!-- RELATED:END -->
