---
title: >-
  [Paper Note] Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal
description: >-
  [ECCV 2024][Image Restoration][Raindrop removal] The authors propose Raindrop Clarity, a large-scale real-world raindrop removal dataset containing 15,186 high-quality image pairs/triplets. For the first time, it covers raindrop-focused (clear raindrops with blurred background) and nighttime raindrop scenarios, both of which are missing from existing datasets.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Raindrop removal"
  - "Dataset"
  - "Night images"
  - "Dual-focus"
date: 2026-05-08
content_hash: c2815468886bb888
---

# Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal

**Conference**: ECCV 2024  
**arXiv**: [2407.16957](https://arxiv.org/abs/2407.16957)  
**Code**: Yes ([https://github.com/jinyeying/RaindropClarity](https://github.com/jinyeying/RaindropClarity))  
**Area**: Others  
**Keywords**: Raindrop removal, Dataset, Night images, Dual-focus, Image restoration

## TL;DR

The authors propose Raindrop Clarity, a large-scale real-world raindrop removal dataset containing 15,186 high-quality image pairs/triplets. For the first time, it covers raindrop-focused (clear raindrops with blurred background) and nighttime raindrop scenarios, both of which are missing from existing datasets.

## Background & Motivation

Raindrops adhering to lenses or windshields severely degrade visual quality, affecting applications such as surveillance, autonomous driving, and object detection. Existing raindrop removal datasets suffer from two key limitations:

**Only containing background-focused images**: Existing datasets (e.g., Qian et al., RainDS, RobotCar) are captured with the camera focused on the background, leaving raindrops blurred. In practice, however, camera autofocus can shift to the adhering raindrops, resulting in clear raindrops and a blurred background—a scenario that has been completely overlooked.

**Lack of nighttime data**: Existing datasets are fully captured during the daytime. Due to the significant domain gap between day and night (artificial lighting, low-light conditions), models trained on daytime data struggle to handle nighttime raindrop images.

These two limitations severely restrict the generalization ability of raindrop removal algorithms.

## Method

### Overall Architecture

The core contribution of this work is the dataset rather than a new algorithm. The Raindrop Clarity dataset is designed to cover four scenarios:

| Scenario | Time Period | Focus | Raindrop State | Background State |
|------|--------|------|----------|----------|
| 1 | Day | Background-focused | Blurry | Clear |
| 2 | Day | Raindrop-focused | Clear | Blurry |
| 3 | Night | Background-focused | Blurry | Clear |
| 4 | Night | Raindrop-focused | Clear | Blurry |

The dataset provides two annotation formats:
- **Image pairs** $(\tilde{\mathbf{x}}, \mathbf{b}_0)$: Raindrop image + clear background (for background-focused scenarios)
- **Triplets** $(\tilde{\mathbf{x}}, \mathbf{x}_0, \mathbf{b}_0)$: Raindrop image + blurry clean background + clear background (for raindrop-focused scenarios)

### Key Designs

**Data Acquisition Pipeline**:
1. Stabilize the camera using a ball head tripod to ensure static shooting.
2. Capture equipment: Sony FDR-AX33 4K video camera, Sony Alpha 7R III, and iPhone 14 Pro/15 Pro Max.
3. Install a glass plate with a distance of 5-25 cm from the camera.
4. Spray water or collect natural rain to form raindrops on the glass plate, then capture images with the focus on the raindrops.
5. Remove the glass plate and keep the camera focused on the near plane to capture the blurry clean background.
6. Adjust the focus to the distant background to capture the clear background image.

**Difference Map**: For triplet data, a pixel-level difference map $\tilde{\mathbf{m}} = \tilde{\mathbf{x}} - \mathbf{x}_0$ can be calculated to precisely locate the raindrop regions.

**Data Characteristics**:
- Approximately 38,816 images in total, containing 15,186 high-quality paired groups.
- Daytime: 5,442 groups (3,606 triplets + 1,836 pairs).
- Nighttime: 9,744 groups (4,838 triplets + 4,906 pairs).
- Covers various backgrounds such as cities, villages, campuses, roads, and aerial views.
- Diverse raindrop morphologies: elliptical shapes, water streaks, and varying densities.

### Loss & Training

The authors benchmark standard image restoration methods. For the background-focused raindrop degradation model:

$$\tilde{\mathbf{x}} = (1 - \mathbf{M}) \odot \mathbf{x}_0 + \mathbf{D}$$

where $\mathbf{M}$ is the binary raindrop mask, and $\mathbf{D}$ represents the optical effect of the blurry raindrop.

## Key Experimental Results

### Main Results

| Method | Day PSNR↑ | Day SSIM↑ | Day LPIPS↓ | Night PSNR↑ | Night SSIM↑ | Night LPIPS↓ |
|------|-----------|-----------|------------|-----------|-----------|------------|
| Input | 21.92 | 0.560 | 0.247 | 24.78 | 0.726 | 0.209 |
| AtGAN | 23.62 | 0.658 | 0.200 | 24.38 | 0.773 | 0.185 |
| RDdiff | 26.05 | 0.736 | 0.141 | 26.81 | 0.851 | 0.125 |
| Uformer | **26.08** | **0.748** | 0.131 | **26.87** | **0.848** | 0.123 |
| DiT | 26.03 | 0.752 | **0.106** | 26.23 | 0.826 | **0.111** |
| Restormer | 25.52 | 0.734 | 0.111 | 26.48 | 0.831 | 0.112 |

### Dataset Comparison

| Dataset | No. of Images | Real/Synthetic | Day BG-Focus | Night BG-Focus | Day Raindrop-Focus | Night Raindrop-Focus |
|--------|--------|-----------|------------|------------|------------|------------|
| **Raindrop Clarity (Day)** | 14,490 | Real | ✓ | ✗ | ✓ | ✗ |
| **Raindrop Clarity (Night)** | 24,326 | Real | ✗ | ✓ | ✗ | ✓ |
| Raindrop Qian | 1,838 | Real | ✓ | ✗ | ✗ | ✗ |
| RainDS-Real | 992 | Real | ✓ | ✗ | ✗ | ✗ |
| RobotCar | 9,636 | Real | ✓ | ✗ | ✗ | ✗ |
| Windshield | 3,390 | Real | ✓ | ✗ | ✗ | ✗ |

### Key Findings

1. Existing state-of-the-art (SOTA) raindrop removal methods perform significantly worse in raindrop-focused and nighttime scenarios.
2. Even the best methods (e.g., Uformer, DiT) still exhibit obvious failure cases on Raindrop Clarity.
3. Artificial light sources in nighttime scenarios (streetlights, headlights, neon signs) increase the complexity of the appearance of raindrops.
4. The raindrop-focused scenario requires simultaneous raindrop removal and restoration of the blurred background, making it much more challenging than the traditional setting.
5. General image restoration backbones (Restormer, Uformer) exhibit competitive performance compared to specialized raindrop removal methods.

## Highlights & Insights

- **Forward-looking problem definition**: Identifying the neglected practical scenario of "raindrop-focused" images, which fills a gap in existing datasets.
- The difference map provided by the **triplet annotations** offers a new supervisory signal for precise raindrop detection and segmentation.
- **Comprehensive daytime and nighttime coverage** enables the trained models to possess stronger all-weather capabilities.
- The data acquisition method cleverly utilizes optical refraction models and focus switching to produce paired data.

## Limitations & Future Work

- Data collection is conducted in controlled environments with static cameras and glass plates, which differs from real driving scenarios.
- No new algorithm tailored to the specific characteristics of the dataset is proposed; only benchmarking is conducted.
- No effective solution yet exists for the combined challenge of nighttime and raindrop-focus (blur + raindrops + low light).
- Integrating depth information could be considered to assist in distinguishing foreground raindrops from background blur.
- Temporal raindrop removal data at the video level is missing.

## Related Work & Insights

- Similar to the RainDS dataset, it provides a combination of various weather degradations, but with far greater scale and diversity.
- The proposed concept of difference maps can inspire other degradation detection tasks.
- It holds direct application value for systems requiring all-weather operations, such as autonomous driving.
- The dual-focus design of this dataset inspires further exploration of other "neglected degradation patterns."

## Rating

- **Novelty**: ★★★★☆ — The dataset definition is novel, covering raindrop-focused and nighttime scenarios for the first time.
- **Value**: ★★★★★ — High-quality large-scale real dataset, extremely valuable to the community.
- **Experimental Thoroughness**: ★★★☆☆ — Only benchmarking is performed, without proposing a new method.
- **Writing Quality**: ★★★★☆ — The dataset construction process is clearly described.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[ECCV 2024\] Spatially-Variant Degradation Model for Dataset-free Super-resolution](spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [\[ECCV 2024\] Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising](exploiting_dual-correlation_for_multi-frame_time-of-flight_denoising.md)
- [\[CVPR 2026\] From Events to Clarity: The Event-Guided Diffusion Framework for Dehazing](../../CVPR2026/image_restoration/from_events_to_clarity_the_event-guided_diffusion_framework_for_dehazing.md)
- [\[ECCV 2024\] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)

</div>

<!-- RELATED:END -->
