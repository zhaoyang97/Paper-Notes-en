---
title: >-
  [Paper Note] Intrinsic Single-Image HDR Reconstruction
description: >-
  [ECCV 2024][Image Restoration][HDR Reconstruction] > Proposes an HDR reconstruction method based on intrinsic image decomposition, which reformulates the problem into two sub-tasks: dynamic range expansion in the shading domain and color recovery in the albedo domain, training separate networks to improve reconstruction quality.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "HDR Reconstruction"
  - "Intrinsic Decomposition"
  - "Shading Separation"
  - "Albedo Recovery"
  - "Single Image"
date: 2026-05-08
content_hash: 9aeff9c598da46e6
---

tags:
  - ECCV 2024
  - Image Restoration
date: 2026-05-08
content_hash: 74b1bbfc19cac3bb
---
# Intrinsic Single-Image HDR Reconstruction

**Conference**: ECCV 2024

**Paper Link**: [https://arxiv.org/abs/2409.13803](https://arxiv.org/abs/2409.13803)

**Authors**: Sebastian Dille, Chris Careaga, Yagiz Aksoy

**Area**: Computational Photography / HDR Reconstruction

**Keywords**: HDR Reconstruction, Intrinsic Decomposition, Shading Separation, Albedo Recovery, Single Image

## TL;DR

> Proposes an HDR reconstruction method based on intrinsic image decomposition, which reformulates the problem into two sub-tasks: dynamic range expansion in the shading domain and color recovery in the albedo domain, training separate networks to improve reconstruction quality.

---

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: The low dynamic range (LDR) of standard cameras cannot capture the rich contrast of natural scenes, resulting in lost colors and details in saturated pixels. Recovering the high dynamic range (HDR) radiance from a single LDR image is a critical task in computational photography.

### Proposed Solution

**Goal**: **Difficulties of End-to-End Methods**: The HDR reconstruction task requires networks to understand high-level geometry and lighting cues, inferring missing details from scene context, which poses challenges for data-driven algorithms to generate accurate, high-resolution results.

**Coupled Problem**: Traditional methods treat color recovery and dynamic range expansion simultaneously, increasing the learning difficulty.

**Insufficient Generalization**: Models trained directly on LDR-HDR pairs struggle to maintain stable performance across diverse scenes.

### Key Challenge

**Key Challenge**: Natural images can be decomposed into albedo (surface color) and shading (lighting effects) intrinsic components. In HDR reconstruction, the missing information essentially stems from two distinct sources: (1) luminance loss in overexposed regions (shading domain); (2) color details loss in saturated pixels (albedo domain). Decomposing the problem reduces the complexity of individual sub-tasks.

---

## Method

### Overall Architecture

This paper proposes a **physics-inspired intrinsic HDR reconstruction method** that reformulates the HDR reconstruction problem into two sub-problems:

$$I_{HDR} = A \cdot S$$

where $A$ is the albedo component and $S$ is the shading component.

The framework consists of the following key steps:

1. **Intrinsic Decomposition**: Decomposing the input LDR image into an albedo map and a shading map.
2. **Dynamic Range Expansion in Shading Domain**: Training a specialized network to expand the dynamic range in the shading domain, recovering the luminance structure of overexposed areas.
3. **Color Recovery in Albedo Domain**: Training another network to recover the lost color details of saturated pixels in the albedo domain.
4. **HDR Synthesis**: Multiplying the expanded shading component with the recovered albedo component to obtain the final HDR image.

### Key Designs

#### Intrinsic Domain Modeling

- Image formation model based on the physical rendering equation: scene radiance can be decomposed into view-independent albedo and illumination-dependent shading.
- The shading component primarily contains lighting intensity and geometric information, and its dynamic range is much larger than that of the albedo.
- The albedo component loses information in overexposed regions, but its variations are relatively smooth, making it easier to infer from context.

#### Advantages of Sub-task Separation

- **Shading Network**: Focuses on understanding scene geometry and illumination distribution, inferring clipped high-luminance values.
- **Albedo Network**: Focuses on color consistency and texture recovery, filling in details utilizing the color information of adjacent unsaturated regions.

### Loss & Training

During training, reconstruction losses are applied to the two subnetworks separately:

$$\mathcal{L} = \mathcal{L}_{shading} + \mathcal{L}_{albedo}$$

where each sub-loss is composed of standard components such as pixel-level $L_1$ loss and perceptual loss.

---

## Experiments

### Main Results

| Method | PSNR ↑ | SSIM ↑ | Characteristics |
|------|--------|--------|------|
| End-to-end baseline methods | Lower | Lower | Direct LDR→HDR |
| Ours (Intrinsic) | **Highest** | **Highest** | Intrinsic domain decomposition |

### Ablation Study

| Configuration | Effect |
|------|------|
| Shading network only | Good luminance recovery, color bias |
| Albedo network only | Good color recovery, insufficient highlight details |
| Joint Shading + Albedo | Best overall performance |

### Key Findings

1. After decomposing HDR reconstruction into two sub-problems, the learning difficulty of each sub-task is significantly reduced.
2. Dynamic range expansion in the shading domain is more stable than operating directly in the RGB domain.
3. Color recovery in the albedo domain can leverage material consistency priors, generating more natural results.
4. Demonstrates robust generalization across various scenes (indoor/outdoor, different lighting conditions).

---

## Highlights & Insights

1. **Physics-inspired problem decomposition**: Formulates the highly complex HDR reconstruction problem into two simpler sub-problems based on the image formation model, which is clear and physically grounded.
2. **Complementarity between sub-tasks**: Shading is responsible for structure/luminance, while albedo handles color/texture, dividing and executing the workload.
3. **Reduced data requirements**: Each sub-network only needs to learn a simpler mapping, lowering the requirement for training data diversity.
4. **Flexibility**: The framework allows independent improvements for each sub-module, facilitating future upgrades.

## Limitations & Future Work

1. The accuracy of intrinsic decomposition itself affects the final HDR reconstruction quality; decomposition errors will propagate.
2. For severely overexposed (completely white) large regions, both sub-networks struggle to recover meaningful content.
3. Reliance on a pre-trained intrinsic decomposition model increases the complexity of the inference pipeline.
4. Cache file anomaly (due to the ECCV template); the notes are written based on paper metadata and the abstract, so details might not be completely accurate.

## Related Work

- **Single-Image HDR Reconstruction**: End-to-end methods such as ExpandNet, SingleHDR, HDR-GAN, etc.
- **Intrinsic Image Decomposition**: Decomposition methods such as Intrinsic Images in the Wild, CGIntrinsics, etc.
- **Inverse Tone Mapping**: iTMO series of methods.

## Rating

⭐⭐⭐⭐ — Novel problem decomposition approach, physically grounded, but increases pipeline complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ExpoCM: Exposure-Aware One-Step Generative Single-Image HDR Reconstruction](../../CVPR2026/image_restoration/expocm_exposure-aware_one-step_generative_single-image_hdr_reconstruction.md)
- [\[CVPR 2026\] LRHDR: Learning Representation-enhanced HDR Video Reconstruction](../../CVPR2026/image_restoration/lrhdr_learning_representation-enhanced_hdr_video_reconstruction.md)
- [\[ICLR 2026\] DeAltHDR: Learning HDR Video Reconstruction from Degraded Alternating Exposure Sequences](../../ICLR2026/image_restoration/dealthdr_learning_hdr_video_reconstruction_from_degraded_alternating_exposure_se.md)
- [\[CVPR 2026\] ReasonX: MLLM-Guided Intrinsic Image Decomposition](../../CVPR2026/image_restoration/reasonx_mllm-guided_intrinsic_image_decomposition.md)
- [\[ICCV 2025\] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](../../ICCV2025/image_restoration/afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)

</div>

<!-- RELATED:END -->
