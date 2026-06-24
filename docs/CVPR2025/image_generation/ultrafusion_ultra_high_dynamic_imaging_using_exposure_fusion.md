---
title: >-
  [Paper Note] UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion
description: >-
  [CVPR 2025][Image Generation][High Dynamic Range] UltraFusion reformulates exposure fusion as a guided inpainting problem for the first time. By leveraging under-exposed images as soft guidance rather than hard constraints for over-exposed regions, it achieves ultra-high dynamic range imaging with a 9-stop exposure difference while maintaining robustness against alignment errors and illumination variations.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "High Dynamic Range"
  - "Exposure Fusion"
  - "Guided Inpainting"
  - "Diffusion Models"
  - "Ultra-large Exposure Difference"
date: 2026-05-08
content_hash: 8084a166cf77bc12
---

# UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion

**Conference**: CVPR 2025  
**arXiv**: [2501.11515](https://arxiv.org/abs/2501.11515)  
**Code**: [Project Page](https://openimaginglab.github.io/UltraFusion)  
**Area**: Image Generation / HDR Imaging  
**Keywords**: High Dynamic Range, Exposure Fusion, Guided Inpainting, Diffusion Models, Ultra-large Exposure Difference

## TL;DR

UltraFusion reformulates exposure fusion as a guided inpainting problem for the first time. By leveraging under-exposed images as soft guidance rather than hard constraints for over-exposed regions, it achieves ultra-high dynamic range imaging with a 9-stop exposure difference while maintaining robustness against alignment errors and illumination variations.

## Background & Motivation

HDR imaging is a fundamental problem in camera design. Leading approaches improve dynamic range by fusing images captured at different exposures, but severe practical limitations persist:
- Prior methods can only handle a 3-4 stop exposure difference (e.g., HDR+ only adds 3 stops), which is far from sufficient for ultra-high dynamic range scenes.
- **Alignment issues**: Under large exposure differences, the input luminance varies drastically, making optical flow alignment highly challenging and leading to ghosting artifacts.
- **Illumination inconsistency**: Under-exposed images are not merely darkened versions of normally exposed images; object appearance changes with varying exposure.
- **Tone mapping artifacts**: Traditional HDR methods first generate HDR images and then compress them to LDR for display. In high dynamic range scenarios, the tone mapping stage introduces additional artifacts.
- Directly applying ControlNet cannot determine which frame should serve as the reference, leading to horizontally inconsistent reference frame selections across different regions.
- There is a lack of large-scale exposure fusion training data on dynamic scenes.

## Method

### Overall Architecture

UltraFusion is a two-stage framework: (1) Pre-alignment stage—aligning the under-exposed image to the over-exposed image and masking occluded areas; (2) Guided inpainting stage—based on Stable Diffusion, using the over-exposed image as the main reference and the under-exposed image as soft guidance to reconstruct highlight information in over-exposed regions. The guided inpainting stage includes a decomposition fusion control branch and a fidelity control branch.

### Key Designs

#### Key Design 1: Guided Inpainting Paradigm

**Function**: Redefining exposure fusion as an inpainting problem to directly output tone-mapped LDR results.

**Mechanism**: Using the normally exposed (over-exposed) image $I_{oe}$ as the baseline to reconstruct missing information in its highlight regions. The under-exposed image $I_{ue}$ acts as a soft guidance rather than a hard constraint, providing ground-truth contents for highlights. During the pre-alignment stage, RAFT is employed to estimate bidirectional optical flow. An occlusion mask $\mathcal{M}$ is obtained via consistency checks, resulting in the aligned output $I_{ue \to oe} = (1-\mathcal{M}) \cdot \mathcal{W}(I_{ue}, f_{oe \to ue})$.

**Design Motivation**: Soft guidance is robust to alignment errors and illumination variations (unlike hard constraints, which magnify errors). Directly generating LDR avoids cascaded errors from HDR-to-LDR conversion. The generative priors of diffusion models ensure natural-looking and plausible outputs.

#### Key Design 2: Decomposition Fusion Control Branch

**Function**: Extracting structure and color information robust to luminance variations from extremely dark under-exposed images to effectively guide diffusion-based inpainting.

**Mechanism**: Decomposing the under-exposed image into a structural component $S_{ue} = (Y_{ue} - \mu(Y_{ue})) / \sigma(Y_{ue})$ (normalized YUV luminance channel) and a color component (UV chroma channels). Structure and color features are extracted through separate convolutional extractors to obtain multi-scale features, which are then fused with the over-exposed image features using **multi-scale cross-attention**. The control branch replicates the U-Net encoder structure but updates weights independently, and its outputs are injected into the main U-Net via zero convolutions.

**Design Motivation**: Extremely dark under-exposed images are easily ignored by the model if used directly as guidance. After decomposing them into luminance-independent structure and color information, the effectiveness of the guiding signals is substantially improved.

#### Key Design 3: Fidelity Control Branch + Training Data Synthesis

**Function**: Alleviating texture distortions introduced by the VAE decoder; synthesizing training data for dynamic scene exposure fusion.

**Mechanism**: The Fidelity Control Branch (FCB) has a similar structure to the decomposition fusion control branch, but its main extractor adopts a VAE encoder structure (instead of U-Net) to provide skip connections for the VAE decoder. During training, the latent code encoded from the ground truth (GT) is used to simulate the denoising output, optimized via the $\|I_{gt} - \hat{I}_{gt}\|_1$ reconstruction loss. For data synthesis, frame pairs are sampled from video datasets to simulate large motions, under-exposed patches are sampled from a static multi-exposure dataset (SICE), and pseudo-occlusion masks are used to simulate dynamic occlusions, enabling the model to learn to handle dynamic scenes using only static data.

**Design Motivation**: VAE decoding tends to introduce undesirable texture alterations; furthermore, no large-scale dynamic HDR training dataset is readily available.

### Loss & Training

Standard diffusion denoising loss + L1 reconstruction loss $\|I_{gt} - \hat{I}_{gt}\|_1$ for the fidelity control branch.

## Key Experimental Results

### Main Results: Static MEFB Dataset

| Method | MUSIQ ↑ | DeQA-Score ↑ | PAQ2PIQ ↑ | HyperIQA ↑ | MEF-SSIM ↑ |
|------|---------|-------------|-----------|-----------|-----------|
| **UltraFusion** | **68.82** | **3.881** | **73.80** | **0.6482** | 0.9385 |
| HSDS-MEF | 66.76 | 3.544 | 72.60 | 0.6026 | **0.9520** |
| HDR-Transformer | 63.10 | 2.983 | 71.36 | 0.5996 | 0.8626 |

### Ablation Study: Dynamic RealHDRV and UltraFusion Benchmark

| Method | RealHDRV TMQI ↑ | RealHDRV MUSIQ ↑ | Benchmark MUSIQ ↑ | Benchmark DeQA ↑ |
|------|----------------|-----------------|------------------|-----------------|
| **UltraFusion** | **0.8925** | **67.51** | **68.41** | **3.830+** |
| HSDS-MEF | 0.8323 | 61.76 | 64.54 | 3.627 |
| HDR-Transformer | 0.8680 | 62.24 | 63.66 | 2.909 |

### Key Findings

- UltraFusion is the first method capable of merging images with a 9-stop exposure difference.
- In user studies, UltraFusion significantly outperforms all baseline methods in both subjective quality and user preference.
- The soft guidance mechanism makes the method robust to both large motion and illumination changes simultaneously.
- Decomposing the under-exposed image into structure and color components is key to effectively extracting and guiding information from dark images.

## Highlights & Insights

- **Paradigm Shift**: Reformulating exposure fusion from the conventional "align-then-merge" paradigm to a "guided inpainting" paradigm.
- **High Practicality**: Directly outputting LDR images bypasses the tone mapping step, producing display-ready, high-quality outputs end-to-end.
- **Ingenious Data Synthesis Strategy**: Simulating dynamic HDR scenes using a combination of video frames and static multi-exposure data.

## Limitations & Future Work

- Inherits the slow inference speed of diffusion model sampling processes.
- The pseudo-occlusions used in training data synthesis may not fully capture the complexity of real-world dynamic scenes.
- Currently handles only two frames (one long and one short exposure) and is not yet scaled to broader exposure brackets.
- Future work could explore real-time inference and video HDR scenarios.

## Related Work & Insights

- Compared to the direct application of ControlNet, the strategy of fixing a reference frame (the over-exposed image) eliminates ambiguity.
- The concept of decomposing under-exposed images into structure and color can be extended to other cross-domain guidance tasks.
- The newly collected 100-scene UltraFusion Benchmark provides standard evaluation data for ultra-high dynamic range imaging.

## Rating

⭐⭐⭐⭐ — The proposed reformulation of exposure fusion as guided inpainting is novel and effective, achieving 9-stop exposure difference fusion for the first time. The method design is comprehensive, with highly ingenious decomposition fusion control branches and data synthesis pipelines. It achieves state-of-the-art quality across several benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] Latent Space Imaging](latent_space_imaging.md)
- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[ICLR 2026\] Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis](../../ICLR2026/image_generation/latent_wavelet_diffusion_for_ultra_high-resolution_image_synthesis.md)

</div>

<!-- RELATED:END -->
