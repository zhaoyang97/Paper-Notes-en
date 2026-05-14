---
title: >-
  [Paper Note] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection
description: >-
  [NeurIPS 2025][Object Detection][Data Augmentation] ReCon proposes a training-free, region-controllable data augmentation framework that enhances the detection data quality of existing structure-controllable generative m…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "Data Augmentation"
  - "Diffusion Models"
  - "Region Control"
  - "ControlNet"
date: 2026-05-08
content_hash: f52253490b8d6921
---

# ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.15783](https://arxiv.org/abs/2510.15783)
**Code**: [https://github.com/haoweiz23/ReCon](https://github.com/haoweiz23/ReCon)
**Area**: Object Detection / Data Augmentation
**Keywords**: Data Augmentation, Object Detection, Diffusion Models, Region Control, ControlNet

## TL;DR

ReCon proposes a training-free, region-controllable data augmentation framework that enhances the detection data quality of existing structure-controllable generative models through Region-Guided Rectification (RGR) and Region-Aligned Cross-Attention (RACA), achieving 35.5 mAP on COCO—surpassing GeoDiffusion, which requires fine-tuning.

## Background & Motivation

Object detection models rely heavily on large-scale annotated datasets, yet annotation costs are prohibitively high (e.g., a single Cityscapes image requires ~60 minutes to annotate). Generative models have drawn increasing attention as data augmentation tools, but suffer from two core problems:

**Limitations of Prior Work**:

**Content–position mismatch**: Structure-controllable generative models (e.g., ControlNet) tend to generate incorrect numbers of objects or place objects at wrong locations under complex layouts.

**Semantic leakage**: In text-to-image generation, text features from different categories interfere with each other, causing semantic inconsistencies between generated regions and annotations.

**High method complexity**: Existing approaches either require post-hoc filtering (e.g., CLIP score filtering) or large-scale fine-tuning (e.g., GeoDiffusion, DetDiffusion), making them unsuitable for data-scarce scenarios.

**Core Idea**: Directly integrate region-level rectification and alignment mechanisms into the diffusion sampling process—without any additional training—to substantially improve the quality and trainability of generated data.

## Method

### Overall Architecture

ReCon builds upon existing structure-controllable models (e.g., ControlNet + Canny edge) and embeds two modules at each step of the diffusion sampling process:
1. Region-Guided Rectification (RGR): Uses a perception model to detect intermediate results and rectify mis-generated regions.
2. Region-Aligned Cross-Attention (RACA): Enables independent interaction between region features and their corresponding text features, preventing semantic leakage.

### Key Designs

1. **Region-Guided Rectification (RGR)**: Employs Grounded-SAM during sampling to detect intermediate generation results. Specifically, a clean prediction $\mathbf{z}_{0|t-N}$ is obtained after $N=5$ steps via cache-accelerated sampling, and objects are detected using Grounded-SAM. IoU matching is performed between detections and ground-truth annotations to identify false-positive and false-negative regions, which are used to construct a binary mask $\mathbf{M}$. Noisy versions of the original image are injected into mis-generated regions:
    $$\mathbf{z}'_t = \mathbf{M} \odot \mathbf{z}_t^{\text{orig}} + (1-\mathbf{M}) \odot \mathbf{z}_t$$
   Rectification is applied at four timesteps (0.75T, 0.50T, 0.25T, 0.10T) to progressively correct spatial layout, semantic content, and generation quality. The overridability of diffusion sampling ensures that regional replacement does not disrupt the overall inference process.

2. **Region-Aligned Cross-Attention (RACA)**: To address semantic leakage, text features are encoded independently for each object category (in the format "[CLASS]"), alongside a global scene description. In the U-Net cross-attention layers, region features are cropped from the latent $\mathbf{z}_t^{in}$, independently cross-attended with the corresponding class text features, and then reassembled into $\mathbf{z}_t^{out}$. This ensures each region is influenced only by its corresponding class text, eliminating cross-category feature interference.

3. **Perception Target Selection**: Three perception targets are compared: directly detecting $x_t$ (too noisy), predicted $x_{0|t}$, and cache-accelerated $x_{0|(t-N)}$. Using $x_{0|(t-N)}$ yields the best results, as cache-accelerated sampling provides more accurate clean image predictions (mAP: 35.0 → 35.3 → 35.5).

### Loss & Training

ReCon involves no training—it is a plug-and-play inference-time method. Downstream detectors follow standard training pipelines (e.g., Faster R-CNN + R-50-FPN, 6 epochs). Generated data is mixed with the original training set. Generation configuration: SD v1.5 + 25-step DDIM + edge-conditioned ControlNet.

## Key Experimental Results

### Main Results (COCO Dataset)

| Method | Type | mAP | AP50 | AP75 | APm | APl |
|--------|------|-----|------|------|-----|-----|
| Real only | - | 34.5 | 55.5 | 37.1 | 37.9 | 44.3 |
| ControlNet | General Control | 34.9 | 55.5 | 37.7 | 38.2 | 45.5 |
| GeoDiffusion | COCO Fine-tuned | 34.8 | 55.3 | 37.4 | 38.2 | 45.4 |
| DetDiffusion | COCO Fine-tuned | 35.4 | 55.8 | 38.3 | 38.5 | 46.6 |
| **ControlNet + ReCon** | **Training-free** | **35.5** | **56.2** | **38.4** | **39.0** | **46.0** |
| **Instance Diff + ReCon** | **Training-free** | **35.6** | **56.0** | **38.4** | **39.0** | **46.4** |

### Ablation Study

| RGR | RACA | FID | mAP | AP50 | AP75 |
|-----|------|-----|-----|------|------|
| ✘ | ✘ | 13.82 | 34.9 | 55.5 | 37.7 |
| ✔ | ✘ | 13.21 | 35.3 | 56.0 | 38.1 |
| ✔ | ✔ | 12.85 | 35.5 | 56.2 | 38.4 |

### Data-Scarce Scenarios

| Method | 1% Data | 5% Data | 10% Data |
|--------|---------|---------|----------|
| Real only | 0.3 | 13.0 | 18.5 |
| ControlNet | 2.5 | 15.9 | 21.2 |
| ReCon | 3.9 | 16.7 | 21.7 |
| ReCon + RandAugment | **4.2** | **17.1** | **22.0** |

### Key Findings

- ReCon without training surpasses GeoDiffusion fine-tuned on COCO (35.5 vs. 34.8 mAP).
- The two components (RGR + RACA) are complementary: RGR improves spatial consistency, while RACA improves semantic consistency.
- Augmentation efficiency is notable: 3× ReCon augmentation outperforms 7× ControlNet baseline augmentation.
- The method is compatible with multiple detectors (Faster R-CNN, RetinaNet, YOLOX, DEIM) and generative models (ControlNet, GLIGEN, Instance Diffusion).
- Effectiveness is also demonstrated on VOC (77.1 → 78.5 mAP), validating cross-dataset generalizability.
- Stronger perception models (Swin-Base vs. Swin-Tiny) yield further improvements.

## Highlights & Insights

- **Training-free plug-and-play**: Leverages existing perception (Grounded-SAM) and generative models (ControlNet) to substantially improve data quality without any fine-tuning.
- **Closed-loop control during sampling**: Rather than post-generation filtering, ReCon performs real-time detection and rectification during sampling, which is more efficient and preserves diversity.
- **Exploiting the overridability of diffusion sampling**: A key mathematical insight that regional substitution at intermediate steps does not disrupt the overall generation process.
- **Data-scarcity friendly**: Particularly suited for annotation-limited scenarios, which are precisely the settings where data augmentation is most needed.

## Limitations & Future Work

- Relies on the detection quality of Grounded-SAM; may fail for categories that are difficult for SAM to handle.
- Multiple perception model invocations during sampling increase generation time.
- RACA requires independent text feature encoding per category, which may introduce overhead when the number of categories is large.
- Validated only on 2D detection; applicability to 3D object detection and dynamic scenes remains unexplored.
- The rectification timestep schedule (0.75T, 0.50T, etc.) is manually defined as a hyperparameter.

## Related Work & Insights

ReCon's key innovation lies in shifting the evaluation of generation quality from post-generation filtering to online control during the generation process. This idea is analogous to the sampling-process guidance in DistDiff, but ReCon employs region-level rectification via a detection model rather than global feature guidance. Compared to the perception-aware loss training in DetDiffusion, ReCon offers a training-free alternative. Its plug-and-play nature allows seamless integration into any diffusion-based data augmentation pipeline.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of region-level in-sampling rectification is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, detectors, baselines, data-scarce settings, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, with rich visual comparisons.
- Value: ⭐⭐⭐⭐ A highly practical training-free augmentation method with outstanding value in data-scarce scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction](recon-gs_continuum-preserved_gaussian_streaming_for_fast_and_compact_reconstruct.md)
- [\[CVPR 2026\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2026/object_detection/mitigating_memorization_in_texttoimage_diffusion_v.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](../../CVPR2026/object_detection/uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)
- [\[ICCV 2025\] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](../../ICCV2025/object_detection/distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)
- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](../../ICCV2025/object_detection/diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)

</div>

<!-- RELATED:END -->
