---
title: >-
  [Paper Note] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion
description: >-
  [CVPR 2026][Segmentation][Thermal imaging super-resolution] This paper proposes 3M-TI, a **calibration-free** multi-camera cross-modal diffusion framework that performs implicit alignment and fusion of uncalibrated RGB–thermal infrared image pairs via a Cross-modal Self-attention Module (CSM) in the VAE latent space. Combined with a misalignment augmentation strategy, the method achieves state-of-the-art performance on mobile thermal imaging super-resolution and significantly improves downstream object detection and semantic segmentation.
tags:
  - CVPR 2026
  - Segmentation
  - Thermal imaging super-resolution
  - cross-modal diffusion
  - calibration-free fusion
  - RGB guidance
  - mobile thermal imaging
date: 2026-05-08
content_hash: 5d3743a78d2a42a7
---

# 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion

**Conference**: CVPR 2026
**arXiv**: [2511.19117](https://arxiv.org/abs/2511.19117)
**Code**: [GitHub](https://github.com/work-submit/3MTI)
**Area**: Image Segmentation
**Keywords**: Thermal imaging super-resolution, cross-modal diffusion, calibration-free fusion, RGB guidance, mobile thermal imaging

## TL;DR

This paper proposes 3M-TI, a **calibration-free** multi-camera cross-modal diffusion framework that performs implicit alignment and fusion of uncalibrated RGB–thermal infrared image pairs via a Cross-modal Self-attention Module (CSM) in the VAE latent space. Combined with a misalignment augmentation strategy, the method achieves state-of-the-art performance on mobile thermal imaging super-resolution and significantly improves downstream object detection and semantic segmentation.

## Background & Motivation

**Hardware bottleneck in mobile thermal imaging**: Miniaturized thermal sensors on mobile platforms suffer from reduced apertures and constrained pixel sizes, yielding blurry, information-deficient outputs (typically 96×96 resolution).

**Limitations of Prior Work — single-image SR**: A single thermal image lacks sufficient high-frequency information to recover fine structures, especially at large upscaling factors.

**Limitations of Prior Work — RGB-guided methods require calibration**: Existing RGB-guided thermal SR methods rely on precise pixel-level cross-camera calibration, which is cumbersome and non-robust in practical deployment.

**Large cross-modal domain gap**: RGB and thermal infrared imaging operate on fundamentally different physical principles; naively merging features tends to introduce unrealistic texture artifacts.

**Limited thermal infrared datasets**: Compared to the RGB domain, thermal infrared datasets are small in scale and lack scene diversity, constraining network training and generalization.

**Spatiotemporal misalignment in practice**: Multi-camera systems inevitably suffer from parallax and temporal desynchronization, to which existing methods lack robustness.

## Method

### Overall Architecture

3M-TI is built upon the single-step diffusion model SD-Turbo. The inputs are a low-resolution thermal image (64×64) and an uncalibrated high-resolution RGB reference image (512×512). The pipeline proceeds as follows: (1) a frozen VAE encoder maps both modalities into the latent space; (2) the CSM replaces the original self-attention layers in the UNet Transformer blocks to enable cross-modal alignment and fusion; (3) misalignment augmentation is applied to RGB images during training to improve robustness; (4) zero-initialized skip connections are added to enhance structural consistency; (5) RAM generates text prompts from the RGB image to provide semantic guidance; (6) LoRA fine-tunes the UNet (rank=16) and VAE decoder (rank=4).

### Key Design 1: Cross-modal Self-attention Module (CSM)

- **Function**: Achieves implicit alignment and fusion of RGB and thermal infrared within UNet Transformer blocks.
- **Mechanism**: The latent tokens of both modalities are concatenated into a joint sequence $\{z_{RGB}, z_{th}\} \in \mathbb{R}^{B \times (M \times H \times W) \times C}$, allowing self-attention to simultaneously model intra-modal (thermal–thermal) and inter-modal (RGB–thermal) dependencies without additional parameters.
- **Design Motivation**: Standard cross-attention captures only inter-modal information while ignoring intra-modal spatial context. CSM models both relationships jointly via self-attention on the concatenated sequence. Compared to static feature concatenation followed by a fully connected projection, CSM is content-adaptive. The design is inspired by video and multi-view diffusion models.

### Key Design 2: Misalignment Augmentation

- **Function**: Applies controllable spatial transformations (translation, scaling, rotation, perspective distortion) to RGB images during training.
- **Mechanism**: Simulates geometric offsets caused by parallax and temporal desynchronization in real multi-camera systems without physical simulation.
- **Design Motivation**: Existing RGB–thermal datasets are strictly pixel-aligned, causing models to overfit specific calibration configurations. Injecting synthetic misalignment forces CSM to learn robust cross-modal correspondences under uncalibrated conditions, bridging the gap between training data and real deployment.

### Key Design 3: Latent-Space Diffusion and Skip Connections

- **Function**: Leverages SD-Turbo's single-step diffusion in the latent space to generate high-fidelity thermal images, with zero-initialized skip connections propagating encoder feature maps to the decoder.
- **Mechanism**: The generative prior of the diffusion model compensates for the scarcity of thermal infrared data by synthesizing realistic high-frequency details; skip connections maintain geometric structural consistency.
- **Design Motivation**: Pure CNN/Transformer methods produce over-smoothed results under severe degradation, while diffusion models can generate high-frequency details but may introduce geometric distortions. Skip connections effectively mitigate this issue.

## Loss & Training

- **Loss function**: $\mathcal{L} = \mathcal{L}_2 + \lambda \cdot \mathcal{L}_{\text{LPIPS}}$, where $\lambda = 1$, combining pixel-level L2 loss and perceptual loss LPIPS.
- **Optimizer**: Adam, learning rate $2 \times 10^{-5}$, batch size = 4.
- **Training time**: Approximately 4 hours (8,000 iterations) on a single A800 GPU (80 GB).
- **LoRA configuration**: UNet rank=16, VAE decoder rank=4.
- **Training data**: 10,922 RGB–thermal image pairs from four datasets: IRVI, LLVIP, M3FD, and PBVS 2025.

## Key Experimental Results

### Table 1: Quantitative Comparison on Public Datasets

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | MANIQA↑ | MUSIQ↑ |
|--------|-------|-------|--------|---------|--------|
| CoReFusion | 30.11 | 0.8588 | 0.3214 | 0.2771 | 28.35 |
| CoRPLE | 30.47 | 0.8642 | 0.3206 | 0.2833 | 30.46 |
| SwinFuSR | 29.85 | 0.8549 | 0.3085 | 0.2740 | 29.86 |
| SeeSR | 29.41 | 0.8495 | 0.1828 | 0.4278 | 35.22 |
| OSEDiff | 28.05 | 0.8422 | 0.2113 | 0.4014 | 36.30 |
| DifIISR | 27.48 | 0.7905 | 0.3484 | 0.4214 | 36.74 |
| **3M-TI (Ours)** | **30.09** | **0.8610** | **0.1787** | **0.4443** | **36.66** |

3M-TI achieves the best perceptual metrics (LPIPS, MANIQA, MUSIQ) while outperforming other diffusion-based methods on fidelity metrics (PSNR, SSIM).

### Table 2: Downstream Object Detection Performance

| Method | Precision↑ | Recall↑ | F1↑ | IoU↑ |
|--------|-----------|---------|-----|------|
| SwinPaste | 0.1800 | 0.2109 | 0.1765 | 0.1941 |
| SeeSR | 0.3832 | 0.4637 | 0.3849 | 0.3022 |
| **3M-TI** | **0.4565** | **0.5455** | **0.4724** | **0.3427** |
| Reference RGB | 0.4322 | 0.5708 | 0.4643 | 0.3359 |
| GT Thermal | 0.4582 | 0.5793 | 0.4887 | 0.3494 |

3M-TI's detection performance slightly surpasses the RGB reference and approaches GT thermal image quality.

### Ablation Study

- Removing the RGB reference: reconstruction becomes blurry; LPIPS degrades from 0.1787 to 0.2106.
- Removing misalignment augmentation: MUSIQ drops from 36.66 to 34.94 with noticeable high-frequency detail degradation.
- Removing skip connections: structural fidelity decreases (PSNR from 30.09 to 29.86) with geometric distortions in objects such as circular wheels.
- CSM outperforms standard cross-attention (LPIPS 0.1787 vs. 0.1953) and feature concatenation (0.1787 vs. 0.2164).

## Highlights & Insights

1. **Calibration-free cross-modal fusion** is the most practically significant contribution — implicit alignment via attention in the VAE latent space completely avoids the calibration challenges encountered in real deployment.
2. **CSM is elegantly simple**: it introduces no additional parameters and captures both intra- and inter-modal dependencies through token concatenation and self-attention, representing a creative adaptation of multi-frame processing ideas from video diffusion models.
3. **The misalignment augmentation strategy** is conceptually novel, replacing complex physical simulation with simple geometric transformations to effectively improve generalization.
4. **Downstream task validation is thorough**: beyond perceptual quality evaluation, the method demonstrates substantive gains in object detection and semantic segmentation, confirming the practical value of the super-resolution output.
5. **Real hardware validation**: a practical system is constructed using a HIKVISION P09 thermal camera module (under $100) and a Xiaomi 15 smartphone, demonstrating strong engineering feasibility.

## Limitations & Future Work

1. **Quality ceiling of single-step diffusion**: while efficient, SD-Turbo-based single-step inference may not match the generation quality of multi-step diffusion methods.
2. **Semantic guidance depends on RAM**: the method requires reasonably high-quality RGB input; degraded references (e.g., low-light or motion-blurred) may produce erroneous semantic prompts.
3. **Insufficient handling of FOV discrepancy**: the RGB and thermal cameras have different fields of view (74°×59° vs. 50°×50°); robustness under larger FOV gaps remains to be validated.
4. **Only 8× super-resolution (64→512) is evaluated**: applicability to other upscaling factors is not discussed.
5. **Inference overhead not fully analyzed**: the cross-modal self-attention in the UNet operates on sequences of length $2HW$, and the computational complexity at higher resolutions warrants attention.

## Related Work & Insights

- **CoReFusion / SwinFuSR / SwinPaste**: Traditional RGB-guided thermal SR methods that rely on calibration and prioritize fidelity at the expense of high-frequency detail.
- **SeeSR / OSEDiff**: Diffusion-based image SR methods capable of generating high-frequency content but lacking cross-modal guidance, prone to artifacts.
- **DifIISR**: Infrared-specific diffusion SR, but dependent on strictly aligned data.
- **Stable Video Diffusion**: The multi-frame joint self-attention paradigm is the direct inspiration for CSM.
- **Insights**: The calibration-free cross-modal fusion concept is extensible to other modality pairs (e.g., depth–RGB, SAR–optical); the misalignment augmentation strategy is broadly applicable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — CSM and misalignment augmentation are novel; the calibration-free setting has significant practical relevance
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers public benchmarks, real mobile hardware, downstream tasks, and ablation studies comprehensively
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated design choices, and rich figures and tables
- **Value**: ⭐⭐⭐⭐⭐ — Calibration-free design, low-cost hardware, and mobile deployment make this highly practical

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels](towards_high-quality_image_segmentation_improving_topology_accuracy_by_penalizin.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)
- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[ICLR 2026\] Universal Multi-Domain Translation via Diffusion Routers](../../ICLR2026/segmentation/universal_multi-domain_translation_via_diffusion_routers.md)
- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](../../NeurIPS2025/segmentation/omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)

</div>

<!-- RELATED:END -->
