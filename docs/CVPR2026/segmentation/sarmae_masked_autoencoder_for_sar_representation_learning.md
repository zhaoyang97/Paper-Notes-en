---
title: >-
  [Paper Note] SARMAE: Masked Autoencoder for SAR Representation Learning
description: >-
  [CVPR2026][Segmentation][SAR] This paper proposes SARMAE, a framework for noise-robust SAR self-supervised pre-training built upon the million-scale SAR-1M dataset, speckle-aware representation enhancement (SARE)…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "SAR"
  - "self-supervised pre-training"
  - "Masked Autoencoder"
  - "speckle noise"
  - "optical-SAR alignment"
  - "remote sensing"
date: 2026-05-08
content_hash: e84f2543d1861b27
---

# SARMAE: Masked Autoencoder for SAR Representation Learning

**Conference**: CVPR2026  
**arXiv**: [2512.16635](https://arxiv.org/abs/2512.16635)  
**Code**: [SARMAE](https://github.com/SARMAE/SARMAE)  
**Area**: Semantic Segmentation / SAR Representation Learning  
**Keywords**: SAR, self-supervised pre-training, Masked Autoencoder, speckle noise, optical-SAR alignment, remote sensing

## TL;DR

This paper proposes SARMAE, a framework for noise-robust SAR self-supervised pre-training built upon the million-scale SAR-1M dataset, speckle-aware representation enhancement (SARE), and semantic anchor representation constraint (SARC). SARMAE achieves state-of-the-art performance across multiple downstream tasks including classification, detection, and segmentation.

## Background & Motivation

**Unique Advantages and Challenges of SAR Imaging**: SAR enables all-weather, all-day imaging and is widely used in ocean monitoring, disaster assessment, and urban analysis. However, its inherent speckle noise results in low semantic content and weak structural cues, severely degrading deep learning representation quality.

**Data Scale Bottleneck**: Acquiring SAR data is costly, and existing pre-training datasets are limited in scale—SARATR-X contains only 180k images and SUMMIT only 560k—far insufficient for general-purpose SAR representation learning.

**Inapplicability of Optical Pre-training Strategies**: Existing methods directly adopt optical image pre-training strategies (e.g., MAE, MoCo) without accounting for the physical characteristics of SAR speckle noise, which is multiplicative rather than additive Gaussian noise and thus requires dedicated modeling.

**Semantic Limitations of Unimodal Pre-training**: Pre-training solely on SAR data is constrained by the inherently low semantic discriminability of SAR imagery, resulting in representations that lack semantic richness and generalizability.

**Shortcomings of Existing SAR Foundation Models**: Although SARATR-X and SUMMIT attempt unified pre-training, neither models the physical priors of SAR speckle nor exploits complementary multimodal information.

**Semantic Guidance Potential of Optical Imagery**: Optical images exhibit clearer semantic structures. Leveraging SAR–optical paired data for cross-modal alignment can substantially improve the semantic quality of SAR representations.

## Method

### Overall Architecture

SARMAE is built upon the MAE architecture and comprises two branches:

- **SAR Branch**: A ViT encoder and a Transformer decoder following the MAE pipeline (75% random masking), augmented with the SARE module to handle speckle noise.
- **Optical Branch**: A frozen DINOv3 encoder (sharing the ViT architecture with the SAR branch) that provides semantic anchors for paired data.

For SAR data with paired optical images, both branches operate jointly; unpaired SAR data is processed through the SAR branch alone.

### Key Design 1: Speckle-Aware Representation Enhancement (SARE)

**Core Idea**: Physically-modeled speckle noise is explicitly injected into the pre-training process, training the model to reconstruct clean images from noisy inputs.

- **Speckle Physical Model**: Multi-look SAR intensity images follow a Gamma distribution $Z \sim \text{Gamma}(L, \bar{I}/L)$, where $L$ is the number of looks and $\bar{I}$ is the true backscattering intensity.
- **Synthetic Noise Injection**: For each input patch $x$, a noisier version $x'$ is sampled from the Gamma distribution at a lower synthetic look number $L_{\text{syn}}$, preserving the pixel mean while increasing variance.
- **Denoising Reconstruction Task**: $x'$ is randomly masked and fed into the encoder, and the decoder is trained to reconstruct the original $x$ rather than the noisy version.
- **Multiple Noise Types**: In addition to Gamma noise, Rayleigh, Gaussian, and Uniform noise are incorporated to enhance robustness.
- **Loss Function**: $\mathcal{L}_{\text{SARE}} = \frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} \| D(E_{\text{SAR}}(\tilde{x}'))_p - x_p \|_2^2$

### Key Design 2: Semantic Anchor Representation Constraint (SARC)

**Core Idea**: Semantic features from paired optical images serve as anchor points to guide the alignment of the SAR encoder.

- Masked SAR images are passed through the SAR encoder to obtain visible patch embeddings $f_{\text{SAR}}^i$.
- Unmasked optical images are processed by the frozen DINOv3 encoder to obtain complete patch embeddings $f_{\text{OPT}}^i$.
- A patch-wise cosine distance loss is applied to spatially corresponding patch pairs: $\mathcal{L}_{\text{SARC}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \left(1 - \frac{f_{\text{SAR}}^i \cdot f_{\text{OPT}}^i}{\|f_{\text{SAR}}^i\|_2 \|f_{\text{OPT}}^i\|_2}\right)$

### Total Pre-training Loss

$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{SARE}} + \lambda \mathcal{L}_{\text{SARC}}, \quad \lambda = 0.1$$

### SAR-1M Dataset

- The first million-scale SAR dataset, aggregating 18 public datasets spanning 57 categories.
- 1.3 million SAR images paired with 1 million optical images, totaling 2.3 million samples.
- Covers multiple sensors including Sentinel-1, Gaofen-3, RadarSat-2, and TerraSAR-X.
- Multi-band (C/X/Ku/Ka), multi-polarization (HH/HV/VV/VH), and multi-resolution (0.1 m–60 m).

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | SARMAE (ViT-B) | SARMAE (ViT-L) | Prev. SOTA |
|------|---------|--------|-----------------|-----------------|------------|
| Classification | FUSAR-SHIP (40-shot) | Top1 Acc | **89.30%** | **90.86%** | 87.61% (Copernicus FM) |
| Classification | FUSAR-SHIP (30%) | Top1 Acc | **92.92%** | 92.80% | 71.91% (SUMMIT) |
| Classification | MSTAR (40-shot) | Top1 Acc | **96.70%** | **97.24%** | 91.60% (SAR-JEPA) |
| Detection | SARDet-100k | mAP | 57.90% | **63.10%** | 57.30% (SARATR-X) |
| Detection | SSDD | mAP | **68.10%** | **69.30%** | 67.50% (SARATR-X) |
| Rotated Detection | RSAR | mAP | 66.80% | **72.20%** | 64.82% (O-RCNN) |
| Segmentation | AIR-PolSAR-Seg (multi-class) | mIoU | 66.53% | **67.51%** | 52.58% (ANN) |
| Segmentation | AIR-PolSAR-Seg (water) | IoU | 92.31% | **93.06%** | 89.29% (DANet) |

### Ablation Study

| Model | Pre-training Data | SARE | SARC | FUSAR | SSDD | AIR-PolSAR-Seg |
|-------|-------------------|------|------|-------|------|----------------|
| MAE (Baseline) | ImageNet-1K | ✗ | ✗ | 75.40 | 64.00 | 60.28 |
| MAE | SAR-1M (SAR only) | ✗ | ✗ | 82.22 | 64.20 | 64.36 |
| MAE + Noise | SAR-1M (SAR only) | ✓ | ✗ | 86.80 | 64.40 | 65.15 |
| SARMAE | SAR-1M (SAR/OPT) | ✓ | ✓ | **89.30** | **68.10** | **66.53** |

### Key Findings

- **Substantial Gain from In-domain Pre-training**: SAR-1M pre-training outperforms ImageNet pre-training by +6.82% on FUSAR, confirming a significant distribution gap between SAR and natural images.
- **Contribution of SARE**: Incorporating speckle noise modeling yields a +4.58% classification gain; attention maps show the model more accurately focuses on semantic targets and even captures subtle semantically relevant objects.
- **Contribution of SARC**: SARC brings a +3.7% mAP improvement on SSDD detection, effectively alleviating false alarms caused by speckle interference; reconstruction visualizations demonstrate that SARC recovers local scene structure.
- **Good Scalability**: Scaling from ViT-B to ViT-L yields a +5.4 mAP gain on rotated object detection.
- **Direct Fine-tuning of DINOv3 is Ineffective**: Directly fine-tuning a frozen DINOv3 on SAR data performs poorly (74.25%), indicating that SARC's effectiveness stems from explicit SAR–optical alignment.

## Highlights & Insights

- SAR-1M, the first million-scale SAR dataset, fills a critical gap in large-scale SAR pre-training data.
- The physically-grounded speckle noise injection design, based on the Gamma distribution, directly adapts the pre-training process to the physical properties of SAR imaging.
- SARE and SARC are complementary: the former enables the model to understand noise, while the latter provides clear semantic guidance.
- SARMAE achieves comprehensive state-of-the-art results across classification, detection, and segmentation on multiple benchmarks, demonstrating strong generalizability.

## Limitations & Future Work

- Pre-training is computationally expensive (300 epochs, batch size 1024, A800 GPUs), making reproduction difficult for most research groups.
- SARC relies on SAR–optical paired data; SAR data from regions without optical counterparts (e.g., polar or high-latitude areas) cannot benefit from cross-modal alignment.
- The optical branch uses a frozen DINOv3 encoder; joint training or alternative optical teacher models have not been explored.
- Although SAR-1M spans multiple sensors, differences in annotation standards and quality across 18 source datasets may introduce bias.
- Downstream evaluation does not cover additional SAR application scenarios such as change detection and object tracking.

## Related Work & Insights

- **SAR Pre-training**: SARATR-X (HiViT + two-stage self-supervision), SUMMIT (MAE + multiple auxiliary tasks), SAR-JEPA (masked autoencoding + local reconstruction).
- **Remote Sensing Pre-training**: SeCo (MoCo), RVSA (MAE + rotated window attention), SatMAE (multi-spectral/multi-temporal), ScaleMAE (scale-aware masking).
- **General Visual Pre-training**: MAE, BEiT, DINOv3, CROMA (contrastive learning), Copernicus FM (DINO distillation).

## Rating

- Novelty: ⭐⭐⭐⭐ — The physically-grounded noise modeling in SARE and the cross-modal alignment design in SARC are both innovative; SAR-1M is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three tasks, multiple datasets, complete ablations, with consistent and significant results.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous physical modeling formulations, and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ — The combination of dataset, framework, and state-of-the-art results represents an important advancement for the SAR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](mrm_masked_representation_modeling_domain_adaptive.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[CVPR 2026\] AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](../../ICLR2026/segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[ICML 2026\] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation](../../ICML2026/segmentation/semir_semantic_minor-induced_representation_learning_on_graphs_for_visual_segmen.md)

</div>

<!-- RELATED:END -->
