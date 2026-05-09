---
title: >-
  [Paper Note] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing
description: >-
  [AAAI 2026][Medical Imaging][Text-guided image editing] This paper proposes FIA-Edit, an inversion-free text-guided image editing framework based on frequency-interactive attention. It introduces a Frequency Representation Interaction (FRI) module that performs frequency-domain fusion of source/target features within self-attention, and a Feature Injection (FIJ) module that explicitly incorporates source image features into cross-attention. The framework achieves precise semantic editing while maintaining high background fidelity, and for the first time applies a general image editing method to clinical surgical bleeding image augmentation.
tags:
  - AAAI 2026
  - Medical Imaging
  - Text-guided image editing
  - inversion-free editing
  - frequency-domain interaction
  - Diffusion Transformer
  - medical data augmentation
  - surgical bleeding classification
date: 2026-05-08
content_hash: a3f98cf5ecb6e6ad
---

# FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing

**Conference**: AAAI 2026
**arXiv**: [2511.12151](https://arxiv.org/abs/2511.12151)
**Code**: [kk42yy/FIA-Edit](https://github.com/kk42yy/FIA-Edit)
**Area**: Medical Imaging / Image Editing
**Keywords**: Text-guided image editing, inversion-free editing, frequency-domain interaction, Diffusion Transformer, medical data augmentation, surgical bleeding classification

## TL;DR

This paper proposes FIA-Edit, an inversion-free text-guided image editing framework based on frequency-interactive attention. It introduces a Frequency Representation Interaction (FRI) module that performs frequency-domain fusion of source/target features within self-attention, and a Feature Injection (FIJ) module that explicitly incorporates source image features into cross-attention. The framework achieves precise semantic editing while maintaining high background fidelity, and for the first time applies a general image editing method to clinical surgical bleeding image augmentation.

## Background & Motivation

**Background**: Text-guided image editing is an important application of diffusion models. Existing methods fall into two categories: inversion-based methods (e.g., P2P, PnP, MasaCtrl) that first invert the source image into noise space before editing—yielding high fidelity but at significant computational cost—and inversion-free methods (e.g., FlowEdit, FlowAlign) that directly construct editing trajectories via velocity field differencing, offering speed but poor background preservation.

**Limitations of Prior Work**:
- Inversion-based methods require mapping images to Gaussian noise space, which is time-consuming and complex (P2P requires 34s per image).
- Inversion-free methods, while efficient (FlowEdit takes only 3.5s per image), lack explicit integration of source image features, leading to background drift, spatial inconsistency, and over-editing.
- In the inversion-free pipeline, interaction between source and target velocity fields is only implicit, resulting in weak source image constraints.

**Key Challenge**: A fundamental trade-off exists between editing efficiency and fidelity—fast inversion-free methods fall far short of time-consuming inversion-based methods in background preservation.

**Goal**: To significantly improve background fidelity and semantic alignment quality while retaining the efficiency advantage of inversion-free methods.

**Key Insight**: Explicitly introduce source image feature interaction constraints into the target velocity field computation of the inversion-free framework—leveraging the natural decoupling of structure and semantics in the frequency domain to design feature interaction mechanisms in both self-attention and cross-attention.

**Core Idea**: Through selective fusion of high-frequency (source structure) and low-frequency (target semantics) components in the frequency domain, combined with source feature injection, achieve explicit source–target feature interaction within the inversion-free editing pipeline.

## Method

### Overall Architecture

FIA-Edit is built upon SD3.5-Medium (Diffusion Transformer), using FlowEdit's inversion-free Rectified Flow framework as its backbone. The core innovation lies in introducing FIA Constraints into the computation of the target velocity field $v_\theta(\mathbf{x}_t^{tar}, \mathcal{P}^{tar}, t)$ to enable explicit source–target feature interaction.

### Backbone: Inversion-Free Editing (Rectified Flow)

1. At editing timestep $\sigma_t$, noise is injected into the source image: $\mathbf{x}_t^{src} = (1-\sigma_t)\cdot\mathbf{X}^{src} + \sigma_t\cdot\epsilon_t$
2. Source velocity field $v_\theta(\mathbf{x}_t^{src}, \mathcal{P}^{src}, t)$ and target velocity field $v_\theta(\mathbf{x}_t^{tar}, \mathcal{P}^{tar}, t)$ are computed.
3. The editing direction is defined by velocity differencing: $v_t^\Delta = v^{tar} - v^{src}$
4. Edited features are iteratively updated: $\mathbf{x}_{t-1}^{FE} = \mathbf{x}_t^{FE} + (\sigma_{t-1} - \sigma_t)\cdot v_t^\Delta$

### Key Design 1: Frequency Representation Interaction (FRI)

FRI operates within the self-attention layer. The core idea is that structure and semantics can be naturally decoupled in the frequency domain:
- Low-frequency components → coarse spatial layout and background structure
- High-frequency components → fine-grained texture and semantic detail

**Procedure**:
1. Intermediate features $f_t^{src}, f_t^{tar} \in \mathbb{R}^{C \times H \times W}$ are extracted from source/target velocity fields.
2. 2D FFT is applied to both to obtain spectra $\mathcal{F}^{src}, \mathcal{F}^{tar}$.
3. A Gaussian low-pass filter $\mathcal{L}$ decomposes each into high- and low-frequency components.
4. Cross-weighted fusion: $\mathcal{F}^{fused} = \lambda_1(\mathcal{F}^{src}_{high} + \mathcal{F}^{tar}_{low}) + \lambda_2(\mathcal{F}^{src}_{low} + \mathcal{F}^{tar}_{high})$
5. With $\lambda_1=0.8, \lambda_2=0.2$, emphasizing high-frequency source structure and low-frequency target semantics.
6. The fused result is converted back to the spatial domain via IFFT and injected into the self-attention layer.

**Design Intuition**: Preserve high-frequency structural information (edges, textures) from the source image while allowing low-frequency semantic changes from the target to propagate—achieving content modification without structural alteration.

### Key Design 2: Feature Injection (FIJ)

FIJ operates within the cross-attention layer, drawing inspiration from feature injection strategies in inversion-based methods (PnP, MasaCtrl):
- In the latter half of DiT layers (layers 13–23), source Q, K, V, and text embeddings replace their counterparts in the target branch during cross-attention.
- Applied only during early generation steps (first 27 of 50 steps), when source and target features remain similar.
- Early-stage fusion allows the target features to smoothly absorb source information, avoiding abrupt transitions.

$$Q^{tar} \leftarrow Q^{src},\quad K^{tar} \leftarrow K^{src},\quad V^{tar} \leftarrow V^{src},\quad \mathbf{e}^{tar} \leftarrow \mathbf{e}^{src}$$

### Loss & Training

FIA-Edit is a tuning-free method involving no training loss. The core mechanism applies constraints during inference within the velocity field computation:
$$v_t^\Delta = v_\theta(\mathbf{x}_t^{tar}, \mathcal{P}^{tar}, t, \text{FIA}(\{f_t^{src}\}, \{f_t^{tar}\})) - v_\theta(\mathbf{x}_t^{src}, \mathcal{P}^{src}, t)$$

## Key Experimental Results

### Evaluation Benchmark

- **PIE-Bench**: 700 image–prompt pairs covering 10 categories of editing tasks.
- **Baselines**: 13 state-of-the-art methods (5 LDM-based, 4 FLUX-based, 4 DiT-based).
- **Metrics**: Structure Distance, PSNR, LPIPS, MSE, SSIM (background preservation) + CLIP Similarity (semantic alignment).

### Main Results (PIE-Bench)

| Method | Structure Dist.↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ | CLIP-Whole↑ | CLIP-Edit↑ | Avg Rank↓ |
|--------|-------------------|-------|--------|------|-------|-------------|------------|-----------|
| FlowEdit | 23.62 | 23.21 | 93.81 | 69.95 | 85.09 | **26.78** | **23.73** | 6.1 |
| DNAEdit | 14.19 | 26.66 | 74.57 | 32.76 | 88.63 | 25.63 | 22.71 | 3.1 |
| **FIA-Edit** | **10.34** | **27.32** | **55.02** | **28.66** | **89.21** | 25.89 | 22.82 | **1.7** |

FIA-Edit achieves state-of-the-art performance across all background preservation metrics and ranks first overall (Avg Rank = 1.7).

### Efficiency Comparison

| Method | GPU Memory (GB) | Inference Time (s) |
|--------|-----------------|---------------------|
| P2P | 10.95 | 34.84 |
| FlowEdit | 17.93 | 3.49 |
| **FIA-Edit** | 17.93 | **6.30** |

FIA-Edit adds only ~3s overhead over FlowEdit (~6s per image, 512×512, RTX 4090) while substantially improving fidelity.

### Ablation Study

| FIJ | FRI | Struct.Dist.↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ |
|-----|-----|---------------|-------|--------|------|-------|
| ✗ | ✗ | 23.62 | 23.21 | 93.81 | 69.95 | 85.09 |
| ✓ | ✗ | 14.89 | 25.59 | 70.18 | 41.74 | 87.51 |
| ✓ | add | 16.50 | 25.93 | 85.44 | 38.72 | 86.51 |
| ✓ | **freq** | **10.34** | **27.32** | **55.02** | **28.66** | **89.21** |

FIJ alone already yields significant improvements in background preservation; the frequency-domain fusion design of FRI outperforms naive spatial feature addition (add).

### Medical Application: Surgical Bleeding Classification

- **Dataset**: Laparoscopic surgery videos (140 videos, 770K frames); only 44K bleeding frames in training set (severe class imbalance).
- **Task**: Generate augmented data with varying bleeding severity by editing early-stage bleeding frames.
- **Results** (ConvNeXt-T classifier):

| Method | AUC (%) | Recall (%) | F1 (%) |
|--------|---------|------------|--------|
| No augmentation | 81.54 | 29.49 | 37.35 |
| FlowEdit augmentation | 83.83 | 31.44 | 38.86 |
| **FIA-Edit augmentation** | **85.05** | **32.90** | **40.89** |

FIA-Edit significantly improves downstream classification performance through high-fidelity bleeding editing, marking the first application of a general image editing method to clinical data augmentation.

### Key Findings

1. Explicitly introducing source feature interaction in inversion-free methods is critical for improving fidelity.
2. Frequency-domain fusion decouples structure and semantics more effectively than simple spatial-domain operations.
3. Restricting FIJ to early steps and latter-half layers is essential for preserving editing flexibility.
4. The medical application validates the practical value of general editing methods for data augmentation.

## Highlights & Insights

1. **Elegant frequency-domain decoupling**: The natural separation of structure and semantics in the frequency domain enables high-quality cross-domain feature fusion at negligible memory cost.
2. **Lightweight and efficient design**: Both FRI and FIJ are lightweight modules that add only ~3s to inference time without increasing GPU memory usage.
3. **Pioneering medical application**: This is the first work to apply general text-guided image editing to clinical surgical images, generating bleeding variation data to address medical data imbalance.
4. **Insight from frequency weight design**: $\lambda_1=0.8$ (source high-frequency + target low-frequency) greatly exceeds $\lambda_2=0.2$ (source low-frequency + target high-frequency), demonstrating that effective editing requires preserving source structure while transferring target semantics.
5. **Early injection strategy**: FIJ injects source features only during the first 27 of 50 steps, reflecting a "stabilize first, then transform" editing philosophy.

## Limitations & Future Work

1. Based on SD3.5-Medium, the model is large and difficult to deploy on edge devices.
2. Accurate source/target prompts must be provided manually; prompt quality directly affects editing results.
3. The frequency fusion hyperparameters ($\lambda_1=0.8, \lambda_2=0.2$) may require adjustment across different editing tasks.
4. The selection of FIJ layer range (layers 13–23) and step count (first 27 steps) lacks an adaptive mechanism.
5. The medical application is validated only on laparoscopic bleeding augmentation; generalizability to other clinical scenarios remains unknown.
6. CLIP semantic editing scores are slightly lower than FlowEdit (trade-off), indicating a residual tension between background preservation and semantic editing quality.

## Related Work & Insights

- **Inversion-based methods**: P2P (attention replacement), PnP (feature injection), MasaCtrl, FlexiEdit (frequency domain), FDS (wavelet decomposition).
- **Inversion-free methods**: InfEdit (DDCM consistency sampling), FlowEdit (velocity field differencing), FlowAlign (trajectory regularization).
- **Frequency-domain operations**: FlexiEdit (suppressing high-frequency DDIM latents), FDS (adaptive frequency band selection in wavelet domain).
- **DiT-based methods**: FTEdit (AdaLN semantic replacement), DNAEdit (reducing inversion bias).

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Frequency-domain interaction approach is novel; medical application is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparison against 13 baselines, thorough ablation, with medical application validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Code is publicly available; fast inference (~6s per image) makes it practically deployable.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services](egoems_a_high-fidelity_multimodal_egocentric_dataset_for_cognitive_assistance_in.md)
- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)

<!-- RELATED:END -->
