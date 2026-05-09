---
title: >-
  [Paper Note] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset
description: >-
  [CVPR 2026][Image Restoration][infrared-super-resolution] This paper proposes Real-IISR, a visual autoregressive framework guided by thermal-structural cues, which achieves real-world infrared image super-resolution via a conditionally adaptive codebook and a thermal order consistency loss. The first real-world infrared SR dataset, FLIR-IISR, is also introduced.
tags:
  - CVPR 2026
  - Image Restoration
  - infrared-super-resolution
  - autoregressive
  - codebook
  - thermal-guidance
  - benchmark
date: 2026-05-08
content_hash: 798dc1343f0ee604
---

# Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset

**Conference**: CVPR 2026
**arXiv**: [2603.04745](https://arxiv.org/abs/2603.04745)
**Code**: N/A
**Area**: Image Restoration
**Keywords**: infrared-super-resolution, autoregressive, codebook, thermal-guidance, benchmark

## TL;DR

This paper proposes Real-IISR, a visual autoregressive framework guided by thermal-structural cues, which achieves real-world infrared image super-resolution via a conditionally adaptive codebook and a thermal order consistency loss. The first real-world infrared SR dataset, FLIR-IISR, is also introduced.

## Background & Motivation

Infrared image super-resolution (IISR) is critical in applications such as object detection, tracking, and autonomous driving. Existing visible-light SR methods face fundamental challenges in the infrared domain:

1. **Lack of real-world degradation datasets**: Most existing IISR methods are trained on downsampled infrared-visible fusion datasets, failing to capture real optical-sensor coupled degradations.
2. **Lack of infrared-aware degradation modeling**: Diffusion models rely on fixed degradation priors, ignoring spatially heterogeneous blur and noise; visual autoregressive models are confined to visible-light images and lack infrared-specific constraints.
3. **Misalignment between thermal radiation and structural edges**: In infrared imaging, thermal intensity and object boundaries are often misaligned, leading to boundary distortion and thermal drift.

## Method

### Overall Architecture

Real-IISR adopts the VAR (Visual AutoRegressive) backbone, generating super-resolved results scale-by-scale via next-scale prediction. It comprises three core modules.

### 1. Thermal-Structural Guidance Module (TSG)

Two auxiliary representations are constructed from the low-resolution input: a thermal map $\mathbf{I}_{\text{Heat}}$ and an edge map $\mathbf{I}_{\text{Edge}}$, with features extracted via a DINOv3 pretrained encoder. An adaptive attention gate fuses the two modalities:

$$\mathbf{F}_{\text{Fused}} = \mathbf{F}_{\text{Heat}} \odot \mathbf{W} + \mathbf{F}_{\text{Edge}} \odot (1 - \mathbf{W})$$

where $\mathbf{W} = \sigma(L(\mathbf{A}) + G(\mathbf{A}))$, and $L(\cdot)$, $G(\cdot)$ denote local and global attention operators, respectively. The fused features are propagated to LR features via a cross-attention module.

### 2. Conditionally Adaptive Codebook (CAC)

To address VQ-VAE quantization bias, a low-rank perturbation is introduced to dynamically modulate codebook embeddings:

$$\mathbf{Z}'(g)[i] = \mathbf{Z}[i] + \tanh(\alpha)[(\mathbf{U}_i \odot \mathbf{h}(g))\mathbf{V}^\top]$$

where $\mathbf{h}(g)$ is a conditioning vector derived from TSG features and $\mathbf{U}_i \in \mathbb{R}^r$ is a low-rank basis vector. This allows the same discrete index to decode different embeddings under varying degradation conditions.

### 3. Thermal Order Consistency Loss ($\mathcal{L}_{\text{TOC}}$)

This loss enforces a monotonic relationship of thermal intensity between SR and HR outputs:

$$\mathcal{L}_{\text{TOC}} = \frac{1}{|\Omega|}\sum_{(i,j)\in\Omega} \text{ReLU}\left(-\left[(\mathbf{I}_{\text{SR}}^p(i) - \mathbf{I}_{\text{SR}}^p(j)) \times (\mathbf{I}_{\text{HR}}^p(i) - \mathbf{I}_{\text{HR}}^p(j))\right]\right)$$

It constrains the relative brightness ordering between adjacent patch pairs rather than absolute values, providing robustness to spatial misalignment between LR and HR.

### Overall Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{MSE}} + \lambda_2 \mathcal{L}_{\text{TOC}}$$

where $\lambda_1=0.2$ and $\lambda_2=0.8$.

### FLIR-IISR Dataset

Captured using a FLIR T1050sc camera at 1024×768 resolution, comprising 1,457 LR-HR image pairs across 6 cities, 3 seasons, and 12 scene categories, with two types of real degradation: defocus blur and motion blur.

## Key Experimental Results

### No-Reference Metric Comparison (FLIR-IISR & M³FD)

| Method | Type | FLIR-Set5 MUSIQ↑ | FLIR-Set5 MANIQA↑ | FLIR-Set15 MUSIQ↑ | M³FD-Set5 MUSIQ↑ |
|--------|------|----------|---------|----------|----------|
| DifIISR | IISR | 54.79 | 0.3672 | 53.16 | 40.46 |
| VARSR | R-ISR | 52.76 | 0.2948 | 51.99 | 38.94 |
| SinSR | R-ISR | 54.16 | 0.3719 | 53.09 | 40.91 |
| **Real-IISR** | **R-IISR** | **59.90** | **0.3776** | **57.06** | **41.58** |

### Full-Reference Metric Comparison (FLIR-IISR)

| Method | Set5 PSNR↑ | Set5 LPIPS↓ | Set15 PSNR↑ | Set15 LPIPS↓ |
|--------|-----------|-----------|-----------|-----------|
| DifIISR | 27.20 | 0.2525 | 28.56 | 0.2739 |
| VARSR | 26.98 | 0.2304 | 28.34 | 0.2003 |
| **Real-IISR** | **28.51** | **0.1615** | **29.51** | **0.1340** |

### Efficiency Analysis

The model has 1,144.6M parameters and runs at 2.45 FPS (A800), 6% faster than VARSR, achieving the best perceptual quality. The autoregressive framework avoids the multi-step denoising bottleneck of diffusion models through deterministic generation.

## Highlights & Insights

- **First real-world infrared SR dataset** FLIR-IISR, capturing genuine optical and motion degradations.
- **Dual thermal-structural guidance** explicitly models thermal radiation and structural edges, addressing the infrared-specific thermal-structural misalignment problem.
- **Conditionally adaptive codebook** adapts discrete representations to varying degradation conditions via low-rank perturbation.
- **Thermal order consistency loss** preserves the physical consistency of the temperature-brightness monotonic relationship.
- Ablation studies validate the effectiveness of each module; the VAR backbone outperforms the diffusion-based backbone.

## Limitations & Future Work

- Large model size (1,144.6M parameters) incurs high deployment cost.
- The dataset is relatively limited in scale (1,457 pairs), and scene diversity warrants further expansion.
- Only 4× super-resolution is supported; generalization to other scale factors has not been verified.
- The DINOv3 pretrained encoder is not fine-tuned on infrared data, potentially introducing domain gaps.

## Rating

⭐⭐⭐⭐ — This work systematically addresses real-world infrared SR for the first time, offering a solid tri-partite contribution of dataset, method, and benchmark. The thermal-structural guidance and thermal order consistency loss are novel with clear physical intuition. Model efficiency and dataset scale leave room for improvement.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](unicac_universal_computational_aberration_correction_benchmark.md)
- [\[CVPR 2026\] FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution](finpercep_rm_fine_grained_reward_model_rl_super_resolution.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_rag_dataset_distillation.md)

<!-- RELATED:END -->
