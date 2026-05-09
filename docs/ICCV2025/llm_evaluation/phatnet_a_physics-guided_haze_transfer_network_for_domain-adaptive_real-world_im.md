---
title: >-
  [Paper Note] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing
description: >-
  [ICCV 2025][LLM Evaluation][image dehazing] This paper proposes PHATNet, a physics-guided haze transfer network that extends the Atmospheric Scattering Model (ASM) to latent space to disentangle and transfer haze patterns, generating domain-adaptive fine-tuning datasets that enable dehazing models to effectively adapt to unseen real-world haze scenes at test time.
tags:
  - ICCV 2025
  - LLM Evaluation
  - image dehazing
  - domain adaptation
  - haze transfer
  - atmospheric scattering model
  - disentangled learning
date: 2026-05-08
content_hash: c40d79adfdf008f7
---

# PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing

**Conference**: ICCV 2025
**arXiv**: [2507.14826](https://arxiv.org/abs/2507.14826)
**Code**: [GitHub](https://github.com/pp00704831/PHATNet)
**Area**: LLM Evaluation
**Keywords**: image dehazing, domain adaptation, haze transfer, atmospheric scattering model, disentangled learning

## TL;DR

This paper proposes PHATNet, a physics-guided haze transfer network that extends the Atmospheric Scattering Model (ASM) to latent space to disentangle and transfer haze patterns, generating domain-adaptive fine-tuning datasets that enable dehazing models to effectively adapt to unseen real-world haze scenes at test time.

## Background & Motivation

Image dehazing aims to remove haze artifacts from images. Existing dehazing models face critical bottlenecks:

**Synthetic-to-real domain gap**: Training data synthesized via ASM fails to accurately simulate real haze distributions.

**Cross-domain performance degradation**: Even models trained on real paired data suffer significant performance drops on real haze images from unseen domains.

**Insufficient existing domain adaptation solutions**: GAN-based methods suffer from mode collapse and training instability, and fail to capture region-specific degradation patterns such as non-uniform haze.

**Core Insight**: Extracting haze patterns is generally easier than recovering haze-free content. Haze typically forms smooth, homogeneous, semi-transparent layered overlays that are more predictable in the ASM parameter space. This motivates a strategy of first disentangling haze patterns and then transferring them onto clean images from the source domain.

## Method

### Overall Architecture

PHATNet is a flexible domain adaptation framework: given a target-domain hazy image and a source-domain clean image, PHATNet transfers the haze patterns of the target domain onto the source-domain clean image to generate a domain-specific paired fine-tuning dataset. This dataset is then used to offline fine-tune the dehazing model, achieving test-time domain adaptation. The entire adaptation process is performed offline, introducing no additional inference latency.

### Key Designs

1. **Parameterized Haze Disentanglement and Transfer Module (PHDT)**: A core dual-branch network:

    - **Top Branch (Haze Disentanglement)**: Disentangles two categories of haze-related features from the hazy image $I^H$:
        - Atmospheric Light Encoder (ALE): Extracts atmospheric light features $F^{AL} = \exp(-\text{ALE}(I^H)) \in \mathbb{R}^{128}$
        - Transmission Map Encoder (TME): Extracts transmission map features $F^{TM} = \exp(-\text{TME}(I^H)) \in \mathbb{R}^{H/8 \times W/8 \times 128}$
        - Normalized to $[0,1]$ via $e^{-x}$ to maintain consistency with the physical interpretation of ASM
    - **Bottom Branch (Content Extraction)**: A content encoder (CE) extracts content features $F^J$ from the clean image $I^C$
    - **ASM-guided Fusion**: Features are fused in latent space following the ASM formulation:

    $F^I = F^J \times F^{TM} + F^{AL} \times (1 - F^{TM})$

   The key distinction is that this operation is performed in latent space, where the transmission map features serve as channel-wise attention maps that are invariant to scene depth variations, thereby avoiding ghosting artifacts. A Rehazing Encoder (RE) then generates the final haze-transferred image.

2. **Multi-scale Haze Transfer**: PHATNet employs a 3-level multi-scale PHDT structure that processes non-uniform haze from coarse to fine:

    $I^O = \text{PHDT}(I^H, I^C) + \text{UP}(\text{PHDT}(I^{H\downarrow}, I^{C\downarrow}) + \text{UP}(\text{PHDT}(I^{H\downarrow\downarrow}, I^{C\downarrow\downarrow})))$

   The 3-level structure is verified as optimal in ablation studies (PSNR: 16.25 → 16.68 → 16.95 → 17.07).

3. **ASM Parameter-domain Data Augmentation**: After generating haze-transferred images, further augmentation can be performed in the ASM parameter domain:

    - Gamma correction applied to $F^{TM}$ to modulate haze density (enhance/weaken)
    - Vertical flipping of $F^{TM}$ to create additional variants
    - Each hazy/clean image pair can yield $M \times N$ training samples

### Loss & Training

- **Haze Transfer Consistency Loss ($\mathcal{L}_{HTC}$)**: Measures the discrepancy between the haze-transferred image $I^O$ and the original hazy image $I^H$ when both correspond to the same scene:

  $$\mathcal{L}_{HTC} = \sum_{s=1}^{3} \|I^O_{i,s} - I^H_{i,s}\|_1$$

- **Content Leakage Loss ($\mathcal{L}_{CL}$)**: A clean image is fed as the haze source input to PHATNet, and the output should equal the target clean image:

  $$\mathcal{L}_{CL} = \sum_{s=1}^{3} \|I^O_{i,j,s} - I^C_{j,s}\|_1$$

  This elegantly avoids the instability of GAN training while preventing ALE and TME from capturing content features.

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{HTC} + \mathcal{L}_{CL}$
- Adam optimizer, learning rate $10^{-4}$ → $10^{-7}$ (cosine annealing), 1000 epochs, resolution $1600 \times 1200$

## Key Experimental Results

### Main Results

**Setting 1 (Source domain: NH-Haze20 → Target domain), Average PSNR (dB)**:

| Dehazing Model | Baseline | +PHATNet | Gain |
|----------------|----------|----------|------|
| FocalNet | 16.25 | 17.07 | +0.82 |
| Dehamer | 16.33 | 17.29 | +0.96 |
| MITNet | 14.83 | 16.31 | +1.48 |
| SGDN | 14.84 | 16.77 | +1.93 |

**Setting 2 (Source domain: HD-NH-Haze → Target domain), Average PSNR (dB)**:

| Dehazing Model | Baseline | +PHATNet | Gain |
|----------------|----------|----------|------|
| FocalNet | 15.38 | 16.22 | +0.84 |
| Dehamer | 15.95 | 16.61 | +0.66 |
| MITNet | 12.44 | 15.79 | +3.35 |
| SGDN | 12.99 | 16.17 | +3.18 |

PHATNet yields consistent improvements across all dehazing models and target domains, with MITNet and SGDN benefiting the most (average gain exceeding 3 dB in Setting 2).

### Ablation Study

| Component Analysis | PSNR |
|--------------------|------|
| Baseline (w/o PHATNet) | 16.25 |
| +CNN (Concatenate) | 16.62 |
| +ALE only | 16.46 |
| +TME only | 16.80 |
| +ALE + TME (full PHDT) | 17.07 |

| Content Leakage Loss | NH-Haze21 | HD-NH-Haze | DenseHaze | I-Haze | O-Haze |
|----------------------|-----------|------------|-----------|--------|--------|
| Baseline | 16.45 | 14.76 | 14.80 | 16.13 | 19.10 |
| w/o $\mathcal{L}_{CL}$ | 16.68 | 15.48 | 15.14 | 16.37 | 18.04 |
| w/ $\mathcal{L}_{CL}$ | 16.90 | 15.87 | 15.60 | 16.96 | 20.01 |

- $\mathcal{L}_{CL}$ has the largest impact on O-Haze (sparse haze, 18.04 → 20.01), as sparse haze scenes are more susceptible to content interference.
- Comparison with competing methods: PHATNet (17.07 dB) > D4+ (16.69) > CNN (16.62) > FocalNet baseline (16.25) > HTFANet (16.21) > PTTD (16.02) > TMD (15.71)
- PHATNet has 26M parameters and requires only 0.153 seconds to generate a single $1600 \times 1200$ haze-transferred image.

### Key Findings

- Operating ASM in latent space is more effective than in the image domain, as it avoids ghosting artifacts caused by scene depth variations.
- The content leakage loss is particularly important for sparse haze scenarios.
- The offline fine-tuning paradigm introduces no inference latency, making it suitable for practical deployment.
- The 3-level multi-scale PHDT structure is optimal.

## Highlights & Insights

- **A well-balanced integration of physical modeling and deep learning**: ASM is no longer used merely as a simple linear transformation in the image domain, but instead serves as an inductive bias for feature fusion in latent space.
- **The insight that "extracting haze is easier than restoring content" is simple yet powerful**: This observation shifts the focus of domain adaptation from "learning to dehaze" to "learning to transfer haze."
- **Elegant design of $\mathcal{L}_{CL}$**: By feeding a clean image as the haze source, the expected output equals the clean target — a simple yet effective mechanism to prevent content leakage.
- **General-purpose framework**: Compatible with arbitrary dehazing models; effectiveness has been validated on four SOTA models.

## Limitations & Future Work

- Prior knowledge of real-world haze distributions remains limited, constraining the scope of parameter-domain augmentation.
- Training requires paired real hazy/clean data from the source domain; the fully unpaired setting remains unexplored.
- Validation is conducted on only 7 datasets; more diverse real-world scenarios (e.g., industrial smoke, sandstorms) warrant further investigation.
- The inherent limitations of ASM (e.g., inapplicability to extremely dense or colored haze) may propagate into PHATNet.

## Related Work & Insights

- Comparison with GAN-based methods (HTFANet, D4+) demonstrates the advantages of PHDT over vanilla ASM.
- The domain adaptation approaches of TMD and PTTD operate online at inference time, while PHATNet's offline paradigm is more practical.
- This work can inspire domain adaptation strategies for other restoration tasks such as deraining and desnowing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of extending ASM to latent space is novel, with a well-integrated combination of physics and learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated on 7 datasets, 4 dehazing models, 2 settings, with complete ablation studies and comparisons against competing methods.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, with intuitive explanations for the loss function design.
- **Value**: ⭐⭐⭐⭐ A general-purpose domain adaptation framework for dehazing with direct practical relevance to real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](forcennet_foreground-centric_network_for_document_image_rectification.md)
- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](a_realworld_display_inverse_rendering_dataset.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](../../NeurIPS2025/llm_evaluation/a_highdimensional_statistical_method_for_optimizing_transfer.md)

</div>

<!-- RELATED:END -->
