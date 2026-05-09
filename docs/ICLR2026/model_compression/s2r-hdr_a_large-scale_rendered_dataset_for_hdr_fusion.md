---
title: >-
  [Paper Note] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion
description: >-
  [Model Compression] This paper proposes S2R-HDR, the first large-scale high-quality synthetic HDR fusion dataset (24,000 samples), and introduces S2R-Adapter, a domain adaptation method that bridges the synthetic-to-real gap, achieving state-of-the-art HDR fusion performance on real-world datasets.
tags:
  - Model Compression
date: 2026-05-08
content_hash: c24777b94a0f1c5e
---

# S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2504.07667](https://arxiv.org/abs/2504.07667)
- **Code**: [Project Page](https://openimaginglab.github.io/S2R-HDR)
- **Area**: Computer Vision / Image Processing
- **Keywords**: HDR Fusion, Synthetic Dataset, Domain Adaptation, Unreal Engine, Sim-to-Real

## TL;DR

This paper proposes S2R-HDR, the first large-scale high-quality synthetic HDR fusion dataset (24,000 samples), and introduces S2R-Adapter, a domain adaptation method that bridges the synthetic-to-real gap, achieving state-of-the-art HDR fusion performance on real-world datasets.

## Background & Motivation

### State of the Field
HDR fusion is critical in computational photography, autonomous driving, and related fields. However, existing HDR datasets are extremely small in scale (the largest contains only 144 images) and are mostly confined to artificially controlled, simple dynamic scenes, failing to cover extreme conditions such as direct sunlight and large-scale motion.

### Limitations of Prior Work

**Extremely small scale**: Kalantari (89 pairs), SCT (144 images), Challenge123 (123 images);

**Limited dynamics**: Most datasets contain only basic human motion, lacking diverse dynamic elements such as animals and vehicles;

**Difficult data collection**: Real HDR ground truth requires per-frame capture at different exposures with manually controlled motion, making collection time-consuming and hard to scale;

**Limited dynamic range**: Beam-splitter-based capture supports only two exposures, unable to cover scenes with extremely high dynamic range.

### Mechanism
Leverage Unreal Engine 5 to render high-quality synthetic HDR data, combined with domain adaptation techniques to bridge the synthetic-to-real gap.

## Method

### 1. S2R-HDR Dataset Construction

#### Rendering Pipeline Design

- Modify UE5's default tone mapping and gamma correction to ensure outputs remain in linear HDR space
- Store data in EXR floating-point format to avoid quantization artifacts
- Simulate handheld camera shake to improve realism

#### Scene Diversity

- **Dynamic elements**: Pedestrians, animals, vehicles, and other moving objects
- **Environment types**: Indoor/outdoor, daytime/dusk/nighttime
- **Extreme lighting**: Ultra-high dynamic range scenes including direct sunlight

#### Dataset Scale

- **1,000 sequences × 24 frames = 24,000 HDR images**
- Resolution 1920 × 1080, EXR format
- **166× larger** than the previously largest dataset

### 2. S2R-Adapter Domain Adaptation

To bridge the synthetic-to-real domain gap, a plug-and-play dual-branch adapter is proposed:

#### Share Branch
Uses a low-rank adapter to retain shared knowledge from synthetic data and prevent catastrophic forgetting:

$$
f_s = U_s V_s x, \quad r_s \ll \min(h_{in}, h_{out})
$$

#### Transfer Branch
Uses a high-rank adapter to learn domain-specific knowledge and adapt to the real data distribution:

$$
f_t = U_t V_t x, \quad r_t \geq \max(h_{in}, h_{out})
$$

#### Final Output
The two branches are fused via weighted scaling factors:

$$
f = W_0 x + \alpha_s \times f_s + \alpha_t \times f_t
$$

#### Test-Time Adaptation (TTA)
When labeled data is unavailable, scaling factors are dynamically adjusted based on model uncertainty:

$$
\alpha_s = 1 - \mathcal{U}(x); \quad \alpha_t = 1 + \mathcal{U}(x)
$$

Higher uncertainty places greater reliance on the Transfer Branch; lower uncertainty preserves more shared knowledge.

### 3. Zero Inference Overhead

Through re-parameterization, no additional computational cost is incurred at inference time.

## Experiments

### Main Results: HDR Fusion on Real-World Datasets

| Method | SCT PSNR-μ | SCT SSIM-μ | Challenge123 PSNR-μ | Challenge123 SSIM-μ |
|------|-----------|-----------|---------------------|---------------------|
| DHDRNet | 40.05 | 0.9794 | 37.83 | 0.9707 |
| AHDRNet | 42.08 | 0.9837 | 40.44 | 0.9877 |
| HDR-Transformer | 42.39 | 0.9844 | 40.70 | 0.9881 |
| SCTNet | 42.55 | 0.9850 | 40.65 | — |
| EHDRNet (S2R-HDR) | 42.93 | 0.9858 | **42.15** | **0.9895** |
| **EHDRNet + S2R-Adapter** | **43.47** | **0.9871** | 41.89 | 0.9891 |

### Ablation Study: Domain Adaptation Component Analysis

| Configuration | SCT PSNR-μ | Challenge123 PSNR-μ |
|------|-----------|---------------------|
| S2R-HDR training only | 41.32 | 39.85 |
| + Share Branch | 42.15 | 40.71 |
| + Transfer Branch | 42.78 | 41.43 |
| + Share + Transfer (S2R-Adapter) | **43.47** | **42.15** |
| Direct fine-tuning on real data | 42.55 | 40.65 |

### Dataset Quality Comparison

| Metric | Kalantari | SCT | Challenge123 | **S2R-HDR** |
|------|----------|-----|-------------|------------|
| FHLP ↑ | 15.07 | 12.43 | 26.91 | **28.02** |
| EHL ↑ | 3.07 | 2.43 | 5.19 | **5.47** |
| SI ↑ | 18.4 | 18.25 | 20.47 | **38.02** |
| DR ↑ | 2.71 | 2.55 | 2.36 | **3.86** |
| # Samples | 89 | 144 | 123 | **24,000** |

### Key Findings

1. **Models trained on S2R-HDR significantly outperform those trained on small-scale real datasets**, even in the presence of a domain gap;
2. **S2R-Adapter effectively bridges the domain gap**, yielding substantial improvements in both labeled and unlabeled adaptation settings;
3. **The dual-branch design outperforms single-branch alternatives**: the Share Branch and Transfer Branch each contribute approximately 1 dB PSNR improvement;
4. **Direct fine-tuning is inferior to S2R-Adapter**: fine-tuning directly on real data leads to overfitting and catastrophic forgetting;
5. **TTA remains effective**: even without ground-truth annotations, test-time adaptation improves performance by approximately 0.5 dB.

## Highlights & Insights

- First large-scale synthetic HDR fusion dataset, with 24,000 samples covering diverse scenes and extreme lighting conditions
- Customized UE5 rendering pipeline preserves linear HDR space and simulates handheld camera shake
- S2R-Adapter is plug-and-play and compatible with both CNN and Transformer architectures
- Supports both labeled domain adaptation and unlabeled test-time adaptation modes
- Zero additional inference overhead through re-parameterization

## Limitations & Future Work

- Synthetic data still exhibits a texture distribution gap relative to real data (visible in t-SNE visualizations)
- Rendered scenes, though diverse, remain limited and may not cover all real-world edge cases
- UE5 rendering requires substantial computational resources and artistic design effort
- The domain adaptation method depends on the representativeness of the calibration dataset

## Related Work & Insights

- **HDR Datasets**: Kalantari et al. (2017), SCT (Tel et al., 2023), Challenge123 (Kong et al., 2024)
- **HDR Fusion Methods**: AHDRNet (Yan et al., 2019), HDR-Transformer (Liu et al., 2022), DiffHDR
- **Sim-to-Real Domain Adaptation**: LoRA (Hu et al., 2021), TTA (Wang et al., 2022)
- **Synthetic Data**: Li et al. (2023) for depth estimation, Yang et al. (2023) for semantic segmentation

## Rating

- Novelty: ⭐⭐⭐⭐ — First large-scale synthetic HDR dataset, filling a significant gap in the field
- Technical Depth: ⭐⭐⭐⭐ — Well-designed rendering pipeline, dual-branch adapter, and TTA mechanism
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark, multi-architecture comparisons with comprehensive ablations
- Value: ⭐⭐⭐⭐⭐ — Both the dataset and method are directly applicable to HDR research and product development

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)

</div>

<!-- RELATED:END -->
