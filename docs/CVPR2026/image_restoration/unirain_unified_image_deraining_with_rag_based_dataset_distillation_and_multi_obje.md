---
title: >-
  [Paper Note] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization
description: >-
  [CVPR 2026][Image Restoration][Unified deraining] UniRain is a unified deraining framework that employs RAG-driven dataset distillation to select high-quality samples from public datasets, and introduces a multi-objective reweighted optimization strategy within an asymmetric MoE architecture to balance learning across different rain degradation types, achieving state-of-the-art performance across four scenarios: daytime/nighttime rain streaks and raindrops.
tags:
  - CVPR 2026
  - Image Restoration
  - Unified deraining
  - RAG-based dataset distillation
  - multi-objective optimization
  - MoE architecture
  - rain streaks/raindrops
date: 2026-05-08
content_hash: 3b2fe1bcf10f9e19
---

# UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization

**Conference**: CVPR 2026
**arXiv**: [2603.03967](https://arxiv.org/abs/2603.03967)
**Code**: [GitHub](https://github.com/QianfengY/UniRain)
**Area**: Image Restoration / Deraining
**Keywords**: Unified deraining, RAG-based dataset distillation, multi-objective optimization, MoE architecture, rain streaks/raindrops

## TL;DR

UniRain is a unified deraining framework that employs RAG-driven dataset distillation to select high-quality samples from public datasets, and introduces a multi-objective reweighted optimization strategy within an asymmetric MoE architecture to balance learning across different rain degradation types, achieving state-of-the-art performance across four scenarios: daytime/nighttime rain streaks and raindrops.

## Background & Motivation

Existing deraining methods are typically designed for specific rain degradation types and fail to generalize across scenarios. Achieving unified deraining faces two major challenges:

1. **Mixed dataset quality imbalance**: Directly merging all publicly available synthetic/real deraining datasets (over 2 million image pairs) introduces substantial heterogeneity in background quality, resolution, and rain formation patterns, with low-quality data introducing erroneous supervision signals.
2. **Optimization imbalance across degradation types**: Different rain degradation types (daytime/nighttime rain streaks/raindrops) vary in difficulty and convergence speed, causing unified training to bias toward simpler types while neglecting harder ones.

## Method

### Overall Architecture

Public deraining datasets → RAG-based dataset distillation (retrieving real-world references + VLM ensemble voting for quality assessment) → distilled high-quality mixed dataset → Soft-MoE encoder + Hard-MoE decoder → multi-objective reweighted optimization for dynamic loss balancing → deraining output.

### Key Designs

1. **RAG-driven Dataset Distillation**:
   - **Function**: Selects reliable training samples from large-scale public datasets.
   - **Mechanism**: Retrieval stage — constructs a real-rain-image database and performs three-level similarity matching (CLIP text semantics → CLIP visual features → SSIM structural similarity) to retrieve the most relevant real-rain references for each query image; Generation stage — query image, retrieved references, and prompt templates are fed into three VLMs for quality assessment, with majority voting determining sample retention.
   - **Design Motivation**: Real rain images serve as references to help VLMs judge whether synthetic data is sufficiently realistic.

2. **Multi-objective Reweighted Optimization**:
   - **Function**: Dynamically balances training across different rain degradation types.
   - **Mechanism**: Linear regression over a sliding window estimates the convergence slope $\alpha$ for each type's loss → Type Balance Score (TBS, higher weight for slower convergence) + Type Stability Score (TSS, lower weight for diverging types) + Adaptive Factor (AF, TBS-dominant in early training, TSS-dominant in later stages) → dynamic loss weights.
   - **Design Motivation**: A single optimization objective causes the model to favor nighttime rain streaks (easier) while neglecting daytime raindrops (harder).

3. **Asymmetric MoE Architecture**:
   - **Function**: Encoder collaboratively preserves diverse degradation cues; decoder precisely reconstructs fine details.
   - **Mechanism**: Soft-MoE encoder (continuous weighted combination of all experts) + Hard-MoE decoder (Top-k routing activates the most relevant experts).
   - **Design Motivation**: Encoding requires comprehensive feature fusion, while decoding requires focused and fine-grained reconstruction.

### Loss & Training

- L1 reconstruction loss + perceptual loss, dynamically weighted per rain type by the multi-objective reweighting strategy.
- Convergence slope estimation via linear regression within a window of size $N$; TBS, TSS, and AF are jointly applied.
- VLM ensemble: InternVL2.5-8B + LLaVA-NeXT-7B + MobileVLM-3B with majority voting.

## Key Experimental Results

### Main Results (Proposed RainRAG Benchmark)

| Method | DRS PSNR | DRD PSNR | NRS PSNR | NRD PSNR | Avg. PSNR |
|--------|----------|----------|----------|----------|-----------|
| Restormer | 28.45 | 23.36 | 33.92 | 25.85 | 27.89 |
| MSDT | 28.60 | 23.31 | 34.56 | 25.28 | 27.94 |
| UniRain | 29.58 | 24.71 | 35.23 | 26.21 | 28.93 |

### Ablation Study

| Configuration | Avg. PSNR | Notes |
|---------------|-----------|-------|
| Direct mixture training | 27.55 | Interference from uneven data quality |
| + RAG distillation | 28.32 | Gains from high-quality data |
| + MoE | 28.61 | Benefits of asymmetric expert design |
| + Multi-objective optimization | 28.93 | Final, balanced across all types |

### Key Findings

- RAG distillation filters approximately 700K high-quality pairs from 2M+ image pairs, yielding a distillation ratio of ~35%.
- Multi-objective optimization substantially narrows the performance gap across rain degradation types.
- The asymmetric combination of Soft-MoE encoder and Hard-MoE decoder outperforms symmetric designs.
- UniRain demonstrates the most consistent visual quality in real-world scene evaluation.

## Highlights & Insights

- First work to introduce RAG into low-level vision data filtering, leveraging retrieval to enhance VLM-based quality assessment.
- The three-indicator design of TBS/TSS/AF for multi-objective reweighting is comprehensive, addressing both convergence speed and training stability.
- Unifying four rain degradation types offers high practical value.
- The dataset distillation pipeline is transferable to other image restoration tasks.

## Limitations & Future Work

- VLM-based quality assessment remains imperfect and may incorrectly retain or discard samples.
- The four-type rain taxonomy is coarse and does not cover other weather degradations such as fog or snow.
- The number of experts and the Top-k value in MoE require manual tuning.
- Comparisons with recent large-scale restoration models (e.g., diffusion-based methods such as DiffIR) are absent.

## Related Work & Insights

- **vs. Restormer/MSDT**: Trained on single degradation types with poor cross-scenario generalization; UniRain jointly handles all four types.
- **vs. URIR**: The earliest unified deraining method, but limited to driving scenarios; UniRain covers a broader range of scenes.
- **vs. ReFIR**: RAG is applied at the inference stage for image restoration; UniRain applies RAG at the training stage for data filtering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of RAG-based dataset distillation and multi-objective optimization is novel in the deraining domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes a self-constructed benchmark, multiple public datasets, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation figures are intuitive, method descriptions are clear, and formulations are detailed.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution of dataset distillation and balanced optimization for unified image restoration.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[CVPR 2026\] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)

<!-- RELATED:END -->
