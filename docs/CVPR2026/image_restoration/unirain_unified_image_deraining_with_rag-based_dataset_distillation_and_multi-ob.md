---
title: >-
  [Paper Note] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization
description: >-
  [CVPR 2026][Image Restoration][image_restoration] This paper proposes UniRain, a unified deraining framework that distills high-quality training samples from over 2 million public image pairs via RAG-driven dataset distillation, combines an asymmetric Mixture-of-Experts (MoE) architecture with a multi-objective adaptive reweighting optimization strategy, and for the first time handles all four degradation types — daytime rain streaks, daytime raindrops, nighttime rain streaks, and nighttime raindrops — within a single model.
tags:
  - CVPR 2026
  - Image Restoration
  - image_restoration
  - deraining
  - mixture_of_experts
  - multi_objective_optimization
  - RAG
date: 2026-05-08
content_hash: be262335dca82d30
---

# UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization

**Conference**: CVPR 2026
**arXiv**: [2603.03967](https://arxiv.org/abs/2603.03967)
**Code**: N/A
**Area**: Image Restoration
**Keywords**: image_restoration, deraining, mixture_of_experts, multi_objective_optimization, RAG

## TL;DR

This paper proposes UniRain, a unified deraining framework that distills high-quality training samples from over 2 million public image pairs via RAG-driven dataset distillation, combines an asymmetric Mixture-of-Experts (MoE) architecture with a multi-objective adaptive reweighting optimization strategy, and for the first time handles all four degradation types — daytime rain streaks, daytime raindrops, nighttime rain streaks, and nighttime raindrops — within a single model.

## Background & Motivation

Existing deraining methods face two core challenges:

1. **Uneven data quality**: Directly mixing all synthetic and real datasets (>2 million pairs) introduces inaccurate supervision signals, which degrades model convergence and generalization. Experiments confirm that naive mixing can even underperform a carefully curated subset.
2. **Training imbalance**: Different rain degradation types — daytime rain streaks (DRS), daytime raindrops (DRD), nighttime rain streaks (NRS), and nighttime raindrops (NRD) — vary substantially in difficulty and convergence rate, causing unified optimization to favor simpler types while neglecting harder ones.

## Method

### 1. RAG-based Dataset Distillation

#### Retrieval Stage
A database is constructed from a large-scale corpus, storing a triplet $(T_r, f_r, I_r)$ for each real rain image (BLIP text description, CLIP visual feature, and image).

A three-level hierarchical similarity matching is applied to each query image:
- **Semantic similarity**: $s_{txt}(q,r) = \|\phi_T(T_q) - \phi_T(T_r)\|_2$, selecting Top-$K_1$
- **Visual feature similarity**: $s_{vis}(q,r') = \frac{f_q^\top f_{r'}}{\|f_q\|_2 \|f_{r'}\|_2}$, selecting Top-$K_2$
- **Structural similarity**: $s_{perc}(q,r'') = SSIM(I_q, I_{r''})$, selecting Top-$K_3$

#### Generation Stage
Retrieved reference images are combined with the query image, and three VLMs (InternVL2.5-8B, LLaVA-NeXT-7B, MobileVLM-3B) vote to assess data quality:

$$\hat{R}_q = \begin{cases} 1 & \text{if } \sum_{i=1}^3 \mathbb{I}(R_q^i = 1) \geq 2 \\ 0 & \text{otherwise} \end{cases}$$

This process distills 52,869 high-quality training pairs from over 2 million image pairs, retaining approximately 2.6% of the original data.

### 2. Multi-objective Adaptive Reweighted Optimization

The convergence slope $\alpha_i$ for each degradation type is estimated via sliding-window linear regression, from which three dynamic weight metrics are derived:

- **Type Balance Score (TBS)**: Shifts attention toward types with slower convergence.
  $$\mathrm{TBS}_i(t) = \text{softmax}_i\left(K \frac{\alpha_i(t)}{\sum_i |\alpha_i(t)|}\right)$$

- **Type Stability Score (TSS)**: Suppresses excessive weights for diverging types.
  $$\mathrm{TSS}_i(t) = \text{softmax}_i\left(-N \frac{\alpha_i(t)}{\sum_{k=t-N+1}^t |\alpha_i(k)|}\right)$$

- **Adaptivity Factor (AF)**: Dynamically adjusts the balance between TBS and TSS.
  $$AF(t) = \min\left(t \cdot \text{softmax}_t\left(-\frac{\tau t \cdot \alpha_{\max}(t)}{\sum_{i=1}^t \alpha_{\max}(i)}\right), 1\right)$$

Final weight: $\omega_i(t) = AF(t) \cdot TBS(t) + (1 - AF(t)) \cdot TSS(t)$

### 3. Asymmetric MoE Architecture

- **Encoder (Soft-MoE)**: Outputs from all experts are aggregated via continuous weighting to comprehensively preserve diverse degradation cues.
  $$y_{en} = \sum_{i=1}^N \mathcal{R}_{soft}^i \otimes y_{en}^i$$

- **Decoder (Hard-MoE)**: Top-k routing selectively activates the most relevant experts to focus on fine-grained texture reconstruction.
  $$y_{de} = \sum_{i=1}^N \mathcal{R}_{hard}^i \cdot y_{de}^i$$

## Key Experimental Results

### Table 1: Unified Evaluation on RainRAG Dataset Across Four Degradation Types

| Method | DRS PSNR | DRD PSNR | NRS PSNR | NRD PSNR | Avg. PSNR↑ | Avg. SSIM↑ |
|--------|----------|----------|----------|----------|------------|------------|
| Restormer | 28.45 | 23.36 | 33.92 | 25.85 | 27.89 | 0.8405 |
| MSDT | 28.60 | 23.31 | 34.56 | 25.28 | 27.94 | 0.8410 |
| NeRD-Rain | 28.11 | 23.30 | 33.88 | 25.31 | 27.65 | 0.8340 |
| URIR | 28.29 | 23.19 | 34.32 | 25.82 | 27.91 | 0.8425 |
| **UniRain** | **29.58** | **24.71** | **35.23** | **26.21** | **28.93** | **0.8515** |

### Table 2: Average Performance on Real-world Public Benchmarks

| Method | Avg. PSNR↑ | Avg. SSIM↑ |
|--------|------------|------------|
| NeRD-Rain | 27.81 | 0.8132 |
| URIR | 27.69 | 0.8061 |
| **UniRain** | **29.42** | **0.8222** |

UniRain achieves consistent and significant improvements across all four degradation types and all real-world benchmarks, surpassing the previous state of the art by approximately 1 dB in average PSNR.

## Highlights & Insights

- The **RAG + VLM dataset distillation** paradigm is novel: it transfers retrieval-augmented generation from NLP to low-level visual data curation, achieving better performance by retaining only 2.6% of the original data.
- The **multi-objective adaptive reweighting** strategy effectively addresses type imbalance in mixed training; the three-level TBS/TSS/AF design is internally coherent.
- The **asymmetric MoE** design — soft routing in the encoder and hard routing in the decoder — is intuitive, reflecting an exploration-versus-exploitation principle.
- UniRain is the first unified deraining framework to simultaneously cover daytime/nighttime rain streaks and raindrops.
- Model complexity is comparable to competing methods (126.5G FLOPs, 24.4M parameters).

## Limitations & Future Work

- The RAG data distillation pipeline requires inference across multiple VLMs, incurring high upfront computational cost.
- The accuracy of VLM-based quality assessment depends on prompt engineering and VLM capability, which may introduce bias.
- The window size $N$ and sensitivity parameter $\tau$ in the multi-objective optimization require manual tuning.
- The framework addresses only rain-related degradations and has not been extended to other adverse weather conditions such as haze or snow.
- Performance gains on the nighttime raindrop (NRD) subset are relatively modest (+0.39 dB), indicating that complex degradations remain challenging.

## Rating

⭐⭐⭐⭐ — The problem formulation is practically motivated, and the combination of RAG-based dataset distillation and multi-objective optimization is logically coherent and effective. The unified framework offers strong practical utility; however, the scalability of the RAG pipeline and its generalizability to other degradation types remain to be validated.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_network_for_expansive_receptive_fields_in_l.md)
- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[CVPR 2026\] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)

<!-- RELATED:END -->
