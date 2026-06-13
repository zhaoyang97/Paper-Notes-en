---
title: >-
  [Paper Note] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models
description: >-
  [CVPR 2026][Image Generation][unified multimodal model acceleration] FlashU conducts the first systematic redundancy analysis of native unified multimodal models…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "unified multimodal model acceleration"
  - "training-free inference optimization"
  - "task-aware pruning"
  - "diffusion head caching"
  - "dynamic layer skipping"
date: 2026-05-08
content_hash: 18f34c385655e618
---

# Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models

**Conference**: CVPR 2026 Findings  
**arXiv**: [2603.15271](https://arxiv.org/abs/2603.15271)  
**Code**: [Available](https://github.com/Rirayh/FlashU)  
**Area**: Image Generation
**Keywords**: unified multimodal model acceleration, training-free inference optimization, task-aware pruning, diffusion head caching, dynamic layer skipping

## TL;DR

FlashU conducts the first systematic redundancy analysis of native unified multimodal models, identifying parameter specialization and computational heterogeneity. Based on these findings, it proposes a training-free, task-aware acceleration framework that achieves 1.78×–2.01× speedup on Show-o2 while maintaining SOTA performance, through FFN pruning, dynamic layer skipping, adaptive guidance scaling, and diffusion head caching.

## Background & Motivation

**Background**: Native unified multimodal models (e.g., Show-o2) integrate understanding and generation into a single architecture, incurring substantial computational overhead. Existing acceleration methods adopt static, unified strategies.

**Core Problem**:
- Generation tasks (iterative denoising, multi-step ODE) and understanding tasks (single forward pass, layer-wise feature abstraction) differ fundamentally in computational characteristics.
- Unified strategies are forced to compromise and cannot optimize either task sufficiently.
- A systematic understanding of the internal mechanisms of unified models is lacking.

**Key Findings**:
- **Parameter Specialization**: A large proportion of FFN neurons are exclusively important for either generation or understanding, with a small fraction of shared critical neurons.
- **Computational Heterogeneity**: Generation tasks exhibit extremely high inter-layer feature redundancy, while understanding tasks show gradual evolution of critical token features with depth.

## Method

### Overall Architecture

FlashU is a training-free, task-aware acceleration framework consisting of:

1. **Task-Specific Network Pruning**: FFN pruning + dynamic layer skipping (shared base for both tasks).
2. **Generation Acceleration Path**: Adaptive guidance scaling + diffusion head caching.
3. **Understanding Acceleration Path**: V-Norm proxy dynamic token pruning.

### Key Designs

#### 1. Parameter Specialization Analysis

Inspired by OBD, importance scores are computed by zeroing each FFN neuron and measuring the resulting reconstruction error. Scores are computed separately on generation and understanding data, revealing that the majority of neurons are critical for only one task.

#### 2. Task-Specific FFN Pruning + Hybrid FFN

Importance is computed using the Wanda method (activation norm × weight magnitude).

For generation tasks, a **Hybrid FFN** is employed:
- Both a full path and a pruned path coexist.
- The full path is used during early denoising steps ($t \leq 0.2T$) and the pruned path for later steps.
- Understanding tasks use a static pruning mask directly.

#### 3. Dynamic Layer Skipping

Redundancy is quantified by the cosine similarity between a layer's input and output. For generation tasks, the mean over the full sequence is computed; for understanding tasks, only the last token is used. The skip list is recomputed every $T_{LS}$ steps.

#### 4. Adaptive Guidance Scaling (Generation-Specific)

An incremental strategy replaces static CFG: low guidance is applied early to preserve diversity (mode selection), while high guidance is applied later for refinement (concentration). This is the largest individual contributor to acceleration.

#### 5. Diffusion Head Caching (Generation-Specific)

The diffusion head is fully computed only every $T_{\text{cache}}$ steps; intermediate steps reuse cached hidden states, exploiting temporal coherence.

#### 6. V-Norm Proxy Token Pruning (Understanding-Specific)

A negative correlation is observed between attention scores and the L2 norm of value vectors. The v-norm is computed at a shallow layer (layer 2), and the top-$\rho$ fraction of tokens (least important) are pruned. The method is compatible with Flash Attention.

### Loss & Training

Fully training-free: FFN pruning uses a small calibration set (20 samples); dynamic layer skipping is computed online.

## Key Experimental Results

### Main Results

**Multimodal Understanding (Table 1, Show-o2 7B)**:

| Method | MME | MMB | MMMU | MMStar | Latency |
|--------|-----|-----|------|--------|---------|
| Show-o2 7B | 1620.5 | 79.3 | 48.9 | 56.6 | 1.71s |
| Emu3 8B | — | 58.5 | 31.6 | — | 1.29s |
| **+ FlashU 7B** | **1560.5** | **75.3** | **45.1** | **48.3** | **0.96s (1.78×)** |

**Image Generation GenEval (Table 2)**:

| Method | Overall | Latency |
|--------|---------|---------|
| Emu3 8B | 0.66 | 110.5s |
| Show-o2 1.5B | 0.73 | 10.61s |
| + FlashU 1.5B | 0.71 | 5.28s (2.01×) |
| Show-o2 7B | 0.76 | 22.74s |
| + FlashU 7B | 0.72 | 11.82s (1.92×) |

### Ablation Study

**7B DPG-Bench Ablation (Table 4)**:

| Strategy | Latency | Score |
|----------|---------|-------|
| Show-o2 | 11.30s | 86.14 |
| FlashU (full) | 5.89s (1.92×) | 84.39 |
| w/o Dynamic Layer Pruning | 6.70s (1.69×) | 85.56 |
| w/o Hybrid Network | 5.65s (2.00×) | 83.68 |
| w/o Adaptive Guidance | 8.86s (1.28×) | 85.34 |
| FFN Pruning Only | 9.21s (1.23×) | 85.08 |

### Key Findings

- Adaptive guidance scaling contributes the largest speedup (removal reduces acceleration from 1.92× to 1.28×).
- Hybrid Network is critical for generation quality (removal causes a 0.71-point score drop).
- After acceleration, FlashU 7B still outperforms unaccelerated Emu3 8B on MMMU (45.1 vs. 31.6).
- Understanding achieves 1.78× speedup; generation achieves up to 2.01×.

## Highlights & Insights

1. **First Systematic Analysis of Unified Models**: The findings on parameter specialization and computational heterogeneity lay the groundwork for future research.
2. **Task-Aware Orthogonal Design**: Separate acceleration paths for generation and understanding share a common underlying model.
3. **Fully Training-Free**: Requires only a small calibration set; plug-and-play deployment.
4. **V-Norm Proxy**: An elegant solution to the unavailability of explicit attention scores under Flash Attention.

## Limitations & Future Work

1. Generation performance degrades slightly (GenEval: 0.76 → 0.72).
2. Validation is limited to Show-o2; generalizability to other unified models remains to be verified.
3. Periodic recomputation of the dynamic layer skip list introduces additional overhead.
4. The robustness of the negative-correlation assumption underlying V-Norm requires further validation.
5. The method is orthogonal to quantization and can be combined for further speedup.

## Related Work & Insights

- **Show-o2**: The target unified model.
- **Wanda**: Importance metric for FFN pruning.
- **Flash Attention**: Motivates the necessity of the V-Norm proxy.
- **Insight**: Parameter specialization implies that a "single model, two paths" paradigm may represent an efficient paradigm for unified AI.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4.5 | First systematic analysis + task-aware paradigm |
| Technical Depth | 4 | Multi-component synergy; V-Norm is elegant |
| Experimental Thoroughness | 4 | Dual-task, multi-benchmark evaluation |
| Writing Quality | 4.5 | In-depth analysis with intuitive figures |
| Value | 4.5 | Training-free, plug-and-play |
| Overall | 4.3 | |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts](tag-moe_task-aware_gating_for_unified_generative_mixture-of-experts.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)

</div>

<!-- RELATED:END -->
