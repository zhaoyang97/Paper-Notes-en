---
title: >-
  [Paper Note] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer
description: >-
  [CVPR2026][Model Compression][Diffusion Transformer] SODA is proposed to achieve controllable-speedup high-fidelity generation for Diffusion Transformers without training, via offline fine-grained sensitivity modeling, dynamic-programming-based cache schedule optimization, and a unified adaptive pruning strategy.
tags:
  - CVPR2026
  - Model Compression
  - Diffusion Transformer
  - training-free acceleration
  - caching
  - pruning
  - sensitivity modeling
  - dynamic programming
date: 2026-05-08
content_hash: 660ab3cd21b1097c
---

# SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer

**Conference**: CVPR2026  
**arXiv**: [2603.07057](https://arxiv.org/abs/2603.07057)  
**Code**: [leaves162/SODA](https://github.com/leaves162/SODA)  
**Area**: Model Compression  
**Keywords**: Diffusion Transformer, training-free acceleration, caching, pruning, sensitivity modeling, dynamic programming

## TL;DR

SODA is proposed to achieve controllable-speedup high-fidelity generation for Diffusion Transformers without training, via offline fine-grained sensitivity modeling, dynamic-programming-based cache schedule optimization, and a unified adaptive pruning strategy.

## Background & Motivation

**State of the Field**: Diffusion Transformers excel at image and video generation, but repeated sampling steps and Transformer blocks result in low inference efficiency, severely hindering deployment.

**Limitations of Prior Work**: Training-based acceleration methods such as distillation and fine-tuning incur high computational costs and limited generalizability, motivating the pursuit of training-free acceleration. Caching offers high efficiency at the cost of fidelity, while pruning is flexible but less efficient; combining both is desirable. Existing methods such as ToCa and DuCa rely on fixed or heuristic configurations for cache intervals and pruning rates, capturing only coarse-grained sensitivity trends and failing to account for fine-grained variation across timesteps, layers, and module dimensions.

**Root Cause**: Heuristic schemes inevitably skip computations that are highly sensitive to acceleration, leading to degraded generation fidelity and poor cross-model generalizability.

**Paper Goals**: To design a training-free acceleration framework that models sensitivity at fine granularity and optimizes caching and pruning decisions jointly and adaptively.

## Method

### Overall Architecture

SODA consists of three modules: **(1) OFS** — offline fine-grained sensitivity modeling; **(2) DCS** — dynamic cache schedule optimization; **(3) UAS** — unified adaptive strategy formulation. The inference pipeline proceeds as follows: offline sensitivity priors are first computed → dynamic programming solves for the optimal cache interval combination → during pruning/cache reuse, pruning timing and rate are decided adaptively.

### Key Designs

#### 1. Offline Fine-Grained Sensitivity Modeling (OFS)

- **Cache sensitivity error** $\mathcal{E}_c(t,l,m,n) = 1 - \text{Cos}(\mathcal{D}_{t+n,l,m}(x), \mathcal{D}_{t,l,m}(x))$: measures the cosine distance error of reusing the cache from step $t+n$ at timestep $t$, covering four dimensions: timestep, layer, module, and cache interval.
- **Pruning sensitivity error** $\mathcal{E}_p(t,l,m,\alpha)$: analogously measures feature error under different pruning rates $\alpha$.
- **Offline strategy**: Sensitivity errors are averaged over 100 random generations (10 for video models), stored as model-specific priors, and reused permanently after a single offline pass. The offline overhead is minimal (approximately 160s for DiT-XL/2, with only 0.56GB additional memory).

#### 2. Dynamic Cache Schedule Optimization (DCS)

- The cache interval combination problem is formulated as a dynamic programming problem with optimal substructure: $dp[t][i+1] = \min_{n \in \mathcal{N}} \{\mathcal{E}_{dp}(t,n) + dp[t+n][i]\}$.
- Given an acceleration budget (number of cached steps $N_s$), the algorithm minimizes cumulative sensitivity error from $T$ to 1 and recovers optimal cache timesteps and intervals via backtracking.
- The algorithm operates entirely on offline sensitivity errors, incurring no additional inference overhead.

#### 3. Unified Adaptive Strategy Formulation (UAS)

- **Adaptive pruning timing**: Pruning is applied at step $t+1$ only when the pruning error $\delta_{t+1,l,m} = \mathcal{E}_p(t+1,l,m,\alpha)$ is smaller than the cache error $\mathcal{E}_c(t,l,m,n)$; otherwise, the cache is directly reused, ensuring that pruning always reduces overall error.
- **Adaptive pruning rate**: $\alpha_{t+1,l,m} = \lambda \cdot \mathcal{E}_c(t,l,m,n) + \beta$, where $\lambda$ is a scaling coefficient and $\beta$ is a base pruning rate (adjusted adaptively according to the acceleration budget), enabling pruning to be aware of both the global budget and local sensitivity.
- **Token importance metric**: Feature mean is used for TopK selection, avoiding compatibility issues arising from the unavailability of attention weights under Flash Attention.

### Loss & Training

No additional training loss is introduced. The core optimization objective is minimizing the cumulative sensitivity error $dp[1][N_s]$ via dynamic programming.

## Key Experimental Results

### Main Results

| Model / Setting | Speedup | FID↓ | sFID↓ | IS↑ | Notes |
|---|---|---|---|---|---|
| DiT-XL/2 DDPM baseline | 1.00× | 2.23 | 4.57 | 275.65 | — |
| ToCa (DDPM) | 2.75× | 2.58 | 5.74 | 256.26 | — |
| DuCa (DDPM) | 2.73× | 2.59 | 5.68 | 256.36 | — |
| **SODA (DDPM, $N_s$=72)** | **2.73×** | **2.47** | **5.09** | **262.30** | sFID −0.65, IS +6 |
| DiT-XL/2 DDIM baseline | 1.00× | 2.25 | 4.33 | 239.97 | — |
| DuCa (DDIM) | 2.48× | 3.05 | 4.66 | 233.21 | — |
| **SODA (DDIM, $N_s$=18)** | **2.49×** | **2.75** | **4.56** | **235.65** | FID −0.30 |

| Model | Speedup | FID-30K↓ | CLIP↑ | Notes |
|---|---|---|---|---|
| PixArt-α baseline | 1.00× | 28.10 | 16.29 | — |
| DuCa | 1.87× | 28.05 | 16.42 | — |
| **SODA ($N_s$=8)** | **1.88×** | **27.33** | **16.42** | FID −0.72 vs. DuCa |
| **SODA ($N_s$=7)** | **2.21×** | **27.72** | **16.44** | Higher speedup still outperforms baseline |

### Ablation Study

| Configuration | FID↓ | sFID↓ | IS↑ |
|---|---|---|---|
| Vanilla (fixed cache) | 3.83 | 5.24 | 213.12 |
| OFS + DCS | 2.78 | 4.63 | 234.78 |
| OFS + UAS | 2.89 | 4.75 | 235.01 |
| **SODA (full)** | **2.75** | **4.56** | **235.65** |

- DCS alone contributes FID −1.05 and IS +21.66; UAS alone contributes FID −0.94 and IS +21.89; combining both yields the best performance.
- Cosine distance outperforms L1/L2 as the sensitivity metric.

### Key Findings

1. **Low speedup can improve over baseline**: At 1.55×, FID decreases from 2.23 to 2.21 (DDPM), possibly because skipping redundant computations leads to more stable denoising.
2. **Offline modeling exhibits high consistency**: Offline sensitivity distributions closely match those observed during actual inference (Fig. 6), with stable variance, confirming that sensitivity is an intrinsic model property.
3. **Robustness to hyperparameters**: Across the tested ranges of $\lambda$ and $\beta$, ΔFID ≤ 0.02 and ΔIS ≤ 2.2.
4. **Effective on video tasks**: At 2.50× speedup on OpenSora, VBench drops by only 0.64%, outperforming ToCa, DuCa, PAB, and FORA.

## Highlights & Insights

- SODA is the first to push sensitivity modeling to the granularity of timestep × layer × module, and to solve for the globally optimal cache schedule via dynamic programming with theoretical guarantees.
- Pruning and caching decisions are unified through sensitivity error, enabling adaptive retention of high-sensitivity tokens without manual design, while maintaining generalizability.
- Offline modeling is required only once per model with negligible cost (< 1 hour); at runtime, only a 0.16MB prior is loaded, incurring zero additional inference overhead.
- Cross-task generalization: the same framework applies without modification to class-conditional image generation, text-to-image, and text-to-video tasks.

## Limitations & Future Work

- As a training-free method, the achievable acceleration still falls short of training-based approaches such as distillation.
- Integration with training-based techniques such as distillation remains unexplored.
- Offline modeling requires one execution per new model (though with low overhead), so the method is not fully plug-and-play.
- The pruning position selection relies on a feature-mean heuristic, which may not be the optimal token importance metric.
- The dynamic programming search space grows with the size of the cache interval candidate set, and solver efficiency may become a concern in extreme large-step scenarios.

## Related Work & Insights

- **Cache-based acceleration**: FORA, FasterDiffusion, ToCa (ICLR 2025), DuCa — exploit inter-timestep similarity to reuse intermediate features, but rely on fixed or heuristic cache schedules.
- **Token pruning**: ToMe, AT-EDM — exploit token redundancy for pruning; flexible but less efficient than caching.
- **Combined caching and pruning**: ToCa, DuCa — perform full computation at anchor steps and cache, then apply pruning with cache reuse at intermediate steps, but with manually designed strategies.
- **Sensitivity analysis inspiration**: This work is inspired by sensitivity analysis in LLM quantization (e.g., SqueezeLLM) and extends it to multi-dimensional acceleration decisions for diffusion models.

## Rating

- Novelty: ⭐⭐⭐⭐ — Combining sensitivity modeling with dynamic programming for cache optimization is novel; the unified pruning/caching decision mechanism is conceptually clear.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three models, three task types, comprehensive ablations, offline analysis, hyperparameter sensitivity, and qualitative comparisons; very thorough.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, problem-driven, with an effective motivation figure (Fig. 1).
- Value: ⭐⭐⭐⭐ — Practical and generalizable method with open-source code for reproducibility; directly contributes to the DiT acceleration community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[CVPR 2026\] PPCL: Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](ppcl_pluggable_pruning_dit_distillation.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[CVPR 2026\] Batch Loss Score for Dynamic Data Pruning](batch_loss_score_for_dynamic_data_pruning.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)

<!-- RELATED:END -->
