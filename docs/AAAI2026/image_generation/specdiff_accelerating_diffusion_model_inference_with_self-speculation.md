---
title: >-
  [Paper Note] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation
description: >-
  [AAAI 2026][Image Generation][Diffusion model acceleration] This paper proposes SpecDiff, a training-free multi-level feature caching strategy based on self-speculation. By leveraging a small number of speculative steps…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "feature caching"
  - "self-speculation"
  - "DiT"
  - "information utilization"
date: 2026-05-08
content_hash: 8ad3a7cfd71f1303
---

# SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation

**Conference**: AAAI 2026
**arXiv**: [2509.13848](https://arxiv.org/abs/2509.13848)  
**Code**: Unavailable  
**Area**: Image Generation
**Keywords**: Diffusion model acceleration, feature caching, self-speculation, DiT, information utilization

## TL;DR

This paper proposes SpecDiff, a training-free multi-level feature caching strategy based on self-speculation. By leveraging a small number of speculative steps to introduce **future information** for token importance selection, SpecDiff overcomes the accuracy–speed bottleneck of methods that rely solely on historical information, achieving 2.80×/2.74×/3.17× speedup on Stable Diffusion 3/3.5 and FLUX with negligible quality loss.

## Background & Motivation

### State of the Field

The Diffusion Transformer (DiT) architecture (SD3, SD3.5, FLUX) has become the dominant design paradigm for diffusion models. However, as model scale grows (driven by scaling laws), inference latency and memory requirements have become critical bottlenecks. In DiT inference, the number of rows in the input matrix is comparable to that of the model weight matrix (e.g., $M=4096$, $N=1536$ in SD3), making matrix multiplications compute-bound with high arithmetic intensity. Reducing actual computation is therefore a direct and effective approach to accelerating inference.

### Limitations of Prior Work

**Feature caching** has emerged as a promising acceleration technique, exploiting the similarity of intermediate features across denoising steps to cache and reuse relatively stable features in place of redundant computation:
- **FORA**: Directly caches and reuses features from adjacent steps → 1.8× speedup but >60% quality degradation
- **ToCa**: Selects important tokens based on high attention scores → 2.0× speedup, ~30% degradation
- **RAS**: Selects tokens using two-step adjacent feature similarity → 2.3× speedup, ~20% degradation
- **TaylorSeer**: Approximates current features via finite differences over all historical features → 2.6× speedup, ~15% degradation

### Root Cause

From an information-utilization perspective, all existing methods **rely exclusively on historical information**. DiT inference is inherently a Markov process—the next-step output depends only on the current-step output. Consequently, accumulated historical information is always restricted to a subset of past steps, meaning **historical information can only capture local changes** and fails to anticipate abrupt future image variations. Recall analysis shows that regardless of whether 1-step, 2-step, or $N$-step historical information is used, a significant gap from the theoretical upper bound remains.

### Starting Point

Inspired by speculative decoding in LLMs, the authors observe that **information within the same timestep is highly similar across different iteration counts**. A small number of speculative steps can therefore be used to obtain "future information" in advance, enabling token selection from a global perspective. Experiments show that a substantial fraction of important tokens that historical information fails to identify can be captured via speculative information.

### Core Idea

A new paradigm is proposed: a small number of self-speculative steps with the original model are used to introduce future information at negligible cost (<5% additional inference time, <0.1% additional memory), assisting feature caching token selection and upgrading from local change capture to global change capture.

## Method

### Overall Architecture

SpecDiff is built on two technical components:
1. **Speculation-information-based feature selection algorithm (T1)**: Computes token importance scores using both future and historical attention information.
2. **Multi-level feature classification algorithm based on importance scores (T2)**: Classifies tokens into three levels and handles each accordingly.

Inference pipeline:
- First, a small number of speculative denoising steps (default: 2) are executed to obtain attention information at future timesteps.
- At each step of the main inference pass, token importance scores are computed by combining historical and future attention scores.
- Tokens are classified into three levels—C1 (computed), C2 (directly reused), and C3 (weighted approximation)—according to their importance scores.

### Key Designs

#### 1. Speculation-Information-Based Feature Selection Algorithm

The importance score of token $x_i$ is composed of three factors:

$$Score(x_i) = his(x_i) \cdot fut(x_i) \cdot star(x_i)$$

- **Historical score** $his(x_i)$: Sum of attention scores for the token across all layers in the previous iteration.
- **Future score** $fut(x_i)$: Attention score at the nearest future timestep obtained from the speculative steps.
- **Starvation score** $star(x_i) = e^{cf(x_i)}$, where $cf(x_i)$ denotes the number of times the token has been cached.

Design motivation:
- The multiplicative form of historical × future ensures mutual confirmation from both sources.
- The starvation score addresses distributional skewness: experiments show that when only the top 20% of important tokens are selected, the top 25% most frequently selected tokens account for >75% of all selections, while ~40% of tokens are never selected as important, leading to accumulated approximation error from long-term caching.
- Since the number of speculative steps is fewer than the total number of inference steps, timesteps cannot be perfectly aligned; the score from the nearest future timestep is therefore used.

#### 2. Multi-Level Feature Classification Algorithm

After important tokens are identified, the remaining tokens' features are efficiently approximated. SpecDiff classifies tokens into three levels:

**C1 (network computation)**: Tokens with the highest importance scores (fraction $1-CR$ determined by cache rate CR) participate directly in DiT computation.

**C2 (direct reuse)**: Tokens with the lowest scores (accounting for the bottom 10% of total scores) directly reuse features from the previous iteration:
$$F(x_t^{\text{cached}}) \approx F(x_{t+1}^{\text{cached}})$$

Design motivation: These tokens exhibit low coefficient of variation in ERROR, resulting in small reuse error.

**C3 (weighted approximation)**: Tokens that do not participate in computation but fall within the top 90% of the score distribution are approximated using a weighted combination of the three most recent historical features:

$$F(x_t^{\text{cached}}) = \sum_{i=1}^{3} W_{t+i} \cdot F(x_{t+i}^{\text{cached}})$$

where the weights are exponentially decayed by temporal distance:

$$W_{t+i} = \frac{e^{-i}(T_{t+i} - T_t)}{\sum_{i=1}^{3} e^{-i}(T_{t+i} - T_t)}$$

Design motivation:
- Analysis reveals a strong correlation between token importance scores and the coefficient of variation of ERROR—high-score tokens exhibit large feature variations and cannot be simply reused.
- In high-dimensional space, cosine similarity falls below 95% after three consecutive steps, indicating that features are no longer highly similar beyond this range.
- Earlier timesteps are assigned lower weights to reduce approximation error.

### Loss & Training

SpecDiff is a **training-free method** requiring no training process. All parameter settings:
- Default number of speculative steps: 2
- Number of inference steps: 28
- CFG: 7.0 for SD3/3.5, 3.5 for FLUX
- Cache rate CR is variable, supporting a wide range from 55% to 99%.

## Key Experimental Results

### Main Results

Performance on Stable Diffusion 3 (COCO 2014 val, 5,000 pairs, 1024×1024):

| Method | Cache Rate | FID↓ | Clip Score↑ | VQA Score↑ | Speedup |
|--------|-----------|------|------------|-----------|---------|
| RFlow (baseline) | 0 | 29.31 | 0.3176 | 0.9110 | 1.00× |
| RAS | 50% | 27.26 | 0.3162 | 0.9005 | 1.61× |
| **SpecDiff** | 55% | **27.57** | **0.3168** | **0.9057** | **1.61×** |
| RAS | 75% | 27.38 | 0.3149 | 0.8849 | 2.09× |
| **SpecDiff** | 92% | **29.52** | **0.3160** | **0.8888** | **2.43×** |
| RAS | 87.5% | 40.92 | 0.3044 | 0.8611 | 2.40× |
| **SpecDiff** | 99% | **29.75** | **0.3152** | **0.8822** | **2.80×** |

Comparison with TaylorSeer on FLUX.1 Dev:

| Method | Config | FID↓ | Clip↑ | VQA↑ | SSIM↑ | PSNR↑ | Memory (GB)↓ | Speedup |
|--------|--------|------|------|------|------|------|-------------|---------|
| RFlow | - | 27.68 | 0.3093 | 0.8986 | - | - | 38.36 | 1.00× |
| TaylorSeer | N5O1 | 27.87 | 0.3090 | 0.8909 | 0.7098 | 16.93 | 42.66 | 2.47× |
| **SpecDiff** | 85% | **28.61** | **0.3125** | **0.8925** | **0.7101** | **19.20** | **41.46** | 2.52× |
| TaylorSeer | N6O1 | 28.83 | 0.3107 | 0.8822 | 0.6570 | 16.07 | 42.66 | 2.63× |
| **SpecDiff** | 95% | **29.24** | **0.3124** | **0.8834** | **0.6963** | **19.02** | **41.46** | **3.17×** |

### Ablation Study

Effect of the number of speculative steps (SD3, cache rate 99%):

| Speculative Steps | FID↓ | Clip Score↑ | VQA Score↑ | Speedup |
|-------------------|------|------------|-----------|---------|
| 2 | 29.75 | 0.3152 | 0.8822 | **2.80×** |
| 3 | 29.87 | 0.3154 | 0.8825 | 2.49× |
| 4 | 29.64 | 0.3157 | 0.8834 | 2.25× |

Component-wise ablation (SD3, speed and quality decomposition):

| Module | ΔFID | ΔClip | Speedup |
|--------|------|------|---------|
| + Feature prediction (future information) | FID ↓18% | Clip +2.8% | 2.36× |
| + Token-level classification | FID ↓3.6% further | Clip +0.35% further | 3.17× (total) |

### Key Findings

1. **Pareto frontier advancement**: SpecDiff successfully pushes the Pareto frontier in the speed–quality trade-off, consistently outperforming RAS and TaylorSeer at equivalent computational budgets.
2. **Quality preservation at 99% cache rate**: On SD3, using a 99% cache rate (i.e., only 1% of tokens participate in computation), FID increases by only 0.44 and Clip Score decreases by only 0.7%.
3. **Best preservation of text–image alignment**: Clip Score and VQA Score are maintained more robustly than FID.
4. **Two speculative steps is optimal**: Additional speculative steps yield marginal quality improvements but incur significant speed penalties (~20% per step); 2 steps is the best trade-off overall.
5. **Minimal memory overhead**: Less than 0.1% additional memory is required, far below the extra memory consumed by TaylorSeer.

## Highlights & Insights

1. The **information-utilization perspective** as an analytical framework is highly novel—it unifies diverse caching methods under a common analysis showing a positive correlation between information utilization and performance.
2. The **cross-domain transfer from LLM speculative decoding to diffusion model feature caching** is an elegant and insightful adaptation.
3. The **three-level token classification strategy** is elegantly designed: high-score tokens are computed, low-score tokens are directly reused, and intermediate tokens are approximated via weighted history—each choice is theoretically motivated.
4. The **starvation score** mechanism that prevents long-tail tokens from being permanently ignored is simple yet effective.
5. The **training-free** nature allows direct application to any DiT model without additional training overhead.

## Limitations & Future Work

1. **Suboptimal FID metric**: The authors acknowledge that high CFG values homogenize image styles, which may inflate FID scores.
2. **DiT-architecture-only**: Validation on UNet-based diffusion models has not been conducted.
3. **Fixed number of speculative steps**: Two speculative steps may not be universally optimal across all models and scenarios; adaptive speculation strategies warrant exploration.
4. **Manual cache rate setting**: Different image contents may require different cache rates; dynamic cache rate adjustment is a promising direction.
5. **No combination with orthogonal compression methods**: Feature caching can potentially be combined with quantization and distillation techniques for further gains.

## Related Work & Insights

- **FORA, ToCa, RAS, TaylorSeer**: A progression of feature caching methods; SpecDiff builds upon them by introducing future information.
- **Eagle/SpecPIM** (LLM speculative decoding): Cross-domain source of inspiration.
- **MDTv2** (Gao et al., 2024): Demonstrates that DiT feature attention is strongly correlated with timestep parameters, motivating the utilization of same-timestep information across iterations.
- **Key Takeaway**: In iterative inference systems, "peeking into the future" via cheap approximate speculation is a general-purpose strategy for improving efficiency. The Markov property does not imply that historical information is sufficient—prior knowledge from analogous tasks (same timestep, across iterations) can complement the limitations of historical information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The information-utilization perspective and self-speculation paradigm represent novel analytical and design contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers SD3/3.5/FLUX, provides comprehensive comparisons with multiple baselines, and includes ablation and Pareto analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, figures are intuitive (recall analysis, Pareto frontier), and the analysis is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free 3× speedup is of significant practical value for DiT deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](../../NeurIPS2025/image_generation/tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[ICCV 2025\] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](../../ICCV2025/image_generation/from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)
- [\[CVPR 2026\] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration](../../CVPR2026/image_generation/lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)
- [\[CVPR 2026\] Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective](../../CVPR2026/image_generation/accelerating_diffusion_model_training_under_minimal_budgets_a_condensation-based.md)

</div>

<!-- RELATED:END -->
