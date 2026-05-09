---
title: >-
  [Paper Note] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration
description: >-
  [AAAI 2026][Image Generation][Diffusion Transformer Acceleration] This paper proposes ProCache, a training-free dynamic feature caching framework that achieves 2.90× speedup on DiT-XL/2 and 1.96× speedup on PixArt-α with negligible image quality degradation, through constraint-aware non-uniform caching pattern search and selective computation, significantly outperforming existing caching methods.
tags:
  - AAAI 2026
  - Image Generation
  - Diffusion Transformer Acceleration
  - Feature Caching
  - Training-Free Inference
  - Dynamic Scheduling
  - Token Selection
date: 2026-05-08
content_hash: 907037aa3ef11692
---

# ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration

**Conference**: AAAI 2026
**arXiv**: [2512.17298](https://arxiv.org/abs/2512.17298)
**Code**: [https://github.com/macovaseas/ProCache](https://github.com/macovaseas/ProCache)
**Area**: Image Generation
**Keywords**: Diffusion Transformer Acceleration, Feature Caching, Training-Free Inference, Dynamic Scheduling, Token Selection

## TL;DR

This paper proposes ProCache, a training-free dynamic feature caching framework that achieves 2.90× speedup on DiT-XL/2 and 1.96× speedup on PixArt-α with negligible image quality degradation, through constraint-aware non-uniform caching pattern search and selective computation, significantly outperforming existing caching methods.

## Background & Motivation

### State of the Field

Diffusion Transformers (DiTs) have achieved state-of-the-art performance in generative modeling, but their substantial computational overhead severely limits real-time deployment. Acceleration approaches include:
- **Training-required methods**: pruning, quantization, and knowledge distillation — these incur additional training costs and degrade at high speedup ratios.
- **Training-free methods**: feature caching — exploiting temporal redundancy between denoising steps by caching computed features for reuse in subsequent steps, enabling plug-and-play deployment.

### Limitations of Prior Work

Existing caching-based methods suffer from two key limitations:

**Uniform caching intervals mismatched with non-uniform temporal dynamics**: All prior methods adopt fixed intervals (e.g., full computation every $N$ steps), yet DiT feature variations across the denoising process are non-uniform — features change slowly in early-to-middle stages and change sharply in later stages (with an approximately exponential growth trend). Uniform strategies waste computation during stable phases and introduce excessive error during rapid-change phases.

**Naive feature reuse leads to error accumulation**: As the caching interval increases, feature similarity decays exponentially, and directly reusing stale features causes severe quality degradation. Existing methods (e.g., Δ-DiT, FORA) apply no correction to reused features, leading to noticeable quality drops at high speedup ratios.

### Core Findings

Through quantitative analysis, the authors reveal two key observations:

**Error accumulation is depth-dependent**: Feature errors in deeper blocks (e.g., Blocks 25–28) are significantly larger than in shallower blocks (e.g., Blocks 1–4), and grow progressively over timesteps.

**Output variation is time-dependent**: Feature changes are gradual in early-to-middle stages and increase sharply in later stages, exhibiting an approximately exponential growth trend.

### Starting Point

**Core Idea**: Since DiT feature evolution is non-uniform and error propagation is depth-dependent, the caching strategy should also be non-uniform and computational correction should be selective. By offline-searching for an optimal caching pattern and selectively refreshing high-importance tokens in deep blocks during caching steps, the method achieves maximum speedup while preserving quality.

## Method

### Overall Architecture

ProCache consists of two core components:

1. **Constraint-Aware Caching Pattern Search**: Offline search for a non-uniform optimal activation schedule.
2. **Selective Computation**: Lightweight partial computation during caching steps to suppress error accumulation in deep blocks.

### Key Designs

#### 1. Constraint-Aware Caching Pattern Search

**Function**: Replaces uniform-interval caching with a non-uniform activation pattern tailored to the model's temporal characteristics.

**Mechanism**: The caching pattern is represented as a binary sequence $\mathbf{s} = [s_1, s_2, \ldots, s_T] \in \{0,1\}^T$, where $s_t = 1$ indicates full computation at step $t$ and $s_t = 0$ indicates cache reuse. Three constraints are defined:

1. **Budget constraint**: The total number of active steps does not exceed budget $B$: $M = \sum_{t=1}^T s_t \leq B$
2. **Monotonicity constraint**: Reuse intervals are non-increasing over time (computation should be more frequent in later stages): $v_{i+1} \leq v_i$
3. **Boundedness constraint**: Each interval lies within a feasible range: $v^{\min} \leq v_i \leq v^{\max}$

Here the reuse interval is $v_i = t_{i+1} - t_i - 1$, representing the number of consecutive caching steps between two computations.

**Search Procedure**:
- Sample $K$ candidate patterns from the constrained space $\mathcal{C}$ (default $K=5$)
- Evaluate each candidate on a small representative dataset (e.g., via FID)
- Select the optimal pattern $\mathbf{s}^*$
- Fully training-free; completes in under one hour on a single GPU

**Design Motivation**: The monotonicity constraint directly follows from the empirical observation in Figure 3 — feature variations increase sharply in later stages, and caching intervals should monotonically decrease to match this dynamic. The boundedness constraint prevents error accumulation from overly long intervals and efficiency loss from overly short ones.

#### 2. Selective Computation

**Function**: Performs partial computation during caching steps with minimal overhead (~3% additional latency) to suppress error accumulation.

**Mechanism**: Selective updates are applied at even positions within each consecutive zero block. Only a **subset of deep blocks** and a **subset of high-importance tokens** are recomputed; all others retain cached values.

$$\mathbf{x}_i^{(l)} = \begin{cases} f^{(l)}(\mathbf{x}_i^{(l-1)}), & \text{if } l \in \mathcal{U}^{\text{cmpt}} \text{ and } i \in \mathcal{I}^{\text{cmpt}} \\ \text{Cache}^{(l)}(\mathbf{x}_i), & \text{otherwise} \end{cases}$$

**Layer selection — focusing on deep blocks**: Analysis shows that errors are concentrated in deeper layers (high-level semantic refinement layers), while shallow layers exhibit strong temporal stability with minimal deviation. The deepest $D = r \times L$ layers are selected for selective computation (default $r = 25\%$).

**Token selection — prioritizing high-importance tokens**: The L2 norm of the attention module output is used as an importance metric: $T(\mathbf{x}_i) = \|\mathbf{v}_i\|_2$, and the top-$p\%$ tokens ($7$–$30\%$) are recomputed.

$$\mathcal{I}^{\text{cmpt}} = \{i \mid \text{rank}(T(\mathbf{x}_i)) \leq p\% \times N\}$$

Note: Token selection is not applied to self-attention modules (since each token attends to all others, partial computation can propagate errors); it is applied only in cross-attention and FFN modules.

### Loss & Training

ProCache is a fully training-free method requiring no loss functions or training procedures. Its key advantage is plug-and-play deployment:
- Offline search requires only a small number of inference evaluations
- Orthogonally compatible with existing samplers (DDIM, DPM-Solver++, Rectified Flow)

## Key Experimental Results

### Main Results

**DiT-XL/2 ImageNet class-conditional generation** (50K images, 256×256):

| Method | Latency (s) ↓ | FLOPs (T) ↓ | Speedup ↑ | FID ↓ | sFID ↓ | Precision ↑ | IS ↑ |
|--------|--------------|------------|----------|------|-------|------------|------|
| DDIM-50 steps | 4.549 | 23.74 | 1.00× | 2.43 | 4.40 | 0.80 | 241.25 |
| Δ-DiT (N=3) | 2.572 | 16.46 | 1.47× | 3.75 | 5.70 | 0.77 | 207.57 |
| FORA (N=3) | 2.191 | 8.59 | 2.76× | 3.88 | 6.43 | 0.79 | 229.02 |
| ToCa (N=3) | 2.087 | 10.23 | 2.32× | 3.04 | 5.14 | 0.79 | 230.70 |
| **ProCache** | **1.725** | **8.18** | **2.90×** | **2.96** | **4.93** | **0.80** | **232.85** |

**PixArt-α COCO30K text-to-image generation**:

| Method | Latency (s) ↓ | Speedup ↑ | FID ↓ | CLIP ↑ |
|--------|--------------|----------|------|--------|
| DPM-Solver++ 20 steps | 2.142 | 1.00× | 28.12 | 16.29 |
| FORA (N=3) | 1.301 | 2.79× | 29.84 | 16.42 |
| ToCa | 1.473 | 1.77× | 28.02 | 16.43 |
| **ProCache** | **1.215** | **1.96×** | **27.66** | **16.45** |

**FLUX.1-dev/schnell PartiPrompts**: ProCache achieves 1.54× speedup on FLUX.1-dev (Image Reward 1.207 vs. baseline 1.202) and 1.56× speedup on FLUX.1-schnell (Image Reward 1.138 vs. baseline 1.133), outperforming both FORA and ToCa.

### Ablation Study

| Configuration | Speedup | FID ↓ | sFID ↓ | Notes |
|--------------|---------|------|-------|-------|
| Default uniform (B=13) | 2.93× | 4.75 | 8.43 | Uniform interval baseline |
| + Search pattern | 2.93× | 3.15 | 5.12 | Zero-cost improvement |
| + Selective computation | 2.90× | 3.28 | 5.95 | Selective computation only |
| **ProCache** | **2.90×** | **2.96** | **4.93** | Combined optimum |

Ablation over deep block ratio (DiT-XL/2):

| Ratio r | FLOPs (T) ↓ | FID ↓ | IS ↑ |
|---------|-----------|------|------|
| 50% | 9.138 | 45.32 | 184.18 |
| 75% | 8.594 | 45.31 | 183.61 |
| 90% | 8.344 | 45.38 | 182.46 |

75% achieves the best trade-off; the main experiments use 25% to maximize speedup.

### Key Findings

1. The searched non-uniform pattern reduces FID from 4.75 to 3.15 (a 33.7% improvement) without any change in speed, demonstrating the significance of pattern design.
2. ProCache improves FID by nearly 30% over FORA while achieving a higher speedup ratio (2.90× vs. 2.76×).
3. Selective computation introduces only ~3% additional latency (due to partial caching steps, 25% of layers, and 7–30% of tokens being recomputed).
4. The search process is robust: FID and IS variance across 5 independent runs is negligible, and a minimal sampling budget of $K=5$ is sufficient.
5. At extreme speedup ratios exceeding 4.53×, ProCache reduces quality degradation by 56.2% compared to prior methods.

## Highlights & Insights

1. **Empirical observation directly drives method design**: The analyses in Figures 1 and 3 precisely reveal the non-uniformity of error accumulation and feature evolution, and the three constraints correspond perfectly to these observations.
2. **Elegant search space design**: The monotonicity and boundedness constraints compress the exponentially large search space to a tractable scale, and good patterns can be found with $K=5$.
3. **Well-calibrated selective computation**: Layer-level focus on deep blocks and token-level focus on high-importance tokens, with token pruning deliberately excluded from self-attention to avoid error propagation — a carefully considered design.
4. **Broad compatibility**: Applicable to DiT-XL/2, PixArt-α, FLUX.1-dev/schnell, and various samplers.

## Limitations & Future Work

1. Although the offline search is fast (<1 hour), it must be re-run for each model/sampler/step-count combination.
2. The token selection strategy is based on attention output L2 norms, an importance metric with limited theoretical grounding; more principled alternatives may exist.
3. Excluding token selection from self-attention is a conservative design choice; a safer partial update scheme for self-attention could enable further acceleration.
4. Validation is limited to image generation tasks; the additional complexity introduced by the temporal dimension in video generation remains unexplored.
5. The hyperparameters $v^{\min}$ and $v^{\max}$ in the search constraints are manually specified; adaptive determination may be preferable.

## Related Work & Insights

- **FORA** (Selvaraju et al.): Uniform caching across all steps; ProCache demonstrates the overwhelming advantage of non-uniform strategies.
- **ToCa** (Zou et al.): Introduces token-level caching but retains uniform intervals; ProCache further advances the pattern search dimension.
- **Δ-DiT** (Chen et al.): MLP sharing strategy with limited speedup.
- **DeepCache/Faster Diffusion**: Caching methods specific to U-Net architectures, not applicable to DiTs.
- The constraint-based sampling search paradigm introduced in this work is generalizable to other adaptive scheduling scenarios (e.g., inference step scheduling, dynamic precision scheduling).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of constraint-aware search and selective computation is innovative
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 models, multi-dimensional metrics, thorough ablations, robustness analysis
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from analysis-driven design is clear, with highly informative figures and tables
- Value: ⭐⭐⭐⭐ — A practical training-free acceleration method, though the contribution feels somewhat incremental

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[AAAI 2026\] Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback](playmate2_training-free_multi-character_audio-driven_animation_via_diffusion_tra.md)
- [\[CVPR 2026\] LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration](../../CVPR2026/image_generation/lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)

</div>

<!-- RELATED:END -->
