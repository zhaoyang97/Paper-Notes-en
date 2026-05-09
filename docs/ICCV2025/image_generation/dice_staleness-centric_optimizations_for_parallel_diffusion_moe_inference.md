---
title: >-
  [Paper Note] DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference
description: >-
  [ICCV 2025][Image Generation][MoE] DICE is a framework targeting the *staleness* problem in parallel inference of MoE-based diffusion models. Through three levels of optimization — step-level interweaved parallelism, layer-level selective synchronization, and token-level conditional communication — DICE achieves a 1.26× speedup on DiT-MoE with negligible quality degradation.
tags:
  - ICCV 2025
  - Image Generation
  - MoE
  - Diffusion Models
  - Expert Parallelism
  - Staleness Optimization
  - Communication Optimization
date: 2026-05-08
content_hash: 35c318a40cf26f95
---

# DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference

**Conference**: ICCV 2025
**arXiv**: [2411.16786](https://arxiv.org/abs/2411.16786)
**Code**: [https://github.com/Cobalt-27/DICE](https://github.com/Cobalt-27/DICE)
**Area**: Image Generation / Parallel Inference for Diffusion Models
**Keywords**: MoE, Diffusion Models, Expert Parallelism, Staleness Optimization, Communication Optimization

## TL;DR

DICE is a framework targeting the *staleness* problem in parallel inference of MoE-based diffusion models. Through three levels of optimization — step-level interweaved parallelism, layer-level selective synchronization, and token-level conditional communication — DICE achieves a 1.26× speedup on DiT-MoE with negligible quality degradation.

## Background & Motivation

- **Background**: Mixture-of-Experts-based diffusion models (e.g., DiT-MoE) scale to 16 billion parameters and demonstrate outstanding generation quality, but their reliance on expert parallelism introduces severe communication bottlenecks. On 8 GPUs, all-to-all communication accounts for 61.7%–73.3% of total inference time for DiT-MoE-XL.
- **Limitations of Prior Work**: Displaced Parallelism (proposed by DistriFusion) mitigates blocking by overlapping computation and communication via asynchronous execution, but introduces a critical issue — **staleness**: using outdated activations from earlier timesteps, causing FID to degrade from 5.31 to 8.27.
- **Key Challenge**: Displaced parallelism in the expert parallelism setting incurs **2-step staleness** (dispatch delayed by 1 step + combine delayed by 1 step), which is particularly harmful to generation quality in MoE architectures. Meanwhile, the authors observe high **routing redundancy** across adjacent diffusion steps in DiT-MoE (token-to-expert assignments are highly similar), which provides a feasibility basis for asynchronous communication.

## Method

### Overall Architecture

DICE optimizes staleness at three granularities — step-level, layer-level, and token-level — forming a collaborative optimization framework.

### Key Designs

1. **Interweaved Parallelism** — Step-Level Optimization:

   Conventional displaced parallelism incurs 2-step staleness (dispatch: 1 step + combine: 1 step). Interweaved parallelism restructures the scheduling of communication and computation to achieve:
   - Asynchronous dispatch completing within the current step during interleaved execution (0-step delay)
   - The next combine operation initiated while processing expert outputs
   - Combine results becoming available in the next step (1-step delay)

   $$\text{Staleness}_{\text{interweaved}} = \underbrace{0}_{\text{dispatch}} + \underbrace{1}_{\text{combine}} = 1\text{-step}$$

   Compared to displaced parallelism, staleness is halved, cache size is halved (only store combine results), and no additional overhead is introduced — a **free-lunch** optimization. Used alone, it improves FID from 8.27 to 6.97.

2. **Selective Synchronization** — Layer-Level Optimization:

   Analysis reveals **layer-wise asymmetry** in staleness sensitivity: shallow-layer experts extract low-level features and are naturally robust to asynchronous communication, while deep-layer experts handle high-level semantics and are highly sensitive to activation staleness. This is consistent with DeepSpeed-MoE's observation that deeper layers benefit more from MoE in language models.

   Accordingly, DICE applies synchronous communication (preserving data freshness) only to deeper layers, while shallow layers continue to execute asynchronously. Ablation experiments confirm that synchronizing deep layers yields the best result (FID 5.74), far outperforming synchronizing shallow layers (FID 6.55).

3. **Conditional Communication** — Token-Level Optimization:

   This design leverages an intrinsic property of MoE routing: tokens with high routing scores dominate outputs through weighted summation and are thus more susceptible to staleness perturbations. Specifically, the propagation of staleness-induced activation perturbation $\Delta \mathbf{h}_i^e$ to the output is proportional to the routing score $s_i^e$:

   $$\frac{\partial \|\mathbf{y}_i\|}{\partial \mathbf{h}_i^e} = \frac{s_i^e \cdot \mathbf{y}_i}{\|\mathbf{y}_i\|}$$

   Therefore, high-score tokens are transmitted every step to remain fresh, while low-score tokens reuse cached values, reducing communication frequency. This strategy requires no training.

### Loss & Training

DICE is an inference optimization framework and does not involve training. The core mechanism lies in rationally scheduling communication during inference to reduce staleness.

## Key Experimental Results

### Main Results (ImageNet 256×256, DiT-MoE-XL, 50 steps)

| Method | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|--------|------|-------|-----|-----------|---------|
| Expert Parallelism (Sync) | 5.31 | 10.10 | 235.89 | 0.75 | 0.60 |
| DistriFusion | 7.79 | 12.13 | 206.24 | 0.72 | 0.59 |
| Displaced Expert Para. | 8.27 | 11.58 | 204.07 | 0.71 | 0.59 |
| Interweaved Para. (alone) | 6.97 | 11.01 | 216.62 | 0.72 | 0.59 |
| **DICE** | **6.11** | **10.93** | **225.65** | **0.73** | 0.59 |

Few-step experiments (10/20 steps):

| Steps | Method | FID↓ | Speedup↑ |
|-------|--------|------|----------|
| 10 | Expert Para. | 10.24 | - |
| 10 | Displaced Expert Para. | 27.61 | 1.28× |
| 10 | **DICE** | **15.13** | **1.20×** |
| 20 | Expert Para. | 6.41 | - |
| 20 | Displaced Expert Para. | 15.27 | 1.33× |
| 20 | **DICE** | **8.60** | **1.24×** |

Speedup: DICE achieves up to 1.26× acceleration at batch size 32; DistriFusion runs out of memory in most configurations.

### Ablation Study

| Interweaved | Selective Sync | Conditional Comm | FID↓ | IS↑ |
|-------------|---------------|-----------------|------|-----|
| ✓ | × | × | 6.97 | 216.62 |
| ✓ | Deep | × | **5.74** | **230.23** |
| ✓ | Shallow | × | 6.55 | 221.61 |
| ✓ | Staggered | × | 5.95 | 227.78 |
| ✓ | × | Low Score | 7.24 | 214.10 |
| ✓ | × | High Score | 7.51 | 211.40 |
| ✓ | × | Random | 7.37 | 212.84 |

### Key Findings

- **Interweaved parallelism is a free lunch**: staleness halved, cache halved, same overlap degree, no extra overhead.
- **Deep layers are more sensitive than shallow layers**: synchronizing deep layers yields FID 5.74 vs. 6.55 for shallow layers, a gap of 0.81.
- **Routing scores serve as effective token importance signals**: reducing communication frequency for low-score tokens outperforms reducing it for high-score or randomly selected tokens.
- **DICE's advantage is more pronounced in few-step settings (10/20 steps)**: the fewer the steps, the greater the relative impact of per-step staleness.
- DistriFusion runs out of memory directly on DiT-MoE-G (33GB parameters), confirming that expert parallelism is essential for large-scale MoE diffusion models.
- All-to-all communication accounts for 73.3% of inference time at batch size 16, making communication optimization critical.

## Highlights & Insights

1. **Precise problem formulation**: introducing the concept of "staleness" provides a unified analytical framework for asynchronous parallel inference.
2. **Systematic multi-granularity optimization**: the three dimensions — step-level, layer-level, and token-level — are each targeted and work synergistically.
3. **Elegance of interweaved parallelism**: staleness is halved purely through communication scheduling changes, with no additional computation or communication overhead.
4. **Strong practicality**: all optimizations are model-transparent, require no retraining, and the code is open-sourced.

## Limitations & Future Work

- Validation is limited to DiT-MoE; other MoE diffusion architectures (e.g., Switch-DiT) are not covered.
- The stride parameter (communication frequency) for conditional communication requires manual tuning.
- The shallow/deep layer boundary for selective synchronization is currently a 50/50 split; finer-grained adaptive strategies remain to be explored.
- Experiments are conducted only on PCIe-connected RTX 4090 GPUs; performance under NVLink/InfiniBand environments may differ.
- More complex generation scenarios such as video diffusion are not addressed.

## Related Work & Insights

- **Displaced Parallelism (DistriFusion)**: the direct improvement target of DICE.
- **FasterMoE, DeepSpeed-MoE**: pioneers of MoE communication optimization, but not tailored for diffusion models.
- **Cache-based methods (DeepCache, Learn2Cache)**: complementary to communication optimization.
- **PipeFusion, AsyncDiff**: similarly exploit activation similarity; DICE offers a more systematic treatment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The systematic analysis of staleness and the design of interweaved parallelism are elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple configurations (batch size / step count / model scale) with sufficient ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figures are exceptionally clear, particularly the execution flow comparison in Figure 2.
- **Value**: ⭐⭐⭐⭐ Direct engineering value for deploying large-scale MoE diffusion models.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](inference-time_diffusion_model_distillation.md)
- [\[NeurIPS 2025\] Accelerating Parallel Diffusion Model Serving with Residual Compression](../../NeurIPS2025/image_generation/accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](../../NeurIPS2025/image_generation/gspn-2_efficient_parallel_sequence_modeling.md)
- [\[ICCV 2025\] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers](genhancer_imperfect_generative_models_are_secretly_strong_vision-centric_enhance.md)

<!-- RELATED:END -->
