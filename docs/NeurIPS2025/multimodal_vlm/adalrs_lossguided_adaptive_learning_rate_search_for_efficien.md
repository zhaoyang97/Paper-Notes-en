---
title: >-
  [Paper Note] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining
description: >-
  [NeurIPS 2025][Multimodal VLM][learning rate search] AdaLRS is proposed as a plug-and-play online learning rate search algorithm that adaptively adjusts the learning rate by monitoring the loss descent velocity…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "learning rate search"
  - "online optimization"
  - "loss velocity"
  - "foundation model pretraining"
  - "cosine scheduler"
date: 2026-05-08
content_hash: c872c2e3c6b317eb
---

# AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining

**Conference**: NeurIPS 2025
**arXiv**: [2506.13274](https://arxiv.org/abs/2506.13274)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: learning rate search, online optimization, loss velocity, foundation model pretraining, cosine scheduler

## TL;DR
AdaLRS is proposed as a plug-and-play online learning rate search algorithm that adaptively adjusts the learning rate by monitoring the loss descent velocity, reducing the cost of learning rate hyperparameter search from multiple independent training runs to a single run, achieving approximately 50% savings in training cost.

## Background & Motivation
The learning rate is the most critical hyperparameter in foundation model pretraining, yet finding the optimal learning rate typically requires extensive proxy model experiments or multiple independent training runs. Existing methods either restrict the search to small proxy models (which may not align with larger models) or require numerous independent runs. For cost-intensive foundation model pretraining (LLM/VLM), a method capable of automatically identifying the optimal learning rate within a single training run is needed.

## Core Problem
How to online identify a near-optimal learning rate within a single training run, while remaining compatible with modern learning rate schedulers (cosine, WSD)?

## Method

### Overall Architecture
The method monitors the slope of the training loss curve (loss descent velocity) and increases the learning rate when the slope decelerates. The slope is estimated via least squares over a $k$-step window, with a validation mechanism to prevent excessive adjustment.

### Key Designs
1. **Core Theoretical Insight**:

    - It is proved that the training loss $L(\eta)$ and the loss descent velocity $V(\eta)$ are both convex functions of the learning rate and share the same optimal learning rate $\eta^*$.
    - Theoretical derivation: under SGD, $\mathbb{E}[L_{t+1}-L_t] \approx -\eta\|\nabla L_t\|^2 + \frac{C_{Lip}}{2}\eta^2\|\nabla L_t\|^2$, with optimal $\eta^* = 1/C_{Lip}$.
    - This implies that optimizing the online-estimable slope can serve as an indirect proxy for optimizing the loss.

2. **Online Adjustment Algorithm**:

    - The loss slope $v(\eta)$ is estimated via least squares over every $k$-step window.
    - Adjustment rule: tentatively scale up by $\alpha'\eta$, then compare $v(\alpha'\eta)$ with $v(\eta)+2e$ to decide whether to maintain, increase, or decrease the learning rate.
    - Scaling factors decay over time: $\alpha' = \max(\lambda^t\alpha, 1)$, $\beta' = 1/\max(\lambda^t\beta, 1)$, with default $\lambda=0.99$.
    - Search is restricted to the $[0.1, 0.4]$ phase of training; the standard scheduler resumes control thereafter.

3. **Stability Mechanisms**:

    - Backtracking: upon a failed scale-up, model and optimizer states are restored to prevent corruption from destructive updates.
    - Early stopping: scale-up is halted if the loss exceeds the historical maximum.
    - Boundary condition: if the loss increases over two consecutive windows, the learning rate is reduced.

4. **Convergence Guarantees**:

    - Theorem 2.1: $\lim_{t\to\infty}\mathbb{P}(|\eta_t-\eta^*|<e)=1$ (almost sure convergence to an $e$-neighborhood).
    - Theorem 2.4: geometric error decay $|\eta_{t+k}-\eta^*|\leq\gamma|\eta_t-\eta^*|$, with complexity $O(\log R)$.

## Key Experimental Results

### Main Results: LLM Pretraining

| Setting | Model | Training Loss (baseline→AdaLRS) | PPL (val) |
|---------|-------|---------------------------------|-----------|
| Small LR init | Qwen2.5-1.5B | 2.56→improved | 12.66→improved |
| Large LR init | Qwen2.5-1.5B | 5.21→improved | 183.94→improved |
| Small LR init | Qwen2.5-7B | 2.38→improved | 10.61→improved |

- Small LR init: ~50% training cost savings to reach baseline loss.
- Large LR init: >30% training cost savings.
- Training budget: 120B–160B tokens, 10,000–20,000 910B NPU hours.

### VLM Pretraining (2B SAIL-VL, 7 Benchmarks)

| LR Setting | Avg. Score (Baseline / AdaLRS) |
|------------|-------------------------------|
| Fit LR | 56.16 / 55.80 |
| Small LR | 57.34 / 53.77 |
| Large LR | 48.96 / 47.67 |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| With / without backtracking | Backtracking is **critical**: without it, loss under large LR stagnates at 5.0–5.2 |
| Hyperparameter combinations ($\alpha$, $\beta$, $\lambda$) | Combinations such as (3/2, 0.99), (2/1.67, 0.99), (1.5/1.43, 0.99) are all effective, demonstrating strong robustness |
| Cosine vs. WSD scheduler | Both are compatible, validating plug-and-play applicability |
| Continual pretraining | Effective under small LR (0.8851→0.8286); large LR is constrained by catastrophic forgetting |

## Highlights & Insights
- **Theoretical elegance**: The convexity proof showing that the loss and loss slope share the same optimal learning rate provides a solid theoretical foundation for the method.
- **Strong practicality**: Plug-and-play design, compatible with mainstream schedulers, requiring no modification to the training pipeline.
- **Multi-scenario validation**: Covers LLM/VLM, varying scales, different initial learning rates, and both pretraining and continual pretraining settings.
- **Significant savings**: 50% reduction in training cost carries substantial practical significance for large-scale model pretraining.

## Limitations & Future Work
- The method fails when the initial learning rate is excessively large — parameters are already corrupted by destructive updates and cannot be recovered by simply lowering the learning rate.
- Convergence is only guaranteed to an $e$-neighborhood rather than the exact optimum; the magnitude of $e$ depends on estimation quality.
- Catastrophic forgetting under large learning rates cannot be mitigated during continual pretraining.
- The "Fit LR" baseline is determined via grid search, which may introduce comparison bias.
- The method's adaptability to changes in batch size is insufficiently analyzed.

## Related Work & Insights
- **vs. Chinchilla-style LR search**: Requires hundreds of independent training runs to establish scaling laws; AdaLRS achieves the same goal in a single run.
- **vs. μP/μTransfer**: Transfers hyperparameters from proxy models, but the proxy search itself remains costly; AdaLRS requires no proxy models.
- **vs. standard cosine scheduling**: AdaLRS applies adaptive adjustment on top of the cosine schedule — the two are complementary rather than mutually exclusive.
- The approach has direct engineering value for practical large-scale model training by reducing computational waste due to suboptimal learning rate selection.

## Rating
- Novelty: ⭐⭐⭐⭐ — The theoretical insight regarding convexity of the loss slope is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-model, multi-scenario coverage with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ — Theory and practice are integrated clearly.
- Value: ⭐⭐⭐⭐⭐ — Significant practical impact on large-scale model pretraining.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](../../ICCV2025/multimodal_vlm/babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)
- [\[NeurIPS 2025\] Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding](enhancing_visionlanguage_model_reliability_with_uncertaintyg.md)
- [\[NeurIPS 2025\] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [\[CVPR 2026\] Revisiting Model Stitching in the Foundation Model Era](../../CVPR2026/multimodal_vlm/revisiting_model_stitching_in_the_foundation_model.md)
- [\[NeurIPS 2025\] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)

</div>

<!-- RELATED:END -->
