---
title: >-
  [Paper Note] Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization
description: >-
  [ACL 2025][LLM Alignment][Machine Unlearning] In the SemEval 2025 LLM Unlearning Shared Task, this paper combines Negative Preference Optimization (NPO) with Low-Rank Adaptation (LoRA). By leveraging the structural properties of LoRA, the authors acquire the original model distribution with zero additional overhead to compute KL divergence regularization, significantly stabilizing the unlearning process and outperforming the task baselines.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Machine Unlearning"
  - "NPO"
  - "LoRA"
  - "KL Regularization"
  - "OLMo"
date: 2026-05-08
content_hash: 33855a3e8861637b
---

# Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization

**Conference**: ACL 2025  
**arXiv**: [2503.13690](https://arxiv.org/abs/2503.13690)  
**Code**: [https://github.com/XelfXendr/peft_unlearning](https://github.com/XelfXendr/peft_unlearning)  
**Area**: LLM Alignment / Machine Unlearning  
**Keywords**: Machine Unlearning, NPO, LoRA, KL Regularization, OLMo

## TL;DR
In the SemEval 2025 LLM Unlearning Shared Task, this paper combines Negative Preference Optimization (NPO) with Low-Rank Adaptation (LoRA). By leveraging the structural properties of LoRA, the authors acquire the original model distribution with zero additional overhead to compute KL divergence regularization, significantly stabilizing the unlearning process and outperforming the task baselines.

## Background & Motivation
**Background**: LLMs memorize sensitive information (such as copyrighted content and personal privacy) in the training data after being trained on massive datasets. Regulations like GDPR mandate the "right to be forgotten". Retraining from scratch is extremely expensive, and machine unlearning aims to remove specific knowledge through fine-tuning.

**Limitations of Prior Work**: Unlearning methods based on Gradient Ascent (GA) easily destroy the overall utility of the model. NPO alleviates this but requires maintaining a copy of the original model to compute reference probabilities, which doubles the memory overhead. Meanwhile, most methods are based on full-parameter fine-tuning, which is computationally expensive.

**Key Challenge**: Achieving effective unlearning while maintaining model utility, and being as efficient as possible in terms of computation and memory.

**Goal**: Combine NPO with LoRA to achieve high-quality unlearning with zero extra memory overhead by leveraging the structural advantages of LoRA.

**Key Insight**: A key insight is that the original weights in LoRA are frozen and preserved. When computing the original model's output, one only needs to disable the LoRA layers, eliminating the need to store an extra model copy.

**Core Idea**: The original weights are naturally preserved in LoRA. Disabling LoRA allows retrieving the reference model's output distribution, thereby achieving NPO with KL regularization unlearning with zero extra overhead.

## Method

### Overall Architecture
Unlearning the fine-tuned OLMo-7B model:
- Add LoRA adapters (rank=5) to the attention layers of the model.
- Only train LoRA parameters while freezing the original weights.
- Perform two forward passes per batch: (1) Enable LoRA to compute the retain loss; (2) Disable LoRA to get the reference model output, computing the NPO loss and KL regularization.

### Key Designs

1. **NPO + LoRA Combination**:

    - **Function**: Replace simple gradient ascent with NPO loss for unlearning, while using LoRA for parameter-efficient fine-tuning.
    - **Mechanism**: The NPO loss is $\mathcal{L}_{NPO}(\theta;\beta) = \mathbb{E}_{\mathcal{D}_{FG}}[\frac{2}{\beta}\log(1+(\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)})^\beta)]$, where $\pi_{ref}$ requires the output of the original model. Traditional approaches require storing a copy of the original model, but the frozen original weights in LoRA are exactly the $\pi_{ref}$—which can be accessed simply by disabling the LoRA layers.
    - **Design Motivation**: NPO is proven to be more stable than GA, and LoRA reduces memory. Combining them addresses the memory bottleneck of requiring dual models.

2. **KL Divergence Regularization ($\mathcal{K}_{RT}$)**:

    - **Function**: Minimize the KL divergence of the output patterns between the current model and the original model on the retain set.
    - **Mechanism**: $\mathcal{K}_{RT}(\theta) = \mathbb{E}_{\mathcal{D}_{RT}}[KL(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x))]$, which requires the full output distribution (not just specific token probability) and thus cannot be computed in advance.
    - **Design Motivation**: The retain loss $\mathcal{L}_{RT}$ only ensures accurate predictions on the retain set but does not constrain the overall shape of the output distribution. KL regularization further ensures no distribution shift, stabilizing long-term training.

3. **Total Loss Function**:

    - $\mathcal{L}(\theta;\beta,\gamma,\delta) = \mathcal{L}_{NPO}(\theta;\beta) + \gamma\mathcal{L}_{RT}(\theta) + \delta\mathcal{K}_{RT}(\theta)$
    - The three terms are responsible for: unlearning target data, retaining the accuracy of non-target data, and keeping the model distribution from shifting, respectively.

### Loss & Training
- $\beta=0.5$, LoRA rank $r=\alpha=5$, learning rate $10^{-4}$, batch size 4.
- Search in the $(\gamma, \delta)$ space: (1,0), (1,0.5), (1,1.0), (0,1.0).
- Run 5 different seeds for each configuration and report the mean and standard deviation.

## Key Experimental Results

### Main Results (OLMo-7B)

| Configuration | Epoch | Task Score↑ | MIA Score↑ | MMLU↑ | Final Score↑ |
|---|---|---|---|---|---|
| γ=1,δ=0 (No KL) | 10 | 0.431 | 0.657 | 0.461 | 0.516 |
| γ=1,δ=0.5 | 20 | 0.434 | 0.594 | 0.439 | 0.489 |
| γ=1,δ=1.0 | 20 | **0.453** | 0.620 | 0.449 | **0.507** |
| γ=0,δ=1.0 (KL Only) | 10 | 0.369 | **0.699** | 0.441 | 0.503 |
| Baseline NPO | - | 0.021 | 0.080 | **0.463** | 0.188 |
| Baseline GD | - | 0.000 | 0.382 | 0.348 | 0.243 |

### Ablation Study: Effect of KL Regularization

| Configuration | 10 epoch Final | 20 epoch Final | Stability (σ) |
|---|---|---|---|
| γ=1,δ=0 (No KL) | 0.516 | 0.429↓ | High Variance |
| γ=1,δ=1.0 (With KL) | 0.327 | 0.507↑ | More Stable |

### Key Findings
- All configurations significantly outperform the task baselines (Final Score 0.489-0.516 vs 0.188-0.243).
- KL regularization ($\delta>0$) stabilizes the unlearning process—training for long (20 epochs) without KL causes degradation, whereas with KL, 20 epochs yields better results.
- The retain loss $\mathcal{L}_{RT}$ is critical for maintaining MMLU scores—removing it leads to a larger drop in MMLU.
- The "two models in one" characteristics of LoRA is key—achieving zero extra memory overhead to obtain the reference model's output in the NPO+KL scenario.
- The variance across different random seeds is large (especially for MIA Score), indicating the inherent instability of the unlearning process.

## Highlights & Insights
- **Clever Exploitation of LoRA's Structure**: Not just for parameter efficiency; it leverages the "frozen original weights" property to acquire the reference model's output for free. This is a highly elegant insight—where previous scenarios required two model copies, LoRA achieves it with just one.
- **KL Regularization Stabilizes Unlearning**: Unlearning and retaining are naturally conflicting goals. The KL divergence constraint acts as a "seatbelt", preventing the model from drifting too far during long-duration training.
- **Ultra-small LoRA rank (r=5) is Sufficient**: This suggests that the parameter change space required for unlearning is quite small, further rendering parameter-efficient methods suitable for unlearning tasks.

## Limitations & Future Work
- Only validated on OLMo-7B, without testing other architectures (e.g., Llama, Mistral).
- Large variance across different random seeds, indicating that the stability of the method still needs improvement.
- The selection of hyperparameters $(\beta, \gamma, \delta)$ lacks theoretical guidance and relies heavily on grid search.
- Only tested on English unlearning tasks; the performance in multilingual scenarios is unknown.
- Variants like larger LoRA rank or QLoRA were not explored.

## Related Work & Insights
- **vs. Original NPO (Zhang et al.)**: The original version uses full-parameter fine-tuning and stores an extra reference model. This work uses LoRA to significantly reduce memory and leverage structural features to avoid dual models.
- **vs. TOFU (Maini et al.)**: TOFU also uses KL regularization, but this work demonstrates how LoRA can make KL computations zero-overhead.
- **vs. Gradient Ascent**: GA achieves a Final Score of only 0.243 in the shared task baselines, while NPO+LoRA reaches 0.516, showing that the advantage of NPO is highly significant in practice.
- The structural advantage of LoRA in unlearning tasks (free reference model) can be generalized to other preference optimization scenarios that require DPO/RLHF.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Although the NPO+LoRA combination of components is not entirely new, the integration strategy is ingenious. "Disabling LoRA to obtain the reference model" is an elegant engineering insight.
- **Experimental Thoroughness**: ⭐⭐⭐ Validated only on a single model, with multiple seeds and hyperparameters tested but yielding high variance.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations, and the motivation of the method is well articulated.
- **Value**: ⭐⭐⭐⭐ Provides direct reference value for practitioners in the LLM unlearning community; the exploitation of LoRA's structure is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)
- [\[ACL 2025\] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
