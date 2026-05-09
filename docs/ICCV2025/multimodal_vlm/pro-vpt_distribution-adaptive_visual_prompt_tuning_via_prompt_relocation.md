---
title: >-
  [Paper Note] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation
description: >-
  [ICCV 2025][Multimodal VLM][Visual prompt tuning] This paper proposes PRO-VPT, a framework that co-designs Adaptive Distribution Optimization (ADO) with Visual Prompt Tuning (VPT) via nested optimization. By iteratively relocating prompts through idleness score-based pruning and a reinforcement learning-based allocation strategy, PRO-VPT achieves gains of 1.6 pp and 2.0 pp over VPT on VTAB-1k and FGVC, respectively.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Visual prompt tuning
  - parameter-efficient fine-tuning
  - adaptive distribution optimization
  - prompt relocation
  - reinforcement learning
date: 2026-05-08
content_hash: 9c20fbad495e4750
---

# PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation

**Conference**: ICCV 2025  
**arXiv**: [2503.06901](https://arxiv.org/abs/2503.06901)  
**Code**: [https://github.com/ckshang/PRO-VPT](https://github.com/ckshang/PRO-VPT)  
**Area**: Multimodal VLM  
**Keywords**: Visual prompt tuning, parameter-efficient fine-tuning, adaptive distribution optimization, prompt relocation, reinforcement learning

## TL;DR

This paper proposes PRO-VPT, a framework that co-designs Adaptive Distribution Optimization (ADO) with Visual Prompt Tuning (VPT) via nested optimization. By iteratively relocating prompts through idleness score-based pruning and a reinforcement learning-based allocation strategy, PRO-VPT achieves gains of 1.6 pp and 2.0 pp over VPT on VTAB-1k and FGVC, respectively.

## Background & Motivation

### State of the Field

**State of the Field**: Visual Prompt Tuning (VPT) is one of the mainstream parameter-efficient fine-tuning (PEFT) methods, which adapts pre-trained models by inserting lightweight learnable prompt tokens into the input space of Transformer blocks. Existing VPT methods typically employ a fixed prompt distribution — either shallow (first layer only) or deep (uniformly distributed across all layers).

However, recent studies have revealed a critical phenomenon: **the importance of each pre-trained block varies substantially across tasks**, implying that indiscriminately applying a fixed prompt distribution cannot fully exploit the potential of VPT.

This paper systematically investigates two core questions:
1. How should Adaptive Distribution Optimization (ADO) be properly defined?
2. How should an adaptive distribution strategy be designed based on this definition?

Through empirical analysis, the authors identify three key insights:

### Limitations of Prior Work

**Limitations of Prior Work**: Appropriately adjusting prompt distribution can significantly improve performance.

### Root Cause

**Root Cause**: Effective adjustments vary as prompts are updated during training (non-static).

### Starting Point

**Starting Point**: The effect of distribution adjustment can only be accurately evaluated after prompt tuning has been performed (nested relationship).

## Method

### Overall Architecture

PRO-VPT adopts a nested optimization framework that alternates between ADO and VPT:
- **Outer optimization**: Adjusts the prompt distribution $D$
- **Inner optimization**: Tunes prompt parameters $P$ under the current distribution
- **Iterative cycle**: "Distribution adjustment → Prompt tuning → Adjustment evaluation → New adjustment"

### Key Designs

**1. Prompt Relocation (PR) Strategy**

Relocation is decomposed into two sequential steps, reducing the action space from $L^2$ to $L$:

**(a) Idleness Score-based Pruning**

An idleness score $I_k$ is defined to measure whether prompt $p_k$ is redundant in its current block. Intuitively, if removing a prompt improves performance, the prompt is considered "idle" in that block and is a candidate for relocation.

For efficient computation, a first-order Taylor expansion is used as an approximation: $I_k \approx g_k^T \cdot p_k$, where $g_k$ is the gradient with respect to the prompt. The prompt with the largest $I_k$ is selected for pruning.

**(b) RL-based Allocation**

A PPO algorithm determines which block the pruned prompt should be allocated to:
- **State**: Idleness scores of all blocks + current distribution + position of the pruned prompt
- **Action**: Select a target layer $a \in [L]$
- **Reward**: Performance improvement after prompt tuning following the allocation

**2. Fundamental Distinction from Pruning Methods**

Existing pruning methods (e.g., NOAH) can only remove prompts from saturated blocks but cannot add prompts to under-served blocks. PRO-VPT achieves true relocation through "pruning + allocation," maximizing the overall effectiveness of the prompt distribution.

### Loss & Training

- **Task loss**: Standard cross-entropy classification loss
- **Pruning evaluation**: Idleness score approximated via Taylor expansion
- **Allocation optimization**: PPO policy gradient with a reward incorporating an idleness score correction term
- **Overall pipeline**: PR is executed periodically (once after multiple VPT iterations per epoch)

## Key Experimental Results

### Main Results

Comparison on VTAB-1k (ViT-B/16, pre-trained on ImageNet-21k):

| Method | Params (M) | Natural | Specialized | Structured | Overall Mean |
|--------|------------|---------|-------------|------------|--------------|
| Full FT | 85.8 | 78.6 | 86.3 | 57.8 | 74.2 |
| VPT-Deep | 0.60 | 82.5 | 84.6 | 62.1 | 76.4 |
| LoRA | 0.29 | 82.4 | 84.3 | 60.1 | 75.6 |
| Adapter | 0.35 | 83.3 | 86.2 | 63.3 | 77.6 |
| **PRO-VPT** | 0.60 | **83.3** | **86.2** | **64.5** | **78.0** |

PRO-VPT achieves state-of-the-art performance with the same parameter budget as VPT-Deep, outperforming all PEFT baselines including Adapter.

### Ablation Study

Performance variation of distribution adjustment applied at different training epochs:

| Adjustment Timing | Epoch 25 | Epoch 50 | Epoch 75 |
|-------------------|----------|----------|----------|
| Optimal adjustment block | Different | Different | Different |
| Performance gain | Yes | Yes | Yes |

This validates Finding 2: effective distribution adjustment varies as prompts are updated, necessitating an iterative process.

On the FGVC benchmark, PRO-VPT achieves a mean accuracy of 91.7%, surpassing VPT by 2.0 pp.

### Key Findings

- Prompt distribution matters: appropriate distribution adjustment yields significant performance gains.
- The nested relationship holds: distribution optimization must be nested within prompt tuning.
- First-order approximation is effective: Taylor expansion-based idleness scores are highly correlated with ground-truth values.
- Reducing the RL allocation action space from $L^2$ to $L$ substantially accelerates convergence.

## Highlights & Insights

1. **First systematic definition of the ADO problem**: Empirically motivated, revealing the nested relationship between ADO and VPT.
2. **Decomposed pruning + allocation strategy**: Resolves the action space explosion while providing an intuitive interpretation (transferring prompts from saturated blocks to under-served ones).
3. **Taylor expansion approximation**: Reduces idleness score computation from $O(N)$ forward passes to a single gradient computation.
4. **Unifies existing pruning methods under a theoretical framework**: Demonstrates that pruning-only methods are special cases of PRO-VPT.

## Limitations & Future Work

- RL training introduces additional hyperparameters and computational overhead.
- Validation is limited to ViT-B/16; larger-scale models have not been tested.
- Only one prompt is relocated per step, which may lead to slower convergence.
- Additive distribution adjustment (directly increasing the number of prompts) was found to be unstable and was excluded in favor of relocation only.

## Related Work & Insights

- Consistent with findings in layer-wise fine-tuning that different layers require different fine-tuning intensities.
- The nested optimization framework is general and can be extended to hyperparameter optimization for other PEFT methods.
- The Taylor expansion-based pruning idea originates from neural network pruning and is innovatively applied in the prompt tuning setting.

## Rating

- Novelty: ⭐⭐⭐⭐ (Systematic definition of the ADO problem and nested optimization framework are innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (All 19 datasets of VTAB-1k + FGVC + comprehensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐⭐ (Problem-driven, progressively structured, and rigorously analyzed)
- Value: ⭐⭐⭐⭐ (Substantial advancement in the VPT field, though absolute performance gains are moderate)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)
- [\[ICLR 2026\] Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)

<!-- RELATED:END -->
