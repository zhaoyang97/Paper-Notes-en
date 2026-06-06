---
title: >-
  [Paper Note] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models
description: >-
  [ACL 2026][Model Compression][Structured Pruning] GRASPrune proposes a global-budget-constrained structured pruning framework that enforces hard mask budget constraints at each training step through a Projected Straight-…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Structured Pruning"
  - "Global Budget"
  - "Gating Learning"
  - "KV Head Pruning"
  - "Projected STE"
date: 2026-05-08
content_hash: 2a37f41d6ff4e6a4
---

# GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.19398](https://arxiv.org/abs/2604.19398)  
**Code**: [GitHub](https://github.com/ZiY-Wang/GRASPrune)  
**Area**: Robotics  
**Keywords**: Structured Pruning, Global Budget, Gating Learning, KV Head Pruning, Projected STE

## TL;DR
GRASPrune proposes a global-budget-constrained structured pruning framework that enforces hard mask budget constraints at each training step through a Projected Straight-Through Estimator (Projected STE). By jointly pruning FFN channels and KV head groups, it achieves 12.18 PPL on LLaMA-2-7B with 50% parameter retention, requiring only 6 minutes of training on a single A100 GPU.

## Background & Motivation

**Background**: The inference costs of LLMs are prohibitive—model parameters, attention computation, and KV cache all impose significant memory and latency overhead. Structured pruning produces smaller dense checkpoints by removing channels or head groups, which can be directly deployed using standard inference stacks.

**Limitations of Prior Work**: (1) FFN channels and KV head groups are typically pruned separately using different criteria, despite sharing the same deployment budget and representation capacity; (2) Many methods use predefined layer-wise retention rates or depth-dependent schedules, hard-coding budget allocation rather than learning globally optimal distributions; (3) Existing pipelines estimate importance scores before applying budgets, leading to unconstrained training and constrained selection—creating a disconnection between score learning and the final mask.

**Key Challenge**: The primary issue is not the lack of better saliency metrics, but a mismatch between how scores are learned and how final masks are selected—scores learned without constraints may lead to suboptimal allocation under constrained selection.

**Goal**: To enforce budget feasibility within the optimization loop, ensuring that gating scores are learned under the same constraints as the final deployment mask.

**Key Insight**: Formalize structured pruning as a joint optimization problem under a single global budget constraint, where FFN channels and KV head groups compete for the same budget with different unit costs.

**Core Idea**: Use Projected STE to perform budget projection $\rightarrow$ hard mask forward pass $\rightarrow$ soft score backward pass at each training step. Combined with post-processing scaling calibration, this generates smaller dense checkpoints without additional inference overhead.

## Method

### Overall Architecture
Three stages: (1) Gating learning—freeze backbone weights and optimize scalar gating scores using Projected STE, performing budget-feasible hard mask projections at each step; (2) Scaling calibration—freeze the mask and learn scalar multipliers for retained units to mitigate scale shifts caused by pruning; (3) Checkpoint compilation—fold scaling factors into sliced weights to output smaller dense checkpoints.

### Key Designs

1.  **Global Budget Joint Pruning**:
    - **Function**: Allows FFN channels and KV head groups to compete optimally under a unified budget.
    - **Mechanism**: Each prunable unit $i \in \mathcal{S}$ has a binary variable $z_i$ and a cost $c_i$. For FFN channels $c_i=1$, and for KV head groups $c_i=\alpha$, where $\alpha = \frac{(2G+2)d_h}{3}$ is the parameter approximation ratio. The global budget is $B = \rho \sum c_i$. The optimization is $\min_{\mathbf{z}} \mathcal{L}(\theta; \mathbf{z})$ s.t. $\sum c_i z_i \leq B$, where $\theta$ is fixed and only $\mathbf{z}$ is optimized.
    - **Design Motivation**: Separate pruning of FFN and KV cannot allocate budgets optimally at a global level—joint optimization allows the system to automatically determine where to retain more capacity.

2.  **Projected Straight-Through Estimator (Projected STE)**:
    - **Function**: Enables differentiable optimization in discrete mask selection while satisfying the budget at every step.
    - **Mechanism**: At each step, project continuous gating probabilities $\mathbf{p}$ into a budget-feasible hard mask $\mathbf{m} = \text{Project}(\mathbf{p}, \mathbf{c}, B)$ by sorting $p_i$ in descending order and greedily selecting units until the budget is exhausted. The forward pass uses the hard mask $m_i$, and the backward pass uses the STE of soft probabilities $p_i$: $\tilde{z}_i = m_i + (p_i - \text{stopgrad}(p_i))$. **Key Design**: Sorting by $p_i$ instead of $p_i/c_i$—cost-normalization biases allocation towards cheaper units.
    - **Design Motivation**: Typical pipelines learn scores and then select masks; GRASPrune imposes constraints during every training step, ensuring scores are learned under those constraints.

3.  **Budget-Preserving Scaling Calibration**:
    - **Function**: Alleviates output scale shifts caused by pruning.
    - **Mechanism**: After freezing the hard mask $\mathbf{m}$, introduce a scalar multiplier $\gamma_i$ for each retained unit, optimized on the same calibration set with frozen backbone weights. The number of trainable parameters is $O(|\mathcal{S}|)$, and FLOPs remain unchanged. Finally, fold $\gamma$ into the sliced weights.
    - **Design Motivation**: Pruning attention heads changes the output scale; lightweight rescaling is significantly more efficient than full fine-tuning.

### Loss & Training
The language modeling loss is used to optimize gating scores on the calibration set. It utilizes 512 unlabeled sequences over 4 epochs, taking approximately 6 minutes on a single A100. No full model fine-tuning is required.

## Key Experimental Results

### Main Results

| Parameter Ratio | Method | Wiki PPL↓ | Zero-shot Avg Acc |
| :--- | :--- | :--- | :--- |
| 50% | LLM-Pruner | ~18 | 0.61 |
| 50% | SliceGPT | ~15 | - |
| 50% | **Ours** | **12.18** | **Strong** |
| 40% | **Ours** | **16.65** | - |

### Ablation Study

| Configuration | Description |
| :--- | :--- |
| Sorting by $p_i/c_i$ | Bias towards cheap units, performance drops |
| No scaling calibration | PPL increases |
| Perturbing $\alpha$ | Moderately insensitive to $\alpha$ |
| Projection overhead | Sorting time is only 0.11% of total training time |

### Key Findings
- Achieving a PPL of 12.18 at 50% parameter retention outperforms all baseline methods.
- Sorting by $p_i$ (rather than $p_i/c_i$) yields better results—counter-intuitive but justified because scores are learned under constraints.
- Training is highly efficient—gating learning and calibration are completed in 6 minutes on a single GPU.
- KV cache reduction leads to actual inference speedups.

## Highlights & Insights
- The insight of **"learning under constraint" vs. "constraining after learning"** is profound—addressing a widely overlooked issue in structured pruning.
- Joint FFN+KV pruning using a unified cost model $\alpha$ elegantly handles fair competition between heterogeneous structures.
- Extremely low training cost (6 minutes on a single GPU) makes the method highly practical.

## Limitations & Future Work
- The cost model is based on parameter approximations and does not directly optimize for latency or throughput.
- Evaluations were only conducted on the LLaMA-2 series; applicability to newer models (LLaMA-3+) needs verification.
- Lack of a subsequent fine-tuning stage—for extremely high compression rates (<30%), fine-tuning may be necessary.

## Related Work & Insights
- **vs. LLM-Pruner**: LLM-Pruner uses Taylor scores and layer-wise scheduling; GRASPrune uses a global budget and Projected STE.
- **vs. ZipLM**: ZipLM also performs global sorting but ignores cost differences; GRASPrune explicitly models heterogeneous costs.
- **vs. DISP-LLM**: DISP-LLM performs dimension-independent architecture search; GRASPrune is simpler and more training-efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ (Innovation in Projected STE and in-budget training)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple compression rates, detailed ablation, and efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Incisive problem analysis and rigorous derivation)
- Value: ⭐⭐⭐⭐ (Efficient and practical LLM structured pruning solution)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Two-Stage Regularization-Based Structured Pruning for LLMs](two-stage_regularization-based_structured_pruning_for_llms.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)

</div>

<!-- RELATED:END -->
