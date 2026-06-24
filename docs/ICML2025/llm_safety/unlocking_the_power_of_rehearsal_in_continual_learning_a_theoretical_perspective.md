---
title: >-
  [Paper Note] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective
description: >-
  [ICML 2025][LLM Safety][continual learning] This work rigorously proves the effectiveness mechanism of the rehearsal strategy in continual learning from a theoretical perspective. Rehearsal approximates multi-task sequential learning as joint training by controlling the gradient direction bias. The forgetting bound grows sublinearly at $O(\sqrt{T/m})$ with respect to the buffer size $m$, providing precise guidance of $O(d/\epsilon^2)$ for buffer configuration in practical sys…
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "continual learning"
  - "rehearsal"
  - "experience replay"
  - "catastrophic forgetting"
  - "theoretical analysis"
date: 2026-05-08
content_hash: 310fac98dd9bc38d
---

# Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective

**Conference**: ICML 2025  
**arXiv**: [2506.00205](https://arxiv.org/abs/2506.00205)  
**Code**: None  
**Area**: Continual Learning  
**Keywords**: continual learning, rehearsal, experience replay, catastrophic forgetting, theoretical analysis

## TL;DR
This work rigorously proves the effectiveness mechanism of the rehearsal strategy in continual learning from a theoretical perspective. Rehearsal approximates multi-task sequential learning as joint training by controlling the gradient direction bias. The forgetting bound grows sublinearly at $O(\sqrt{T/m})$ with respect to the buffer size $m$, providing precise guidance of $O(d/\epsilon^2)$ for buffer configuration in practical systems.

## Background & Motivation

### Background
**Background**: Continual Learning (CL) faces the challenge of catastrophic forgetting, where the model's performance on old tasks degrades rapidly when learning new tasks sequentially. Rehearsal methods (experience replay) are among the most effective practical approaches, mitigating forgetting by storing a subset of old task samples and replaying them during new task training. Representative methods like ER (Experience Replay) and DER (Dark Experience Replay) have achieved excellent performance across multiple benchmarks.

### Limitations of Prior Work
**Limitations of Prior Work**: (1) **Severe Lack of Theoretical Understanding**: While rehearsal methods perform well in practice, they lack rigorous theoretical guarantees. Why is replay effective? How large should the buffer be? How does forgetting scale with the number of tasks? (2) **Lack of Guidance on Optimal Buffer Configuration**: In practice, buffer sizes are determined through trial and error, lacking a theoretical foundation. (3) **Missing Theoretical Comparison of Buffer Management Strategies**: It remains unclear which strategy is superior between uniform and progressive allocation, and under what conditions.

**Key Challenge**: The huge gap between the empirical success of rehearsal methods and the lack of theoretical analysis—systematic improvements are hindered without understanding their optimality and limitations.

### Research Goal & Strategy
**Goal**: To establish a theoretical analysis framework for rehearsal strategies to (1) quantify the control mechanism of rehearsal on forgetting, (2) provide the precise relationship between buffer size and forgetting, and (3) compare different buffer management strategies.

**Key Insight**: Analyzing the forgetting bound of SGD with rehearsal under a convex optimization framework, and decomposing the total forgetting into gradient bias and optimization error.

**Core Idea**: The essence of rehearsal is gradient correction. The gradient signals from the old task samples in the buffer correct the current gradient direction, approximating sequential learning as joint training. Perfect rehearsal (with an infinite buffer) is equivalent to joint training.

## Method

### Overall Architecture
Consider $T$ sequentially arriving tasks, where each task $t$ has a data distribution $\mathcal{D}_t$ and a loss function $\ell_t$. After completing task $t$, $m$ samples are preserved in the buffer $\mathcal{B}$ (total buffer size is $m$) for replay during subsequent training. The goal is to analyze the average performance (i.e., average forgetting metric) of the final model across all historical tasks.

### Key Designs

1. **Forgetting Bound Decomposition Theorem**:

    - **Function**: Precisely quantifying the mechanism of rehearsal in controlling forgetting.
    - **Mechanism**: Decomposing the total forgetting into two terms: $\text{Forgetting} = \underbrace{\text{梯度偏差项}}_{\text{排练缓冲不完美}} + \underbrace{\text{优化误差项}}_{\text{SGD 本身}}$. The core insight is that the bias $\|\hat{g}_t - g^*\|$ between the gradient signal $\hat{g}_t$ provided by the samples in the rehearsal buffer and the true joint gradient $g^*$ can be reduced by increasing the buffer size. When the buffer size is infinite, the bias is zero, and rehearsal is equivalent to joint training.
    - **Design Motivation**: This decomposition reveals the fundamental reason behind the effectiveness of rehearsal—it is not merely simple "data replay" but a systematic correction of the gradient direction.

2. **Precise Relationship Between Buffer Size and Forgetting**:

    - **Function**: Providing theoretical guidance for buffer configuration in practical systems.
    - **Mechanism**: Proving that when the buffer size satisfies $m \ge O(d/\epsilon^2)$ (where $d$ is the feature dimension), forgetting can be controlled within $\epsilon$. The average forgetting after $T$ tasks grows at $O(\sqrt{T/m})$—crucially, a **sublinear** rate! Compared to the linear growth of $O(T)$ without rehearsal, rehearsal provides a fundamental improvement.
    - **Design Motivation**: $O(d/\epsilon^2)$ establishes the relation between buffer requirements and the feature dimension—higher-dimensional problems require larger buffers, which is consistent with practical observations.

3. **Comparison of Buffer Management Strategies**:

    - **Function**: Theoretically comparing the optimality of different allocation strategies.
    - **Mechanism**: Analyzing three strategies—(a) **Reservoir Sampling**: Uniform sampling for preservation, theoretically simple but suboptimal; (b) **Uniform Allocation**: Fixed quota $m/T$ per task, where early tasks are underestimated in later stages; (c) **Progressive Allocation**: Allocating more samples to recent tasks and gradually fewer to older tasks. It is proved that under non-stationary settings (with large shifts in task distribution), progressive allocation is optimal, improving the forgetting bound from $O(\sqrt{T/m})$ to $O(\sqrt{T\log T/m})$.
    - **Design Motivation**: Providing a theoretical foundation for selecting buffer update strategies in online continual learning systems.

### Loss & Training
SGD with rehearsal under convex loss is analyzed. The training objective of each step is the weighted sum of the current task loss and the buffer replay loss: $\mathcal{L} = (1-\beta)\ell_t(x; \mathcal{D}_t) + \beta \cdot \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \ell_i(x)$, where $\beta$ controls the replay intensity.

## Key Experimental Results

### Main Results: Validation of Theoretical Bounds

| Buffer Strategy | Average Forgetting Bound | Condition | Growth Rate |
|-----------|-----------|------|--------|
| No Rehearsal (Naive SGD) | $O(T)$ | — | Linear |
| Uniform Rehearsal (m samples) | $O(\sqrt{T/m} + d/m)$ | Convex Loss | Sublinear |
| Progressive Rehearsal | $O(\sqrt{T\log T/m})$ | Non-stationary | Sublinear (Superior) |
| Perfect Rehearsal (m=∞) | 0 | — | Equivalent to Joint Training |

### Ablation Study: Theoretical Predictions vs. Empirical Matching

| Experiment | Theoretical Prediction | Empirical Observation | Consistency |
|------|---------|---------|--------|
| Buffer Size Sweep | Forgetting ∝ $1/\sqrt{m}$ | $\sqrt{1/m}$ Decreasing | ✓ Match |
| Number of Tasks Sweep | Forgetting ∝ $\sqrt{T}$ | $\sqrt{T}$ Increasing | ✓ Match |
| Uniform vs. Progressive | Progressive is superior in non-stationary settings | Progressive is superior | ✓ Match |

### Theoretical Comparison with Other Continual Learning Methods

| Method Category | Forgetting Bound | Assumptions |
|---------|--------|---------|
| No Rehearsal (Naive SGD) | $O(T)$ — Linear | Convex |
| **Rehearsal (Ours)** | $O(\sqrt{T/m})$ | Convex, Buffer m |
| Regularization (EWC theoretical bound) | $O(T/\lambda)$ | Strongly convex, regularization strength λ |
| Experience Replay + Knowledge Distillation | No theoretical bound | — |

### Key Findings
- The essence of rehearsal is **gradient bias control**; perfect rehearsal is equivalent to joint training.
- The $d/m$ term reveals the direct impact of feature dimension on buffer requirements, indicating that dimensionality reduction or feature selection can reduce memory usage.
- The $\sqrt{T}$ growth rate indicates that rehearsal does not completely eliminate forgetting, but significantly slows down its rate of increase.
- Progressive buffer allocation exhibits a clear advantage when there are significant shifts in the task distribution.

## Highlights & Insights
- **Theoretical Validation of Practical Intuition**: Rehearsal is effective because gradient correction approximates sequential CL to joint training—the larger the buffer, the better the approximation.
- **Precise Buffer Size Guidelines**: $O(d/\epsilon^2)$ directly informs engineers of the required buffer size to achieve the desired upper bound on forgetting.
- **Encouraging Signal of Sublinear Growth**: $\sqrt{T}$ implies that the average forgetting after 100 tasks is only $1/10$ of that without rehearsal.
- **Theoretical Support for Progressive Allocation**: Recent tasks are more important in non-stationary environments, which aligns with both intuition and empirical experience.

## Limitations & Future Work
- **Convex Loss Assumption**: Non-convex optimization in deep networks does not satisfy this condition, making the theoretical bound an upper bound rather than a tight bound in practice.
- **Bounded Task Distribution Shift Assumption**: The conclusions may not hold under extreme distribution shifts.
- **Buffer Sample Selection Strategies Omitted**: Selection based on difficulty or diversity might improve the bound, but it complicates the theoretical analysis.
- **Lack of Unified Theoretical Comparison with Regularization Methods (e.g., EWC)**: The conditions for the optimal combination of these two paradigms remain to be explored.

## Related Work & Insights
- **vs ER (Rolnick et al.)**: A classical representative of rehearsal-based methods; this work provides the first rigorous theoretical support for it.
- **vs EWC (Kirkpatrick et al.)**: A regularization-based method; this work focuses solely on the rehearsal paradigm, leaving a unified analysis of both as a promising future direction.
- **vs GEM (Lopez-Paz & Ranzato)**: A gradient-constrained method that restricts gradients of new tasks to avoid interfering with old tasks, which shares theoretical connections with the gradient bias decomposition in this work.
- **vs DER (Buzzega et al.)**: A hybrid approach combining knowledge distillation and rehearsal; the theoretical framework of this work can be extended to analyze the contribution of distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ First rigorous theoretical analysis of rehearsal methods; precise buffer size guidance offers both theoretical and practical value.
- Experimental Thoroughness: ⭐⭐⭐ Features verification matching theoretical predictions with empirical experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations with intuitive findings.
- Value: ⭐⭐⭐⭐ Provides much-needed theoretical guidance for the continual learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[ICML 2025\] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)

</div>

<!-- RELATED:END -->
