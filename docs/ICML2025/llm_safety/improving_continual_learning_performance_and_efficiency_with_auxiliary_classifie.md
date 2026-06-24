---
title: >-
  [Paper Note] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers
description: >-
  [ICML 2025][LLM Safety][continual learning] This paper for the first time explores the application of early-exit networks (EENs) in continual learning, discovering that early classifiers inherently suffer from less catastrophic forgetting. It proposes the Task-wise Logits Correction (TLC) method to balance task bias, matching the accuracy of standard methods with less than 70% of the computational cost in class-incremental learning.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "continual learning"
  - "early-exit networks"
  - "catastrophic forgetting"
  - "task-recency bias"
  - "dynamic inference"
date: 2026-05-08
content_hash: 9e134d176aa20b53
---

# Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers

**Conference**: ICML 2025  
**arXiv**: [2403.07404](https://arxiv.org/abs/2403.07404)  
**Code**: [https://anonymous.4open.science/r/ContinualEE](https://anonymous.4open.science/r/ContinualEE) (to be released)  
**Area**: Continual Learning / Efficient Inference  
**Keywords**: continual learning, early-exit networks, catastrophic forgetting, task-recency bias, dynamic inference

## TL;DR
This paper for the first time explores the application of early-exit networks (EENs) in continual learning, discovering that early classifiers inherently suffer from less catastrophic forgetting. It proposes the Task-wise Logits Correction (TLC) method to balance task bias, matching the accuracy of standard methods with less than 70% of the computational cost in class-incremental learning.

## Background & Motivation
**Background**: Continual learning (CL) studies how to sequentially learn on non-i.i.d. data streams without forgetting old knowledge. Early-exit networks (EENs) add internal classifiers (ICs) to intermediate layers of the network, allowing "easy" samples to exit early to save computation.

**Limitations of Prior Work**: Existing CL methods only consider a single final classifier, ignoring the prediction potential of intermediate layers. EENs are only studied under i.i.d. settings, without considering dynamic data distributions. The two fields remain isolated from each other.

**Key Challenge**: In CL, deeper layers suffer severe forgetting, yet traditional methods only utilize predictions from the final layer; in EENs, task-recency bias prevents old task samples from exiting early.

**Goal**: (1) Explore the behavioral characteristics of EENs in CL; (2) Address the negative impact of task bias on dynamic inference.

**Key Insight**: Analysis reveals that early ICs suffer less forgetting and "overthinking" is more severe in CL. Based on this, early exiting is leveraged to improve both efficiency and performance.

**Core Idea**: Early exit + Task-wise logits correction = Reducing computation while mitigating forgetting.

## Method

### Overall Architecture
Input: Classification data $\{(\mathcal{X}_t, \mathcal{Y}_t)\}_{t=1}^T$ arriving sequentially by task.  
Output: A continual learning model with early-exit capabilities, allowing dynamic selection of the exit layer during inference.

Pipeline:
1. Place 6 internal classifiers (ICs) in the intermediate layers of a standard ResNet, positioned at approximately 15%, 30%, 45%, 60%, 75%, and 90% of the computational budget following the SDN architecture.
2. When each task arrives, train all ICs using a weighted joint loss: $\min_\theta \sum_{i=1}^{M+1} w_i \cdot \mathcal{L}_i$
3. During inference, if the prediction confidence of any IC exceeds a threshold $\tau$, the prediction is returned early.
4. After training all tasks, apply TLC to correct the logit bias across tasks.

### Key Designs

1. **Early-Exit Continual Learning Architecture**:
    - **Function**: Adapts the SDN (Shallow-Deep Network) architecture to class-incremental learning settings.
    - **Mechanism**: Each IC consists of a feature reduction (FR) layer + a multi-head fully connected (FC) layer. A new classification head is added to each IC when a new task arrives.
    - **Design Motivation**: The multi-head design naturally fits the class-incremental setting, and the ICs share backbone features.

2. **Task-wise Logits Correction (TLC)**:
    - **Function**: Corrects task bias so that old task predictions can obtain sufficiently high confidence to exit early.
    - **Mechanism**: Add a correction term $c_t = a \cdot (T - t) + b$ to the logits of the $t$-th task, where $a, b$ are optimized by minimizing the average maximum logit difference across tasks:
      $$E(a,b) = \sum_{i=1}^{N} \sum_{t=1}^{T} (M_j - m_t^{i,j})^2$$
      where $m_t^{i,j} = \max(\bar{y}_t^{i,j} + a(T-t) + b)$, and $M_j$ is the average maximum logit across all tasks and classifiers.
    - **Design Motivation**: Task bias causes low confidence for old task samples, preventing them from exiting at early ICs, which happen to suffer less forgetting of old tasks. Correcting this bias allows old task samples to exit at early layers that forget less, achieving a win-win.

3. **Overthinking Analysis**:
    - **Function**: Quantitatively analyzes the overthinking phenomenon in CL.
    - **Mechanism**: Overthinking is defined as the difference between oracle accuracy (at least one IC is correct) and the final classifier's accuracy. It is found that overthinking is much more severe in CL than in joint training.
    - **Design Motivation**: This finding provides the theoretical basis for using early-exit strategies—since deeper layers "overthink" and cause errors, it is better to trust early layers.

### Loss & Training
- Main loss: $\sum_{i=1}^{M+1} w_i \cdot \mathcal{L}_i(C_i^t(E_i(\mathcal{X}_t; \theta_i)), \mathcal{Y}_t)$
- $w_i$ adopts the SDN progressive scheduler, which emphasizes early ICs at the beginning of training and gradually increases the weights of deeper ICs later.
- Combination with base CL methods (LwF, ER, BiC, iCaRL, etc.): Duplicate the regularization/distillation logic of the original method for each IC.

## Key Experimental Results

### Main Results (10-Task Class-Incremental Learning)

| Method | CIFAR100 (100% FLOPs) | CIFAR100 (75% FLOPs) | ImageNet-Sub (100%) | ImageNet-Sub (80%) |
|---|---|---|---|---|
| iCaRL (Standard) | 41.23 | — | 38.92 | — |
| iCaRL + EE | 38.80 | 36.81 | 36.58 | 34.34 |
| iCaRL + EE + TLC | **47.98** | **46.22** | **51.90** | **49.52** |
| BiC (Standard) | 46.94 | — | 50.86 | — |
| BiC + EE + TLC | **49.10** | **47.58** | **55.38** | **53.28** |
| LwF (Standard) | 25.03 | — | 24.60 | — |
| LwF + EE + TLC | **30.10** | **29.81** | **29.68** | **29.92** |

### Ablation Study

| Configuration | CIFAR100 Accuracy | FLOPs % | Description |
|---|---|---|---|
| Standard network (No EE) | 41.23 (iCaRL) | 100% | Baseline |
| + Early-Exit (No TLC) | 38.80 | 100% | Effect of task bias |
| + Early-Exit + TLC | 47.98 | 100% | Significant improvement |
| + Early-Exit + TLC | 46.22 | 75% | Saves 25% computation, still surpasses baseline |
| + Early-Exit + TLC | 37.98 | 50% | Saves 50% computation, close to baseline |

### Key Findings
- **Early ICs forget less**: Taking LwF as an example, IC4 achieves about 30% accuracy on Task 1, while the final classifier only gets 13%, verifying that lower layers have "better memory".
- **Overthinking is more severe in CL**: The gap between oracle accuracy and final classifier accuracy is much larger in CL than in joint training.
- **Huge improvement from TLC**: Improves iCaRL by ~7 percentage points on CIFAR100 and ~13 percentage points on ImageNet-Subset.
- Matches the performance of standard methods using full computation with only 70% of the computational cost.

## Highlights & Insights
- Reveals for the first time the structural finding that "early layers forget less" in continual learning, and cleverly exploits it.
- The TLC method is extremely simple (with only two parameters $a, b$), yet highly effective.
- Reducing computation and reducing forgetting are not contradictory—this is a counter-intuitive yet crucial finding.

## Limitations & Future Work
- TLC uses a linear correction model, which may be insufficient for complex bias patterns.
- Currently only validated on ResNet32/18 architectures; its effectiveness on architectures like Transformers remains unknown.
- The selection of the exit threshold $\tau$ remains an open question.

## Related Work & Insights
- Combines early-exit methods like SDN and BranchyNet with CL methods such as iCaRL and BiC.
- Insight: Layers at different network depths exhibit distinct forgetting characteristics; future CL methods can exploit this structural property more granularly.
- Holds significant practical value for resource-constrained scenarios (e.g., continual learning on edge devices).

## Rating
- Novelty: ⭐⭐⭐⭐ Explores the combination of early exit with continual learning for the first time, discovering structural insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three datasets, six CL methods, and multiple computational budgets.
- Writing Quality: ⭐⭐⭐⭐⭐ Deep analysis, clear logic, and rich tables.
- Value: ⭐⭐⭐⭐⭐ Simultaneously improves efficiency and performance, presenting outstanding practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[ICML 2025\] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin.md)

</div>

<!-- RELATED:END -->
