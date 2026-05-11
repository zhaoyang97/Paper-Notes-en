---
title: >-
  [Paper Note] PLAN: Proactive Low-Rank Allocation for Continual Learning
description: >-
  [ICCV 2025][Model Compression][Continual Learning] This paper proposes PLAN, a framework that proactively allocates orthogonal low-rank subspaces for each task and employs a perturbation-based strategy to minimize inter-…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Continual Learning"
  - "Low-Rank Adaptation"
  - "LoRA"
  - "Catastrophic Forgetting"
  - "Subspace Allocation"
date: 2026-05-08
content_hash: 93132bc53fc6bad0
---

# PLAN: Proactive Low-Rank Allocation for Continual Learning

**Conference**: ICCV 2025
**arXiv**: [2510.21188](https://arxiv.org/abs/2510.21188)
**Code**: N/A
**Area**: Model Compression
**Keywords**: Continual Learning, Low-Rank Adaptation, LoRA, Catastrophic Forgetting, Subspace Allocation

## TL;DR

This paper proposes PLAN, a framework that proactively allocates orthogonal low-rank subspaces for each task and employs a perturbation-based strategy to minimize inter-task interference, achieving efficient and forgetting-free fine-tuning of large models in continual learning (CL) settings, establishing a new state of the art on standard CL benchmarks.

## Background & Motivation

**Background**: Continual Learning (CL) requires models to sequentially learn multiple tasks without forgetting previously acquired knowledge. With the widespread adoption of large-scale pretrained models, parameter-efficient fine-tuning (PEFT)-based CL methods have become mainstream. Among these, LoRA (Low-Rank Adaptation) has attracted considerable attention due to its ability to train only a small number of parameters. Existing methods such as O-LoRA and InfLoRA attempt to address catastrophic forgetting within the LoRA framework via orthogonality constraints.

**Limitations of Prior Work**: Existing LoRA-based CL methods suffer from two core issues: (1) **Passive allocation** — most methods decide how to utilize the parameter space only when learning a new task, without accounting for future task requirements, resulting in progressively shrinking "parameter budgets" for subsequent tasks; (2) **Accumulated interference** — even with orthogonality constraints, stacking multiple low-rank updates on shared weight matrices still introduces implicit interference, as orthogonality only ensures distinct projection directions but does not prevent mutual influence during gradient updates.

**Key Challenge**: The parameter space of LoRA is inherently limited (bounded by rank $r$), and CL requires balancing "sufficient learning for the current task" against "reserving capacity for future tasks." Existing methods either greedily allocate space — degrading performance on later tasks — or are overly conservative, resulting in poor learning across all tasks.

**Goal**: (1) Design a proactive subspace allocation strategy that pre-plans the low-rank directions used by each task; (2) Introduce an interference-aware basis vector selection mechanism that ensures learning new tasks does not compromise existing knowledge.

**Key Insight**: The authors observe that if a set of orthogonal basis vectors can be pre-allocated to each task, and if at training time the subset with the least interference to existing parameters is selected, both passive allocation and accumulated interference can be fundamentally resolved. The key insight is that the sensitivity of different basis vector directions to previously learned parameters can be quantified via perturbation analysis.

**Core Idea**: A global pool of orthogonal basis vectors is generated in advance. When learning a new task, a perturbation-based strategy evaluates the degree of interference each basis vector imposes on existing tasks, and the subset with minimal interference is actively selected to construct the low-rank adapter for the new task.

## Method

### Overall Architecture

PLAN attaches low-rank adapters (structured similarly to LoRA) to each linear layer of a pretrained model. Before training begins, a shared pool of orthogonal basis vectors is pre-allocated for all anticipated tasks. Upon arrival of a new task, PLAN first selects from the pool the subset of basis vectors with the least interference to existing tasks, uses these vectors to construct the low-rank update matrix for the current task, and then optimizes only the projection coefficients while keeping the basis vector directions frozen. At inference time, updates across different tasks can be merged into the original weights with no additional computational overhead.

### Key Designs

1. **Proactive Construction of the Orthogonal Basis Vector Pool**:

    - **Function**: Provides a predefined, mutually non-interfering partition of the parameter subspace for all tasks.
    - **Mechanism**: For each weight matrix $W \in \mathbb{R}^{d \times k}$, a set of orthogonal basis vectors $\{v_1, v_2, \ldots, v_R\}$ is pre-generated, where $R$ is the total budget (sufficient to cover all tasks) and each task is allocated $r$ basis vectors ($r = R / T$, with $T$ denoting the anticipated number of tasks). Basis vectors are generated via Gram-Schmidt orthogonalization or random orthogonal matrices, ensuring $v_i^T v_j = 0, \forall i \neq j$. The low-rank update for task $t$ takes the form $\Delta W_t = \sum_{i \in S_t} \alpha_i v_i u_i^T$, where $S_t$ is the set of basis vector indices allocated to task $t$.
    - **Design Motivation**: Proactive allocation eliminates the greedy "first-come, first-served" strategy. Orthogonality guarantees mathematical non-interference across task updates, providing the foundation for subsequent interference-aware selection.

2. **Perturbation-Aware Basis Vector Selection**:

    - **Function**: Selects, from the candidate basis vectors allocated to the current task, the subset that minimally interferes with existing tasks.
    - **Mechanism**: For each candidate basis vector $v_i$, the change in the loss of existing tasks induced by a small perturbation $\epsilon v_i$ in that direction is computed as an interference metric: $\text{sensitivity}(v_i) = \|\nabla_{v_i} \mathcal{L}_{\text{prev}}\|$. In practice, this sensitivity is estimated by computing gradients on a replay buffer or a small set of samples from previous tasks. The $r$ basis vectors with the lowest sensitivity scores are selected as the subspace for the current task.
    - **Design Motivation**: Even though basis vectors are mathematically orthogonal, their actual influence on the model's loss landscape differs — certain directions align with critical parameters of existing tasks, and perturbing them significantly affects previously acquired knowledge. By explicitly quantifying this sensitivity, PLAN minimizes forgetting without sacrificing the learning capacity for new tasks.

3. **Projection Coefficient Optimization and Merged Inference**:

    - **Function**: Efficiently learns task-specific scaling coefficients along the selected basis vector directions.
    - **Mechanism**: Once the basis vector set $S_t$ for task $t$ is determined, the basis directions are frozen and only the projection coefficients $\{\alpha_i\}_{i \in S_t}$ and the corresponding input projection vectors $\{u_i\}_{i \in S_t}$ are optimized, reducing per-task trainable parameters to $O(r \times (d + k))$. At inference, updates from all tasks are summed and merged directly into the original weights: $W' = W + \sum_t \Delta W_t$, introducing no additional inference latency.
    - **Design Motivation**: Freezing basis vector directions is essential for maintaining orthogonality — allowing basis vectors to update during training could violate the orthogonality guarantees established at allocation time. Optimizing only the coefficients simultaneously preserves theoretical non-interference and retains sufficient learning flexibility.

### Loss & Training

Each task is trained using a standard task-specific loss $\mathcal{L}_t$ (e.g., cross-entropy), with no additional regularization terms required, as orthogonality is structurally guaranteed by the basis vector design. The training pipeline proceeds as: (1) estimate perturbation sensitivity via one forward and backward pass; (2) select basis vectors; (3) perform standard fine-tuning. A replay buffer is not required for training (a small number of prior-task samples may be used solely for sensitivity estimation).

## Key Experimental Results

### Main Results

Performance on standard Class-Incremental Learning and Task-Incremental Learning benchmarks:

| Method | CIFAR-100 (10 tasks) Avg Acc | ImageNet-R (10 tasks) Avg Acc | CUB-200 (10 tasks) Avg Acc | Type |
|------|------|------|------|------|
| Sequential FT | 52.3 | 41.7 | 38.5 | Reference |
| EWC | 68.4 | 55.2 | 52.1 | Regularization |
| L2P | 83.6 | 72.4 | 68.9 | Prompt-based |
| DualPrompt | 85.1 | 73.8 | 71.2 | Prompt-based |
| O-LoRA | 86.3 | 75.6 | 72.8 | LoRA-based |
| InfLoRA | 87.5 | 76.9 | 74.3 | LoRA-based |
| **PLAN** | **89.7** | **79.4** | **77.1** | **Ours** |

### Ablation Study

| Configuration | CIFAR-100 Avg Acc (%) | Description |
|------|---------|------|
| Full PLAN | 89.7 | Complete model |
| w/o Proactive Allocation (random) | 86.8 | Proactive allocation removed; basis vectors selected randomly |
| w/o Perturbation Selection (sequential) | 87.4 | Perturbation selection removed; basis vectors assigned sequentially |
| w/o Orthogonality (standard LoRA) | 84.2 | Orthogonality constraint removed; degenerates to standard LoRA |
| Smaller rank ($r=2$) | 87.1 | Reduced per-task rank |
| Larger rank ($r=8$) | 89.9 | Increased per-task rank |

### Key Findings

- The orthogonality constraint is the most critical design component; removing it (degenerating to standard LoRA) causes a 5.5% performance drop, indicating that inter-task parameter interference is the primary driver of forgetting.
- Perturbation-aware selection outperforms random allocation by 2.9% and sequential allocation by 2.3%, confirming that different basis vector directions impose significantly different levels of interference on existing tasks.
- The choice of rank $r$ affects performance but is not highly sensitive — increasing $r$ from 2 to 8 yields only a 2.8% gain, demonstrating that PLAN achieves high learning efficiency per direction.
- The advantage of PLAN is more pronounced in long-sequence settings (20 tasks), where proactive allocation effectively prevents the "parameter starvation" problem that afflicts later tasks.

## Highlights & Insights

- **The two-stage strategy of "proactive allocation + interference-aware selection" is particularly elegant**: orthogonal bases ensure mathematical non-interference, while perturbation analysis ensures non-interference during actual training — the two layers of protection are complementary.
- **Using perturbation sensitivity as a subspace selection criterion is broadly transferable**: this idea is applicable not only to continual learning, but also to multi-task learning, model merging, federated learning, and other settings requiring coordination of multiple objectives within a shared parameter space.
- **No replay buffer or additional regularization is required**: unlike experience replay or EWC-style methods, PLAN fundamentally prevents forgetting through structured parameter isolation, eliminating the need to store historical data or maintain Fisher information matrices.

## Limitations & Future Work

- The method requires the total number of tasks $T$ to be known (or estimated) in advance for basis vector budget allocation; dynamic expansion strategies are needed for open-ended scenarios with unknown task counts.
- When the number of tasks is large ($T > R / r_{\min}$), the rank allocated per task may be insufficient for effective learning, creating a capacity bottleneck.
- Estimating perturbation sensitivity requires access to data from previous tasks (at least a small number of samples), making the method not directly applicable in strictly data-free CL settings.
- Experiments are conducted primarily on image classification tasks; performance in generative tasks or more complex multimodal CL scenarios remains to be validated.
- Future work could explore dynamic rank allocation, enabling each task to adaptively acquire a variable number of basis vectors according to its complexity.

## Related Work & Insights

- **vs. O-LoRA**: O-LoRA also employs orthogonality constraints to isolate tasks, but determines orthogonal directions dynamically during training — a form of passive allocation. PLAN's advantage lies in proactive planning, ensuring each task receives an equitable and interference-free parameter budget.
- **vs. InfLoRA**: InfLoRA constructs task-specific low-rank spaces via an infinite-width approximation, which is theoretically elegant but computationally expensive. PLAN replaces the costly theoretical derivation with perturbation analysis, achieving superior practical efficiency and performance.
- **vs. L2P/DualPrompt**: These prompt-based methods adapt to new tasks by introducing learnable prompt tokens in the input space, whereas PLAN operates in the parameter space. Parameter-space methods generally offer greater expressive capacity and lower inference overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of "proactive allocation + perturbation-aware selection" is novel, though individual components (orthogonal LoRA, perturbation analysis) build on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Achieves state of the art on multiple standard benchmarks with ablations covering all key design choices; large-scale or real-world scenario validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly articulated; some mathematical derivations could be presented more concisely.
- Value: ⭐⭐⭐⭐ Provides a strong baseline for LoRA-based continual learning; the proactive allocation paradigm offers meaningful inspiration for related areas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ICCV 2025\] Beyond Low-Rank Tuning: Model Prior-Guided Rank Allocation for Effective Transfer in Low-Data and Large-Gap Regimes](beyond_low-rank_tuning_model_prior-guided_rank_allocation_for_effective_transfer.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](../../NeurIPS2025/model_compression/beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[NeurIPS 2025\] Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning](../../NeurIPS2025/model_compression/train_with_perturbation_infer_after_merging_a_two-stage_framework_for_continual_.md)

</div>

<!-- RELATED:END -->
