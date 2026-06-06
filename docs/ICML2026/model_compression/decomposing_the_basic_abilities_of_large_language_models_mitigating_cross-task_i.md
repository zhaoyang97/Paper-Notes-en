---
title: >-
  [Paper Note] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning
description: >-
  [ICML 2026][Model Compression][Multi-Task Instruction Tuning] Addressing the cross-task gradient conflict in multi-task instruction tuning, this paper proposes Badit. It first decomposes pre-trained weights into a set of…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Multi-Task Instruction Tuning"
  - "Cross-Task Interference"
  - "SVD-LoRA"
  - "Spherical Clustering"
  - "Orthogonal Basic Abilities"
date: 2026-05-08
content_hash: 0138d00e8cad7a66
---

# Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning

**Conference**: ICML 2026  
**arXiv**: [2605.05676](https://arxiv.org/abs/2605.05676)  
**Code**: https://github.com/wangbing1416/BADIT  
**Area**: LLM Efficiency / Multi-Task Fine-Tuning / LoRA / MoE  
**Keywords**: Multi-Task Instruction Tuning, Cross-Task Interference, SVD-LoRA, Spherical Clustering, Orthogonal Basic Abilities

## TL;DR
Addressing the cross-task gradient conflict in multi-task instruction tuning, this paper proposes Badit. It first decomposes pre-trained weights into a set of naturally orthogonal high-singular-value LoRA "basic ability" experts using SVD, and then employs spherical K-means for dynamic orthogonal grouping of rank-1 components during training. This shifts the paradigm from "parameter isolation by task" to "decoupling by basic abilities," achieving an average improvement of 2.68 Rouge over GainLoRA across six LLMs.

## Background & Motivation
**Background**: The robust multi-task capabilities of current LLMs primarily rely on multi-task instruction tuning. However, multi-task training inevitably leads to *cross-task interference*—where different tasks generate gradients in opposite directions on shared parameters, overriding each other. Existing solutions follow two paths: (1) Task-specific neuron/parameter selection (e.g., Leng & Xiong 2025), using gradient attribution to identify and train only the parameter subsets exclusive to each task; (2) MoE-style expert isolation (LoRAMoE, OLoRA, etc.), assigning independent LoRA experts to each task under orthogonal constraints.

**Limitations of Prior Work**: Empirical analysis on 15 tasks of SuperNI (Fig. 1, Fig. 2) reveals that both categories of methods fail: whether it is "task-activated neurons" or "routed experts in MoE," the **vast majority are simultaneously activated by multiple tasks**—with some neurons even shared by all 15 tasks. In other words, "task-exclusivity" is often an illusion, and cross-task interference is never truly eliminated.

**Key Challenge**: Using "tasks" as the granularity for isolation is inherently flawed—tasks are surface labels, not the functional segmentation units actually existing at the parameter level. Forcing groups by task results in parameter re-sharing due to overlapping capabilities between tasks.

**Goal**: (1) Identify more fundamental, truly separable capability units than "tasks"; (2) Design an MoE architecture that maintains the orthogonality of these units during training.

**Key Insight**: Fig. 1 also reveals an inverse signal—"certain neurons/parameters are consistently co-activated by multiple tasks." These co-activation sets recur, forming a few "basis sets." The authors propose a metaphor: LLMs encode several **orthogonal basic abilities**, and each task is a linear combination of these abilities. Therefore, parameters should be isolated by "basic ability" rather than by "task."

**Core Idea**: Utilize SVD to decompose pre-trained weights into multiple naturally orthogonal LoRA experts (each corresponding to a basic ability). During training, use spherical clustering to continuously regroup rank-1 components, forcing gradient directions of different experts to remain orthogonal.

## Method

### Overall Architecture
Badit transforms each LLM weight matrix $\mathbf{W}^0 \in \mathbb{R}^{m \times n}$ into the form $\mathbf{W}^0 = \sum_{k=1}^{K}\alpha_k \mathbf{A}_k\mathbf{B}_k + \widehat{\mathbf{W}}$. Here, $K$ LoRA experts cover the top $rK$ largest singular values (each with rank=$r$), while the residual $\widehat{\mathbf{W}}$ contains the remaining smaller singular values. The former are interpreted as "basic ability" experts with learnable routing coefficients $\alpha_k$; the latter remains frozen throughout fine-tuning. The process consists of two stages: **BAD** provides the initial orthogonal decomposition, and **DOG** maintains orthogonality during training. The final routing weights allow the model to adaptively mix these basic abilities per task.

### Key Designs

1.  **Basic Ability Decomposition (BAD)**:
    *   **Function**: Performs SVD on each pre-trained weight $\mathbf{W}^0$, taking the top $rK$ singular values/vectors to construct $K$ rank-$r$ LoRA experts as the initialization for basic abilities.
    *   **Mechanism**: $\mathbf{U}_{[:rK]}\boldsymbol{\Sigma}_{[:rK]}\mathbf{V}_{[:rK]}^\top$ is partitioned into $K$ segments by columns. The $k$-th segment constructs $\mathbf{A}_k = \mathbf{U}_{[r(k-1):rk]}\,\mathrm{diag}(\sqrt{\boldsymbol{\Sigma}_{[r(k-1):rk]}})$ and a symmetric $\mathbf{B}_k$; remaining low-singular values form the frozen residual. The paper proves (Appendix A) that these $K$ LoRA experts are **naturally orthogonal at initialization**—as SVD left/right singular vectors inherently occupy different subspaces. Finally, a learnable router $\alpha_k$ (initialized to 1) is added to each expert for MoE compatibility.
    *   **Design Motivation**: Unlike traditional LoRA with zero initialization or MoE with random expert assignment, BAD's key insight is that "basic abilities are not assigned post-hoc but exist as directions within pre-trained weights." SVD provides an training-free dictionary of orthogonal basic abilities, shifting the burden of "expert differentiation" from training to initialization.

2.  **Dynamically Orthogonal Grouping (DOG)**:
    *   **Function**: Since gradient updates destroy expert orthogonality during training, DOG regroups $rK$ rank-1 components every few steps to restore orthogonality of the $K$ LoRA experts' gradients.
    *   **Mechanism**: Concatenated gradients $\mathbf{g}_i$ of each rank-1 component $[\mathbf{a}_i;\mathbf{b}_i^\top]$ are normalized to $\widehat{\mathbf{g}}_i$. The objective is to find a binary assignment matrix $\boldsymbol{\Pi}\in\{0,1\}^{rK\times K}$ (where each new expert receives exactly $r$ components) to satisfy: "gradients within the same expert are similar, while gradients across experts are orthogonal": $\max_{\boldsymbol{\Pi}}\sum_k \|\sum_i \pi_{i,k}\widehat{\mathbf{g}}_i\|^2$. This is solved iteratively: (1) Spherical K-means provides initial assignments; (2) Cluster centroids $\mathbf{c}_k^{(\tau)}=\sum_i \pi_{i,k}^{(\tau)}\widehat{\mathbf{g}}_i$ are computed, and their matrix $\mathbf{C}^{(\tau)}=\mathbf{U}_c\boldsymbol{\Sigma}_c\mathbf{V}_c^\top$ undergoes SVD to yield $\mathbf{Q}^{(\tau)}=\mathbf{U}_c\mathbf{V}_c^\top$ as forced orthogonal target directions; (3) $\boldsymbol{\Pi}$ is updated via $\langle \widehat{\mathbf{g}}_i, \mathbf{q}_k^{(\tau)}\rangle$ solving an integer assignment problem constrained by $\sum_i \pi_{i,k}=r$. Appendix C proves that MoE output remains mathematically invariant before and after regrouping, preventing model disturbance.
    *   **Design Motivation**: SVD orthogonality only holds at $t=0$ and drifts quickly (Fig. 4 shows LoRAMoE's inter-expert angles deviating from 90°). DOG avoids the pitfalls of direct orthogonal penalties (which disrupt loss geometry) by converting the constraint into a discrete optimization of "regrouping by direction," maintaining end-to-end stability while locking inter-expert angles near 90°.

### Loss & Training
Standard SFT loss $\mathcal{L}(\mathcal{F}(\mathbf{x};\boldsymbol{\theta}), y)$ is used without additional orthogonal regularization. The number of experts $K=8$ and rank $r$ are aligned with LoRAMoE. DOG is triggered periodically during training (clustering and integer optimization are performed on CPU). Two evaluation paradigms are used: mixed training (15 tasks simultaneously) and sequential training (average of 5 task orders, focusing on forgetting rates).

## Key Experimental Results

### Main Results
Comparing five baselines across 15 SuperNI tasks and 6 LLMs (Qwen3-8B/4B, Llama3-8B/3B, Gemma2-9B/2B). Results for Qwen3-8B and Llama3-8B in mixed and sequential settings:

| Model / Setup | Method | Mixed Rouge↑ | Seq Rouge↑ | Seq Forget Rate↓ | Seq Backward↑ |
|:---|:---|:---|:---|:---|:---|
| Qwen3-8B | LoRA | 54.22 | 47.08 | 9.21 | -8.11 |
| Qwen3-8B | LoRAMoE | 54.64 | 48.07 | 8.47 | -6.23 |
| Qwen3-8B | GainLoRA (SOTA) | 54.33 | 48.44 | 8.96 | -6.42 |
| Qwen3-8B | **Badit** | **55.87** | **50.86** | **6.95** | **-5.43** |
| Llama3-8B | LoRA | 52.58 | 44.88 | 12.41 | -10.23 |
| Llama3-8B | GainLoRA | 52.71 | 45.04 | 12.70 | -10.94 |
| Llama3-8B | **Badit** | **54.75** | **48.83** | **8.57** | **-3.49** |

Badit outperforms GainLoRA by an average of **2.68 Rouge** across 6 LLMs, achieving the best for forgetting rate and backward transfer. On Llama3-8B, the forget rate dropped from 12+ to 8.57, and backward transfer improved from -10 to -3.49, indicating a significant reduction in catastrophic forgetting in sequential scenarios.

### Ablation Study
The contribution of BAD and DOG (Δ represents the total decline relative to full Badit):

| Model | Config | Seq Rouge | Seq Forget | Mixed Rouge | Δ |
|:---|:---|:---|:---|:---|:---|
| Qwen3 | Badit (full) | 50.86 | 6.95 | 55.87 | - |
| Qwen3 | w/o BAD | 50.07 | 7.34 | 55.02 | 2.03 |
| Qwen3 | w/o DOG | 49.70 | 8.20 | 54.85 | 3.43 |
| Qwen3 | w/o BAD & DOG | 48.07 | 8.47 | 54.64 | 5.54 |
| Llama3 | Badit | 48.83 | 8.57 | 54.75 | - |
| Llama3 | w/o BAD | 47.92 | 9.30 | 54.02 | 2.37 |
| Llama3 | w/o DOG | 47.33 | 9.74 | 53.74 | 3.68 |

### Key Findings
*   **DOG is more critical than BAD**: Removing DOG (w/o DOG) results in a larger performance drop than removing SVD initialization (w/o BAD), suggesting that maintaining orthogonality throughout training is more important than a good initialization.
*   **Basic Ability Hypothesis Validated**: As shown in Fig. 4, Badit's inter-expert gradient angles stay consistently near 90°, while intra-expert angles are around 60° (~20° lower than LoRAMoE), proving it effectively achieves "inter-expert orthogonality and intra-expert consistency."
*   **Acceptable Overhead**: Table 3 shows Badit's total training time is approximately $1.22\times$ that of LoRAMoE. The extra cost primarily stems from DOG’s spherical K-means and integer optimization on the CPU; BAD may actually accelerate convergence.

## Highlights & Insights
*   **Shifting Isolation Granularity from "Task" to "Ability"**: This is a conceptual leap. While traditional MoE assumes one expert per task, the authors argue that tasks are linear combinations of abilities, which are the fundamental atoms. This explains why task-level isolation often fails—tasks inherently overlap in the ability dimension.
*   **SVD as a "Training-Free Expert Differentiator"**: The challenge of "expert differentiation," which usually requires specialized losses, is offloaded to a pure linear algebra operation (SVD). This is efficient, provably orthogonal, and steps beyond the PiSSA paradigm.
*   **The "Cluster-then-Orthogonalize" Strategy**: Directly penalizing gradients for non-orthogonality often destabilizes the loss landscape. DOG's approach—grouping similar directions and then orthogonalizing them via discrete assignment—is a robust "soft-grouping + hard-orthogonality" workflow that could be extended to continual learning or multi-domain adaptation.
*   **Invariance via Discrete Assignment**: Appendix C proves the MoE output remains invariant after regrouping, which is crucial. It allows "representation surgery" while maintaining end-to-end equivalence.

## Limitations & Future Work
*   DOG introduces a 1.22× training cost, and its scalability to larger batches or more experts ($K$) remains to be tested in industrial settings.
*   Spherical K-means and integer assignment on the CPU may become a bottleneck for GPU-bound training; more efficient GPU implementations are needed.
*   The assumption "Basic Ability = Major SVD Singular Directions" is mathematically elegant, but whether these directions correspond to interpretable semantics (e.g., arithmetic, reasoning) lacks confirmation via probing experiments.
*   The residual $\widehat{\mathbf{W}}$ is fully frozen, potentially discarding long-tail task information; lightweight adaptation for the residual could be explored.

## Related Work & Insights
*   **vs LoRAMoE / OLoRA**: These methods use randomly initialized LoRA experts and task-based routing, relying on training penalties for orthogonality. Badit extracts experts from SVD and maintains orthogonality via regrouping.
*   **vs PiSSA**: PiSSA is a single-expert SVD-LoRA focusing on major singular values. Badit segments the SVD spectrum into $K$ experts, moving from single-expert acceleration to multi-expert decoupling.
*   **vs GainLoRA**: GainLoRA remains in the task-to-parameter assignment paradigm. The authors refute the "task-separable" assumption of GainLoRA with empirical evidence, leading to a consistent 2.68 Rouge gain.

## Rating
*   Novelty: ⭐⭐⭐⭐ Re-conceptualizing isolation granularity is insightful; the combination of SVD and dynamic clustering is novel, though it builds on existing blocks like PiSSA and OLoRA.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 LLMs, 15 tasks, and 2 training paradigms with detailed auxiliary analysis of gradient angles and overhead.
*   Writing Quality: ⭐⭐⭐⭐ The motivation is clearly derived from counter-intuitive findings in Fig. 1/2.
*   Value: ⭐⭐⭐⭐ Multi-task tuning is central to LLM post-training. An average 2.68 Rouge improvement is attractive for industrial pipelines; the 22% overhead is a manageable trade-off.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](../../ACL2026/model_compression/tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ICML 2026\] Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models](jailbreak_to_protect_buffering_and_reinforcing_via_temporary_jailbreaking_for_sa.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](../../CVPR2026/model_compression/frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)

</div>

<!-- RELATED:END -->
