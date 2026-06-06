---
title: >-
  [Paper Note] Energy-Structured Low-Rank Adaptation for Continual Learning
description: >-
  [ICML 2026][Model Compression][continual learning] E2-LoRA shifts the perspective from parameter or input feature spaces to "task-induced output feature drift" $\Delta \mathbf{Y}_t = \Delta \mathbf{W}_t \mathbf{X}_t$. By…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "continual learning"
  - "LoRA"
  - "orthogonal subspace"
  - "energy concentration"
  - "dynamic rank allocation"
date: 2026-05-08
content_hash: 9281bec9f3a93cfb
---

# Energy-Structured Low-Rank Adaptation for Continual Learning

**Conference**: ICML 2026  
**arXiv**: [2605.27482](https://arxiv.org/abs/2605.27482)  
**Code**: Not yet released  
**Area**: Model Compression / LoRA / Continual Learning  
**Keywords**: continual learning, LoRA, orthogonal subspace, energy concentration, dynamic rank allocation  

## TL;DR
E2-LoRA shifts the perspective from parameter or input feature spaces to "task-induced output feature drift" $\Delta \mathbf{Y}_t = \Delta \mathbf{W}_t \mathbf{X}_t$. By performing SVD on this drift, LoRA parameters are rearranged onto an energy-concentrated and rank-ordered basis. This allows for discarding low-energy ranks to recycle capacity for new tasks. Combined with an adaptive rank allocation strategy based on energy retention rates, it achieves SOTA performance across multiple continual learning benchmarks.

## Background & Motivation

**Background**: Mainstream Continual Learning (CL) based on Pre-trained Models (PTMs) typically freezes the backbone and adds Parameter-Efficient Fine-Tuning (PEFT) modules (prompt / adapter / LoRA) for each new task. To reduce inter-task interference, a family of "orthogonalization" methods has emerged: either forcing LoRA parameters of different tasks to be orthogonal (O-LoRA, Param-Param) or performing SVD on historical input features to keep new task parameters orthogonal to principal singular vectors (GPM / DualGPM / InfLoRA, Input-Param).

**Limitations of Prior Work**: The authors observe that both orthogonalization routes suffer from "excessive energy dispersion." In the parameter space, old task knowledge is randomly scattered across the columns of $\mathbf{B}$, causing the occupied subspace to grow linearly and squeeze the capacity available for new tasks. In the input space, PTM features are inherently high-dimensional and diverse; constraining new tasks based on input principal directions overly restricts learning directions, leading to a sharp drop in plasticity (see Figure 1 in the original paper). Fundamentally, these restricted subspaces are "rigid" and cannot be recycled.

**Key Challenge**: The trade-off between stability (preventing forgetting) and plasticity (learning new things) is amplified by the constraint that "orthogonal subspaces are non-recyclable"—once a subspace is allocated to a task, it is locked.

**Goal**: Find a low-dimensional, energy-concentrated, and intrinsically ordered subspace representation for old task knowledge, allowing low-energy directions to be released and recycled for new tasks while keeping the old task outputs nearly unchanged.

**Key Insight**: Instead of focusing on parameters or inputs, one should monitor the intermediate product where LoRA truly impacts the model—the output feature drift $\Delta y_t(x) = \Delta \mathbf{W}_t x$. Empirical and theoretical findings suggest that although the input $\mathbf{X}_t$ may be high-rank, the $\Delta \mathbf{Y}_t$ induced by $\Delta \mathbf{W}_t$ is usually concentrated in very few principal directions (as task semantics are inherently low-dimensional).

**Core Idea**: Perform PCA / SVD on $\Delta \mathbf{Y}_t$ and rewrite the LoRA parameters $\mathbf{B}_t, \mathbf{A}_t$ onto this set of orthogonal bases sorted by descending energy. Thus, the "top-$r$ ranks" provide the optimal low-rank approximation of the task knowledge (theoretically provable, see Prop 3.1/3.2), while remaining low-energy ranks can be safely discarded to release capacity.

## Method

### Overall Architecture
Let $\mathbf{W}_0$ be the pre-trained linear layer weights. When the $t$-th task arrives, a LoRA module $\Delta \mathbf{W}_t = \mathbf{B}_t \mathbf{A}_t$ is allocated. E2-LoRA executes four stages per task: (1) Dynamic rank allocation—pruning old task LoRA modules to $r_k^{(t)}$ ranks based on an energy retention threshold $\rho$, and concatenating released columns to form the initial frozen basis for the new task $\mathbf{B}_t$; (2) Training—optimizing only $\mathbf{A}_t$ with self-distillation to prevent forgetting; (3) Energy-structured transformation—performing SVD on $\Delta \mathbf{Y}_t$ and rotating $(\mathbf{B}_t, \mathbf{A}_t)$ onto the energy basis to be rank-ordered; (4) Classifier head alignment—fine-tuning the classifier using synthetic features generated from class statistics.

### Key Designs

1. **Output-Drift-Induced Orthogonalization**:

    - **Function**: Changes the core question of "what to be orthogonal to" by using a reference space with the fewest constrained dimensions and highest energy concentration.
    - **Mechanism**: Calculates the output drift matrix $\Delta \mathbf{Y}_t = \mathbf{B}_t \mathbf{A}_t \mathbf{X}_t \in \mathbb{R}^{d_\text{out}\times n}$ on a proxy input batch $\mathbf{X}_t$, and performs SVD to obtain $\Delta \mathbf{Y}_t = \mathbf{U}_t \mathbf{\Sigma}_t \mathbf{V}_t^\top$, where columns of $\mathbf{U}_t$ are output directions in descending energy order. The new task $\mathbf{B}_t$ is initialized by concatenating and freezing "low-energy columns" from previous tasks, ensuring the output of $\mathbf{B}_t \mathbf{A}_t$ falls into the zero-energy subspace of old task drifts, equivalent to hard orthogonalization in the output space.
    - **Design Motivation**: In parameter-level orthogonalization (O-LoRA), columns of $\mathbf{B}$ lack energy meaning, making the constraint $\|\mathbf{B}_i^\top \mathbf{B}_t\|_F^2$ "blind." In input-level orthogonalization (DualGPM/InfLoRA), PTM input representations are naturally high-rank/dispersed, meaning constraints shut down most directions for new tasks. $\Delta \mathbf{Y}_t$ reflects which directions the model output actually changed and remains energy-concentrated due to low-dimensional task semantics, making it the ideal space for orthogonalization.

2. **Energy-Structured Transformation + Rank Truncation Optimality**:

    - **Function**: Rewrites $(\mathbf{B}_t, \mathbf{A}_t)$ such that "the first few ranks carry almost all task knowledge" without changing the mathematical equivalence of $\Delta \mathbf{W}_t = \mathbf{B}_t \mathbf{A}_t$.
    - **Mechanism**: After training the current task, perform $\mathbf{B}_t \leftarrow \mathbf{U}_t[:,:r_t]$ and $\mathbf{A}_t \leftarrow (\mathbf{U}_t[:,:r_t])^\top \mathbf{B}_t \mathbf{A}_t$. The paper proves (Prop 3.1) that among all updates with rank $\le r$, $\mathbf{B}_t[:,:r]\mathbf{A}_t[:r,:]$ minimizes the expected output reconstruction error $\mathbb{E}_x \|\mathbf{B}_t[:,:r]\mathbf{A}_t[:r,:]x - \mathbf{B}_t\mathbf{A}_t x\|^2$. (Prop 3.2): After truncating to the top $r$ ranks, the expected error equals the sum of squares of discarded singular values $\sum_{i=r+1}^{d_\text{out}} \sigma_i^2$.
    - **Design Motivation**: This step elevates the observation that "low-energy ranks can be discarded" to an optimal truncation with theoretical guarantees. It is also the key to making "output-drift-induced orthogonalization" executable—only by sorting and concentrating energy can "pruning the tail and yielding columns to the next task" be done without damaging learned knowledge.

3. **Dynamic Rank Allocation**:

    - **Function**: Dynamically schedules the current rank of each task within a shared $d_\text{out}$ capacity pool based on "how much old tasks must retain" and "how much the new task needs."
    - **Mechanism**: For each old task $k$, select the minimum $r_k^{(t)}$ such that the energy retention ratio $\sum_{i=1}^{r_k^{(t)}} \sigma_{k,i}^2 / \sum_{i=1}^{r_k} \sigma_{k,i}^2 \ge \rho$. The new task is set with a minimum rank threshold $r_t^\text{min} = \lceil d_\text{out}/t \rceil$. If pruning by energy is still insufficient for the new task, ranks are uniformly pruned from the tails of all old tasks until the goal is met.
    - **Design Motivation**: Fixed rank allocation either compresses too early (losing accuracy) or occupies too much (collapsing plasticity). Binding pruning to a specific energy threshold $\rho$ allows each task to "occupy space as needed"—simple tasks may take 1-2 ranks while complex ones take more, allowing the entire $d_\text{out}$ capacity to be reused efficiently.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_\text{ce} + \lambda \mathcal{L}_\text{distill}$. $\mathcal{L}_\text{distill}$ is KL self-distillation of old class logits at temperature $T=2$ (where the teacher is the same network with frozen historical LoRA). $\mathcal{L}_\text{ce}$ is the cross-entropy for new classes. The final classifier head alignment uses intra-class statistical synthetic features.

## Key Experimental Results

### Main Results
Using ViT-B/16-IN21K as the backbone, class-incremental continual learning was performed on ImageNet-R / CIFAR-100 / CUB-200 / Cars-196 (Last-Acc = accuracy after all tasks; Inc-Acc = average accuracy across the incremental process). Key baselines from Table 1:

| Method | ImageNet-R Last-Acc | CIFAR-100 Last-Acc | CUB-200 Last-Acc | Cars-196 Last-Acc |
|------|--------------------|--------------------|------------------|-------------------|
| L2P | 66.49 | 82.76 | 62.21 | 38.18 |
| DualPrompt | 68.50 | 85.56 | 66.00 | 40.14 |
| SLCA | 77.00 | 91.53 | 84.71 | 67.73 |
| EASE | 77.75 | 86.54 | 79.90 | 35.46 |
| BiLoRA | 77.95 | 87.46 | — | — |
| SSIAT | 79.38 | 91.35 | 88.75 | 71.02 |
| MOS | 78.10 | 90.53 | 89.91 | 67.76 |
| **E2-LoRA (Ours)** | **New SOTA** | **New SOTA** | **New SOTA** | **New SOTA** |

> On all benchmarks, Last-Acc and Inc-Acc are superior to known SOTA (specific values in the paper; gains are most significant on long, fine-grained sequences like Cars-196).

### Ablation Study

| Configuration | Description |
|------|----------|
| Param-Param Orthogonalization (O-LoRA style) | Subspace is scattered, new task directions are exhausted quickly, drops accuracy on long sequences |
| Input-Param Orthogonalization (InfLoRA style) | PTM input features are high-rank; constraints are too strong, reducing plasticity |
| Output-Drift Orthogonalization (Ours) | Single-task energy concentrates in few directions; old tasks are preserved + sufficient space for new tasks |
| No Energy-Structured Transformation | $\mathbf{B}$ columns are unordered, preventing rank-based pruning; degrades to fixed-rank LoRA |
| No Dynamic Rank Allocation (Fixed $r_t$) | Insufficient capacity for later tasks, leading to reduced plasticity |
| No self-distillation | Logit drift for old classes, increasing forgetting |

### Key Findings
- **Perspective shift is more important than extra regularization**: Simply changing the constraint space from "parameter/input" to "output drift" achieves consistent improvements without adding trainable parameters—indicating the problem lies in the energy distribution of the subspace, not the constraint method itself.
- **Truncation error has a closed-form upper bound** $\sum_{i>r} \sigma_i^2$, meaning the energy retention rate $\rho$ directly corresponds to the "upper bound of output reconstruction error," making parameter tuning interpretable.
- On the most challenging setting, Cars-196 (fine-grained + many tasks), the gap with other SOTA is the widest, highlighting the value of "dynamic capacity recycling" in long task sequences.

## Highlights & Insights
- The observation that "knowledge is hidden in the output rather than parameters," combined with a simple PCA on $\Delta \mathbf{Y}_t$, pushes LoRA-based continual learning to a new level with almost no extra runtime overhead. This "problem reformulation" is often more powerful than adding new regularizers.
- Linking the physically interpretable knob, energy retention $\rho$, directly to rank allocation is very user-friendly—one can derive $\rho$ by knowing "I can tolerate a 5% output error."
- Energy-structured transformation is essentially a "Principal Component Alignment" for LoRA: it suggests that any PEFT method (IA³, AdaLoRA, prefix tuning) in CL scenarios can be "aligned to task-intrinsic low-dimensional bases" before storage.
- Releasing low-energy bases to subsequent tasks essentially treats continual learning as an "auction of a finite capacity pool," which is closer to human learning under capacity constraints than "linearly growing subspaces."

## Limitations & Future Work
- SVD on output drift is performed once per task; larger proxy batches are more accurate but slower. If $d_\text{out}$ is very large (e.g., LLM hidden dim 4096+), SVD costs may become non-negligible.
- The empirical observation "output drift is low-rank" comes from ViT classification; whether it remains concentrated in scenarios with complex output distributions (LLM generation, retrieval, RLHF) is unverified.
- Dynamic rank allocation assumes all tasks share one $d_\text{out}$ pool, but cross-layer rank coupling or cross-module balancing in multi-layer LoRA is not explicitly modeled.
- The threshold $\rho$ is shared across the whole network and all tasks; whether layer-wise or task-adaptive $\rho$ could provide further gains is unexplored.

## Related Work & Insights
- **vs O-LoRA**: O-LoRA performs orthogonalization in parameter space $\|\mathbf{B}_i^\top \mathbf{B}_t\|_F^2$; subspaces grow linearly and are non-recyclable. E2-LoRA "compresses and sorts" each task's LoRA by energy, allowing low-energy parts to be yielded.
- **vs GPM / DualGPM / InfLoRA**: They perform SVD on input features, but PTM input energy is dispersed. E2-LoRA uses the low-rank nature of output drift, offering more precise constraints and less damage to plasticity.
- **vs AdaLoRA**: AdaLoRA also uses energy-based pruning on LoRA but targets parameter efficiency for a single task; E2-LoRA extends this to multi-task settings with a "release to subsequent tasks" recycling mechanism.
- **vs EASE / MOS / TUNA** (adapter route): These rely on task-specific adapters + routing/fusion, where parameters grow linearly with tasks. E2-LoRA reuses ranks within a fixed $d_\text{out}$ capacity, making it more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ Clean perspective shift to "output drift" with accompanying PCA and proof of truncation optimality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across four classification benchmarks, class/domain-incremental settings, and various orthogonalization route ablations.
- Writing Quality: ⭐⭐⭐⭐ Uses Prop 3.1/3.2 to formalize intuition; Figure 1 clearly illustrates energy distribution differences.
- Value: ⭐⭐⭐⭐ Achieves SOTA in the mainstream CL + PEFT route; the method is plug-and-play for any LoRA-CL framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)

</div>

<!-- RELATED:END -->
