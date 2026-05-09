---
title: >-
  [Paper Note] Continuous Subspace Optimization for Continual Learning (CoSO)
description: >-
  [NeurIPS 2025][Self-Supervised Learning][continual learning] This paper proposes CoSO, a framework that dynamically derives continuous subspaces from per-step gradient SVD (rather than LoRA's fixed subspace), combined with orthogonal projection onto historical task subspaces to prevent interference and Frequent Directions for efficient gradient information aggregation. CoSO achieves 78.19% final accuracy on ImageNet-R with 20 tasks, surpassing the best baseline by 2.77 percentage points.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - continual learning
  - catastrophic forgetting
  - Dynamic Subspace
  - Orthogonal Projection
  - Frequent Directions
  - LoRA
  - GaLore
date: 2026-05-08
content_hash: a52f6c778aee945e
---

# Continuous Subspace Optimization for Continual Learning (CoSO)

**Conference**: NeurIPS 2025  
**arXiv**: [2505.11816](https://arxiv.org/abs/2505.11816)  
**Authors**: Quan Cheng, Yuanyu Wan, Lingyu Wu, Chenping Hou, Lijun Zhang (Nanjing University, Zhejiang University, National University of Defense Technology)  
**Area**: Continual Learning / Parameter-Efficient Fine-Tuning  
**Keywords**: continual learning, catastrophic forgetting, Dynamic Subspace, Orthogonal Projection, Frequent Directions, LoRA, GaLore

## TL;DR

This paper proposes CoSO, a framework that dynamically derives continuous subspaces from per-step gradient SVD (rather than LoRA's fixed subspace), combined with orthogonal projection onto historical task subspaces to prevent interference and Frequent Directions for efficient gradient information aggregation. CoSO achieves 78.19% final accuracy on ImageNet-R with 20 tasks, surpassing the best baseline by 2.77 percentage points.

## Background & Motivation

**Background**: The core challenge in continual learning is catastrophic forgetting—the sharp decline in performance on previously learned tasks when a model acquires new ones. PEFT methods based on pre-trained ViTs (e.g., LoRA) have become mainstream, constraining parameter updates to fixed low-rank subspaces to mitigate inter-task interference.

**Limitations of Prior Work**: (a) LoRA's fixed-rank constraint limits learning capacity and underperforms full-rank fine-tuning; (b) InfLoRA mitigates interference within a pre-specified subspace, and SD-LoRA decouples magnitude from direction, but both confine weight updates to a single low-rank subspace; (c) performance degrades noticeably on long task sequences, as fixed subspaces lack the expressiveness to accommodate continuously shifting gradient structures.

**Key Challenge**: How can high learning capacity (flexibility) be maintained while effectively mitigating catastrophic forgetting (stability)? There is a fundamental tension between the "safety" of fixed subspaces and the "flexibility" of dynamic subspaces.

**Key Insight**: Inspired by GaLore (gradient low-rank projection for offline learning), the subspace is dynamically derived from the SVD of the gradient itself rather than fixed in advance, while orthogonal bases of historical task subspaces are maintained to decouple update directions across tasks.

**Core Idea**: Every $K$ steps, a projection matrix is derived from the SVD of the current gradient for low-rank optimization, forced into the orthogonal complement of historical task subspaces, and Frequent Directions is used to efficiently maintain historical information.

## Method

### Overall Architecture

For each new task $\tau$, the following procedure is executed at each training step:

1. Compute the current gradient $G_{\tau,t}$
2. Orthogonal projection: $G'_{\tau,t} = G_{\tau,t} - \mathcal{M}_{\tau-1}\mathcal{M}_{\tau-1}^T G_{\tau,t}$ (removing components aligned with the historical subspace)
3. Truncated SVD: $P_{\tau,t} = U[:, :r_1]$ (obtaining the current low-rank projection matrix)
4. Forward projection → Adam optimization → back-projection for parameter update
5. Simultaneously aggregate gradient information into sketch matrix $S_{\tau,t}$ incrementally via Frequent Directions
6. After the task ends, perform SVD on $S_{\tau,T}$ to extract principal directions, appended to the historical orthogonal basis $\mathcal{M}_\tau$

### Key Designs

**Design 1: Continuous Subspace Optimization**

- **Function**: Dynamically derive low-rank projection matrices to replace LoRA's fixed matrices
- **Mechanism**: Every $K$ steps, perform truncated SVD on the current orthogonalized gradient to obtain a rank-$r_1$ projection matrix $P_{\tau,t}$, then optimize within this subspace using Adam. Unlike LoRA, the subspace evolves continuously with the gradient, allowing the final learned weights to be full-rank
- **Procedure**: $R_{\tau,t} = P_{\tau,t}^T G'_{\tau,t}$ (forward projection) → $N_{\tau,t} = \text{Adam}(R_{\tau,t})$ (low-dimensional optimization) → $\tilde{G}_{\tau,t} = P_{\tau,t} N_{\tau,t}$ (back-projection) → $W_{\tau,t} = W_{\tau,t-1} - \eta \tilde{G}_{\tau,t}$
- **Design Motivation**: Fixed subspaces cannot adapt to changes in gradient direction during training; optimizing across multiple continuous subspaces breaks the learning capacity ceiling imposed by low-rank constraints
- **Memory Advantage**: Compared to LoRA-type methods, memory requirements are reduced from $mn + 3mr_1 + 3nr_1$ to $mn + mr_1 + 2nr_1$

**Design 2: Historical Task Orthogonal Projection**

- **Function**: Ensure parameter updates for new tasks do not interfere with old tasks
- **Mechanism**: An orthogonal basis matrix $\mathcal{M}_{\tau-1}$ is maintained, consolidating the gradient subspaces of all historical tasks. At each step, the current gradient is projected onto the orthogonal complement: $G'_{\tau,t} = G_{\tau,t} - \mathcal{M}_{\tau-1}\mathcal{M}_{\tau-1}^T G_{\tau,t}$
- **Principle**: Since $P_{\tau,t}$ is derived from $G'_{\tau,t}$, all parameter updates lie in the null space of the historical subspace, producing no effect on the linear layer outputs of prior tasks
- **Design Motivation**: Provides principled protection against forgetting. Ablation studies show that removing orthogonal projection results in a drop of 8.52 percentage points in final accuracy on the 20-task benchmark

**Design 3: Frequent Directions Gradient Aggregation**

- **Function**: Efficiently maintain task-specific gradient covariance information
- **Mechanism**: The FD algorithm incrementally aggregates gradient information across all training steps at $O(mnr_2T)$ complexity (versus $O(m^2nT)$ for direct covariance computation), producing a sketch matrix $S_{\tau,T}$
- **Procedure**: First apply rank-$r_2$ truncated SVD to the gradient to obtain $Q_{\tau,t}$, then incrementally update $S_{\tau,t} = \text{FD}([S_{\tau,t-1}, Q_{\tau,t}])$
- **At task end**: Perform SVD on $S_{\tau,T}$, select $k$ principal directions according to $\sum_{i=1}^k \sigma_i^2 / \sum_{j=1}^{r_2} \sigma_j^2 \leq \epsilon_{th}$, and append to $\mathcal{M}_\tau = [\mathcal{M}_{\tau-1}, U_\tau[:, :k]]$
- **Theoretical Guarantee**: Proposition 1 establishes an upper bound on the approximation error, which becomes negligible when $r_2$ exceeds the intrinsic rank of the gradient

### Loss & Training

- **Loss Function**: Standard cross-entropy with temperature parameter set to 3 to prevent overfitting
- **Backbone**: ViT-B/16 (pre-trained on ImageNet-21K, fine-tuned on ImageNet-1K); DINO self-supervised pre-trained ViT-B/16 is also evaluated
- **Optimization Scope**: Only the output projection layer of Multi-Head Attention is optimized (not QKV transformations)
- **Optimizer**: Adam ($\beta_1=0.9, \beta_2=0.999$)
- **Key Hyperparameters**: $r_1$ (projection rank), $r_2$ (FD rank, set $> r_1$), $K$ (SVD update interval), $\epsilon_{th}$ (information retention threshold, uniformly 0.98)

| Hyperparameter | CIFAR100 | ImageNet-R | DomainNet |
|:---:|:---:|:---:|:---:|
| $r_1$ | 15 | 50 | 70 |
| $r_2$ | 100 | 120 | 160 |
| $K$ | 1 | 1 | 20 |
| Epochs | 20 | 40 | 5 |

## Key Experimental Results

### Main Results

Comparison with 6 SOTA methods on ImageNet-R (L2P, DualPrompt, CODA-Prompt, InfLoRA, VPT-NSP², SD-LoRA), averaged over 3 independent runs with standard deviation:

| Setting | CoSO Final Acc | Best Baseline | Gain |
|:---:|:---:|:---:|:---:|
| 5 Tasks | — | — | +2.38% |
| 10 Tasks | — | — | +3.23% |
| 20 Tasks | **78.19%** | 75.42% (SD-LoRA) | **+2.77%** |

- Average accuracy on 20 tasks: CoSO 83.69% vs. best baseline 81.32% (+2.37%)
- CoSO's advantage grows with the number of tasks, demonstrating robustness on long-sequence challenging scenarios
- Training curves show CoSO maintains the best performance at intermediate and final stages, with noticeably slower accuracy decay than competing methods

### CIFAR100 and DomainNet

- DomainNet (5 Tasks): CoSO final accuracy exceeds the best baseline by 1.75%, average accuracy by 1.37%
- CIFAR100 (10 Tasks): CoSO achieves the best performance as well

### Ablation Study (ImageNet-R)

| Variant | 5 Tasks Drop | 10 Tasks Drop | 20 Tasks Drop |
|:---:|:---:|:---:|:---:|
| w/o Orth (remove orthogonal projection) | — | — | **−8.52%** |
| w/o FD (replace FD aggregation with final subspace only) | −1.65% | −1.89% | −1.59% |

- Orthogonal projection is the core contribution; its removal causes a sharp performance drop, indicating that task interference is the primary cause of catastrophic forgetting
- FD aggregation is also indispensable; aggregating gradients throughout training captures richer task information compared to using only the terminal subspace

### Computation and Memory Overhead (ImageNet-R 10 Tasks)

| Method | GFLOPs | Memory (GB) |
|:---:|:---:|:---:|
| L2P / DualPrompt / CODA-P | 70.24 | 12.90–12.97 |
| InfLoRA | 35.12 | 13.44 |
| SD-LoRA | 35.12 | 15.62 |
| **CoSO** | **35.12** | **13.61** |

- Computation is half that of prompt-based methods (no second forward pass required)
- Memory is comparable to InfLoRA, far below SD-LoRA

### DINO Self-Supervised Backbone

On DINO pre-trained ViT-B/16 (ImageNet-R 10 Tasks), CoSO also outperforms all baselines by a significant margin, confirming the generality of the method.

## Highlights & Insights

- **Organic combination of dynamic subspace and orthogonal constraint**: Dynamic subspace ensures learning capacity (expressiveness), while orthogonality ensures anti-forgetting (safety); the two are complementary rather than conflicting
- **Elegant application of Frequent Directions**: Importing a streaming matrix sketching algorithm from data analysis into continual learning elegantly resolves the key question of how to efficiently summarize the full gradient information of a task
- **Theoretically grounded task isolation mechanism**: Orthogonal projection has a clear geometric interpretation—"reserving" mutually non-interfering subspaces in parameter space for each task

## Limitations & Future Work

- Evaluation is limited to visual classification; more complex scenarios such as multimodal learning, domain-incremental learning, and NLP have not been explored. The authors explicitly identify extension to multimodal settings as an open problem
- SVD computation introduces non-trivial overhead on high-dimensional parameter matrices, even when performed only every $K$ steps
- The orthogonal basis $\mathcal{M}_\tau$ grows continuously with the number of tasks, potentially progressively squeezing the optimization space available for new tasks
- Four hyperparameters ($r_1, r_2, K, \epsilon_{th}$) require dataset-specific tuning; while $\epsilon_{th}$ is uniformly set to 0.98, the optimal projection rank varies substantially across datasets (15 vs. 70)

## Key Findings

- **Orthogonal projection is the core**: Removing it causes an 8.52 percentage point drop, fully releasing task interference
- **Advantage grows with more tasks**: +2.38% at 5 tasks → +2.77% at 20 tasks, demonstrating increasing superiority on longer sequences
- **Frequent Directions contributes a stable 1.5–2%**: Aggregating all intermediate gradients consistently outperforms using only the final-step gradient
- **Computational overhead comparable to InfLoRA**: Identical GFLOPs, memory only 0.17 GB higher

### Main Results

| Dataset | Tasks | CoSO Final Acc | Best Baseline | Gain |
|--------|------|---------------|------------|------|
| ImageNet-R | 5 | **82.10%** | VPT-NSP² 79.72% | +2.38 |
| ImageNet-R | 10 | **81.10%** | 77.87% | +3.23 |
| ImageNet-R | 20 | **78.19%** | 75.42% | +2.77 |
| CIFAR100 | 10 | **88.77%** | 88.09% | +0.68 |
| DomainNet | 5 | **74.27%** | 72.52% | +1.75 |

### Ablation Study (ImageNet-R, 20 Tasks)

| Configuration | Final Acc | Avg Acc |
|------|----------|---------|
| CoSO (full) | **78.27%** | **83.62%** |
| w/o orthogonal projection | 69.75% (−8.52) | 78.88% |
| w/o Frequent Directions | 76.68% (−1.59) | 82.41% |

## Related Work & Insights

- **vs. LoRA/InfLoRA/SD-LoRA**: Fixed subspace → dynamic subspace, representing a qualitative improvement in learning capacity
- **vs. OGD (orthogonal gradient descent)**: CoSO performs orthogonal projection within low-rank subspaces, which is more efficient; the key improvement lies in using FD to aggregate gradient information across the entire training process (rather than a single checkpoint) to estimate the task subspace
- **Inspiration**: The paradigm of orthogonality combined with dynamic subspaces may generalize to continual learning and multi-task adaptation in LLMs

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of dynamic subspace, orthogonal projection, and FD is novel with clear geometric intuition
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets, varying task counts, and detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐ Method derivation is clear with complete algorithmic pseudocode
- **Value**: ⭐⭐⭐⭐ Makes an important contribution to the continual learning community, especially for long task sequence scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](m-grpo_stabilizing_self-supervised_reinforcement_learning_for_large_language_mod.md)
- [\[ICLR 2026\] PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)
- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](../../ICLR2026/self_supervised/fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)

</div>

<!-- RELATED:END -->
