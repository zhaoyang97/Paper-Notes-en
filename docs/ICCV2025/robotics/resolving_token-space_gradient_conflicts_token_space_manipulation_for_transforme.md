---
title: >-
  [Paper Note] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning
description: >-
  [ICCV 2025][Robotics][Multi-Task Learning] This paper proposes DTME-MTL, a framework that identifies and categorizes gradient conflicts in token space into range-space conflicts and null-space conflicts, and addresses them via Token Modulation (affine transformation) and Token Expansion (task-specific token insertion), respectively, to mitigate negative transfer in Transformer-based multi-task learning with minimal parameter overhead.
tags:
  - ICCV 2025
  - Robotics
  - Multi-Task Learning
  - Gradient Conflict
  - Token Space
  - Transformer
  - Dynamic Network Expansion
date: 2026-05-08
content_hash: 16624bc700a95778
---

# Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning

**Conference**: ICCV 2025
**arXiv**: [2507.07485](https://arxiv.org/abs/2507.07485)
**Code**: [GitHub](https://github.com/wooseong97/DTME-MTL)
**Area**: Robotics
**Keywords**: Multi-Task Learning, Gradient Conflict, Token Space, Transformer, Dynamic Network Expansion

## TL;DR

This paper proposes DTME-MTL, a framework that identifies and categorizes gradient conflicts in token space into range-space conflicts and null-space conflicts, and addresses them via Token Modulation (affine transformation) and Token Expansion (task-specific token insertion), respectively, to mitigate negative transfer in Transformer-based multi-task learning with minimal parameter overhead.

## Background & Motivation

Multi-task learning (MTL) improves generalization and efficiency by jointly learning multiple tasks within a shared network, but divergent task objectives can cause **negative transfer**, where learning one task degrades performance on another. Pretrained Transformer-based MTL architectures (e.g., Task Prompter, MoE) offer strong generalization, yet their **fixed network capacity** and **rigid structure** limit adaptability.

Limitations of existing approaches:

**Multi-task optimization methods** (e.g., PCGrad, Nash-MTL) mitigate negative transfer by adjusting task loss weights or modifying gradients, but are constrained by fixed network designs and cannot expand model capacity.

**Dynamic network architectures** (e.g., Recon) directly convert shared parameters into task-specific ones, which leads to parameter inefficiency, excessive computational overhead, and increased overfitting risk in Transformers.

3. Naively scaling the Transformer backbone cannot leverage pretrained networks and requires training large models from scratch at prohibitive computational cost.

**Core Insight**: Rather than operating directly in parameter space (which is prone to overfitting), it is more effective to operate in token space—treating tokens as learnable parameters, analyzing token space structure via SVD, identifying gradient conflict types, and resolving them adaptively. This approach is more efficient and avoids the overfitting issues associated with parameter-level manipulation.

## Method

### Overall Architecture

DTME-MTL (Dynamic Token Modulation and Expansion) is a plug-and-play framework applicable to any Transformer-based MTL architecture. The core pipeline is as follows:

1. Compute the non-centered covariance matrix of shared tokens at each Transformer layer.
2. Define the range space and null space of the token space via SVD decomposition.
3. Project task gradients onto these two subspaces to detect different types of gradient conflicts.
4. Adaptively apply Token Modulation or Token Expansion based on the conflict type.

### Key Designs

1. **SVD-Based Token Space Definition**:

    - Function: Construct the vector space of shared tokens to provide a mathematical foundation for gradient conflict classification.
    - Mechanism: For the shared token $\mathcal{T}_s^{l,d}$ at layer $d$, compute the non-centered covariance matrix $\widetilde{\mathcal{T}}_s^d = \frac{1}{n}\sum_{l=1}^{n}(\mathcal{T}_s^{l,d})(\mathcal{T}_s^{l,d})^T$, then perform SVD decomposition $\widetilde{\mathcal{T}}_s^d = \mathcal{U}\Lambda\mathcal{V}^T$, and partition eigenvalues into range space $\Lambda_\mathcal{R}$ and null space $\Lambda_\mathcal{N}$ based on a variance ratio threshold $r$.
    - Design Motivation: In practice, eigenvalues of the token covariance matrix are never exactly zero; thus, the paper adopts the SVD variance ratio criterion $r = \frac{\sum_{\lambda \in \Lambda_\mathcal{N}} \lambda}{\sum_{\lambda \in \Lambda_\mathcal{R}} \lambda}$ to automatically determine the boundary between the two subspaces.

2. **Gradient Conflict Classification and Projection**:

    - Function: Decompose task gradients into range-space and null-space components to detect two distinct types of conflicts.
    - Mechanism: For task $\tau_i$, the gradient $g_i = \nabla_{\mathcal{T}_{s,k}} \mathcal{L}_i$ is decomposed into a range-space component $g_{\mathcal{R},i} = (\mathcal{U}_\mathcal{R}\mathcal{U}_\mathcal{R}^T)g_i$ and a null-space component $g_{\mathcal{N},i} = (\mathcal{U}_\mathcal{N}\mathcal{U}_\mathcal{N}^T)g_i$. A range-space conflict occurs when $g_{\mathcal{R},i} \cdot g_{\mathcal{R},j} \leq 0$, and a null-space conflict when $g_{\mathcal{N},i} \cdot g_{\mathcal{N},j} \leq 0$.
    - Design Motivation: In transfer learning settings, pretrained weights confine the model to the same loss landscape basin. Range-space conflicts indicate that the network already possesses relevant representational capacity but requires rotation or scaling; null-space conflicts indicate that new features are needed to increase model capacity.

3. **Token Modulation (Range-Space Conflict Resolution)**:

    - Function: When range-space gradient conflicts are detected, introduce task-specific affine transformation modulators for conflicting task pairs.
    - Mechanism: The modulator $\mathcal{M}$ applies an affine transformation $W \odot \mathcal{T}_{s,i} + b$ to shared tokens, where $W, b \in \mathbb{R}^p$. Proposition 1 provides the theoretical guarantee: when input tokens span the range space, optimizing the modulator reduces range-space gradient conflicts and decreases multi-task loss.
    - Design Motivation: Range-space conflicts imply that existing features already contain relevant information; lightweight channel-wise transformations suffice to provide differentiated feature representations for different tasks.

4. **Token Expansion (Null-Space Conflict Resolution)**:

    - Function: When null-space gradient conflicts are detected, introduce new task-specific tokens to expand the feature space.
    - Mechanism: Task-specific tokens $\{\mathcal{T}_i\}_{i=1}^\mathcal{K}$ are concatenated with shared tokens and fed into the Transformer block, extending the scope of attention computation. Proposition 2 provides the theoretical guarantee: when input tokens span the null space, token expansion alleviates the increase in multi-task loss caused by null-space gradient conflicts.
    - Design Motivation: Null-space conflicts indicate that the network lacks necessary feature dimensions to distinguish task-specific requirements, necessitating additional information channels via token augmentation.

### Loss & Training

- Standard weighted multi-task loss is employed: $\Theta^* = \arg\min_\Theta \sum_{i=1}^\mathcal{K} w_i \mathcal{L}_i(\Theta_s, \Theta_i)$.
- Optimal timing for network expansion: experiments show that expansion at the early stage of training yields the best results, consistent with the design philosophy of leveraging pretrained backbones.
- SVD and gradient conflict computation serve as one-time preprocessing steps with relatively low computational cost (approximately 1 hour for ViT-L).

## Key Experimental Results

### Main Results

**Ablation on NYUD-v2 (4 tasks) and PASCAL-Context (5 tasks):**

| Method | NYUD Semseg mIoU↑ | NYUD Depth RMSE↓ | PASCAL Semseg mIoU↑ | PASCAL Normal mErr↓ | Param. Increase |
|------|-------------------|------------------|--------------------|--------------------|---------|
| Baseline (ST) | 39.35 | 0.6611 | 67.96 | 15.65 | - |
| Baseline (MT) | 34.13 | 0.6732 | 54.47 | 16.22 | - |
| TM | 37.85 | 0.6490 | 64.28 | 15.40 | 0.24% |
| TE | 37.25 | 0.6553 | 60.51 | 15.55 | 0.30% |
| TM+TE | 38.27 | 0.6370 | 66.18 | 15.26 | 0.30% |

**Comparison with SOTA optimization methods on Taskonomy (11 tasks):**

| Method | Multi-Task Performance △m↑ | Note |
|------|-------------|------|
| GD | -7.83% | Gradient descent baseline |
| PCGrad | -8.29% | Gradient projection method |
| Nash-MTL | -5.01% | Nash bargaining method |
| FAMO | -7.87% | Loss balancing method |
| DTME-MTL | **+4.67%** | Only +0.118% parameters |

### Ablation Study

| Configuration | NYUD △m↑ | PASCAL △m↑ | Note |
|------|---------|-----------|------|
| TM+TE (highest-conflict layers) | 0.044 | -1.289 | Conflict-detection-based layer selection (optimal) |
| TM+TE (random layers) | Below optimal | Below optimal | Random layer selection |
| TM+TE (lowest-conflict layers) | Worst | Worst | Reversed selection to validate strategy |

**Applied to SOTA MTL methods (NYUD-v2):**

| Baseline | Semseg mIoU↑ | + DTME-MTL | Depth RMSE↓ | + DTME-MTL |
|---------|-------------|-----------|------------|-----------|
| InvPT | 53.56 | **54.38** | 0.5183 | **0.5020** |
| Taskprompter | 55.30 | **56.36** | 0.5152 | **0.5122** |

### Key Findings

1. Adding only 0.2–0.3% parameters is sufficient to lift multi-task performance from a negative transfer regime to near single-task baseline levels.
2. Conflict-detection-based layer selection significantly outperforms random selection, validating the effectiveness of token-space gradient conflict analysis.
3. Methods such as Recon that operate directly in parameter space suffer from severe overfitting on Transformers, whereas DTME-MTL avoids this issue through token-space manipulation.
4. Expanding the network at the early stage of training yields the best results, consistent with the characteristics of pretrained models.

## Highlights & Insights

- **Strong Theoretical Contributions**: Gradient conflicts are decomposed into range-space and null-space types, each supported by formal mathematical guarantees (Propositions 1 & 2), providing a rigorous theoretical foundation for the proposed operations.
- **Plug-and-Play Design**: The framework integrates seamlessly into existing SOTA multi-task Transformer architectures (InvPT, Taskprompter) without modifying the base architecture.
- **High Efficiency**: Parameter increase is less than 0.3%, inference time increases by approximately 13.4%, and SVD preprocessing takes about 1 hour.
- **Deep Insight**: The work reveals that gradient conflicts in parameter space are not always reliable indicators of negative transfer; token-space analysis provides a more effective perspective.

## Limitations & Future Work

1. The variance ratio threshold $r$ requires manual specification, which may affect the quality of range/null-space partitioning.
2. SVD computation requires a full forward pass over the entire training set, which may pose efficiency challenges for very large-scale datasets.
3. The method currently focuses on dense visual prediction tasks; its generalizability to NLP multi-task settings remains to be verified.
4. The attention computation overhead introduced by Token Expansion scales with the number of tasks, potentially incurring significant cost when the task count is large.

## Related Work & Insights

- The comparison with Recon reveals an important finding: directly converting shared parameters to task-specific parameters in Transformers leads to overfitting, providing an important cautionary signal for future dynamic network architecture design.
- The affine transformation idea underlying Token Modulation shares conceptual connections with parameter-efficient fine-tuning methods such as LoRA, suggesting potential for broader application.
- Elevating SVD analysis from a conventional dimensionality reduction tool to a gradient conflict detection instrument represents a methodologically novel contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ Shifting gradient conflict analysis from parameter space to token space is a genuinely novel perspective, though affine transformations and token concatenation are individually well-established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, Taskonomy with 11 tasks, extensive ablation studies and comparative experiments.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though the dense mathematical notation leaves some room for improved readability.
- Value: ⭐⭐⭐⭐ The plug-and-play design confers high practical utility, and the theoretical analysis offers a new lens for understanding multi-task learning.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[ICLR 2026\] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[ICCV 2025\] TransiT: Transient Transformer for Non-line-of-sight Videography](transit_transient_transformer_for_non-line-of-sight_videography.md)

<!-- RELATED:END -->
