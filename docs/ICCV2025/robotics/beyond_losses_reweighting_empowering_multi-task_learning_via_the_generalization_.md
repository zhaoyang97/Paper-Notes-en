---
title: >-
  [Paper Note] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective
description: >-
  [ICCV 2025][Robotics][Multi-task learning] From a generalization perspective, this paper introduces Sharpness-Aware Minimization (SAM) into multi-task learning (MTL). By decomposing each task's SAM gradient into a "low-loss direction" and a "flat direction" and aggregating them separately, the method reduces gradient conflicts and guides the model toward a jointly flat low-loss region shared across tasks.
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Multi-task learning"
  - "flat minima"
  - "sharpness-aware minimization"
  - "gradient conflict"
  - "generalization"
date: 2026-05-08
content_hash: 0642482e04460059
---

# Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective

**Conference**: ICCV 2025
**arXiv**: [2211.13723](https://arxiv.org/abs/2211.13723)  
**Code**: None  
**Area**: Multi-Task Learning / Optimization
**Keywords**: Multi-task learning, flat minima, sharpness-aware minimization, gradient conflict, generalization

## TL;DR

From a generalization perspective, this paper introduces Sharpness-Aware Minimization (SAM) into multi-task learning (MTL). By decomposing each task's SAM gradient into a "low-loss direction" and a "flat direction" and aggregating them separately, the method reduces gradient conflicts and guides the model toward a jointly flat low-loss region shared across tasks.

## Background & Motivation

MTL employs a shared backbone to optimize multiple objectives simultaneously, reducing computational cost and facilitating cross-task knowledge sharing. Its central challenge, however, is **gradient conflict**—gradients from different tasks may oppose each other in direction or magnitude, causing some tasks to be under-optimized (negative transfer).

Existing gradient manipulation methods (e.g., PCGrad, CAGrad, IMTL) focus on finding a common descent direction that reduces all task losses simultaneously, yet they share a common blind spot: **they target only empirical error minimization and ignore the geometry of the loss landscape**. In deep learning, empirical risk minimization (ERM) tends to converge to sharp minima with poor generalization. This issue is further amplified in MTL:

- Sharp minima for different tasks may reside at **different locations**, making simultaneous generalization more difficult.
- Gradient conflicts arise not only between tasks (inter-conflict), but also—once flatness is introduced as an objective—between the loss direction and the flat direction within the same task (intra-conflict).

**Core Idea**: Since flat minima improve generalization in single-task settings, seeking a **jointly flat low-loss region** for all tasks in MTL should simultaneously reduce generalization error and gradient conflict.

## Method

### Overall Architecture

The paper proposes a model-agnostic framework (denoted F-Method) that can be composed with any existing gradient-based MTL method. The core mechanism is to define a worst-case perturbed loss for each task, then decompose the resulting gradients into two components that are handled separately.

### Key Designs

1. **SAM Objective for MTL**: For each task $i$, a bilevel maximization problem is defined—seeking the worst-case loss within a neighborhood of the shared parameters $\theta_{sh}$ and task-specific parameters $\theta_{ns}^i$:

$$\max_{\|\epsilon_{sh}\|_2 \leq \rho_{sh}} \left[\max_{\|\epsilon_{ns}^i\|_2 \leq \rho_{ns}} \mathcal{L}_S^i(\theta_{sh} + \epsilon_{sh}, \theta_{ns}^i + \epsilon_{ns}^i)\right]_{i=1}^m$$

The approximate SAM gradient is obtained via first-order Taylor expansion and relaxation.

2. **Gradient Decomposition Strategy**: This is the paper's most critical design. For the shared-parameter gradient of each task $i$:

    - **Loss gradient** $\boldsymbol{g}_{sh}^{i,loss}$: the standard gradient at the current parameters, pointing toward lower loss.
    - **SAM gradient** $\boldsymbol{g}_{sh}^{i,SAM}$: the gradient evaluated at the perturbed parameters.
    - **Flat gradient** $\boldsymbol{g}_{sh}^{i,flat} = \boldsymbol{g}_{sh}^{i,SAM} - \boldsymbol{g}_{sh}^{i,loss}$: points toward flatter regions.

   Loss gradients and flat gradients across all tasks are then aggregated separately:

   $\boldsymbol{g}_{sh}^{loss} = \text{gradient\_aggregate}(\boldsymbol{g}_{sh}^{1,loss}, ..., \boldsymbol{g}_{sh}^{m,loss})$

   $\boldsymbol{g}_{sh}^{flat} = \text{gradient\_aggregate}(\boldsymbol{g}_{sh}^{1,flat}, ..., \boldsymbol{g}_{sh}^{m,flat})$

   Final update: $\boldsymbol{g}_{sh}^{SAM} = \boldsymbol{g}_{sh}^{loss} + \boldsymbol{g}_{sh}^{flat}$

   The rationale is that gradients of the same type (loss-to-loss, flat-to-flat) are more likely to be mutually consistent, so separate aggregation reduces conflict.

3. **Update for Non-Shared Parameters**: Each task-specific head is updated directly with standard SAM, as no cross-task conflict exists there.

4. **Theoretical Support (Theorem 1)**: The paper proves that the generalization error of each task is upper-bounded by the worst-case perturbed loss plus a term related to parameter norms, providing theoretical justification for jointly minimizing loss and sharpness.

### Loss & Training

Each iteration requires two forward passes (original parameters + perturbed parameters) and one gradient aggregation step, incurring roughly $2\times$ the computational cost of the base method. The framework is insensitive to the hyperparameters (perturbation radii $\rho_{sh}$, $\rho_{ns}$).

## Key Experimental Results

### Main Results

Results on Multi-MNIST (3 variants):

| Method | MultiFashion Avg | MultiMNIST Avg | MultiFashion+MNIST Avg |
|--------|:---:|:---:|:---:|
| STL | 86.65 | 94.74 | 93.91 |
| MGDA | 86.27 | 95.05 | 92.72 |
| **F-MGDA** | **87.73** | **95.68** | **93.28** |
| PCGrad | 86.57 | 95.06 | 92.78 |
| **F-PCGrad** | **87.76** | **95.92** | **93.50** |
| CAGrad | 86.51 | 95.01 | 92.68 |
| **F-CAGrad** | **87.82** | **95.95** | **93.54** |

Relative improvement $\Delta m\%$ on NYUv2 three-task benchmark (semantic segmentation + depth estimation + surface normals):

| Method | mIoU↑ | Abs Err↓ | Mean Angle↓ | $\Delta m\%$↓ |
|--------|:---:|:---:|:---:|:---:|
| CAGrad | 39.79 | 0.5486 | 26.31 | +0.20 |
| **F-CAGrad** | **40.93** | **0.5285** | **25.43** | **-3.78** |
| IMTL | 39.35 | 0.5426 | 26.02 | -0.76 |
| **F-IMTL** | **40.42** | **0.5389** | **25.03** | **-4.77** |

### Ablation Study

Comparison of aggregation strategies on CityScapes:

| Strategy | mIoU↑ | Abs Err↓ | Rel Err↓ | $\Delta m\%$↓ |
|----------|:---:|:---:|:---:|:---:|
| ERM (no SAM) | 68.84 | 0.0309 | 33.50 | 44.14 |
| Direct SAM gradient aggregation | 68.93 | 0.0130 | 31.37 | 6.43 |
| **Decomposed aggregation (Ours)** | **73.77** | **0.0129** | **27.44** | **0.67** |

The decomposition strategy outperforms direct aggregation by nearly 5 percentage points in segmentation mIoU, confirming the importance of handling loss gradients and flat gradients separately.

### Key Findings

- F-Method **consistently** improves all baseline methods across all datasets.
- The gradient conflict ratio **approaches 0%** as training progresses (vs. rising above 50% under standard ERM).
- The benefit is not merely from applying SAM to individual tasks—F-LS and F-STL cannot surpass conflict-aware methods such as F-IMTL.
- When flatness is incorporated into all methods, performance gaps among different MTL approaches narrow.

## Highlights & Insights

- **First systematic examination of MTL through the lens of loss landscape geometry**: the generalization strategy of seeking flat minima is rigorously introduced into multi-task optimization.
- **Intuitive gradient decomposition**: the "low-loss direction" and the "flat direction" represent fundamentally distinct optimization objectives and should be handled separately.
- **Theoretical contribution**: employs a more general PAC-Bayesian bound (supporting bounded losses rather than only 0-1 losses), going beyond a straightforward extension of SAM theory.
- **Model-agnostic design**: the framework serves as a plug-in that can enhance any gradient-based MTL method.

## Limitations & Future Work

- Computational cost is approximately $2\times$ that of the base method, requiring an additional forward-backward pass at the perturbed parameters.
- The relaxation of the shared perturbation in the theoretical analysis (from a single shared $\epsilon_{sh}$ to per-task independent $\epsilon_{sh}^i$) may not be tight.
- Validation is limited to computer vision tasks; applicability to NLP or multimodal MTL has not been tested.
- Applicability to non-gradient-based MTL methods (e.g., loss-weighting approaches) is limited.

## Related Work & Insights

- Represents a multi-task generalization of SAM (Foret et al., 2021); the core difficulty lies in handling the shared perturbation under multiple objectives.
- Shares motivational similarities with applications of SAM to continual learning (flatness vs. forgetting), though the underlying problems differ fundamentally.
- The gradient decomposition idea is generalizable to other multi-objective optimization settings (e.g., fairness-constrained optimization).

## Rating

- **Novelty**: 7/10 — The angle of introducing SAM into MTL is novel; the key innovation is the gradient decomposition strategy.
- **Technical Quality**: 8/10 — Theoretical derivations are complete and experiments provide broad coverage.
- **Practicality**: 7/10 — Plug-and-play but incurs $2\times$ computational overhead.
- **Writing Quality**: 7/10 — Notation is heavy, but the logical chain is clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)
- [\[ICLR 2026\] Empowering Multi-Robot Cooperation via Sequential World Models](../../ICLR2026/robotics/empowering_multi-robot_cooperation_via_sequential_world_models.md)
- [\[NeurIPS 2025\] Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention](../../NeurIPS2025/robotics/beyond_parallelism_synergistic_computational_graph_effects_in_multi-head_attenti.md)

</div>

<!-- RELATED:END -->
