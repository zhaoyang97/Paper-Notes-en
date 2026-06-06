---
title: >-
  [Paper Note] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning
description: >-
  [AAAI 2026][Model Compression][continual learning] This paper proposes FLAD, a framework that decomposes the sharpness-aware perturbation direction into a gradient-aligned component and a stochastic-noise component…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "continual learning"
  - "Sharpness-Aware Minimization"
  - "Flatness Decomposition"
  - "catastrophic forgetting"
  - "Loss Landscape"
date: 2026-05-08
content_hash: cd61f279525e9ae5
---

# Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning

**Conference**: AAAI 2026
**arXiv**: [2601.07636](https://arxiv.org/abs/2601.07636)  
**Code**: Not available  
**Area**: Model Compression
**Keywords**: continual learning, Sharpness-Aware Minimization, Flatness Decomposition, catastrophic forgetting, Loss Landscape

## TL;DR

This paper proposes FLAD, a framework that decomposes the sharpness-aware perturbation direction into a gradient-aligned component and a stochastic-noise component, retaining only the noise component for regularization. By combining zeroth-order and first-order sharpness, FLAD improves generalization in continual learning with minimal additional computational overhead.

## Background & Motivation

- **Core challenge in continual learning**: Models suffer from catastrophic forgetting when learning multiple tasks sequentially, where new task training causes severe performance degradation on previously learned tasks.
- **Limitations of existing paradigms**: Replay-based, regularization-based, and architecture-based methods are effective but often require storing large amounts of raw data or model parameters, limiting scalability.
- **Promise of flat minima**: Research indicates that optimizing toward flat loss minima can improve generalization and robustness to distribution shift; methods such as SAM and GAM have demonstrated promise in transfer learning and few-shot learning.
- **First limitation of existing flatness methods**: Sharpness regularization is treated as a unified signal without distinguishing the contributions of different perturbation components (gradient-aligned vs. gradient-orthogonal directions), which is an oversimplification.
- **Second limitation of existing flatness methods**: Double gradient computation or multiple forward-backward passes introduce substantial computational overhead, severely limiting practical scalability.
- **Key Insight**: Can the sharpness perturbation direction be decomposed so that only the component genuinely beneficial to generalization is retained, while significantly reducing computational cost?

## Method

### Overall Architecture

FLAD (**Fla**tness **D**ecomposition) decomposes the perturbation direction in sharpness-aware optimization into a gradient-aligned component and a stochastic-noise component, demonstrating that retaining only the noise component more effectively escapes sharp minima. The framework unifies zeroth-order and first-order sharpness minimization under a single objective:

$$L(w^T) = L^{R^0_\rho}(w^T) + \gamma \cdot L^{R^1_\rho}(w^T)$$

where the zeroth-order and first-order terms respectively encourage flat minima. The framework integrates seamlessly into replay-based, regularization-based, and expansion-based continual learning paradigms.

### Key Design 1: Perturbation Direction Decomposition

- **Function**: Decomposes the batch gradient $\hat{g}_B$ into a component aligned with the global gradient $g$, i.e., $\text{Proj}_g(\hat{g}_B)$, and an orthogonal stochastic-noise component $\text{Proj}_g^\perp(\hat{g}_B)$; only the noise component is used to construct the perturbation direction.
- **Mechanism**: The orthogonal component is approximated as $\hat{g}_B - \sigma m_t$, where $\sigma$ is the cosine similarity between gradients (treated as a constant during training) and $m_t$ is the EMA estimate of the gradient; removing the gradient-aligned direction avoids conflict with the optimization trajectory.
- **Design Motivation**: Experiments show that full-gradient-aligned perturbation actually degrades performance (conflicting with learning dynamics), whereas the stochastic-noise component consistently outperforms all other variants across settings and is the key driver of improved generalization.

### Key Design 2: Noise Decomposition for First-Order Sharpness

- **Function**: Applies the same decomposition to first-order sharpness (curvature of the gradient norm), decomposing $\nabla\|\hat{g}_B\|$ into components aligned with and orthogonal to the EMA direction.
- **Mechanism**: The first-order sharpness gradient $\nabla\|g\| = \nabla^2 L(w) \cdot \frac{\nabla L(w)}{\|\nabla L(w)\|}$ is efficiently computed via Hessian-vector products; the EMA $n_t$ approximates the global direction, from which the noise component is extracted.
- **Design Motivation**: First-order sharpness captures the sharpness of the gradient itself (i.e., the curvature of the function) and is complementary to the zeroth-order term; the noise component similarly benefits generalization more than the complete direction.

### Key Design 3: Lightweight Scheduling Strategy

- **Function**: Rather than applying FLAD at every epoch, FLAD is activated only during a subset of epochs (e.g., 10–20%), with vanilla SGD used for the remainder.
- **Mechanism**: A scheduling scheme is explored in which FLAD is applied for only a fraction of epochs per task, and it is found that sparse application suffices to yield significant gains.
- **Design Motivation**: Each FLAD step requires 2 forward passes and 4 backward passes; the scheduling strategy reduces computational overhead by at least 50% while maintaining or exceeding the performance of full-epoch FLAD.

## Loss & Training

- The total loss unifies zeroth-order and first-order sharpness regularization, with $\gamma$ controlling the weight of the first-order term.
- Zeroth-order perturbation $\delta_0$ and first-order perturbation $\delta_1$ are each constructed from their respective noise components.
- Parameter update: $w = w - \eta(g_0 + \gamma g_1)$, where $g_0$ is the gradient at the zeroth-order perturbed point and $g_1$ is the Hessian-vector product at the first-order perturbed point.
- EMA momentum parameters $\lambda_0, \lambda_1 \in (0,1)$; $\sigma$ is a fixed cosine similarity constant.
- Convergence rate is $\mathcal{O}(\log n^T / \sqrt{n^T})$ under a non-convex setting.

## Key Experimental Results

### Main Results: 6 CL Methods × 4 Settings (Table 1)

| Method | CIFAR-10 N=5 Acc | CIFAR-100 N=10 Acc | Tiny-ImageNet N=8 Acc | Avg. Gain |
|------|:-:|:-:|:-:|:-:|
| Replay → w/FLAD | 41.68→**43.13** | 29.07→**31.74** | 19.09→**22.91** | +2.18% |
| iCaRL → w/FLAD | 50.63→**52.53** | 30.05→**30.36** | 20.38→**23.04** | +1.24% |
| WA → w/FLAD | 61.95→**62.46** | 47.29→**48.35** | 36.40→**39.00** | +1.90% |
| FOSTER → w/FLAD | 56.43→**68.98** | 39.71→**39.13** | 38.10→**38.54** | +1.41% |
| MEMO → w/FLAD | 57.80→**61.40** | 53.27→**53.92** | 44.29→**45.06** | +1.83% |
| PODNet → w/FLAD | 57.03→**57.62** | 31.87→**34.85** | 33.13→**35.04** | +0.97% |

FLAD consistently outperforms SAM, GAM, and C-Flat across all 6 methods, 3 datasets, and multiple task-split configurations.

### Efficiency Analysis (Figure 5)

| Scheduling Ratio | Gain over SGD | Overhead vs. Full FLAD |
|:-:|:-:|:-:|
| 10% epochs with FLAD | Already surpasses SGD | Close to SGD runtime |
| 20% epochs with FLAD | Surpasses GAM/SAM/C-Flat | ~50% overhead |
| 20 epochs of FLAD | Surpasses 50-epoch C-Flat and 200-epoch other optimizers | — |

Applying FLAD for only 30% of epochs reduces training overhead by at least 50% while preserving performance.

## Highlights & Insights

- **Novel decomposition perspective**: This is the first work to decompose sharpness perturbations into gradient-aligned and noise components, demonstrating that the noise component is the key driver of generalization.
- **Plug-and-play**: Seamlessly integrates into all three major CL paradigm families—replay, regularization, and expansion.
- **High efficiency**: The lightweight scheduling strategy yields significant gains with only 10–20% of epochs, offering strong practical utility.
- **Theoretical support**: Provides a convergence guarantee of $\mathcal{O}(\log n / \sqrt{n})$ in the non-convex setting.
- **Comprehensive experiments**: Covers 6 methods across 3 datasets with multiple task splits, supplemented by Hessian analysis and loss landscape visualizations.

## Limitations & Future Work

- Experiments are limited to CIFAR-10/100 and Tiny-ImageNet; validation on larger-scale datasets (e.g., full ImageNet) is absent.
- The effectiveness of FLAD in NLP or other modalities for continual learning has not been explored.
- $\sigma$ is simplified to a fixed constant during training; adaptive adjustment may further improve performance.
- Although Hessian-vector products are more efficient than full Hessian computation, the overhead of 2 forward + 4 backward passes remains non-negligible in resource-constrained environments.
- The work focuses exclusively on class-incremental learning; task-incremental and domain-incremental settings are not addressed.

## Related Work & Insights

- **SAM (Foret et al., 2021)**: Zeroth-order sharpness minimization that seeks flat minima via worst-case perturbations.
- **GAM (Zhang et al., 2023)**: First-order sharpness minimization that leverages the curvature of the gradient norm to improve feature reuse.
- **C-Flat (Bian et al., NeurIPS 2024)**: Combines zeroth-order and first-order sharpness to find globally flatter solutions; the most direct baseline for this work.
- **Li et al., 2024**: Analyzes the role of perturbation components in SAM; this paper extends that analysis from standard training to continual learning.
- **FS-DGPM / SAM-CL series**: Pioneering works applying sharpness-aware methods to continual learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perturbation decomposition perspective is original; the finding that the noise component outperforms the full perturbation is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 6 methods and multiple datasets, with extensive ablation studies and visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Logic is clear and mathematical derivations are complete, though some notations are relatively dense.
- Value: ⭐⭐⭐⭐ — The combination of plug-and-play integration and efficient scheduling offers substantial practical value for continual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](../../ICML2026/model_compression/task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/model_compression/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](../../ICLR2026/model_compression/ider_idempotent_experience_replay_for_reliable_continual_learning.md)
- [\[NeurIPS 2025\] Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning](../../NeurIPS2025/model_compression/train_with_perturbation_infer_after_merging_a_two-stage_framework_for_continual_.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)

</div>

<!-- RELATED:END -->
