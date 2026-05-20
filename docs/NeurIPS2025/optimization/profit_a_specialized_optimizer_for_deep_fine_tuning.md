---
title: >-
  [Paper Note] PROFIT: A Specialized Optimizer for Deep Fine Tuning
description: >-
  [NeurIPS 2025][Optimization][Fine-tuning optimizer] PROFIT frames fine-tuning as a multi-task learning problem across the time dimension…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Fine-tuning optimizer"
  - "catastrophic forgetting"
  - "gradient orthogonalization"
  - "multi-task learning"
  - "proximal fine-tuning"
date: 2026-05-08
content_hash: 6ca0221b3d9af928
---

# PROFIT: A Specialized Optimizer for Deep Fine Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2412.01930](https://arxiv.org/abs/2412.01930)  
**Code**: Unavailable  
**Area**: Optimization
**Keywords**: Fine-tuning optimizer, catastrophic forgetting, gradient orthogonalization, multi-task learning, proximal fine-tuning

## TL;DR

PROFIT frames fine-tuning as a multi-task learning problem across the time dimension, and achieves forgetting-resistant fine-tuning without additional data or parameters by orthogonally projecting new-task gradients onto the direction of a "regression equilibrium point."

## Background & Motivation

Fine-tuning pre-trained models has become the dominant paradigm in deep learning—from autonomous driving to LLMs, the cost of training from scratch continues to rise. However, existing optimizers such as SGD, Adam, and AdamW are designed for **training from scratch** and make no assumptions about the initialization state. Fine-tuning presents an entirely different setting: the starting point is already a well-converged model, and this prior should be exploited.

The central challenge of fine-tuning is **catastrophic forgetting**: training on new data causes the model to rapidly lose prior knowledge. Existing solutions each have their limitations:
- **LWF** (Learning Without Forgetting) requires storing old model responses and performing distillation, increasing data pipeline complexity.
- Parameter-efficient methods such as **LoRA** reduce computation but do not improve accuracy.
- **Freezing the backbone with a small learning rate** limits the model's adaptability.

The authors' key insight is as follows: since fine-tuning begins from a converged point, the model naturally tends to "return" after drifting from it—the negative of the displacement $\Delta$, i.e., $-\Delta$, approximates the gradient direction of the old loss. This constitutes a natural "temporal multi-task learning" problem: Task 1 (preserving the old model) and Task 2 (adapting to new data) may exhibit gradient conflicts.

## Method

### Overall Architecture

PROFIT (PROximal FIne Tuning) is an optimizer wrapper that accepts two standard optimizers: a main optimizer $\mathbf{O}$ and a reference optimizer $\mathbf{O}^{(\text{ref})}$. The procedure at each training step is as follows:

1. Save the current parameters $\theta_{\text{ref}} \leftarrow \theta$
2. Take $n_{\text{ref}}$ small steps on the new data using the reference optimizer, arriving at $\theta'$
3. Compute the displacement $\Delta = \theta' - \theta_{\text{ref}}$
4. Sample a new batch and compute the gradient $\mathbf{g}$
5. If $\langle \Delta, \mathbf{g} \rangle < 0$ (i.e., gradient conflict), orthogonalize $\mathbf{g}$ with respect to $\Delta$
6. Restore the parameters $\theta \leftarrow \theta_{\text{ref}}$ and take a step along the orthogonalized $\mathbf{g}$

### Key Designs

1. **Temporal gradient orthogonalization**: Inspired by PCGrad from the multi-task learning literature. $\Delta$ represents the implicit gradient of the old task (since the direction back to the converged point is a descent direction for the old loss), and $\mathbf{g}$ represents the gradient of the new task. When the two conflict (inner product $\omega < 0$), $\mathbf{g}$ is projected onto the orthogonal complement of $\Delta$: $\mathbf{g} \leftarrow \mathbf{g} - \frac{\langle \mathbf{g}, \Delta \rangle}{\|\Delta\|^2} \Delta$. Crucially, only $\mathbf{g}$ is orthogonalized, not $\Delta$, since the old data may be inaccessible, making the information encoded in $\Delta$ more valuable.

2. **Reference step mechanism**: The reference optimizer $\mathbf{O}^{(\text{ref})}$ (SGD is recommended) explores the loss surface with a small learning rate, and the cumulative displacement $\Delta$ over $n_{\text{ref}}$ steps encodes local surface geometry. $\lambda_{\text{ref}}$ controls the exploration rate, typically set between $\lambda_{\text{main}}/10$ and $\lambda_{\text{main}}/10000$. All experiments use $n_{\text{ref}}=1$.

3. **Theoretical guarantees**:

    - **Theorem 3.1 (correctness on old data)**: Near the converged point, the loss surface satisfies $L(\mathbf{x}) \approx L(\mathbf{x}_0) + 0.5(\mathbf{x}-\mathbf{x}_0)^T \mathbf{H}_0 (\mathbf{x}-\mathbf{x}_0)$, so $-\Delta$ is a valid gradient descent direction, and one step of PROFIT decreases the old data loss.
    - **Theorem 3.2 (stationary points)**: PROFIT produces a zero update only when the loss surface is perfectly linear between $\theta$ and $\theta'$—a condition that is virtually impossible to satisfy in high-dimensional deep networks.
    - **Theorem 3.4 (convergence)**: PROFIT converges to the convergence point of the main optimizer $\mathbf{O}$ or to a stationary point as characterized by Theorem 3.2.

### Loss & Training

- The original task loss function is used without modification.
- The method assumes that the pre-training and fine-tuning data come from **proximal distributions** (distributional overlap).
- For non-proximal settings (e.g., ImageNet→VTAB), a warmup strategy is provided: fine-tune with AdamW for 10 epochs to reduce the distributional distance, then switch to PROFIT.
- Memory overhead increases by approximately 25% (for storing $\theta_{\text{ref}}$), but this can be largely eliminated by approximating via the momentum buffer.

## Key Experimental Results

### Main Results

| Task | Method | Old Data Metric | New Data Metric |
|------|--------|----------------|----------------|
| 2D Regression (MLP) | Full fine-tuning | 0.705 error | 0.504 error |
| 2D Regression (MLP) | Head fine-tuning | 0.110 error | 0.572 error |
| 2D Regression (MLP) | **PROFIT** | **0.046** error | **0.501** error |
| CIFAR10→100 (ViT-Tiny) | Adam | 56.00% / 58.99% | — |
| CIFAR10→100 (ViT-Tiny) | Lookahead | 55.64% / 61.35% | — |
| CIFAR10→100 (ViT-Tiny) | **PROFIT** | **58.53%** / **62.20%** | — |
| CIFAR10→100 (ViT-Small) | Adam | 58.60% / 63.93% | — |
| CIFAR10→100 (ViT-Small) | **PROFIT** | **59.02%** / **65.44%** | — |

### Ablation Study

| Setting | ADE@8s | FDE@8s | Notes |
|---------|--------|--------|-------|
| Waymo Car→Car baseline | 1.327m | 2.581m | No fine-tuning |
| Car→Car full fine-tuning | 1.322m | 2.548m | Marginal improvement |
| Car→Car **PROFIT** | **1.299m** | **2.489m** | Superior |
| Car→Ped full fine-tuning | 0.621m | 1.242m | Domain transfer |
| Car→Ped head fine-tuning | 0.724m | 1.544m | Inferior to full fine-tuning |
| Car→Ped **PROFIT** | **0.579m** | **1.145m** | Significant improvement |

| Task | AdamW | PROFIT | Notes |
|------|-------|--------|-------|
| DriveLM VQA Accuracy | 62.21% | **67.88%** | VLM fine-tuning |
| DriveLM Final Score | 56.98 | **59.16** | Overall score |

### Key Findings

- PROFIT achieves superior CIFAR10/100 trade-offs across all backbone architectures (ResNet-18, ViT-Tiny, ViT-Small) and all baseline optimizers.
- In the non-proximal VTAB-1K setting, naive PROFIT fails (Clevr-Count: only 12.6%); however, with the AdamW warmup, it surpasses full fine-tuning on 15 out of 19 tasks.
- In the cross-domain Car→Ped fine-tuning setting, PROFIT achieves a 7.8% improvement in FDE@8s (1.242m→1.145m), substantially outperforming all other methods.

## Highlights & Insights

- **Novel perspective**: Reframing fine-tuning as temporal multi-task learning opens a new direction in optimizer design.
- **High practicality**: Requires only wrapping an existing optimizer, with no architectural modifications, no old data, and no additional parameters.
- **Concise and compelling theory**: The correctness proof based on positive definiteness of the Hessian is both intuitive and persuasive.
- The paper proposes that "PROFIT can serve as a standard procedure for model maintenance"—it yields improvements even when training continues on the same data.

## Limitations & Future Work

- **Proximal distribution assumption**: The method requires that the pre-training and fine-tuning data distributions be similar; non-proximal settings require an additional warmup procedure.
- **Memory overhead**: Storing $\theta_{\text{ref}}$ increases GPU memory usage by approximately 25%, which may become a bottleneck for full-parameter fine-tuning of large models.
- **Additional forward pass cost**: The $n_{\text{ref}}$ reference computation steps increase training time, though $n_{\text{ref}}=1$ is typically used.
- The optimal value of $\lambda_{\text{ref}}$ is highly problem-dependent and is sensitive to the loss surface geometry, with no automatic tuning mechanism currently available.

## Related Work & Insights

- The core inspiration is PCGrad (gradient conflict resolution in multi-task learning), extended from the spatial dimension to the temporal dimension.
- The approach shares conceptual similarity with the data-driven anchor idea in LWF, but avoids the need to store old data or model snapshots.
- PROFIT is complementary rather than competitive with LoRA: PROFIT focuses on improving fine-tuning accuracy, while LoRA targets efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of fine-tuning as temporal multi-task learning is original; the orthogonalization design is elegant and concise
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Experiments span from 2D toy problems to CIFAR, VLMs, and autonomous driving
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, figures are intuitive, and the theory-experiment connection is natural
- Value: ⭐⭐⭐⭐⭐ A plug-and-play fine-tuning optimizer with extremely high practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)

</div>

<!-- RELATED:END -->
