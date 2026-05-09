---
title: >-
  [Paper Note] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning
description: >-
  [NeurIPS 2025][Model Compression][Continual Learning] This paper proposes DEAL, a framework that leverages wavelet kernel feature filtering to preserve core historical knowledge in LoRA low-rank matrices, combined with a controlled knowledge update module and asymmetric regularization, enabling LLMs to acquire new knowledge without forgetting old tasks under few-shot continual fine-tuning.
tags:
  - NeurIPS 2025
  - Model Compression
  - Continual Learning
  - LoRA
  - Wavelet Kernel
  - Knowledge Retention
  - Parameter-Efficient Fine-Tuning
date: 2026-05-08
content_hash: 0cd5cc30fe39b49d
---

# Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2509.18942](https://arxiv.org/abs/2509.18942)
**Code**: [GitHub](https://github.com/zzm-black/DEAL-Continuous-Low-Rank-Fine-Tuning)
**Area**: Model Compression
**Keywords**: Continual Learning, LoRA, Wavelet Kernel, Knowledge Retention, Parameter-Efficient Fine-Tuning

## TL;DR

This paper proposes DEAL, a framework that leverages wavelet kernel feature filtering to preserve core historical knowledge in LoRA low-rank matrices, combined with a controlled knowledge update module and asymmetric regularization, enabling LLMs to acquire new knowledge without forgetting old tasks under few-shot continual fine-tuning.

## Background & Motivation

LoRA-tuned LLMs must integrate new knowledge through continual learning. However, conventional continual learning methods face two core challenges:

**Catastrophic Forgetting**: Fine-tuning on new tasks leads to performance degradation on prior tasks.

**Data Inefficiency**: Small-scale domain-specific data is insufficient for effective adaptation.

Limitations of existing solutions:
- **Direct editing** (ROME/MEMIT): Requires additional experiments to locate neurons, incurring high cost.
- **Stacked adapters**: Increases inference overhead.
- **Orthogonal subspace constraints** (O-LoRA): Restricts beneficial cross-task transfer.

Core Problem: Can one design a method that continuously fine-tunes LoRA with limited new data, preserving performance on all historical tasks, **without introducing any additional inference latency**?

## Method

### Overall Architecture

DEAL consists of two core modules:
1. **Wavelet Kernel Knowledge Retention Module**: Extracts and filters singular values from LoRA low-rank matrices to preserve core representations of historical knowledge.
2. **Controlled Knowledge Update Module**: Regulates the integration of new knowledge through higher-order regularization constraints on parameter updates.

At inference time, the updated low-rank matrices directly replace the original LoRA modules, leaving **inference latency unchanged**.

### Key Designs

**Wavelet Kernel Feature Filtering**: The low-rank matrix $\mathbf{Y}$ (i.e., $\mathbf{A}$ or $\mathbf{B}$) is decomposed into a task-relevant core component $\mathbf{X}$ and a redundant/noisy component $\mathbf{D}$: $\mathbf{Y} = \mathbf{X} + \mathbf{D}$.

Under the white noise assumption ($\mathbf{D}^\top \mathbf{D} = \sigma_D^2 I$, $\mathbf{X}^\top \mathbf{D} = 0$), the minimum-variance estimate of the core features is:

$$\hat{\mathbf{X}} = \sum_{k=1}^{r_x} \frac{\sigma_k^2 - \sigma_D^2}{\sigma_k} \mathbf{u}_k \mathbf{v}_k^\top$$

Since $\sigma_D^2$ is unknown, a heat kernel is employed as a low-pass filter to define a multi-scale wavelet network:

$$\phi_{\sigma_j^2, c_j}(\mathbf{X}) = \exp\left(-\frac{1}{2\sigma_j^2}\|\mathbf{X} - c_j\|^2\right)$$

The wavelet neural network extracts core features as:

$$\mathbf{H}_{:,i}^{k+1} = \delta\left(\sum_j \phi_{\sigma_j^2, c_j} \mathbf{g}_j \phi_{-\sigma_j^2, c_j} \mathbf{H}_{:,j}^k\right)$$

where $\mathbf{g}_j$ is a learnable diagonal matrix and $c_j$ is a learnable center.

**Controlled Knowledge Update**: An MLP superimposes new knowledge onto the core features to produce the updated low-rank matrices $\mathbf{A}'$ or $\mathbf{B}'$.

### Loss & Training

$$\mathcal{L} = \text{MSE}(\mathcal{A}_{\mathbf{W}, \Delta\mathbf{W}'}(\mathbf{Q}), \mathbf{G}) + \lambda_1 \|\boldsymbol{\theta}_1\|_a^a + \lambda_2 \|\boldsymbol{\theta}_2\|_b^b$$

where $\boldsymbol{\theta}_1 = \{\mathbf{g}, \mathbf{C}\}$ are the retention module parameters and $\boldsymbol{\theta}_2 = \{\boldsymbol{\Omega}, \mathbf{B}\}$ are the update module parameters. The key constraint $a \geq b$ ensures that the regularization strength on the retention module is no less than that on the update module, thereby **minimizing perturbation to core features**.

## Key Experimental Results

### Main Results

**Continual Learning Performance Comparison** (Average Accuracy / ROUGE-1):

| Method | 3-Task TC | 4-Task Standard | 15-Task Large |
|--------|-----------|----------------|---------------|
| | AA / R-1 | AA / R-1 | AA / R-1 |
| T5 + SeqLoRA | 52.4 / 52.8 | 44.6 / 44.6 | 42.1 / 44.0 |
| T5 + O-LoRA | 85.2 / 87.1 | 71.2 / 73.3 | 70.8 / 80.3 |
| T5 + PerTaskFT (oracle) | 90.3 / 91.7 | 70.0 / 73.0 | 76.5 / 78.2 |
| **T5 + DEAL** | **87.7 / 89.3** | **78.5 / 82.5** | **73.9 / 79.1** |
| LLaMA + SeqLoRA | 54.1 / 55.9 | 47.6 / 54.8 | 45.2 / 53.2 |
| LLaMA + O-LoRA | 86.4 / 88.1 | 75.3 / 80.8 | 73.2 / 77.4 |
| LLaMA + PerTaskFT | 88.2 / 90.0 | 77.5 / 79.4 | 77.1 / 82.5 |
| **LLaMA + DEAL** | **88.9 / 90.2** | **78.9 / 81.3** | **74.6 / 78.9** |

On the 4-task benchmark, DEAL outperforms O-LoRA by 7.3 pp in AA (T5) and SeqLoRA by 33.9 pp, approaching the oracle PerTaskFT upper bound.

### Ablation Study

**Adapter Update Strategy**: Jointly updating A+B achieves the highest AA (75.6%), outperforming updating A alone (72.8%) or B alone (70.2%).

**Sensitivity to LoRA Rank**:

| Rank | AA (%) |
|------|--------|
| 4 | 71.5 |
| 8 | 84.3 |
| 16 | 84.5 |
| 32 | 84.6 |

Rank 8 already captures the majority of task-specific variation; further increases yield diminishing returns.

**Regularization Weights** ($(a, b)$ grid search):

| $a$ | $b$ | AA (%) |
|-----|-----|--------|
| 1 | 1 | 74.8 |
| 5 | 1 | 83.9 |
| **10** | **2** | **85.5** |
| 10 | 5 | 84.1 |
| 20 | 2 | 82.7 |

**Robustness to Task Order**: Across 3 random permutations, the AA range spans only 73.1%–75.6%, with a variance of < 3 pp.

### Key Findings

1. DEAL surpasses SeqLoRA by over 29 pp on the 15-task large-scale benchmark, demonstrating strong scalability.
2. Asymmetric regularization ($a > b$) is critical: the retention module requires stronger constraints to protect historical knowledge.
3. Inference latency remains completely unchanged, as the updated low-rank matrices directly replace the original modules.

## Highlights & Insights

1. **Wavelet kernel feature filtering for LoRA** is a novel and well-motivated design, grounded in the signal-processing intuition of SVD-based frequency decomposition.
2. Zero inference overhead is a key practical advantage, avoiding the latency cost associated with stacked adapters.
3. The asymmetric regularization design has clear intuition: protecting historical knowledge should be prioritized over accommodating new knowledge updates.
4. Applicability to large-scale models is validated on LLaMA-3.1-8B.

## Limitations & Future Work

- Assumes a fixed task order and static model capacity; ambiguous task boundaries are not addressed.
- The white noise assumption ($\mathbf{D}^\top \mathbf{D} = \sigma_D^2 I$) may not strictly hold for real-world low-rank matrices.
- The wavelet kernel introduces additional training parameters that, while absent at inference time, incur overhead during training.
- No comparison with replay-based methods is conducted; baselines are limited to memory-free approaches.

## Related Work & Insights

- DEAL is complementary to O-LoRA (orthogonal subspace constraints): O-LoRA reduces interference but also restricts cross-task transfer, whereas DEAL preserves core features via filtering while permitting updates to non-core components.
- Compared to CLoRA (angular regularization), DEAL's wavelet kernel feature filtering provides finer-grained control.
- This work raises the question of whether wavelet kernel methods can be extended to other PEFT modules (e.g., Adapter, Prefix-tuning) in continual learning settings.

## Rating

- ⭐ Novelty: 4/5 — Applying wavelet kernel feature filtering to LoRA continual learning is a genuinely novel technical direction.
- ⭐ Experimental Thoroughness: 4/5 — 15 datasets, two model backbones, and comprehensive ablations.
- ⭐ Writing Quality: 3/5 — Derivations are detailed but notation is occasionally overloaded, limiting readability.
- ⭐ Value: 4/5 — Addresses a practical pain point in LoRA continual learning with zero inference overhead.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)

<!-- RELATED:END -->
