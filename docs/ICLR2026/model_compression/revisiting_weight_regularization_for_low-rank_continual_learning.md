---
title: >-
  [Paper Note] Revisiting Weight Regularization for Low-Rank Continual Learning
description: >-
  [Model Compression] This paper reintroduces Elastic Weight Consolidation (EWC) into low-rank continual learning by estimating the Fisher Information Matrix in the full-dimensional space to regularize a shared LoRA module, achieving effective forgetting mitigation under constant memory overhead.
tags:
  - Model Compression
date: 2026-05-08
content_hash: 668f17ad6c099dea
---

# Revisiting Weight Regularization for Low-Rank Continual Learning

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2602.17559](https://arxiv.org/abs/2602.17559)
- **Code**: [GitHub](https://github.com/yaoyz96/low-rank-cl)
- **Area**: Continual Learning / Model Compression
- **Keywords**: Continual Learning, EWC, LoRA, Fisher Information, Parameter-Efficient Learning

## TL;DR

This paper reintroduces Elastic Weight Consolidation (EWC) into low-rank continual learning by estimating the Fisher Information Matrix in the full-dimensional space to regularize a shared LoRA module, achieving effective forgetting mitigation under constant memory overhead.

## Background & Motivation

### State of the Field
With the rise of large-scale pre-trained models (PTMs), the continual learning paradigm has shifted from training from scratch to continually adapting PTMs. Parameter-Efficient Continual Learning (PECL) has become mainstream, typically alleviating task interference by assigning independent LoRA modules to each task.

### Limitations of Prior Work

**Linearly growing storage**: Existing low-rank CL methods (e.g., InfLoRA, SD-LoRA) maintain separate LoRA branches per task, causing memory overhead to grow linearly with the number of tasks.

**Neglected weight regularization**: Classical regularization methods such as EWC have not been sufficiently explored in the PTM era—naively applying EWC to PTMs requires memory three times the model size to store the old model copy and the Fisher matrix.

**Suboptimal naive integration**: Simply combining EWC with LoRA by regularizing the $\mathbf{A}$ and $\mathbf{B}$ matrices independently ignores their interaction, leading to distorted importance estimates.

### Core Idea
Low-rank parameterization enables efficient EWC: by estimating Fisher information in the full-dimensional space $\Delta\mathbf{W} = \mathbf{AB}$, the method accurately captures parameter importance while maintaining constant memory cost.

## Method

### Overall Architecture

EWC-LoRA trains a single shared LoRA module across all tasks, regularizes the update direction via the Fisher Information Matrix, and merges the update into the base weights upon task completion.

### 1. Problem Formulation

For task $\mathcal{T}_t$, weight updates are constrained to a low-rank subspace:

$$
\mathbf{W}_t = \mathbf{W}_{t-1} + \Delta\mathbf{W} = \mathbf{W}_{t-1} + \mathbf{AB}
$$

where $\mathbf{A} \in \mathbb{R}^{d_O \times r}$, $\mathbf{B} \in \mathbb{R}^{r \times d_I}$, and $r \ll \min(d_I, d_O)$.

### 2. Full-Dimensional Fisher Regularization

**Key innovation**: Rather than computing Fisher information separately on the low-rank matrices $\mathbf{A}$ and $\mathbf{B}$, the method regularizes in the full-dimensional space $\Delta\mathbf{W}$:

$$
\mathcal{L}_t'(\mathbf{A}, \mathbf{B}) = \mathcal{L}_t(\mathbf{A}, \mathbf{B}) + \frac{\lambda}{2} \text{vec}(\mathbf{AB})^\top \mathbf{F}_{t-1}^{\text{cum}} \text{vec}(\mathbf{AB})
$$

where $\mathbf{F}_{t-1}^{\text{cum}}$ is the accumulated diagonal Fisher matrix.

### 3. Fisher Matrix Estimation

The Fisher matrix is estimated after training on task $\mathcal{T}_t$:

$$
F_t^{i,i} = \mathbb{E}_{x \sim \mathcal{D}_t}\left[\mathbb{E}_{y \sim p_{\mathbf{W}_t^*}}\left[\left(\frac{\partial \log p_{\mathbf{W}}(y|x)}{\partial w_i}\bigg|_{\mathbf{W}=\mathbf{W}_t^*}\right)^2\right]\right]
$$

Since only $\Delta\mathbf{W}$ is trainable, gradients in the $\mathbf{W}$ and $\Delta\mathbf{W}$ spaces are equivalent, requiring no additional computation.

### 4. Training Procedure

1. Initialize the LoRA branch ($\mathbf{A}$ zero-initialized, $\mathbf{B}$ uniform-initialized)
2. Freeze base weights $\mathbf{W}_{t-1}$ during training; update only $\mathbf{A}$ and $\mathbf{B}$
3. Regularize $\Delta\mathbf{W} = \mathbf{AB}$ using $\mathbf{F}_{t-1}^{\text{cum}}$
4. Merge upon task completion: $\mathbf{W}_t = \mathbf{W}_{t-1} + \mathbf{AB}$
5. Estimate $\mathbf{F}_t$ and update the cumulative Fisher: $\mathbf{F}_t^{\text{cum}}$
6. Discard task data and per-task Fisher

### Comparison of Three Fisher Estimation Strategies

The paper theoretically and empirically demonstrates that independently regularizing the low-rank matrices is suboptimal:

| Strategy | $\bar{A}_{10}$ | Stability | Plasticity | Extra Memory |
|----------|----------------|-----------|------------|--------------|
| No Fisher | 82.99 | 87.56 | **98.86** | 0 GB |
| Precomputed $\mathbf{F}_{\mathbf{W}}$ | 83.87 | 93.15 | 94.74 | 1 GB |
| Separate $\mathbf{F}_{\mathbf{A}}, \mathbf{F}_{\mathbf{B}}$ | 86.41 | 94.23 | 96.47 | 4 GB |
| **Full-dim $\mathbf{F}_{\Delta\mathbf{W}}$ (Ours)** | **87.91** | **94.45** | 97.99 | 6 GB |

## Experiments

### Main Results (Vision Tasks)

| Method | CIFAR-100 $\bar{A}_{10}$ | DomainNet $\bar{A}_{5}$ | ImageNet-R $\bar{A}_{10}$ | ImageNet-A $\bar{A}_{10}$ |
|--------|--------------------------|------------------------|--------------------------|--------------------------|
| Finetune | 79.09 | 65.57 | 60.42 | 32.85 |
| L2P | 83.18 | 70.26 | 71.26 | 42.94 |
| CODA-Prompt | 86.31 | 70.58 | 74.05 | 45.36 |
| InfLoRA | 86.34 | 71.01 | 74.41 | 50.75 |
| SD-LoRA | 86.77 | — | — | — |
| **EWC-LoRA (Ours)** | **87.91** | **72.13** | **75.20** | **52.48** |

### Ablation Study: Stability–Plasticity Trade-off

| Regularization strength $\lambda$ | Stability (↑) | Plasticity (↑) | $\bar{A}_{10}$ (↑) |
|-----------------------------------|---------------|----------------|-------------------|
| 0 (no regularization) | 87.56 | 98.86 | 82.99 |
| 10 | 92.13 | 98.12 | 86.42 |
| 100 | 94.45 | 97.99 | 87.91 |
| 1000 | 96.21 | 95.34 | 87.15 |

### Key Findings

1. **EWC-LoRA improves vanilla LoRA by 8.92% on average**, achieving state-of-the-art 87.91% on CIFAR-100.
2. **Constant memory**: Unlike InfLoRA and similar methods that require linearly growing LoRA branches, EWC-LoRA maintains only one set of LoRA weights and one diagonal Fisher matrix.
3. **Full-dimensional Fisher is critical**: Independently regularizing $\mathbf{A}$ and $\mathbf{B}$ incurs a 1.5% accuracy drop, confirming the importance of capturing their interaction.
4. **Flexible stability–plasticity trade-off**: Tuning $\lambda$ allows free movement along the entire Pareto frontier, outperforming methods constrained to a fixed operating point.
5. **Generalizes to language tasks**: The method is validated on T5-large and LLaMA-3.2-1B, demonstrating broad applicability.

## Highlights & Insights

- The first systematic study of EWC in low-rank CL, revealing the theoretical deficiencies of naive integration.
- Full-dimensional Fisher estimation elegantly exploits gradient equivalence, requiring no explicit storage of full-dimensional updates.
- Constant memory overhead independent of task count, making it suitable for long task-sequence scenarios.
- Comprehensive theoretical analysis is provided (mathematical proofs in Appendix A).

## Limitations & Future Work

- Fisher matrix estimation still operates in the full-dimensional space, imposing non-trivial memory costs for very large models (>10B parameters).
- The diagonal Fisher assumption neglects inter-parameter correlations, potentially underestimating the importance of certain parameters.
- Vision experiments are conducted solely on ViT-B/16; performance on larger backbone architectures remains unexplored.
- The choice of low-rank dimension $r$ significantly affects performance, yet no automatic selection mechanism is provided.

## Related Work & Insights

- **EWC**: Kirkpatrick et al. (2017) proposed penalizing changes to important parameters via the Fisher Information Matrix.
- **LoRA**: Hu et al. (2022) proposed low-rank adaptation for parameter-efficient fine-tuning.
- **Low-rank CL**: InfLoRA (Liang & Li, 2024), SD-LoRA (Wu et al., 2025), O-LoRA (Wang et al., 2023).
- **Prompt-based CL**: L2P, DualPrompt, CODA-Prompt.

## Rating

- Novelty: ⭐⭐⭐⭐ — A deep and insightful revisitation of a classical method in a new paradigm.
- Technical Depth: ⭐⭐⭐⭐⭐ — Theoretical proofs, empirical validation, and systematic analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both vision and language tasks across multiple benchmarks.
- Practical Value: ⭐⭐⭐⭐ — Constant memory, plug-and-play design, deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

</div>

<!-- RELATED:END -->
