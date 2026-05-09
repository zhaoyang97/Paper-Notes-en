---
title: >-
  [Paper Note] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models
description: >-
  [NeurIPS 2025][Model Compression][LoRA] RefLoRA selects the optimal low-rank factorization form at each iteration by minimizing an upper bound on the loss, thereby addressing the weight update inconsistency and imbalance caused by the non-uniqueness of the LoRA decomposition. It accelerates convergence and improves fine-tuning performance with negligible additional computational overhead.
tags:
  - NeurIPS 2025
  - Model Compression
  - LoRA
  - parameter-efficient fine-tuning
  - low-rank decomposition
  - matrix refactorization
  - large language models
date: 2026-05-08
content_hash: 3751c9f397dd6cc6
---

# RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models

**Conference**: NeurIPS 2025  
**arXiv**: [2505.18877](https://arxiv.org/abs/2505.18877)  
**Code**: [zhangyilang/RefLoRA](https://github.com/zhangyilang/RefLoRA)  
**Area**: Model Compression  
**Keywords**: LoRA, parameter-efficient fine-tuning, low-rank decomposition, matrix refactorization, large language models  

## TL;DR

RefLoRA selects the optimal low-rank factorization form at each iteration by minimizing an upper bound on the loss, thereby addressing the weight update inconsistency and imbalance caused by the non-uniqueness of the LoRA decomposition. It accelerates convergence and improves fine-tuning performance with negligible additional computational overhead.

## Background & Motivation

1. **High cost of fine-tuning large models**: With parameter counts ranging from billions to trillions, full fine-tuning of LLMs imposes prohibitive demands on GPU memory and compute, making it inaccessible to most users and organizations.
2. **LoRA is efficient but has a performance gap**: LoRA assumes the weight increment is a low-rank matrix $\Delta W = AB^\top$, greatly reducing the number of trainable parameters, yet a notable performance gap relative to full fine-tuning remains.
3. **Non-uniqueness of low-rank decomposition causes problems**: Given $AB^\top$, infinitely many equivalent factorizations $(AP, BP^{-\top})$ exist; different factorizations yield different weight updates $\Delta W$, leading to training inconsistency.
4. **Initialization causes factor imbalance**: The standard LoRA initialization $A_0 \sim \mathcal{N}(0, \sigma^2)$, $B_0 = 0$ results in zero gradient for $A$ and non-zero gradient for $B$ at the first step, causing severe factor and gradient imbalance that slows early convergence.
5. **Existing improvements are too costly**: LoRA-Pro requires $O(m^2r)$ time complexity; LoRA-RITE demands custom gradient computation and moment estimation, making implementation complex and expensive.
6. **No principled, optimization-guided selection of the optimal decomposition**: Prior work recognizes the non-uniqueness issue but does not systematically seek the optimal factorization form from an optimization perspective.

## Method

### Core Idea

The core of RefLoRA is to refactorize the current low-rank factors $(A_t, B_t)$ into an optimal form $(\tilde{A}_t, \tilde{B}_t)$ before each optimization step, such that the resulting weight update $\Delta \tilde{W}_t$ minimizes an upper bound on the loss.

### Characterization of the Decomposition Space

Under the full-rank assumption $\text{rank}(A_t) = \text{rank}(B_t) = r$, all equivalent factorizations can be parameterized as:

$$(\tilde{A}_t, \tilde{B}_t) = (A_t P_t, B_t P_t^{-\top}), \quad P_t \in GL(r)$$

The weight update $\Delta \tilde{W}_t$ depends only on the symmetric positive definite matrix $S_t := P_t P_t^\top \in \mathbb{S}_{++}^r$, and is independent of the right singular vectors of $P_t$.

### Loss Upper Bound Minimization

Using the Lipschitz smoothness assumption, the loss is upper-bounded by a quadratic expansion that is further relaxed to separate the $\nabla \ell(W_t)$ factor, yielding an optimization objective depending solely on $S_t$:

$$\min_{S_t \in \mathbb{S}_{++}^r} \left( \|A_t S_t^{1/2}\|_F^2 + \|B_t S_t^{-1/2}\|_F^2 - \frac{1}{L\eta} \right)^2$$

### Closed-Form Global Optimal Solution

**Theorem 3** gives the global optimum $S_t^* = \tilde{S}_t$ (when the learning rate $\eta$ is not too small), where $\tilde{S}_t$ is the matrix geometric mean:

$$\tilde{S}_t = (A_t^\top A_t)^{-1/2} \left[ (A_t^\top A_t)^{1/2} B_t^\top B_t (A_t^\top A_t)^{1/2} \right]^{1/2} (A_t^\top A_t)^{-1/2}$$

### Compatibility with Adaptive Optimizers

To support optimizers such as Adam, RefLoRA performs a "reverse refactorization" after the gradient step, expressing the update equivalently as gradient descent with a preconditioner:

$$A_{t+1} = A_t - \eta \nabla \ell(W_t) B_t \tilde{S}_t^{-1}, \quad B_{t+1} = B_t - \eta \nabla \ell(W_t)^\top A_t \tilde{S}_t$$

This eliminates the need to transform moment estimators, enabling direct use of standard Adam.

### RefLoRA-S: Simplified Variant

By restricting $S_t$ to a scalar $s_t I_r$, the optimal scalar is $s_t^* = \|B_t\|_F / \|A_t\|_F$, reducing time complexity to $O((m+n)r)$ and space overhead to $O(1)$.

### Key Theoretical Properties

- **Balance**: $\tilde{A}_t^\top \tilde{A}_t = \tilde{B}_t^\top \tilde{B}_t$, eliminating factor imbalance.
- **Consistency** (Theorem 6): For any equivalent factorization $(A_t', B_t')$, RefLoRA's weight update satisfies $\Delta \tilde{W}_t' = \Delta \tilde{W}_t$.
- **Riemannian optimization perspective**: RefLoRA is equivalent to steepest descent on a quotient manifold under a specific Riemannian metric.

## Key Experimental Results

### GLUE Natural Language Understanding (DeBERTaV3-base, r=8)

| Method | Params | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE | Avg |
|--------|--------|------|-------|------|-------|-----|------|------|-----|-----|
| LoRA | 1.33M | 69.82 | 94.95 | 89.95 | 91.60 | 91.99 | 90.65 | 93.87 | 85.20 | 88.50 |
| DoRA | 1.33M | 70.85 | 95.79 | 90.93 | 91.79 | 92.07 | 90.29 | 94.10 | 86.04 | 88.98 |
| AdaLoRA | 1.27M | 71.45 | 96.10 | 90.69 | 91.84 | 92.23 | 90.76 | 94.55 | 88.09 | 89.46 |
| LoRA-RITE | 1.33M | 69.55 | 95.41 | 90.93 | 91.79 | 92.02 | 90.22 | 94.42 | 85.20 | 88.69 |
| **RefLoRA** | 1.33M | **71.73** | 95.99 | **91.42** | **92.03** | 92.28 | 90.23 | 94.40 | **88.09** | **89.52** |

### Commonsense Reasoning (LLaMA series, r=32)

| Model | Method | BoolQ | PIQA | SIQA | HS | WG | ARCe | ARCc | OBQA | Avg |
|-------|--------|-------|------|------|-----|-----|------|------|------|-----|
| LLaMA-7B | DoRA | 69.7 | 83.4 | 78.6 | 87.2 | 81.0 | 81.9 | 66.2 | 79.2 | 78.4 |
| LLaMA-7B | **RefLoRA** | 69.60 | 82.48 | **79.53** | **88.25** | **82.56** | 81.57 | **66.64** | 80.20 | **78.85** |
| LLaMA2-7B | LoRA-RITE | 71.04 | 82.43 | 79.79 | 89.12 | 84.53 | 83.88 | 68.77 | 81.20 | 80.10 |
| LLaMA2-7B | **RefLoRA** | **72.54** | 83.79 | **80.04** | 86.94 | **84.85** | **86.36** | **71.50** | 80.20 | **80.78** |
| LLaMA3-8B | LoRA-RITE | 74.19 | 89.44 | 81.52 | 95.44 | 86.74 | 90.45 | 80.12 | 86.60 | 85.56 |
| LLaMA3-8B | **RefLoRA** | **75.35** | 88.74 | 80.91 | **95.71** | 86.66 | 90.49 | 80.20 | **87.40** | **85.68** |

### Computational Overhead Comparison (relative to LoRA baseline)

| Method | Throughput | Extra Memory |
|--------|-----------|--------------|
| LoRA-Pro | 60.2% | +134 MB |
| LoRA-RITE | 72.6% | +140 MB |
| **RefLoRA** | **88.5%** | +132 MB |
| **RefLoRA-S** | **98.7%** | **<1 MB** |

## Highlights & Insights

1. **Closed-form global optimal solution**: Unlike methods requiring iterative inner optimization, RefLoRA admits an analytic solution for the optimal refactorization, avoiding inner optimization loops.
2. **Minimal additional overhead**: RefLoRA-S requires only $O((m+n)r)$ time and $O(1)$ space, with throughput virtually unaffected (98.7%).
3. **Theoretical and empirical consistency**: The theoretical analysis of flat loss landscapes is empirically validated—RefLoRA supports larger learning rates and exhibits more stable convergence.
4. **Multimodal applicability**: Effective not only on NLU and reasoning tasks, but also achieving a 14% reduction in loss on Stable Diffusion image generation fine-tuning.
5. **Riemannian geometry perspective**: Connects LoRA refactorization to steepest descent on a quotient manifold, providing a new theoretical framework for PEFT methods.

## Limitations & Future Work

1. **Dependence on the full-rank assumption**: Assumption 1 requires $A_t, B_t$ to have full column rank, which is violated by the standard LoRA initialization $B_0 = 0$; SVD-based initialization such as PiSSA is required.
2. **Validation limited to language and image generation**: Broader architectures such as vision Transformers and speech models have not been evaluated.
3. **No theoretical convergence rate analysis**: The paper acknowledges the absence of a convergence rate proof for RefLoRA, relying solely on empirical observations.
4. **RefLoRA-S does not guarantee consistency**: The simplified scalar variant cannot guarantee the consistency property of Theorem 6, weakening theoretical guarantees.
5. **Combination with other PEFT techniques**: The interaction of RefLoRA with adapter tuning, prompt tuning, and other PEFT methods remains unexplored.

## Related Work & Insights

- **vs DoRA**: DoRA decomposes weights into magnitude and direction components as a heuristic design; RefLoRA is a principled, optimization-driven approach with no additional parameters.
- **vs LoRA-Pro**: LoRA-Pro aligns gradients with the full fine-tuning direction at $O(m^2r)$ complexity; RefLoRA achieves a similar effect via refactorization at substantially lower complexity.
- **vs LoRA-RITE**: LoRA-RITE requires custom gradient computation and moment estimation with complex implementation; RefLoRA follows standard backpropagation and only modifies the preconditioner.
- **vs PiSSA**: PiSSA addresses early imbalance through SVD-based initialization; RefLoRA dynamically maintains balance at every step throughout training.
- **vs ScaledGD (PrecLoRA/NoRA+)**: ScaledGD is a special case of RefLoRA-S; the matrix variant of RefLoRA is strictly more expressive.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically addresses LoRA decomposition non-uniqueness from an optimization perspective; the closed-form solution is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers NLU, reasoning, and image generation with comprehensive ablation and overhead analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clear; the logical flow from problem formulation to solution is smooth.
- Value: ⭐⭐⭐⭐ Highly practical—improves LoRA performance with negligible overhead; code is publicly available.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)

<!-- RELATED:END -->
