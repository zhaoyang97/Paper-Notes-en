---
title: >-
  [Paper Note] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting
description: >-
  [ACL 2025][Optimization][bilevel optimization] ScaleBiO proposes a fully first-order bilevel optimization algorithm based on penalty function reformulation, applying bilevel optimization to data source reweighting for 30B+ parameter LLMs for the first time, achieving improvements of +9% on GSM8K and +5.8% on MATH for Qwen-2.5-32B.
tags:
  - "ACL 2025"
  - "Optimization"
  - "bilevel optimization"
  - "data reweighting"
  - "LLM training"
  - "scalability"
  - "first-order method"
date: 2026-05-08
content_hash: 3cb5059ba8a29d8e
---

# ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting

**Conference**: ACL 2025  
**arXiv**: [2406.19976](https://arxiv.org/abs/2406.19976)  
**Code**: None  
**Area**: Optimization  
**Keywords**: bilevel optimization, data reweighting, LLM training, scalability, first-order method

## TL;DR
ScaleBiO proposes a fully first-order bilevel optimization algorithm based on penalty function reformulation, applying bilevel optimization to data source reweighting for 30B+ parameter LLMs for the first time, achieving improvements of +9% on GSM8K and +5.8% on MATH for Qwen-2.5-32B.

## Background & Motivation

### Background

**Background**: Data quality significantly affects LLM performance, but finding the optimal weights for different data sources is difficult.

**Limitations of Prior Work**: Classical bilevel optimization requires second-order information (Hessian/Jacobian), which cannot scale to models beyond 3B parameters.

**Key Challenge**: Mathematically, bilevel optimization is the optimal framework for data reweighting, but its computational complexity limits practical application.

**Core Idea**: Reformulate the bilevel problem as a minimax problem using a penalty function, decoupling inner and outer dependencies to achieve a fully first-order optimization.

### Proposed Approach

**Goal**: ### Overall Architecture
The bilevel optimization formula: $\min_\lambda L_1(\lambda, w^*(\lambda))$ s.t. $w^*(\lambda) = \arg\min_w \sum_i (p_i/n_i) \sum_j L_2(w, a_j^i)$, where $\lambda$ represents the data source weights, $L_1$ is the validation loss (outer level), and $L_2$ is the training loss (inner level).


## Method

### Overall Architecture
The bilevel optimization formula: $\min_\lambda L_1(\lambda, w^*(\lambda))$ s.t. $w^*(\lambda) = \arg\min_w \sum_i (p_i/n_i) \sum_j L_2(w, a_j^i)$, where $\lambda$ represents the data source weights, $L_1$ is the validation loss (outer level), and $L_2$ is the training loss (inner level).

### Key Designs

1. **Penalty Function Reformulation**: Reformulated into $\min_{\lambda,w} \max_u L_1(\lambda,w) + \alpha(L_2(\lambda,w) - L_2(\lambda,u))$, where $\alpha$ decouples the dependencies between inner and outer levels, requiring only first-order gradients.
2. **Stochastic Block Coordinate Descent**: Only selected parameter blocks are updated in each step. Combined with LISA to update only the top-k essential layers, this supports the training of 32B models on 8×H100 GPUs.
3. **Convergence Guarantee**: Offers a convergence rate of $O(\epsilon^{-7/2})$, which is consistent with the state-of-the-art theoretical bound.

## Key Experimental Results

### Main Results (Instruction Following, MT-Bench)

| Method | Llama-3-8B | Qwen-2-7B | Gemma-2-9B |
|------|-----------|-----------|------------|
| Uniform | 6.11% | 6.66% | 5.31% |
| LESS | 6.06% | 7.18% | 7.20% |
| RHO-LOSS | 6.89% | 7.34% | 7.38% |
| **ScaleBiO** | **7.12%** | **7.76%** | **7.51%** |

### 30B Model Results (Qwen-2.5-32B)

| Method | GSM8K | MATH |
|------|-------|------|
| Uniform | 78.1% | 54.0% |
| **ScaleBiO** | **87.1 (+9.0)** | **59.8 (+5.8)** |

The only bilevel optimization method successfully scaled to 32B.

### Key Findings
- **Automated Data Source Discovery**: Alpaca-GPT4 (10% of the data) is automatically assigned a weight of over 30, correctly identifying high-quality data.
- **Scalability Breakthrough**: Breaks the 3B parameter barrier for the first time, scaling up to 32B.
- **Cross-Model Variance**: The optimal weights learned by different models vary significantly, reflecting differences in their pre-training data.

## Highlights & Insights
- **Penalty function reformulation** is the key innovation: shifting bilevel optimization from "nested" to "joint", avoiding second-order information. This approach can be generalized to other bilevel optimization scenarios.
- The cross-scale validation **from GPT-2 (124M) to Qwen-2.5 (32B)** is highly convincing.

## Limitations & Future Work
- Not validated in large-scale pre-training; evaluated only on instruction tuning / fine-tuning.
- May introduce bias if the validation set is not representative of the target distribution.
- A single loss metric may neglect other important aspects (e.g., safety, alignment).

## Related Work & Insights
- **vs LESS**: LESS uses data influence functions to select subsets without continuous weight optimization; ScaleBiO achieves more fine-grained results by learning continuous weights.
- **vs RHO-LOSS**: RHO-LOSS scores samples using a reference model and requires logistic regression fitting; ScaleBiO offers a more direct end-to-end optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ Restructuring with a penalty function to achieve large-scale bilevel optimization is a significant breakthrough
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Cross-scale validation from GPT-2 to 32B across multiple tasks and models
- Writing Quality: ⭐⭐⭐⭐ Strong combination of theory and practice
- Value: ⭐⭐⭐⭐⭐ Holds high practical value for LLM training data optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[ICML 2025\] Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](../../ICML2025/optimization/efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)
- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](../../NeurIPS2025/optimization/set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
