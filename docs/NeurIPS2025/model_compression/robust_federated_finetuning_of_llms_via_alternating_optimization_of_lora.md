---
title: >-
  [Paper Note] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA
description: >-
  [NeurIPS 2025][Model Compression][federated learning] This paper proposes RoLoRA, which alternately optimizes the down-projection ($\mathbf{A}$) and up-projection ($\mathbf{B}$) matrices of LoRA to address imprecise aggr…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "federated learning"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "alternating optimization"
  - "LLM"
date: 2026-05-08
content_hash: 8a01701d0767a2b2
---

# Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA

**Conference**: NeurIPS 2025
**arXiv**: [2502.01755](https://arxiv.org/abs/2502.01755)
**Code**: None
**Area**: Model Compression
**Keywords**: federated learning, LoRA, parameter-efficient fine-tuning, alternating optimization, LLM

## TL;DR

This paper proposes RoLoRA, which alternately optimizes the down-projection ($\mathbf{A}$) and up-projection ($\mathbf{B}$) matrices of LoRA to address imprecise aggregation and limited expressiveness in federated learning. RoLoRA significantly outperforms FedAVG of LoRA and FFA-LoRA on RoBERTa-Large and Llama-2-7B.

## Background & Motivation

**Background**: Using LoRA for parameter-efficient fine-tuning in federated learning is a mainstream approach. LoRA decomposes weight updates as $\Delta \mathbf{W} = \alpha \mathbf{A}\mathbf{B}$, where $\mathbf{A} \in \mathbb{R}^{d \times r}$, $\mathbf{B} \in \mathbb{R}^{r \times d}$, $r \ll d$.

**Limitations of Prior Work**:
- **FedAVG of LoRA**: Directly averaging client matrices $\mathbf{A}_i$ and $\mathbf{B}_i$ introduces aggregation interference — $\frac{1}{N}\sum_i \mathbf{A}_i \mathbf{B}_i \neq \frac{1}{N}\sum_i \mathbf{A}_i \cdot \frac{1}{N}\sum_i \mathbf{B}_i$
- **FFA-LoRA**: Freezing $\mathbf{A}$ (down-projection) and only updating $\mathbf{B}$ avoids interference but sacrifices model expressiveness, leading to significant performance degradation with fewer parameters or more clients
- **FlexLoRA/FLoRA**: Recover exact updates via matrix multiplication and truncated SVD, but at substantial computational cost

**Key Challenge**: A three-way tension among exact aggregation, model expressiveness, and computational/communication efficiency.

**Goal**: Design a federated LoRA fine-tuning framework that simultaneously ensures exact aggregation, sufficient expressiveness, and low communication/computation overhead.

**Key Insight**: Inspired by multi-task linear representation learning (MLRL), the paper alternately freezes $\mathbf{A}$ and $\mathbf{B}$ — only one matrix is trained and aggregated per round, naturally guaranteeing exact aggregation.

**Core Idea**: In odd rounds, freeze $\mathbf{A}$ and update $\mathbf{B}$; in even rounds, freeze $\mathbf{B}$ and update $\mathbf{A}$. This alternation achieves both exact aggregation and full expressiveness.

## Method

### Overall Architecture

**RoLoRA Algorithm (Algorithm 1)**:
- Odd communication rounds: All clients freeze the shared $\mathbf{A}^t$ and locally train $\mathbf{B}_i^{t+1}$; the server aggregates $\mathbf{B}^{t+1} = \frac{1}{N}\sum_i \mathbf{B}_i^{t+1}$
- Even communication rounds: All clients freeze the shared $\mathbf{B}^{t+1}$ and locally train $\mathbf{A}_i^{t+1}$; the server aggregates $\mathbf{A}^{t+1} = \frac{1}{N}\sum_i \mathbf{A}_i^{t+1}$
- Since the frozen matrix is globally consistent, aggregation is inherently exact

### Key Designs

**1. Exact Aggregation Guarantee**

In odd rounds, $\mathbf{A}_i^t = \mathbf{A}^t$ is identical across all clients, so:
$$\frac{1}{N}\sum_i \mathbf{A}_i^t \mathbf{B}_i^{t+1} = \mathbf{A}^t \cdot \frac{1}{N}\sum_i \mathbf{B}_i^{t+1}$$
Aggregation is fully exact with no interference.

**2. Theoretical Analysis via Linear Models (Theorem 4.5)**

In federated linear regression ($\mathbf{Y}_i = \mathbf{X}_i \mathbf{a}^* \mathbf{b}^{*\top}$), RoLoRA achieves exponential convergence:
$$\sin\theta(\mathbf{a}^{t+1}, \mathbf{a}^*) \leq \sin\theta(\mathbf{a}^t, \mathbf{a}^*) \sqrt{1 - \eta(1 - \delta_0^2)\|\mathbf{b}^*\|^2}$$
- The angular distance decays exponentially to an arbitrarily small $\epsilon$
- FFA-LoRA's loss lower bound (Proposition 4.6) is $(1 + \tilde{c})\|\mathbf{b}^*\|^2 (\delta_0)^2$, which is bounded away from zero by the initialization angle and can never converge to zero

**3. Non-Convex Convergence Guarantee (Theorem A4.4)**

Under a smooth non-convex setting, RoLoRA achieves a convergence rate of $O(1/\sqrt{T})$, consistent with FedAVG.

### Loss & Training

- Same loss function as standard LoRA
- Trainable parameters per round are halved (only $\mathbf{A}$ or $\mathbf{B}$ is trained), and communication cost is also halved
- Learning rate selected from $\{5e^{-4}, ..., 1e^{-1}\}$

## Key Experimental Results

### Main Results: RoBERTa-Large on GLUE (50 clients, rank 4)

| Method | SST-2 | QNLI | MNLI | QQP | RTE | Avg |
|--------|-------|------|------|-----|-----|-----|
| LoRA | 93.00 | 78.13 | 52.64 | 77.60 | 52.23 | 70.72 |
| FFA-LoRA | 93.23 | 85.05 | 69.97 | 78.44 | 55.72 | 76.48 |
| FlexLoRA | 54.08 | 55.40 | 39.14 | 72.00 | 52.71 | 54.67 |
| **RoLoRA** | **94.80** | **90.00** | **82.98** | **85.71** | **75.57** | **85.81** |

Under the 50-client setting, RoLoRA surpasses LoRA by **+15.09%** and FFA-LoRA by **+9.33%** in average accuracy.

### Llama-2-7B on Commonsense (50 clients, rank 8)

| Method | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|--------|-------|------|------|-----------|------------|-------|-------|------|
| LoRA | 61.42 | 33.19 | 31.88 | 21.23 | 31.36 | 27.36 | 32.03 | 26.07 |
| FFA-LoRA | 53.43 | 35.49 | 10.63 | 11.81 | 1.61 | 6.88 | 7.93 | 15.00 |
| **RoLoRA** | **61.83** | **61.26** | **39.76** | **27.49** | **47.67** | **33.19** | **40.13** | **31.67** |

FFA-LoRA nearly collapses on large models, while RoLoRA maintains a substantial lead.

### Ablation Study

| Ablation Dimension | Key Findings |
|-------------------|--------------|
| Number of clients (3→50) | LoRA/FFA-LoRA degrade sharply; RoLoRA remains stable (3 clients: 88.28 → 50 clients: 85.81) |
| Non-IID (Dir 0.5/1.0) | RoLoRA achieves 82.60% on MNLI; LoRA: 81.19%; FlexLoRA: only 35.45% |
| Reduced parameters | FFA-LoRA degrades significantly with fewer parameters; RoLoRA remains robust |
| Symmetric vs. asymmetric updates | Balanced alternation of A and B is optimal; bias toward either degrades performance |
| Local steps (1→20) | FFA-LoRA: 72.52%→69.97% (degrades with more steps); RoLoRA: 84.39%→82.98% (stable) |

### Key Findings

- As the number of clients increases (3→20→50), the aggregation interference in FedAVG of LoRA deteriorates sharply
- FFA-LoRA's limitation stems from the quality of $\mathbf{A}$ initialization — variance across random seeds is extremely large (PIQA: std=9.55)
- Learning $\mathbf{A}$ is especially critical in the early stages of training (20% RoLoRA + 80% FFA-LoRA already significantly outperforms pure FFA-LoRA)
- RoLoRA's communication cost is only 50% of that of LoRA/FlexLoRA

## Highlights & Insights

- **Simple yet effective**: Alternating freezing is an extremely simple design that simultaneously resolves the tension between exact aggregation and expressiveness
- **Theory–practice alignment**: The linear model theory (exponential convergence vs. saturation) is perfectly validated in nonlinear experiments
- **Outstanding robustness**: Strong performance persists under extreme settings (50 clients, rank 2, non-IID)
- **Deployment-friendly**: Communication and computation costs are halved with no additional SVD operations

## Limitations & Future Work

- The linear model theory assumes homogeneous clients and a single-layer LoRA structure, which diverges from practical multi-layer LoRA setups
- Alternating optimization halves update efficiency per communication round (each of $\mathbf{A}$ and $\mathbf{B}$ is updated only half as many times given the same total rounds)
- Adaptive alternating frequencies remain unexplored (how to determine the optimal alternation ratio between $\mathbf{A}$ and $\mathbf{B}$?)
- Comparison with full-parameter fine-tuning is absent
- Integration with privacy-preserving mechanisms (e.g., differential privacy) is not discussed

## Related Work & Insights

- The connection to MLRL (multi-task low-rank representation learning) provides the theoretical foundation
- LoRA+ explores different learning rates for $\mathbf{A}$ and $\mathbf{B}$; RoLoRA's alternating strategy represents a more aggressive form of asymmetric treatment
- The approach can be extended to heterogeneous rank settings (e.g., different clients using different ranks)
- Insight: The aggregation precision problem in federated learning similarly exists in other decomposed parameter methods (e.g., adapters, prefix tuning)

## Rating

⭐⭐⭐⭐ (4/5)

The method is simple and effective, the theoretical analysis is clear, and the experiments are comprehensive with substantial performance gains. The primary limitation is the gap between theory and practice (linear vs. multi-layer nonlinear settings).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](uni-lora_one_vector_is_all_you_need.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)

</div>

<!-- RELATED:END -->
