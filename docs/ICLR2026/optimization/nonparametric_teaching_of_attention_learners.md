---
title: >-
  [Paper Note] Nonparametric Teaching of Attention Learners
description: >-
  [ICLR 2026][Optimization][nonparametric teaching] This paper proposes AtteNT — a reinterpretation of attention learner (Transformer/ViT) training through the lens of nonparametric teaching theory: it analyzes the importa…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "nonparametric teaching"
  - "attention mechanism"
  - "functional gradient"
  - "training acceleration"
  - "kernel methods"
date: 2026-05-08
content_hash: 70e7672da4bbf90d
---

# Nonparametric Teaching of Attention Learners

**Conference**: ICLR 2026
**arXiv**: [2602.20461](https://arxiv.org/abs/2602.20461)  
**Area**: Training Efficiency / Learning Theory
**Keywords**: nonparametric teaching, attention mechanism, functional gradient, training acceleration, kernel methods

## TL;DR

This paper proposes AtteNT — a reinterpretation of attention learner (Transformer/ViT) training through the lens of nonparametric teaching theory: it analyzes the importance-adaptive role of attention in parametric gradients → proves that the dynamic ANTK converges to an importance-adaptive canonical kernel in functional gradient descent → bridges parameter space and function space → applies a greedy teaching algorithm that selects samples with the largest prediction deviation to accelerate training → achieving 13.01% time savings for LLM fine-tuning and 20.58% for ViT training from scratch, with no degradation in accuracy.

## Background & Motivation

**Background**: Attention learners (Transformers, ViTs, etc.) have achieved remarkable success in NLP and CV, but their training costs are prohibitively high — LLM pre-training requires millions of sentences, and video understanding demands even larger data scales. Reducing training costs has become an urgent need.

**Limitations of Prior Work**:

1. **Limited applicability of nonparametric teaching**: Existing nonparametric teaching theory accelerates learning by selecting teaching samples, but applies only to MLP learners, without accounting for the effect of attention mechanisms.
2. **Gap between parameter space and function space**: Attention neural networks (ANNs) are trained via stochastic gradient descent (SGD) in parameter space, whereas nonparametric teaching uses functional gradient descent (FGD) in function space — the consistency between the two has never been established.
3. **How attention alters learning dynamics**: The attention mechanism invokes the input three times (Q, K, V) to assign varying importance to sequence elements; how this affects the structure of parametric gradients has not been analyzed.

**Key Challenge**: Nonparametric teaching theory has the potential to accelerate attention learner training, but a theoretical gap exists between its mathematical foundation (functional gradient descent) and the actual training procedure (parametric gradient descent), and the introduction of attention makes the extension from MLP to ANN non-trivial.

**Goal**: Systematically analyze the role of attention in parametric gradient descent → prove the consistency between parametric and functional gradient descent for ANNs → directly apply the greedy teaching algorithm (selecting samples with the largest prediction deviation) to accelerate attention learner training.

## Method

### Overall Architecture

The theoretical roadmap of AtteNT:

1. Analyze the role of attention in parametric gradient descent → discover importance-adaptive updates
2. Map the evolution in parameter space to function space via the dynamic ANTK
3. Prove that the dynamic ANTK converges to an importance-adaptive canonical kernel in functional gradient descent
4. Establish the equivalence: "teaching an ANN = teaching an importance-adaptive nonparametric learner"
5. Apply the greedy teaching algorithm to accelerate training

### Key Design 1: Importance-Adaptive Parametric Gradient Analysis of Attention

For a single-layer single-head self-attention network $f_\theta(\mathbf{S}) = \text{softmax}(\frac{\mathcal{Q}(\mathbf{S})\mathcal{K}(\mathbf{S})^\top}{\sqrt{d}})\mathcal{V}(\mathbf{S})$, the authors derive the explicit form of the parametric gradients. Taking the gradient with respect to the Query weight matrix as an example:

$$\frac{\partial f_\theta(\mathbf{S})}{\partial \mathbf{W}^Q_{(:,i)}} = \left[d^{-1/2} \mathbf{S}_{(j,:)} \cdot \omega_j\right]_{S \times d}$$

Key findings:

- The gradient depends not only on sequence element features $\mathbf{S}_{(j,:)}$, but also on an **element-specific scalar** $\omega_j$.
- $\omega_j$ is jointly determined by $\mathcal{Q}, \mathcal{K}, \mathcal{V}$ — reflecting the importance assigned to each element by attention.
- The parametric gradient **does not depend on the input sequence length** $S$ (averaged out), but only on the feature dimension $d$.
- The row order of the gradient is consistent with the order of sequence elements (equivariance) — naturally corresponding to the permutation invariance at inference time.

### Key Design 2: Consistency Between ANTK and Functional Gradient

Via Taylor expansion, the evolution of an ANN in parameter space can be expressed in function space:

$$\frac{\partial f_{\theta^t}}{\partial t} = -\frac{\eta}{NS}\left[\frac{\partial \mathcal{L}}{\partial f_{\theta^t}(\mathbf{S}_1)}, \ldots, \frac{\partial \mathcal{L}}{\partial f_{\theta^t}(\mathbf{S}_N)}\right] \cdot [K_{\theta^t}(\mathbf{S}_i, \cdot)]_N + o(\cdot)$$

where $K_{\theta^t}(\mathbf{S}_i, \cdot) \coloneqq \langle \frac{\partial f_{\theta^t}(\mathbf{S}_i)}{\partial \theta^t}, \frac{\partial f_{\theta^t}(\cdot)}{\partial \theta^t} \rangle$ is the **dynamic Attention Neural Tangent Kernel (ANTK)**.

**Theorem 3 (Core Theorem)**: Given a convex loss function $\mathcal{L}$ and a training set, the dynamic ANTK converges pointwise to the importance-adaptive canonical kernel in functional gradient descent:

$$\lim_{t \to \infty} K_{\theta^t}(\mathbf{S}_i, \cdot) = K(\mathbf{S}_i, \cdot), \quad \forall i$$

This demonstrates that the evolution of an ANN via parametric gradient descent is **consistent** with that via functional gradient descent, thereby legitimizing the application of nonparametric teaching theory to attention learners.

### Key Design 3: AtteNT Greedy Teaching Algorithm

Building on the theoretical bridge above, AtteNT selects teaching samples by maximizing the projection of the functional gradient. Since the norm of the partial derivative of a convex loss is positively correlated with prediction deviation, the selection rule simplifies to:

$$\{\mathbf{S}_i\}_m^* = \arg\max_{\{\mathbf{S}_i\}_m \subseteq \{\mathbf{S}_i\}_N} \|[f_\theta(\mathbf{S}_i) - f^*(\mathbf{S}_i)]_m\|_\mathcal{F}$$

Intuition: prioritize training on samples the model "understands least" (largest prediction deviation) → steepest gradient → fastest convergence.

**Proposition 4 (Sufficient Loss Decrease)**: Under Lipschitz smoothness and bounded kernel conditions, AtteNT guarantees a sufficient decrease in the loss function:

$$\frac{\partial \mathcal{L}}{\partial t} \leq -\frac{\eta\gamma}{2}\left(\frac{1}{NS}\sum_{i,j}\frac{\partial \mathcal{L}}{\partial f_{\theta^t}(\mathbf{S}_i)_{(j,:)}}\right)^2$$

## Key Experimental Results

### Main Results 1: LLM Fine-Tuning (NLG Tasks)

| Model | AtteNT | Avg. Time↓ | GSM8K↑ | MATH↑ | HumanEval↑ | MBPP↑ | MT-Bench↑ |
|------|--------|---------|--------|-------|-----------|-------|----------|
| LLaMA 2-7B | w/o | 246m | 42.96 | 5.06 | 18.35 | 35.65 | 4.58 |
| LLaMA 2-7B | **w** | **213m** | **43.45** | **6.48** | **21.80** | **37.61** | 4.49 |
| Mistral-7B | w/o | 204m | 69.13 | 20.06 | 43.42 | 58.52 | 5.03 |
| Mistral-7B | **w** | **180m** | **71.26** | **23.12** | **46.55** | **61.74** | **5.32** |
| Gemma-7B | w/o | 228m | 75.23 | 30.52 | 53.83 | 65.69 | 5.42 |
| Gemma-7B | **w** | **201m** | **77.74** | **31.40** | **54.26** | **66.28** | **5.44** |

AtteNT reduces training time by an average of 12.78%, while simultaneously improving performance by 1.39–2.42 points on GSM8K, 0.76–2.89 points on MATH, 0.29–3.66% on HumanEval, and 2.08–3.31% on MBPP.

### Main Results 2: ViT Training from Scratch (CV Tasks)

| Model | AtteNT | Pre-training Time↓ | ImageNetS50↑ | NYUv2(S)↑ | NYUv2(D)↑ |
|------|--------|-----------|-------------|----------|----------|
| Multi-Modal MAE | w/o | 1234m | 92.2 | 51.9 | 52.1 |
| Multi-Modal MAE | **w** | **980m (−20.58%)** | **92.3** | **52.6** | **57.2 (+5.1%)** |

Training time is reduced by 20.58%, with performance improvements across all downstream tasks, and the largest gain observed on depth estimation (+5.1%).

### Ablation Study: Data Selection Strategies

| Ratio Strategy | Interval Strategy | Selection Strategy | Training Time | ImageNetS50 | NYUv2(S) | NYUv2(D) |
|----------|-------------|-------------|---------|-------------|----------|----------|
| — | — | — (Standard) | 1234m | 92.2 | 51.9 | 52.1 |
| Cosine | Incremental | Random | 966m | 88.6 | 45.3 | 49.6 |
| Cosine | Incremental | Hard | 972m | 91.8 | 49.5 | 57.3 |
| **Incremental** | **Incremental** | **Soft** | **980m** | **92.3** | **52.6** | **57.2** |
| Incremental | Fixed | Soft | 1319m | 92.4 | 53.7 | 62.1 |

The Soft strategy (Gumbel-Top-k probabilistic sampling) achieves the best balance between time and performance: Random selection disrupts data distribution and degrades accuracy; Hard selection is overly deterministic and lacks robustness; Fixed intervals yield the highest accuracy but double the training time.

## Highlights & Insights

- **Theoretical elegance of "nonparametric teaching → training acceleration"**: Rather than heuristic data selection, the approach is grounded in a complete theoretical framework involving RKHS, functional gradients, and kernel convergence — providing a principled explanation of *why* it works.
- **Theoretical contribution of ANTK**: NTK (Neural Tangent Kernel) was originally developed for fully connected networks; ANTK extends it to attention networks — a significant expansion of an important theoretical tool.
- **Consistency with pedagogical intuition of "teach what is least understood"**: Prioritizing difficult samples → easy samples are naturally learned → aligns with curriculum learning, but with stronger theoretical guarantees.
- **A "free lunch" of 13–21% acceleration without accuracy loss**: Achieving equal or superior performance with less data — nonparametric teaching theory provides principled guidance for data selection.

## Limitations & Future Work

- Theoretical analysis focuses on single-layer single-head self-attention; extension to multi-layer multi-head architectures is posited as a direct generalization but has not been fully proven.
- Evaluating prediction deviation across all data at the start of each epoch introduces selection overhead (though overall training time is still reduced).
- Validation on very large-scale pre-training (e.g., GPT-scale) has not been conducted.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First theoretical framework for nonparametric teaching of attention learners
- Experimental Thoroughness: ⭐⭐⭐⭐ NLP + CV, from-scratch / fine-tuning, multiple models, ablation studies
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, tightly coupled with empirical validation
- Value: ⭐⭐⭐⭐ Dual theoretical and practical contributions to Transformer training efficiency

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parametrized Multi-Agent Routing via Deep Attention Models](../../AAAI2026/optimization/parametrized_multi-agent_routing_via_deep_attention_models.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICLR 2026\] Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [\[ICLR 2026\] Personalized Collaborative Learning with Affinity-Based Variance Reduction](personalized_collaborative_learning_with_affinity-based_variance_reduction.md)

</div>

<!-- RELATED:END -->
