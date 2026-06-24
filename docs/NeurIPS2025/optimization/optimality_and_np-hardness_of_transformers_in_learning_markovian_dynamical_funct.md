---
title: >-
  [Paper Note] Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions
description: >-
  [NeurIPS 2025][Optimization][In-context learning] This paper analyzes the ICL capability of Transformers for learning Markovian dynamical functions from an optimization-theoretic perspective. It derives the global optimal solution (in closed form) for single-layer linear self-attention, proves that recovering Transformer parameters from an extended parameter space is NP-hard, and reveals that multi-layer LSA is equivalent to preconditioned multi-objective optimization.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "In-context learning"
  - "Markov chains"
  - "linear self-attention"
  - "NP-hard"
  - "multi-objective optimization"
date: 2026-05-08
content_hash: 1d003dccd722c193
---

# Optimality and NP-Hardness of Transformers in Learning Markovian Dynamical Functions

**Conference**: NeurIPS 2025
**arXiv**: [2510.18638](https://arxiv.org/abs/2510.18638)  
**Code**: None  
**Area**: Theoretical Machine Learning / Transformer Theory
**Keywords**: In-context learning, Markov chains, linear self-attention, NP-hard, multi-objective optimization

## TL;DR

This paper analyzes the ICL capability of Transformers for learning Markovian dynamical functions from an optimization-theoretic perspective. It derives the global optimal solution (in closed form) for single-layer linear self-attention, proves that recovering Transformer parameters from an extended parameter space is NP-hard, and reveals that multi-layer LSA is equivalent to preconditioned multi-objective optimization.

## Background & Motivation

### Limitations of Prior Work

**Background**: In-Context Learning (ICL) is one of the core capabilities of Transformers, whereby a model learns a new task from input-output demonstrations provided in the prompt. Existing theoretical work on ICL has focused primarily on **linear regression with i.i.d. Gaussian inputs**, establishing that Transformers can implicitly perform gradient descent.

However, many real-world tasks involve **structured sequential data** with temporal dependencies among inputs. This paper focuses on a more challenging setting: learning shared transition dynamics from multiple Markov chain trajectories, and predicting the next token given a new query sequence. This setting introduces two novel challenges:
1. Inputs are no longer i.i.d.; sequences exhibit internal Markovian dependency structure.
2. The input-output relationship is no longer a deterministic linear function but a stochastic transition probability.

**Core Problem**: Can Transformers effectively learn such dynamically driven functions? What are the theoretical limits of their optimization and expressive power?

## Method

### Overall Architecture

The research framework is based on the **linear self-attention (LSA) model**: the softmax nonlinearity is removed and the key/query matrices are merged into $Q = W_k^T W_q$. The input consists of $n$ binary Markov chains of length $d+1$, forming an embedding matrix $Z_0$. The objective is to predict the last token of the query sequence. The training objective is the population mean squared error loss.

### Key Designs

1. **Reparameterization and Convexification**: A mapping $\phi(b,A) = X \in \mathbb{R}^{dm}$ (where $m = \frac{(d+2)(d+1)}{2}$) is defined to project the original non-convex LSA parameter space into an extended space. In this extended space, the objective $\tilde{f}(X)$ is **strictly convex** (Lemma 3.1), and its global minimum admits a closed-form solution $X^* = H^{-1}b$ (Lemma 3.2), where the structure of $H$ is determined by powers of the Markov transition kernel.

2. **Global Optimal Solution for Length-2 Markov Chains** (Theorems 3.3, 3.4): For the simplest case of length-2 Markov chains, analytic global optimal solutions for LSA parameters $(P, Q)$ are derived. A key finding is that all parameters in the optimal solution are **non-zero**, forming a dense structure — in sharp contrast to the sparse structure observed in i.i.d. linear regression. This reflects how the temporal dependency of Markovian inputs compels the optimal solution to simultaneously encode multiple statistical relationships.

3. **NP-Hardness Proof** (Theorem 3.5): For Markov chains of general length $d+1$, while the optimal $X^*$ in the extended space can be solved analytically, recovering the original Transformer parameters $(b, A)$ from $X^*$ is **NP-hard**. The proof proceeds by reduction to the bilinear separability problem (known to be NP-complete). This exposes a fundamental expressive limitation of single-layer LSA: the optimal solution exists in the extended space but may not be realizable by a Transformer.

4. **Multi-Objective Optimization Interpretation of Multi-Layer LSA** (Theorem 4.1): The forward pass of an $L$-layer LSA is equivalent to a **preconditioned multi-objective optimization** process that simultaneously minimizes a squared loss and multiple linear objectives. Specifically, $w_{l+1}^T = w_l^T - b_l^T(\nabla R_1(w_l)\bar{A}_l + \nabla R_2(w_l)[\cdot])$, where $R_1$ contains the squared loss and state interaction terms, and $R_2$ emphasizes the role of the last parameter $w_d$.

### Loss & Training

The population loss is $f(\{P_l, Q_l\}) = \mathbb{E}[(\mathtt{TF}_L(Z_0) - y_{n+1})^2]$, where the expectation is taken over all possible Markov transition kernels and initial distributions.

## Key Experimental Results

### Main Results (Multi-layer LSA Accuracy, Varying State Space Size)

| Model Depth | $|\mathcal{S}|=2$ | $|\mathcal{S}|=3$ | $|\mathcal{S}|=4$ |
|---|---|---|---|
| 1-layer LSA | ~0.50 (random) | ~0.55 | ~0.30 |
| 2-layer LSA | ~0.85 | ~0.80 | ~0.60 |
| 3-layer LSA | ~0.95 | **0.98** | ~0.85 |

Single-layer LSA performs near random chance for all state space sizes, corroborating the theoretical prediction of NP-hardness; adding layers leads to substantial accuracy improvements.

### Ablation Study (Verification of Multi-Objective Optimization Behavior)

| Layer Range | Generational Distance (GD) | R1 Value | R2 Value | Notes |
|---|---|---|---|---|
| Early layers | Decreasing | Improving | Improving | Effectively approaching the Pareto front in early stages |
| Middle layers | Minimal | Optimal | Optimal | Multi-objective optimization most effective |
| Deep layers | Increasing | Degrading | Degrading | Degradation due to constrained parameter space |

A 10-layer LSA trained on Markovian data performs multi-objective optimization during its forward pass, validating Theorem 4.1.

### Key Findings
- Single-layer LSA cannot effectively learn Markovian dynamical functions; even when the optimal solution is known, it may not be realizable (NP-hard gap).
- Increasing depth substantially improves performance, consistent with the multi-objective optimization interpretation.
- Markovian inputs induce a dense optimal solution structure, contrasting with the sparse structure induced by i.i.d. inputs.

## Highlights & Insights
- **Deep theoretical contribution**: The NP-hardness proof reveals a fundamental expressive limitation of Transformers on structured inputs, representing a significant advance in the theory of ICL.
- **Novel multi-objective optimization perspective**: Interpreting the multi-layer Transformer forward pass as simultaneously optimizing multiple objectives (beyond squared loss alone) provides a theoretical understanding of the necessity of depth.
- **Clear i.i.d. vs. Markovian contrast**: The paper precisely characterizes how input structure shapes the mathematical form of the optimal solution.

## Limitations & Future Work
- Only linear self-attention is analyzed; practical components such as softmax attention, MLPs, and positional encodings are not considered.
- The analysis is restricted to first-order binary Markov chains; higher-order chains and large state spaces are not covered.
- A substantial gap remains between the theoretical results and practical Transformer training.
- Empirical validation on real NLP or time-series tasks is absent.

## Related Work & Insights
- **vs. Ahn et al. (2023)**: They show that single-layer LSA admits an analytic solution with a sparse structure under i.i.d. linear regression; this paper reveals a dense structure and NP-hard difficulty under Markovian inputs.
- **vs. Rajaraman et al. (2024)**: They study token-level attention on a single sequence; this paper studies sequence-level attention across multiple sequences.
- **vs. Nichani et al. (2024)**: They prove that a two-layer Transformer can recover single-parent Markov chain transition probabilities via layer-wise training; this paper focuses on optimization hardness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the NP-hardness proof and the multi-objective optimization interpretation are proposed for the first time.
- Experimental Thoroughness: ⭐⭐⭐ Experiments primarily serve to validate theory; scale and diversity are limited.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, though notation is heavy.
- Value: ⭐⭐⭐⭐ Provides an important theoretical perspective for understanding ICL mechanisms and Transformer expressive power.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](../../ICLR2026/optimization/markovian_transformers_for_informative_language_modeling.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](../../ICML2026/optimization/test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)
- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](../../ICML2025/optimization/can_transformers_learn_full_bayesian_inference_in_context.md)

</div>

<!-- RELATED:END -->
