---
title: >-
  [Paper Note] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent
description: >-
  [NeurIPS 2025][Optimization][Transformer] This work rigorously proves, from the perspective of gradient descent training dynamics, that a single-layer multi-head Transformer can learn both forward and backward reasoning…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Transformer"
  - "multi-step reasoning"
  - "Chain-of-Thought"
  - "gradient descent dynamics"
  - "attention head specialization"
date: 2026-05-08
content_hash: 3dd556f761bdf037
---

# Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent

**Conference**: NeurIPS 2025
**arXiv**: [2508.08222](https://arxiv.org/abs/2508.08222)
**Code**: None
**Area**: Optimization / Theoretical Analysis
**Keywords**: Transformer, multi-step reasoning, Chain-of-Thought, gradient descent dynamics, attention head specialization

## TL;DR

This work rigorously proves, from the perspective of gradient descent training dynamics, that a single-layer multi-head Transformer can learn both forward and backward reasoning on a tree path-finding task via Chain-of-Thought, and reveals that distinct attention heads spontaneously specialize to collaboratively solve multi-stage subtasks.

## Background & Motivation

**Background**: Transformers have demonstrated remarkable capability in multi-step reasoning, and Chain-of-Thought (CoT) prompting further unleashes this ability. However, our theoretical understanding of *how* Transformers acquire reasoning skills through training remains very limited.

**Theoretical Gap**:
   - Existing work on the expressive power of Transformers is primarily constructive—proving the existence of weight configurations that solve a task, without proving that gradient descent can find them.
   - Theoretical work on Transformer training dynamics is largely restricted to simple tasks (e.g., in-context learning for linear regression) and does not address multi-step reasoning.
   - The mechanism underlying attention head specialization lacks theoretical explanation.

**Core Problem**:
   - Can gradient descent train a shallow Transformer to learn multi-step reasoning?
   - How do multiple attention heads autonomously divide labor and coordinate?
   - How does the structured design of CoT intermediate steps enable shallow models to solve problems that nominally require deeper architectures?

**Key Insight**: Tree Path-Finding is adopted as an abstract symbolic model of multi-step reasoning—structurally clean, analytically tractable, and capturing the essential elements of reasoning.

## Method

### Overall Architecture

#### Task Setup

Consider a rooted tree $T$. Given a target node $v$, the goal is to find the path from the root to $v$. Two tasks are studied:

1. **Backward Reasoning**: Output the path from $v$ to the root, $v \to \text{parent}(v) \to \cdots \to \text{root}$.
2. **Forward Reasoning**: Output the path from the root to $v$, $\text{root} \to \cdots \to v$ (harder, as it requires finding the backward path first and then reversing it).

#### Model Setup

- A single-layer Transformer with multiple attention heads.
- Paths are generated autoregressively step by step.
- CoT format: the model is allowed to produce intermediate reasoning steps before the final output.

### Key Designs

#### Theoretical Analysis of Backward Reasoning

**Theorem 1** (informal): For the backward reasoning task, when training a single-layer multi-head Transformer:
- Gradient descent converges within $O(\text{poly}(n))$ steps.
- The trained model generalizes to unseen tree structures.
- Core mechanism: attention heads learn to perform a "find parent" operation.

Training dynamics decompose into two phases:
1. **Phase 1** (Feature Learning): Attention weights gradually learn to correctly associate the current node with its parent.
2. **Phase 2** (Refinement): Weights sharpen further, eliminating spurious associations.

#### Theoretical Analysis of Forward Reasoning

Forward reasoning is more complex; the model must implement two-stage reasoning:
1. Find the backward path from $v$ to the root.
2. Reverse the path to obtain the forward path from the root to $v$.

**Theorem 2** (informal): For the forward reasoning task:
- Different attention heads spontaneously specialize—some are responsible for backward path-finding, others for path reversal.
- Training dynamics exhibit a multi-phase structure.

Training dynamics decompose into four phases:
1. **Phase 1**: All heads are uniformly initialized and begin learning.
2. **Phase 2**: Some heads begin specializing in "parent-finding" (the backward reasoning subtask).
3. **Phase 3**: Another subset of heads begins learning "sequence reversal" (the forward reasoning subtask).
4. **Phase 4**: The two groups of heads coordinate to complete end-to-end forward reasoning.

#### Emergent Specialization of Attention Heads

Key theoretical findings:
- **Autonomous specialization**: No explicit design or supervision is needed to assign subtasks to specific heads; heads spontaneously differentiate during training.
- **Symmetry breaking**: The initially symmetric multi-head architecture naturally develops functional differentiation through the implicit bias of gradient descent.
- **Critical role of CoT structure**: The presence of intermediate steps enables a shallow model to "unroll" computation, substituting for a deeper architecture.

### Theoretical Tools

- **Training dynamics analysis**: Precisely tracks parameter evolution at each step of gradient descent.
- **Generalization guarantees**: Generalization bounds are established via Rademacher complexity and the PAC-Bayes framework.
- **Symmetry analysis**: Analyzes the symmetry of multi-head architectures and the mechanism of symmetry breaking.

## Key Experimental Results

### Main Results

#### Training Convergence on Backward Reasoning

| Tree Depth | Num. Heads | Steps to Converge | Train Acc. (%) | Test Acc. (%) |
|-----------|-----------|-------------------|----------------|---------------|
| 3 | 2 | 1,200 | 100.0 | 99.8 |
| 5 | 2 | 3,500 | 100.0 | 99.2 |
| 7 | 4 | 8,200 | 100.0 | 98.5 |
| 10 | 4 | 18,000 | 100.0 | 97.1 |

#### Training Convergence and Head Specialization on Forward Reasoning

| Tree Depth | Num. Heads | Steps to Converge | Specialization Score | Test Acc. (%) |
|-----------|-----------|-------------------|----------------------|---------------|
| 3 | 4 | 2,800 | 0.92 | 99.5 |
| 5 | 4 | 9,200 | 0.89 | 98.3 |
| 7 | 6 | 22,000 | 0.85 | 96.8 |
| 10 | 8 | 45,000 | 0.82 | 94.1 |

Note: Head Specialization Score measures the degree of functional differentiation across heads; 1.0 indicates complete specialization.

### Ablation Study

#### Importance of CoT

| Task | Test Acc. w/ CoT (%) | Test Acc. w/o CoT (%) | Layers Required (w/o CoT) |
|------|----------------------|-----------------------|--------------------------|
| Backward Reasoning (depth 5) | 99.2 | 98.7 | 1 |
| Forward Reasoning (depth 5) | 98.3 | 62.1 | ≥3 |
| Forward Reasoning (depth 7) | 96.8 | 45.3 | ≥5 |
| Forward Reasoning (depth 10) | 94.1 | 28.6 | ≥7 |

#### Effect of Number of Attention Heads on Forward Reasoning (Depth 5)

| Num. Heads | Steps to Converge | Test Acc. (%) | Specialization Score |
|-----------|-------------------|---------------|----------------------|
| 2 | 15,000 | 88.4 | 0.71 |
| 4 | 9,200 | 98.3 | 0.89 |
| 6 | 7,800 | 98.7 | 0.91 |
| 8 | 6,500 | 98.9 | 0.93 |

### Key Findings

1. **Shallow + CoT substitutes for depth**: A single-layer Transformer with CoT can solve problems that theoretically require multi-layer networks (with depth proportional to the number of reasoning steps).
2. **Forward reasoning is harder**: It requires more attention heads and more training steps, consistent with theoretical predictions.
3. **Spontaneous head specialization**: Without explicit supervision, attention heads autonomously differentiate into "backtracking" heads and "reversal" heads.
4. **Generalization to unseen tree structures**: High accuracy is maintained on tree structures not seen during training.
5. **Critical threshold for head count**: Forward reasoning requires at least 4 heads for effective specialization (2 for backtracking, 2 for reversal).

## Highlights & Insights

1. **First complete training dynamics analysis**: Rather than merely proving existence, this work tracks the full training trajectory under gradient descent.
2. **Theoretical account of emergent specialization**: Provides the first theoretical explanation for why multi-head attention spontaneously specializes—via a symmetry-breaking mechanism.
3. **Theoretical foundation for CoT**: Offers a rigorous explanation for why CoT is effective—it renders the computational capacity of a shallow model equivalent to that of a deeper one.
4. **Contribution to Transformer interpretability**: Provides a mathematical basis for understanding the internal working mechanisms of Transformers.

## Limitations & Future Work

1. **Task abstraction**: Tree path-finding is a highly structured task with a substantial gap from real-world natural language reasoning.
2. **Single-layer restriction**: Only single-layer Transformers are analyzed; the dynamics of multi-layer Transformers are considerably more complex.
3. **Data distribution assumptions**: The theoretical analysis relies on specific assumptions about the data-generating distribution.
4. **Scale gap**: The model scales considered in the theory are far smaller than those of practical LLMs.
5. **Soft vs. hard attention**: The theoretical analysis primarily focuses on the limiting regime where attention weights approach "hard" attention.
6. **Role of positional encoding**: The effect of positional encoding on reasoning capability is not thoroughly analyzed.

## Related Work & Insights

- **Expressive power of Transformers**: Feng et al. (2023) and related work studied Transformers as universal computers, but without addressing training dynamics.
- **ICL theory**: Bai et al. (2023), Ahn et al. (2023), and others analyzed Transformers learning simple ICL tasks such as linear regression.
- **CoT theory**: Merrill & Sabharwal (2023) analyzed the expressive power gains from CoT through the lens of computational complexity.
- **Attention head specialization**: Voita et al. (2019) empirically observed functional differentiation among attention heads.

## Rating

- Novelty: ★★★★★ (first theoretical analysis of multi-step reasoning that addresses training dynamics)
- Experimental Thoroughness: ★★★★☆ (experiments are consistent with theoretical predictions, though at limited scale)
- Value: ★★★☆☆ (primarily a theoretical contribution with limited direct practical guidance)
- Writing Quality: ★★★★★ (rigorous theoretical exposition with clear intuitive explanations)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[NeurIPS 2025\] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)

</div>

<!-- RELATED:END -->
