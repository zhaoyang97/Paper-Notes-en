---
title: >-
  [Paper Note] EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph
description: >-
  [ICLR 2026][Reinforcement Learning][Symbolic Regression] This paper proposes Egg-SR, a unified framework that embeds symbolic equivalence via equality graphs (e-graphs) into three categories of symbolic regression methods—MCTS, DRL, and LLM—achieving subtree pruning, policy gradient variance reduction, and feedback prompt enrichment, respectively. Theoretical results prove that Egg-MCTS tightens the regret bound and Egg-DRL reduces gradient estimation variance, while experiments consistently validate improved expression discovery accuracy.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Symbolic Regression
  - Symbolic Equivalence
  - Equality Graph
  - Monte Carlo Tree Search
  - Deep Reinforcement Learning
date: 2026-05-08
content_hash: 50a130ebe5ff3b55
---

# EGG-SR: Embedding Symbolic Equivalence into Symbolic Regression via Equality Graph

**Conference**: ICLR 2026
**arXiv**: [2511.05849](https://arxiv.org/abs/2511.05849)
**Code**: [Project Page](https://nan-jiang-group.github.io/egg-sr)
**Area**: Reinforcement Learning
**Keywords**: Symbolic Regression, Symbolic Equivalence, Equality Graph, Monte Carlo Tree Search, Deep Reinforcement Learning

## TL;DR

This paper proposes Egg-SR, a unified framework that embeds symbolic equivalence via equality graphs (e-graphs) into three categories of symbolic regression methods—MCTS, DRL, and LLM—achieving subtree pruning, policy gradient variance reduction, and feedback prompt enrichment, respectively. Theoretical results prove that Egg-MCTS tightens the regret bound and Egg-DRL reduces gradient estimation variance, while experiments consistently validate improved expression discovery accuracy.

## Background & Motivation

Symbolic Regression (SR) aims to discover closed-form physical laws from experimental data and is a key task in AI-driven scientific discovery. The search space grows exponentially with data dimensionality, posing significant computational challenges.

A promising but underexplored direction is **symbolic equivalence**: many syntactically distinct expressions define the same function. For example:
- $\log(x_1^2 x_2^3)$
- $\log(x_1^2) + \log(x_2^3)$
- $2\log(x_1) + 3\log(x_2)$

These three expressions are mathematically identical, yet existing algorithms treat them as distinct outputs, leading to:

**Redundant exploration**: MCTS independently searches equivalent subtrees, wasting computation.

**Slow training**: DRL cannot share reward signals across equivalent sequences, resulting in high gradient variance.

**Insufficient feedback**: LLMs observe only a single expression form and miss information from equivalent variants.

The core challenge is: how to represent symbolically equivalent expressions in a unified and scalable manner and embed them into modern learning frameworks? Explicitly enumerating equivalent variants is infeasible, as their count grows exponentially with expression length.

## Method

### Overall Architecture

The Egg-SR framework consists of a core module **Egg** (Equality graph for Grammar-based expressions) and three integration interfaces:

1. **Egg-MCTS**: Shares search statistics across equivalent paths and prunes redundant subtrees.
2. **Egg-DRL**: Aggregates probabilities over equivalent sequences to reduce policy gradient variance.
3. **Egg-LLM**: Enriches feedback prompts with equivalent expressions to guide subsequent predictions.

### Key Design 1: The Egg Module — Grammar-Based Equality Graph

**Expression Representation**: Symbolic expressions are represented using a context-free grammar $\langle V, \Sigma, R, S \rangle$, where $V=\{A\}$ denotes non-terminals, $\Sigma$ denotes terminals (variables and constants), and $R$ denotes production rules. Expressions are constructed by sequentially applying rules to the leftmost non-terminal.

**E-graph Data Structure**:
- Composed of equivalence classes (e-classes), each containing a set of equivalent e-nodes.
- Each e-node encodes a mathematical operation and references child e-classes.
- Common subexpressions are shared, enabling compact representation.

**Construction Process (Equality Saturation)**:
1. Initialize the e-graph with the production sequence of the input expression.
2. For each rewrite rule (e.g., $\log(ab) \leadsto \log(a) + \log(b)$), match the LHS pattern.
3. Create new e-classes and e-nodes corresponding to the RHS at matched locations.
4. Merge equivalent e-classes.
5. Iterate until saturation or a maximum number of iterations is reached.

**Extraction Strategies**: (1) Cost-based extraction — selects the most simplified expression; (2) Random walk sampling — generates a batch of equivalent expressions.

### Key Design 2: Egg-MCTS — Equivalence-Aware Backpropagation

Standard MCTS consists of four steps: selection (UCT) → expansion → simulation → backpropagation. Egg-MCTS modifies the **backpropagation** step:

- Converts the current path (partially completed expression) into an e-graph and saturates it.
- Samples multiple equivalent sequences from the saturated e-graph.
- Checks whether corresponding paths exist in the search tree.
- **Simultaneously updates** the reward and visit counts of all equivalent paths.

This changes the semantics of the UCT formula: $\mathtt{visits}(s)$ no longer counts visits to a specific node, but rather visits to **any representative of the equivalence class**, analogous to a transposition table.

**Example**: Path 1 corresponds to $\log(x_1 \times A)$ and Path 2 to $\log(x_1) + \log(A)$; the two are equivalent under $\log(ab) \leadsto \log(a)+\log(b)$. Standard MCTS explores both subtrees independently, whereas Egg-MCTS shares statistics to avoid redundancy.

### Key Design 3: Egg-DRL — Equivalence-Aware Policy Gradient

Standard DRL uses a sequence decoder to sample production sequences $\tau$, with the policy gradient estimator:

$$g(\theta) \approx \frac{1}{N}\sum_{i=1}^N (\mathtt{reward}(\tau_i) - b) \nabla_\theta \log p_\theta(\tau_i)$$

Egg-DRL improvement: for each sampled sequence $\tau_i$, an e-graph is constructed and $K-1$ equivalent sequences are sampled, yielding the equivalence-aware estimator:

$$g_\mathtt{egg}(\theta) \approx \frac{1}{N}\sum_{i=1}^N (\mathtt{reward}(\tau_i) - b') \nabla_\theta \log\left[\sum_{k=1}^K p_\theta(\tau_i^{(k)})\right]$$

The key modification replaces the single-sequence probability with the sum of probabilities over equivalent sequences $\sum_k p_\theta(\tau_i^{(k)})$. Since equivalent sequences share the same reward, averaging over sequences with equal rewards reduces within-group variability.

### Key Design 4: Egg-LLM — Equivalence-Augmented Feedback Prompts

The iterative LLM-based SR pipeline proceeds as: hypothesis generation → data evaluation → experience management. Egg-LLM intervenes at the feedback stage by parsing generated Python expressions into symbolic form, converting them into e-graphs to extract equivalent variants, and incorporating these variants into the feedback prompt, allowing the LLM to observe richer functional equivalences.

### Theoretical Guarantees

**Theorem 3.1 (Egg-MCTS Regret Bound)**: Let $\kappa$ denote the near-optimal branching factor of standard MCTS and $\kappa_\infty$ that of Egg-MCTS; then $\kappa_\infty \leq \kappa$, and the regret bound tightens from $\tilde{\mathcal{O}}(n^{-\frac{\log(1/\gamma)}{\log\kappa}})$ to $\tilde{\mathcal{O}}(n^{-\frac{\log(1/\gamma)}{\log\kappa_\infty}})$.

**Theorem 3.2 (Egg-DRL Variance Reduction)**: The Egg-DRL gradient estimator is unbiased and its variance is strictly no greater than that of standard DRL: $\mathbb{V}\mathrm{ar}[g_\mathtt{egg}(\theta)] \leq \mathbb{V}\mathrm{ar}[g(\theta)]$.

## Key Experimental Results

### Main Results: Trigonometric Dataset NMSE

| Method | (2,1,1) Noiseless | (3,2,2) Noiseless | (4,4,6) Noiseless | (2,1,1) Noisy | (3,2,2) Noisy |
|------|--------------|--------------|--------------|--------------|--------------|
| MCTS | 0.006 | 0.033 | 0.144 | 0.015 | 0.007 |
| **Egg-MCTS** | **<1E-6** | **<1E-6** | **0.006** | **0.005** | **0.012** |
| DRL | 0.030 | 0.277 | 2.990 | 0.09 | 0.44 |
| **Egg-DRL** | **0.020** | **0.161** | **2.381** | **0.07** | **0.35** |

Egg-MCTS reduces NMSE from 0.006/0.033 to <1E-6 on simple noiseless cases, an improvement of several orders of magnitude.

### Main Results: Scientific Benchmark LLM Comparison

| Method | Oscillation I IID | Oscillation II OOD | Bacterial OOD | Stress-Strain OOD |
|------|-----------------|-------------------|-------------|-----------------|
| LLM-SR (GPT3.5) | <1E-6 | 3.81E-5 | 0.0264 | 0.0516 |
| **Egg-LLM (GPT3.5)** | <1E-6 | **<1E-6** | **0.0198** | **0.0419** |
| LLM-SR (Mistral) | <1E-6 | 0.0291 | 0.0037 | 0.0946 |
| **Egg-LLM (Mistral)** | **<1E-6** | **0.0114** | **0.0107** | **0.0754** |

### Efficiency Analysis

**Space Efficiency**: For $\log(x_1 \times \cdots \times x_n)$, there are $2^{n-1}$ equivalent variants; array-based storage grows exponentially, whereas e-graphs achieve significant compression through subexpression sharing.

**Time Efficiency**: The overhead introduced by the Egg module is negligible; compared to the cost of coefficient fitting and parameter updates in DRL, e-graph construction and equivalent sequence sampling account for a minimal fraction of total runtime.

### Key Findings

- Egg consistently improves performance across all three algorithm classes (MCTS/DRL/LLM), validating the framework's universality.
- Egg-MCTS maintains broader and deeper search trees, exploring a larger and more diverse search space.
- The gradient estimation variance of Egg-DRL decreases substantially during training (Figure 3, right).
- Trigonometric expressions benefit most due to the abundance of equivalent variants (trigonometric identities).

## Highlights & Insights

1. **Unified Framework**: A single Egg module serves all three method classes—MCTS, DRL, and LLM—through a consistent interface.
2. **Dual Validation via Theory and Experiment**: Theorems 3.1 and 3.2 formally prove MCTS regret tightening and DRL variance reduction, respectively, and are fully corroborated by experiments.
3. The core advantage of e-graphs lies in **shared subexpressions** enabling exponential compression; at $n=8$ variables, the memory advantage exceeds 100×.
4. Equivalences not captured by e-graph rewriting (e.g., coefficient equivalence in SymNet) point to interesting open problems.

## Limitations & Future Work

1. Rewrite rule sets require manual definition; current coverage includes logarithmic and trigonometric identities, and extension to more complex mathematical identities remains future work.
2. For expressions lacking rich equivalent variants (e.g., pure polynomials), the benefit of Egg is limited.
3. Only grammar-based expression representations are currently supported; layered representations based on SymNet require further extension.
4. How to optimally leverage Egg at inference time (e.g., in LLMs) remains an open question.
5. The termination condition (maximum iteration count) for e-graph saturation requires careful tuning.

## Related Work & Insights

- E-graphs originate from program synthesis and compiler optimization; this work is the first to systematically embed them into the learning algorithms of symbolic regression.
- Unlike the use of e-graphs in GP for deduplication and simplification, Egg-SR exploits equivalent variants for equivalence-aware learning.
- Sharing statistics across equivalent paths in MCTS is analogous to transposition tables in game playing, but requires pattern matching for symbolic equivalence.
- The idea of probability aggregation in DRL generalizes to any sequence generation task with output equivalence.
- Knowledge-guided scientific discovery is an important direction in SR; Egg-SR is orthogonal to symmetry constraints and unit constraints as in AI-Feynman.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Unified framework + theoretical guarantees; first systematic embedding of symbolic equivalence)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers MCTS/DRL/LLM with efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Well-crafted figures; tight theory–experiment integration)
- Value: ⭐⭐⭐⭐ (Substantial advancement for SR; generalizable to other equivalence settings)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)
- [\[NeurIPS 2025\] Distribution Learning Meets Graph Structure Sampling](../../NeurIPS2025/reinforcement_learning/distribution_learning_meets_graph_structure_sampling.md)
- [\[AAAI 2026\] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](../../AAAI2026/reinforcement_learning/tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](../../NeurIPS2025/reinforcement_learning/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[ICLR 2026\] Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning](echo_towards_advanced_audio_comprehension_via_audio-interleaved_reasoning.md)

<!-- RELATED:END -->
