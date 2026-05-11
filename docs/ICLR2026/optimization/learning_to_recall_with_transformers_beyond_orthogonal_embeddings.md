---
title: >-
  [Paper Note] Learning to Recall with Transformers Beyond Orthogonal Embeddings
description: >-
  [ICLR 2026][Optimization][Transformer] This paper analyzes the "early phase" of empirical gradient descent for a single-layer Transformer on a token retrieval task under random (non-orthogonal) embeddings. It derives an explicit formula for storage capacity, revealing a multiplicative dependence among sample size $N$, embedding dimension $d$, and sequence length $L$, and proves that this scaling relation is intrinsic to the information-theoretic lower bound.
tags:
  - ICLR 2026
  - Optimization
  - Transformer
  - "Memory & Retrieval"
  - Storage Capacity
  - Non-Orthogonal Embeddings
  - Gradient Descent Analysis
date: 2026-05-08
content_hash: 65d73e4f9b76c477
---

# Learning to Recall with Transformers Beyond Orthogonal Embeddings

**Conference**: ICLR 2026
**arXiv**: [2603.15923](https://arxiv.org/abs/2603.15923)
**Code**: None
**Area**: Transformer Theory / Optimization Theory
**Keywords**: Transformer, Memory & Retrieval, Storage Capacity, Non-Orthogonal Embeddings, Gradient Descent Analysis

## TL;DR

This paper analyzes the "early phase" of empirical gradient descent for a single-layer Transformer on a token retrieval task under random (non-orthogonal) embeddings. It derives an explicit formula for storage capacity, revealing a multiplicative dependence among sample size $N$, embedding dimension $d$, and sequence length $L$, and proves that this scaling relation is intrinsic to the information-theoretic lower bound.

## Background & Motivation

### Root Cause

**Background**: Large language models (LLMs) excel at tasks requiring knowledge storage and retrieval, such as factual recall and question answering. The Transformer architecture is central to this capability, encoding information during training and retrieving it at inference time.

Understanding how Transformers learn memorization and retrieval patterns is an important direction in deep learning theory. Existing theoretical analyses are largely conducted under the following idealized assumptions:

**Infinite data assumption**: Analysis proceeds under the population gradient, ignoring finite-sample effects.

**Orthogonal embedding assumption**: Token embedding vectors are assumed to be mutually orthogonal, which approximates reality only when the embedding dimension $d$ greatly exceeds the vocabulary size—a condition that does not hold in practice.

In realistic settings, models are trained via **empirical gradient descent** on **finite datasets** with **randomly initialized, non-orthogonal embeddings**. Non-orthogonality introduces inter-token interference, fundamentally altering the learning dynamics and the scaling behavior of storage capacity.

**Goal**: To precisely analyze the memory-retrieval capability of Transformers under these more realistic conditions.

## Method

### Overall Architecture

The paper establishes a concise yet representative theoretical model:
- **Architecture**: Single-layer Transformer (with one attention head)
- **Task**: Token retrieval—identifying an informative token within a sequence of length $L$ and learning a one-to-one mapping from tokens to labels
- **Embeddings**: Random (non-orthogonal) vectors of dimension $d$
- **Training**: Empirical gradient descent on $N$ finite samples

### Key Designs

1. **Formalization of the Token Retrieval Task**:

   Given a token sequence of length $L$ containing exactly one "informative token," the model must:
   - Identify and select this token via the attention mechanism
   - Learn the mapping from the token to its corresponding label

   This task captures the core computational structure of factual retrieval in LLMs: locating relevant information in context and producing the correct output.

2. **Early-Phase Gradient Descent Analysis**:

   Rather than pursuing global convergence results, the paper precisely characterizes the "early phase" of gradient descent—the evolution of the model over the first few iterations from initialization. This phase is typically decisive for whether the model successfully learns, as the "signal directions" established early on are amplified in subsequent training.

   By tracking the evolution of key statistics (e.g., signal-to-noise ratio of attention weights, direction of the value matrix) during the early phase, the paper derives conditions required for successful learning.

3. **Explicit Formula for Storage Capacity**:

   The central result is an explicit scaling relation for storage capacity, revealing a **multiplicative dependence** among three key quantities:

   The condition for successful retrieval roughly requires that $N \cdot d \cdot L$ satisfies a specific scaling relation.

   Specifically:
   - **Sample size $N$**: Sufficiently many samples are needed to learn the token-label mapping.
   - **Embedding dimension $d$**: Higher dimensionality reduces inter-token interference and increases storage capacity.
   - **Sequence length $L$**: Longer sequences increase the difficulty of attention-based selection.

   The multiplicative coupling among the three is a direct consequence of non-orthogonal embeddings—under orthogonal embeddings, these factors can be analyzed independently.

4. **Information-Theoretic Lower Bound**:

   Beyond the upper bound (algorithmic perspective), the paper also establishes an inherent hardness lower bound from a statistical/information-theoretic perspective.

   The results demonstrate that the multiplicative scaling among $N$, $d$, and $L$ is not a limitation of the algorithm but an **intrinsic property** of the problem itself. Any method—whether Transformer-based or otherwise—cannot circumvent this scaling bottleneck under non-orthogonal embeddings.

### Theoretical Tools

- **Random Matrix Theory**: Analyzes the spectral properties of non-orthogonal random embedding matrices and inter-token interference.
- **High-Dimensional Probability**: Handles concentration inequalities for empirical gradients under finite samples.
- **Information Theory**: Establishes statistical lower bounds and proves the optimality of the scaling relations.
- **Dynamical Systems Analysis**: Tracks the evolution of key statistics across gradient descent iterations.

## Key Experimental Results

### Main Results: Validation of Storage Capacity Scaling

The paper verifies the theoretically predicted scaling relations through numerical experiments:

| Dimension $d$ | Sequence Length $L$ | Theoretical Critical $N$ | Observed Critical $N$ | Agreement |
|---|---|---|---|---|
| Small $d$ | Small $L$ | Low | Consistent with theory | ✓ |
| Small $d$ | Large $L$ | High | Consistent with theory | ✓ |
| Large $d$ | Small $L$ | Low | Consistent with theory | ✓ |
| Large $d$ | Large $L$ | Moderate | Consistent with theory | ✓ |

### Ablation Study: Orthogonal vs. Non-Orthogonal Embeddings

| Embedding Type | Storage Capacity Scaling | Remarks |
|---|---|---|
| Orthogonal embeddings | $N$ scales independently with $d$ and $L$ | Classical setting; factors are separable |
| Random (non-orthogonal) embeddings | $N$, $d$, $L$ are multiplicatively coupled | More realistic setting; factors are inseparable |

### Lower Bound Verification

| Setting | Algorithmic Upper Bound (Transformer + GD) | Information-Theoretic Lower Bound | Gap |
|---|---|---|---|
| Non-orthogonal embeddings | $O(f(N,d,L))$ | $\Omega(g(N,d,L))$ | Tight (same order) |

### Key Findings

1. **Multiplicative scaling is intrinsic**: The coupling $N \cdot d \cdot L$ arises from inter-token interference induced by non-orthogonal embeddings, not from algorithmic shortcomings.
2. **The orthogonality assumption leads to overoptimism**: Capacity derived under orthogonal assumptions overestimates the true capacity.
3. **The early phase is critical**: The first few steps of gradient descent determine whether attention successfully locks onto the correct informative token.
4. **Embedding dimension $d$ combats interference**: Increasing the embedding dimension effectively reduces the interference caused by non-orthogonality.
5. **Dual effect of sequence length $L$**: Longer sequences provide richer context but also enlarge the search space for attention-based selection.

## Highlights & Insights

- **Bridges a critical gap between theory and practice**: Relaxing the orthogonal embedding and infinite-data assumptions yields an analysis that more faithfully reflects how real LLMs operate.
- **Elegance of the multiplicative scaling relation**: A single concise formula unifies three seemingly independent factors—data volume, embedding dimension, and sequence length.
- **Significance of the information-theoretic lower bound**: The analysis characterizes not only what Transformers can achieve, but also what no method can achieve.
- **Implications for practical LLM design**: Under a fixed computational budget, there exist optimal trade-offs among increasing embedding dimension, adding training data, and shortening the context window.
- **Elevates the understanding of Transformer "memorization capacity" from empirical intuition to rigorous theory**.

## Limitations & Future Work

1. **Single-layer, single-head Transformer only**: Real LLMs are multi-layer and multi-head; inter-layer interactions and multi-head cooperation may alter capacity scaling.
2. **Early-phase analysis**: Global convergence behavior of the full training trajectory is not covered; later phases may exhibit different dynamics.
3. **Simplified token retrieval task**: Real LLM tasks are far more complex than single-token retrieval, involving composition and reasoning.
4. **Random embedding assumption**: In practice, embeddings are learned and exhibit specific structure (e.g., low-rank, clustered) rather than being uniformly random.
5. **Positional encoding not discussed**: Positional encodings alter the effective embedding structure in attention computation.

## Related Work & Insights

- **Connection to Bietti & Cabannes (2024)**: The latter analyzes similar retrieval tasks under orthogonal embeddings; this work generalizes the setting to non-orthogonal embeddings.
- **Relation to Ahn et al. (2024)**: The latter analyzes in-context learning in linear Transformers, focusing on different aspects.
- **Analogy to associative memory (Hopfield Networks)**: The classical storage capacity analysis (e.g., the $0.14N$ pattern capacity bound) and its Transformer counterpart.
- **Implications for KV Cache design**: The storage capacity scaling relations suggest theoretical limits for KV cache compression.
- **Theoretical grounding for RAG systems**: The core of retrieval-augmented generation is precisely "locating relevant information in context."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Analysis under non-orthogonal embeddings fills an important theoretical gap)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Numerical validation is adequate, though confined to the theoretical setup)
- Writing Quality: ⭐⭐⭐⭐ (Theoretically rigorous with good clarity)
- Value: ⭐⭐⭐⭐ (Makes an important contribution to understanding Transformer memorization capacity)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](../../CVPR2026/optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)
- [\[ICLR 2026\] The Affine Divergence: Aligning Activation Updates Beyond Normalisation](the_affine_divergence_aligning_activation_updates_beyond_normalisation.md)
- [\[AAAI 2026\] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)
- [\[ICLR 2026\] Πnet: Optimizing Hard-Constrained Neural Networks with Orthogonal Projection Layers](pinet_optimizing_hard-constrained_neural_networks_with_orthogonal_projection_lay.md)

</div>

<!-- RELATED:END -->
