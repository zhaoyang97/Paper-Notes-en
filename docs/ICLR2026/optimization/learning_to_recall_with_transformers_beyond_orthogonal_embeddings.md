---
title: >-
  [Paper Note] Learning to Recall with Transformers Beyond Orthogonal Embeddings
description: >-
  [ICLR 2026][Optimization & Theory][Transformer] This work analyzes the "early stage" of empirical gradient descent for a single-layer Transformer on a token retrieval task under random (non-orthogonal) embedding conditions. It derives an explicit formula for the model's storage capacity, revealing a multiplicative dependence between the sample size $N$, embedding di
tags:
  - ICLR 2026
  - Optimization & Theory
  - Transformer
date: 2026-05-08
content_hash: 4611df08eb93d07a
---
# Learning to Recall with Transformers Beyond Orthogonal Embeddings

**Conference**: ICLR 2026  
**arXiv**: [2603.15923](https://arxiv.org/abs/2603.15923)  
**Code**: None  
**Area**: Transformer Theory / Optimization Theory  
**Keywords**: Transformer, Memory and Retrieval, Storage Capacity, Non-orthogonal Embeddings, Gradient Descent Analysis

## TL;DR

This work analyzes the "early stage" of empirical gradient descent for a single-layer Transformer on a token retrieval task under random (non-orthogonal) embedding conditions. It derives an explicit formula for the model's storage capacity, revealing a multiplicative dependence between the sample size $N$, embedding dimension $d$, and sequence length $L$, and proves that this scaling relationship is inherent to information-theoretic lower bounds.

## Background & Motivation

Large Language Models (LLMs) excel at tasks requiring the storage and retrieval of knowledge, such as factual recall and question answering. The core of this capability lies in the Transformer's ability to encode information into parameters during training and retrieve it during inference. Understanding how it "learns to remember and retrieve" is thus a critical problem in deep learning theory.

However, most existing theoretical analyses rely on two ideal assumptions detached from reality. The first is **infinite data**: analysis is conducted under the population gradient, ignoring statistical fluctuations from finite samples. The second is **orthogonal embeddings**: assuming token embedding vectors are pairwise orthogonal—which only holds approximately when the embedding dimension $d$ is much larger than the vocabulary size $V$. In practical models, $d < V$, and embeddings are random non-orthogonal vectors. More subtly, existing work (NLB25) points out that strictly orthogonal embeddings are not capacity-optimal; rather, random (non-orthogonal) embeddings can approach optimal actual storage capacity via "superposition."

The problem is that once the orthogonality assumption is discarded, random embeddings introduce interference between tokens, complicating the optimization trajectory and fundamentally changing how storage capacity scales with various factors. This paper aims to precisely characterize the optimization and sample complexity of a single-layer Transformer learning a factual retrieval task under the more realistic setting of "finite samples + non-orthogonal random embeddings + empirical gradient descent."

## Method

### Overall Architecture

The paper deconstructs "how Transformers remember and retrieve knowledge" into a minimal but representative synthetic task: given a sequence of length $L$, exactly one position $\ell$ contains an "information token," while the rest are noise tokens. The label is obtained by applying a fixed permutation matrix $\Pi^*$ to the information token ($p = \Pi^* x_\ell$). The model must perform two steps: first, use attention to **locate** $\ell$ from $L$ candidates, then learn the **one-to-one mapping** from that token to the label. These steps are handled by a single-layer Transformer: the self-attention head (parameters $W_{KQ}$, utilizing a trigger vector $z_{\text{trig}}$ marking the information token and a sequence-end vector $z_{\text{EOS}}$) is responsible for localization; the subsequent value matrix $V$ (optionally stacked with a fixed random MLP of width $m$) outputs the label via associative memory. The embedding dimension is $d < V$, with random non-orthogonal embeddings.

To analytically track the learning process, the paper focuses on the first few steps of empirical gradient descent rather than global convergence—a **three-step training algorithm**: starting from zero initialization, it updates the value matrix $V$, then the attention $W_{KQ}$, and finally refines $V$, using the empirical gradient of $N$ finite samples at each step. The goal of the analysis is to derive the joint requirements on vocabulary size $V$, sample size $N$, embedding dimension $d$, sequence length $L$, and MLP width $m$ from the evolution of these steps, and prove via a statistical lower bound that these requirements cannot be bypassed by any algorithm accessing only initial gradient information.

### Key Designs

**1. Token Retrieval Task and Single-Layer Architecture: Abstracting Factual Recall into "Localization + Association"**

To analyze memory behavior, the task is compressed into two essential steps. In a sequence of length $L$, only position $\ell$ hides the information token, with the label being its result under a fixed permutation $\Pi^*$. The model segments accordingly: the self-attention head uses $z_{\text{trig}}$ to mark the info token and $z_{\text{EOS}}$ as a query; softmax attention concentrates weights on $\ell$ to complete **contextual localization**. Subsequently, the value matrix $V$ (Attention-only model) or "$V$ + fixed random MLP" (Attention-MLP model, using width $m$ to maintain capacity at small $d$) maps the selected token to the correct label, completing **content output**. This structure reflects the core computation of factual retrieval in LLMs. The realistic complexity arises from $d < V$ and non-orthogonal embeddings—the primary challenge addressed.

**2. Three-Step Early-Stage Training Algorithm: Calculating the Decisive Initial Steps**

Non-orthogonal embeddings cause complex behaviors like oscillations in full training trajectories. Following ORST23, this work characterizes the "early stage" starting from initialization, compressing training into three gradient updates: parameters initialized at $V^{(0)}=0, W_{KQ}^{(0)}=0$, then

$$V^{(1)} = -\eta \cdot \nabla_V \hat{L}, \quad W_{KQ}^{(1)} = -\gamma \cdot \nabla_{W_{KQ}} \hat{L}, \quad V^{(2)} = V^{(1)} - \gamma \cdot \nabla_V \hat{L}$$

where $\hat{L}$ is the empirical cross-entropy on $N$ samples. Focusing on initial steps is justified because signal directions (attention locking and value mapping) are established early and amplified later. Concentration inequalities for high-dimensional probability are used to bound the deviation of empirical gradients from population gradients, translating "learning success" into explicit conditions on key statistics.

**3. Multiplicative Storage Capacity Formula: Coupling of $(V, N, d, L, m)$ and Phase Diagram**

The core conclusion is a **multiplicative** scaling relationship: success depends on how $(V, N, d, L, m)$ couple multiplicatively rather than independent thresholds. $N, d, m$ facilitate learning, while $V, L$ increase difficulty. The paper provides a phase diagram partitioning the required parameter scale $m \cdot d$ based on dominant noise terms—mean bias, gradient noise, and MLP noise. The multiplicative nature stems from non-orthogonality: interference between random embeddings entangles the effects of data, dimension, and context length. This leads to a trade-off: decreasing $d$ enhances superposition and capacity but increases learning difficulty (requiring larger $N$).

**4. Matching Information-Theoretic Lower Bound: The Multiplicative Bottleneck is Inherent**

The paper provides a statistical lower bound for the inherent difficulty of the problem: for any estimator that **only accesses gradient information of the initialized Transformer**, the multiplicative trade-off holds and is of the same order as the upper bound. This proves the scaling is an intrinsic property of the task under non-orthogonal embeddings, not a limitation of the specific training algorithm or architecture.

## Key Experimental Results

### Main Results: Storage Capacity Scaling Verification

Numerical experiments verify the theoretical scaling relationship:

| Dimension d | Sequence Length L | Predicted Critical N | Observed Critical N | Match |
|-------------|-------------------|----------------------|---------------------|-------|
| Small d     | Small L           | Low                  | Consistent          | ✓     |
| Small d     | Large L           | High                 | Consistent          | ✓     |
| Large d     | Small L           | Low                  | Consistent          | ✓     |
| Large d     | Large L           | Moderate             | Consistent          | ✓     |

### Ablation Study: Orthogonal vs. Non-orthogonal Embeddings

| Embedding Type | Storage Capacity Scaling | Description |
|----------------|--------------------------|-------------|
| Orthogonal     | N scales independently with d, L | Classical setting; factors separable |
| Random (Non-ortho) | Multiplicative coupling of N, d, L | Realistic setting; factors inseparable |

### Lower Bound Verification

| Setting | Algorithmic Upper Bound (Transformer+GD) | Information-Theoretic Lower Bound | Gap |
|---------|------------------------------------------|-----------------------------------|-----|
| Non-ortho | $O(f(N, d, L))$                        | $\Omega(g(N, d, L))$              | Tight (Same order) |

### Key Findings

1. **Multiplicative scaling is inherent**: The coupling of $(V, N, d, L, m)$ arises from token interference due to non-orthogonality.
2. **Orthogonality assumption leads to over-optimism**: Capacities derived under orthogonal assumptions overestimate real-world capacity.
3. **Early stage is critical**: The first few gradient updates determine whether attention can lock onto the correct information token.
4. **$d$ is a double-edged sword**: Increasing $d$ reduces interference but decreasing $d$ enhances superposition capacity; a trade-off exists between capacity and learnability.
5. **$V$ and $L$ increase difficulty**: Larger vocabularies and longer sequences require more samples or dimensions to compensate for interference.

## Highlights & Insights

- **Bridging the gap between theory and practice**: Relaxing orthogonality and infinite data assumptions aligns the analysis with how real LLMs function.
- **Elegance of multiplicative scaling**: A concise formula unifies data quantity, dimension, and sequence length.
- **Importance of information-theoretic lower bounds**: Defines the fundamental limits for any method, not just Transformers.
- **Implications for LLM design**: Suggests an optimal trade-off between embedding dimension, training data, and context window under a fixed budget.
- **Elevating memory capacity from intuition to theory**: Provides a rigorous framework for understanding Transformer "memorization."

## Limitations & Future Work

1. **Single-layer, single-head analysis**: Practical LLMs use multi-layer, multi-head architectures where interactions may change scaling.
2. **Early-stage focus**: Does not cover global convergence behaviors or late-stage dynamics.
3. **Simplified task**: Real LLM tasks involve complex reasoning and composition beyond simple token retrieval.
4. **Random embedding assumption**: Practical embeddings are learned and often have specific structures (low-rank, clusters) rather than being purely random.
5. **Positional encoding**: The impact of positional encodings on embedding structures was not discussed.

## Related Work & Insights

- **Connection to Bietti & Cabannes (2024)**: Generalizes their retrieval task analysis from orthogonal to non-orthogonal settings.
- **Relationship to Ahn et al. (2024)**: Complements their analysis of linear Transformers in context learning.
- **Analogy to Associative Memory (Hopfield Networks)**: Map classical storage capacity limits (e.g., $0.14N$ patterns) to Transformer parameters.
- **Insight for KV Cache Design**: Scaling relationships suggest theoretical limits for KV cache compression.
- **Support for RAG systems**: Provides a theoretical foundation for "finding relevant information in context."

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](../../CVPR2025/optimization/scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[ICLR 2026\] Neural Sum-of-Squares: Certifying the Nonnegativity of Polynomials with Transformers](neural_sum-of-squares_certifying_the_nonnegativity_of_polynomials_with_transform.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)
- [\[ICLR 2026\] Cutting the Skip: Training Residual-Free Transformers](cutting_the_skip_training_residual-free_transformers.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)

</div>

<!-- RELATED:END -->
