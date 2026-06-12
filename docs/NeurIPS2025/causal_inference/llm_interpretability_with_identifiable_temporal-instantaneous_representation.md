---
title: >-
  [Paper Note] LLM Interpretability with Identifiable Temporal-Instantaneous Representation
description: >-
  [NeurIPS 2025][Causal Inference][LLM interpretability] This paper proposes an identifiable temporal causal representation learning framework for the high-dimensional activation spaces of LLMs. By adopting a linearized fo…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "LLM interpretability"
  - "causal representation learning"
  - "sparse autoencoders"
  - "temporal causality"
  - "identifiability"
date: 2026-05-08
content_hash: a592a700cbba03f4
---

# LLM Interpretability with Identifiable Temporal-Instantaneous Representation

**Conference**: NeurIPS 2025
**arXiv**: [2509.23323](https://arxiv.org/abs/2509.23323)  
**Code**: None  
**Area**: Causal Inference
**Keywords**: LLM interpretability, causal representation learning, sparse autoencoders, temporal causality, identifiability

## TL;DR

This paper proposes an identifiable temporal causal representation learning framework for the high-dimensional activation spaces of LLMs. By adopting a linearized formulation that jointly models time-lagged and instantaneous causal relationships, it resolves the computational bottleneck that prevents existing CRL methods from scaling to LLM-scale dimensions, while preserving theoretical identifiability guarantees.

## Background & Motivation

### 1. State of the Field

Mechanistic Interpretability (MI) is an important research direction for understanding the internal representations of LLMs. Sparse Autoencoders (SAEs) have become the mainstream tool for extracting interpretable features from LLM activations, decomposing high-dimensional activations into sparse, monosemantic features. Teams such as Anthropic have successfully scaled SAEs to large models, enabling automatic interpretation of millions of features.

### 2. Limitations of Prior Work

Existing SAEs suffer from three key limitations:
- **No temporal dependency modeling**: each feature is treated as an isolated representation, failing to capture causal influences between features across sequence positions (e.g., how a concept in a preceding token affects subsequent tokens).
- **No instantaneous relationship representation**: there is no mechanism to express logical relationships between features within the same time step (e.g., mutual exclusivity or co-occurrence constraints).
- **No theoretical guarantees**: there are no guarantees on the uniqueness of recovered features, meaning extracted features may correspond to arbitrary or unstable transformations.

### 3. Root Cause

The Causal Representation Learning (CRL) community has proposed theoretically grounded frameworks, but these face a severe **scalability bottleneck**: existing methods rely on Jacobian computation, whose time and memory complexity grows super-linearly with dimensionality. When the dimensionality exceeds 1,000, a single Jacobian evaluation requires approximately 10 seconds, while CRL training requires millions of such evaluations—restricting existing methods to dozens to hundreds of latent variables, far short of the thousands to tens of thousands of conceptual features needed for LLMs.

### 4. Paper Goals

To design a computationally efficient temporal causal representation learning framework that scales to the high-dimensional concept spaces of LLMs (thousands to tens of thousands of dimensions), while retaining theoretical identifiability guarantees and capturing both time-lagged and instantaneous causal relationships.

### 5. Starting Point

The paper exploits the **linear representation hypothesis**—that LLM activations are linear generations of underlying causal concepts. Under the linear assumption, the Jacobian reduces to the model parameters themselves, bypassing the computational bottleneck of nonparametric CRL methods. Identifiability is established via autocovariance structure and non-Gaussianity, rather than relying on a "sufficient variability" assumption.

### 6. Core Idea

Replace nonparametric CRL with a linear temporal causal model, and achieve Jacobian-free efficient training through sparsity constraints on parameter matrices, enabling an order-of-magnitude scaling from tens of dimensions to thousands.

## Method

### Overall Architecture

The data-generating process consists of two layers:

1. **Linear mixing layer**: observed activations $x_t = A \cdot z_t$, where $A$ is the linear mixing matrix.
2. **Linear latent variable temporal SEM**: $z_{t,i} = \sum_\tau \sum_j B_{i,j,\tau} \cdot z_{t-\tau,j} + \sum_j M_{i,j} \cdot z_{t,j} + \varepsilon_{t,i}$

where $B_{i,j,\tau}$ encodes time-lagged causal relationships (the causal effect of $z_j$ on $z_i$ at lag $\tau$), $M_{i,j}$ encodes instantaneous causal relationships (the causal effect of $z_j$ on $z_i$ within the same time step), and $\varepsilon_{t,i}$ is spatiotemporally independent noise. $M$ is constrained to be strictly lower triangular, ensuring that the instantaneous causal graph $G_e$ is a DAG.

### Key Designs

#### Module 1: Theoretical Identifiability Guarantee (Theorem 3)

- **Function**: Proves that under four assumptions, model parameters are identifiable up to a signed permutation.
- **Mechanism**: The observed autocovariance matrices $R_x(k)$ are used to establish a Yule–Walker-type recursion, from which the mixed parameters $C_\tau = A(I-M)^{-1}B_\tau A^{-1}$ and $HH^T$ are recovered; non-Gaussianity is then used to constrain the remaining orthogonal ambiguity to a signed permutation.
- **Four assumptions**: A1 (temporally white noise with independent components), A2 ($A$ has full column rank; $B_L$ has full rank), A3 (process stability), A4 (at most one component is Gaussian).
- **Design Motivation**: Existing nonparametric proofs (e.g., the sufficient variability assumption in IDOL) do not hold in the linear case—the coefficient matrix has rank at most $n$ and cannot be full-rank in the presence of instantaneous relationships—necessitating an entirely new proof strategy.

#### Module 2: Component-Level and Subspace-Level Identifiability Corollaries

- **Corollary 1 (Component-level identifiability)**: If each column of $M$ has either an empty support or a unique support element (assumption A5), then $z_t$ is identifiable up to permutation and scaling under sparsity constraints.
- **Corollary 2 (Subspace identifiability)**: If $M$ has a block-diagonal structure (assumption A6), then $z_t$ is identifiable at the subspace level.
- **Design Motivation**: In Theorem 3, the instantaneous relationship $M \neq 0$ expands the ambiguity from orthogonal matrices to general invertible matrices. Imposing sparse structural constraints recovers stronger identifiability.

#### Module 3: Observation Reconstruction (Section 4.1)

- **Function**: A linear autoencoder implements an invertible linear transformation between the observed vector $x_t$ and the latent variable $z_t$.
- **Mechanism**: $L_r = \mathbb{E}[\sum(x_t - \hat{x}_t)^2]$, where $\hat{x}_t = \text{Decoder}(\text{Encoder}(x_t))$.
- **Design Motivation**: The linear mixing assumption directly corresponds to the encode–decode structure of SAEs.

#### Module 4: Independent Noise Estimation (Section 4.2)

- **Function**: Estimates the independent noise $\hat{\varepsilon}_t = \hat{z}_t - \hat{M} \cdot \hat{z}_t - \sum_\tau \hat{B}_\tau \cdot \hat{z}_{t-\tau}$ by inverting the data-generating process.
- **Mechanism**: Noise is modeled as an isotropic Laplace distribution (rather than Gaussian), and KL divergence is minimized.
- **Design Motivation**: In the linear case, the density of an isotropic Gaussian distribution is rotation-invariant (as noted in the linear ICA literature), making it impossible to distinguish rotational transformations; a Laplace distribution must therefore be used.

### Loss & Training

The total loss function consists of three terms:

$$L_\text{total} = L_r + \alpha \cdot L_n + \beta \cdot L_s$$

- $L_r$ **(reconstruction loss)**: mean squared error between $x_t$ and $\hat{x}_t$.
- $L_n$ **(noise independence loss)**: $\ell_1$ norm of the estimated noise $\hat{\varepsilon}_t$ (corresponding to the negative log-likelihood of a Laplace distribution).
- $L_s$ **(sparsity regularization)**: $\sum_\tau \|\hat{B}_\tau\|_1 + \|\hat{M}\|_1$, imposing $\ell_1$ sparsity constraints on both the time-lagged and instantaneous relationship matrices.

A key training constraint is that $M$ is restricted to be strictly lower triangular, so as to match the permutation ambiguity on both sides.

## Key Experimental Results

### Main Results

#### Synthetic Data — Scalability Comparison

| Method | Max Tractable Dim. | Training Time at 1024-d | MCC at 1024-d |
|--------|--------------------|------------------------|---------------|
| iCITRIS | 16 | OOM | N/A |
| IDOL | 200 | OOM | N/A |
| Ours | **1024+** | ~50h | **≈0.9** |

#### Real LLM Activations — SAEBench Quantitative Evaluation

| Model | Recon. Loss ↓ | Sparse Prob. ↑ | Absorp. ↓ | Autointerp ↑ |
|-------|--------------|----------------|-----------|--------------|
| ReLU SAE | 0.0110 | 0.6555 | 0.0141 | 0.6791 |
| TopK SAE | 0.0097 | 0.7141 | 0.0280 | 0.6822 |
| Ours | 0.0108 | 0.6736 | 0.0139 | **0.6883** |

#### Semi-Synthetic Data — Relationship Recovery Score

| Method | Legal | XML | Email |
|--------|-------|-----|-------|
| SAE+regression | 0.54 | 0.94 | 0.74 |
| Ours | **19.95** | **8.63** | **2.66** |

### Ablation Study

#### Jacobian Computation Bottleneck Analysis

| Input Dim. | IDOL Single-Step Jacobian Time | IDOL Single-Step Jacobian Memory |
|-----------|-------------------------------|----------------------------------|
| 100 | ~0.1s | Acceptable |
| 500 | ~2s | Near limit |
| 1000 | ~10s | Exceeds GPU capacity |

#### Linear Model Scalability

From 128 to 1024 dimensions, MCC consistently remains at approximately 0.9, with computation time growing linearly.

### Key Findings

1. **Existing CRL methods are fundamentally incapable of handling LLM-scale dimensionality**: IDOL runs out of memory beyond 200 dimensions, and iCITRIS fails beyond 16 dimensions.
2. **The proposed method performs comparably to standard SAEs on SAEBench**: it matches baseline metrics on concept recovery while additionally providing causal relational structure.
3. **Relationship recovery scores far exceed baselines**: results are 37× higher on Legal text, demonstrating the method's effectiveness in recovering temporal causal relationships between concepts.
4. **Case studies confirm the meaningfulness of both relationship types**: time-lagged relationships such as "appeals→affirmed" (procedural flow in legal text), and instantaneous relationships such as the co-activation of two geographic location concepts.

## Highlights & Insights

1. **Precisely identifies the gap between CRL and MI**: CRL offers theory but cannot scale; SAEs scale but lack theory. The paper elegantly bridges the two via the linear assumption.
2. **Solid theoretical contributions**: independently of existing nonparametric proofs (e.g., IDOL), the paper establishes a novel identifiability theorem for the linear case, and derives component-level and subspace-level identifiability through structured assumptions.
3. **Strategically designed experiments**: the semi-synthetic experiment (comparing legal text against unstructured text) cleverly demonstrates the method's ability to discover domain-specific temporal patterns.
4. **Recovered relationships are intuitively interpretable**: examples such as "nationality adjective → modified noun" and "month of date ⇔ full date" reveal genuine conceptual organization within LLMs.

## Limitations & Future Work

1. **Limitations of the linear assumption**: LLM internals are inherently nonlinear (attention mechanisms, activation functions); the linear model is an approximation. Although the paper cites empirical support for the linear representation hypothesis, nonlinear extensions are a necessary next step.
2. **Single-layer analysis**: only activations from a single layer are analyzed; cross-layer concept transformations are not modeled, even though information processing in LLMs is progressively refined across layers.
3. **Evaluation difficulties**: there is no ground-truth causal structure for real LLMs, so evaluation relies on case studies and indirect metrics, lacking direct causal validation.
4. **Limited model scale**: primary experiments are conducted on Pythia-160M; performance on larger models (e.g., 7B, 70B) remains to be verified.
5. **DAG assumption for instantaneous relationships**: the instantaneous causal graph is required to be a DAG, but LLM internals may involve cyclic dependencies.

## Related Work & Insights

- **SAE series (Anthropic, Cunningham et al.)**: The encoder–decoder structure directly inherits from SAEs; the core improvement is the addition of a temporal SEM module to capture inter-feature relationships.
- **IDOL (Li et al., 2025)**: Appendix A.1 provides a detailed argument for why IDOL's nonparametric proof fails in the linear case (insufficient rank of the coefficient matrix), motivating the development of a new proof strategy.
- **Linear ICA / SSM (Zhang, 2011)**: The proof strategy for Theorem 3 is inspired by this line of work, extending autocovariance-based methods to temporal processes with instantaneous relationships.
- **Attribution Graphs / Sparse Feature Circuits**: These methods infer time-lagged influences through attention scores but lack identifiability guarantees under an independent noise assumption.
- **Insight**: CRL under the linear assumption can serve as a foundation for nonlinear methods—first capturing the primary causal structure with a linear model, then progressively introducing nonlinear corrections.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of CRL and SAEs is novel, and the linearization strategy for scalability is concise and effective; however, the linear assumption itself limits generality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic data validate the theory, semi-synthetic data validate relationship recovery, and real-data case studies provide qualitative evidence; however, large-scale model evaluation and ground-truth verification are absent.
- Writing Quality: ⭐⭐⭐⭐ — The paper is well-structured with a clear theory–algorithm–experiment hierarchy, though the dense notation requires repeated cross-referencing.
- Value: ⭐⭐⭐⭐ — Provides a new causal perspective and theoretical guarantees for LLM interpretability; practical applicability depends on the scope of validity of the linear assumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[ICLR 2026\] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](../../ICLR2026/causal_inference/validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)
- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](../../ACL2026/causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)

</div>

<!-- RELATED:END -->
