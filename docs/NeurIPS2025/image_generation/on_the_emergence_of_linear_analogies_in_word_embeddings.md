---
title: >-
  [Paper Note] On the Emergence of Linear Analogies in Word Embeddings
description: >-
  [NeurIPS 2025][Image Generation][Word Embeddings] A generative model based on binary semantic attributes is proposed to analytically prove the emergence mechanism of linear analogy structures in word embeddings (e.g., $W_{\text{king}} - W_{\text{man}} + W_{\text{woman}} \approx W_{\text{queen}}$), providing a unified explanation for four key empirical observations.
tags:
  - NeurIPS 2025
  - Image Generation
  - Word Embeddings
  - Linear Analogies
  - Word2Vec
  - PMI Matrix
  - Semantic Attributes
date: 2026-05-08
content_hash: 16a712f21590fcd5
---

# On the Emergence of Linear Analogies in Word Embeddings

**Conference**: NeurIPS 2025
**arXiv**: [2505.18651](https://arxiv.org/abs/2505.18651)
**Code**: [GitHub](https://github.com/DJKorchinski/linear-analogies-word-embedding-reproduction)
**Area**: Representation Learning / NLP Theory
**Keywords**: Word Embeddings, Linear Analogies, Word2Vec, PMI Matrix, Semantic Attributes

## TL;DR

A generative model based on binary semantic attributes is proposed to analytically prove the emergence mechanism of linear analogy structures in word embeddings (e.g., $W_{\text{king}} - W_{\text{man}} + W_{\text{woman}} \approx W_{\text{queen}}$), providing a unified explanation for four key empirical observations.

## Background & Motivation

Models such as Word2Vec and GloVe construct word embeddings $W_i$ from co-occurrence probabilities $P(i,j)$, yielding vectors that not only cluster semantically similar words but also exhibit remarkable linear analogy structures. However, the theoretical origin of this geometric regularity remains unclear.

Prior observations indicate: (i) linear analogy structures already exist in the leading eigenvectors of $M(i,j) = P(i,j)/(P(i)P(j))$; (ii) analogy accuracy first increases and then saturates as the embedding dimension (i.e., the number of retained eigenvectors $K$) grows; (iii) using $\log M$ (i.e., the PMI matrix) yields better performance than $M$ itself; (iv) analogy structures persist even when all word pairs associated with a specific analogy relation (e.g., all masculine–feminine pairs) are removed from the corpus.

Previous theories assumed a "true" Euclidean semantic space in which words follow a spherically symmetric distribution. The authors argue this view is problematic: spherical symmetry implies the PMI spectrum should be uniformly spaced, whereas the empirical spectrum is broadly distributed; furthermore, this assumption cannot explain why the top eigenvectors correspond to specific semantic analogies.

## Method

### Overall Architecture

Core assumption: each word $i$ is characterized by $d$ binary semantic attributes $\alpha_i \in \{-1,+1\}^d$ (e.g., gender, royal/non-royal), with different attributes independently influencing co-occurrence statistics.

### Key Designs

1. **Kronecker Product Structure of the Co-occurrence Matrix**

   Under the attribute independence assumption, the co-occurrence probability factorizes as a product of per-attribute contributions:

   $P(i,j) = P(i)P(j)\prod_{k=1}^{d}P^{(k)}(\alpha_i^{(k)}, \alpha_j^{(k)})$

   Each $P^{(k)}$ is a $2 \times 2$ matrix parameterized by a signal strength $s_k$ and an asymmetry factor $q_k = p_k/(1-p_k)$. Consequently, $M(i,j) = \prod_k P^{(k)}$ is exactly a Kronecker product of $d$ such $2 \times 2$ matrices, whose eigenvectors admit closed-form solutions.

   **Core Theorem**: The eigenvectors of $M$ are $v_S = v_{a_1}^{(1)} \otimes v_{a_2}^{(2)} \otimes \cdots \otimes v_{a_d}^{(d)}$, with eigenvalues $\lambda_S = \prod_{k=1}^d \lambda_{a_k}^{(k)}$.

2. **The Key Transition from $M$ to the PMI Matrix**

   Taking the logarithm of $M$ transforms the multiplicative structure into an additive one:

   $\log M(i,j) = \delta + \bm{\eta}^\top \bm{\alpha}_i + \bm{\eta}^\top \bm{\alpha}_j + \bm{\alpha}_i^\top D \bm{\alpha}_j$

   where $D = \text{diag}(\gamma_1, \ldots, \gamma_d)$. This implies that $\log M$ has rank at most $d+1$, and its eigenvectors are affine functions of the attribute vectors. Therefore, when $K \geq d$, the linear analogy $W_A - W_B + W_C = W_D$ (provided attributes satisfy $\bm{\alpha}_D = \bm{\alpha}_A - \bm{\alpha}_B + \bm{\alpha}_C$) holds **exactly**, independent of the distribution of signal strengths $\{s_k\}$.

   This explains why PMI outperforms $M$ itself: the higher-order eigenvectors of $M$ encode products of attributes, introducing interference terms for $K > d+1$ that corrupt analogy structures; whereas the PMI matrix has exactly $d+1$ nonzero eigenvalues.

3. **Robustness Analysis**

   - **Additive noise**: Adding zero-mean noise $\xi(i,j)$ to the PMI matrix yields a perturbation with spectral norm $\|\Delta\|_2 \sim 2\sigma_\xi \sqrt{2^d}$, which is negligible compared to the semantic eigenvalue spacing of $\mathcal{O}(2^d/d)$; analogy structures are thus preserved in the large-$d$ limit.
   - **Vocabulary sparsification**: Retaining only a fraction $f = 0.15$ of words (removing 97%+ of the vocabulary), the Marchenko–Pastur theorem guarantees that the spectral structure of the PMI matrix converges to the full-vocabulary case as long as $m = f \cdot 2^d \gg d$.
   - **Removal of specific analogy word pairs**: After removing all word pairs along a given attribute direction, the resulting perturbation matrix contains only $2^d$ nonzero entries, with spectral norm $\sim \sqrt{2^d}$, which remains negligible.

### Loss & Training

This paper is purely theoretical and involves no model training. Validation is conducted by: (1) computing eigenvalues and eigenvectors on synthetic models and evaluating analogy accuracy; (2) constructing co-occurrence and PMI matrices from Wikipedia text for comparative verification.

## Key Experimental Results

### Wikipedia Analogy Test

| Embedding | Dimension $K$ | Analogy Accuracy Trend |
|-----------|--------------|------------------------|
| $M_{ij}$ | Increasing $K$ | Increases then decreases |
| $\log M_{ij}$ | Increasing $K$ | Monotonically increases to saturation |

### Synthetic Model Validation ($d=8$)

| Matrix | Accuracy at $K=d=8$ | Trend as $K$ Increases |
|--------|---------------------|------------------------|
| $M$ | ~80% (depends on $s_k$ distribution) | Decreases |
| $\log M$ | **100%** | Remains 100% |

### Robustness Experiments

| Perturbation Type | Condition | $\log M$ Analogy Accuracy |
|-------------------|-----------|--------------------------|
| Multiplicative noise $\sigma_\xi = 0.1$ | $d=8, K=d$ | ~100% |
| Vocabulary sparsification $f=0.15$ + noise $\sigma_\xi=0.1$ | $d=12, K=d$ | ~100% |
| Removal of all specified analogy word pairs | Wikipedia | Minimal performance drop |

### Key Findings

1. **Rigorous explanation for the theoretical superiority of PMI**: The finite-rank structure of PMI ($\leq d+1$) enables exact analogy recovery, while the higher-order eigenvectors of $M$ contain attribute product terms that corrupt analogies.
2. **Different analogies emerge at different dimensions**: When the distribution of $s_k$ is broad, analogies associated with stronger-signal attributes emerge at lower embedding dimensions.
3. **Extreme robustness**: Analogy structures in PMI embeddings persist even after removing 97%+ of the vocabulary or all target analogy word pairs.
4. **Spectral structure prediction**: The PMI spectrum predicted by the model ($d+1$ effective eigenvalues) is consistent with observations on Wikipedia data.

## Highlights & Insights

1. **Analytically tractable generative model**: The Kronecker product decomposition enables fully closed-form computation of eigenvalues and eigenvectors, without relying on numerical approximations.
2. **Unified explanation of four phenomena**: A single parsimonious model simultaneously explains empirical observations (i)–(iv), each supported by rigorous mathematical proof.
3. **Implications for large language models**: The theoretical foundations of the linear representation hypothesis (concepts encoded as linear subspaces in LLMs) may similarly be rooted in the attribute independence structure of language statistics.

## Limitations & Future Work

- The binary attribute assumption is overly simplistic; real semantic attributes may be multi-valued or continuous.
- Polysemous words (e.g., "bank" referring to both a financial institution and a riverbank) are not considered.
- Attributes may exhibit hierarchical structure (e.g., random hierarchical models), whereas the current model assumes a flat structure.
- Application to context-dependent embeddings in modern LLMs remains unexplored.

## Related Work & Insights

- **Word embedding theory**: Levy & Goldberg's implicit matrix factorization perspective; Arora et al.'s random walk model.
- **Linear representations in LLMs**: Jiang et al. (2024) on the origins of linear representations in LLMs; Park et al.'s linear representation hypothesis.
- **Analogy evaluation**: Mikolov et al.'s standard analogy benchmark (19,544 quadruples across 13 semantic categories).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Provides the most complete theoretical explanation of linear analogies in word embeddings to date.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Theoretical validation is thorough with both synthetic and real-data experiments, though limited to classical word embedding models.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theorems are clearly stated and proofs are concise and elegant.
- Value: ⭐⭐⭐⭐☆ — Offers profound insights into the fundamental principles of representation learning, though direct connections to modern LLMs remain limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[NeurIPS 2025\] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[NeurIPS 2025\] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)
- [\[NeurIPS 2025\] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)

</div>

<!-- RELATED:END -->
