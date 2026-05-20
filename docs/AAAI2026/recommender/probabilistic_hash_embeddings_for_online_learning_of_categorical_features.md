---
title: >-
  [Paper Note] Probabilistic Hash Embeddings for Online Learning of Categorical Features
description: >-
  [AAAI 2026][Recommender Systems][hash embedding] This paper proposes Probabilistic Hash Embeddings (PHE), which models hash embedding tables as random variables and performs posterior inference via Bayesian online learni…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "hash embedding"
  - "Bayesian online learning"
  - "categorical features"
  - "continual learning"
  - "variational inference"
date: 2026-05-08
content_hash: c3a5396d9b723a7c
---

# Probabilistic Hash Embeddings for Online Learning of Categorical Features

**Conference**: AAAI 2026
**arXiv**: [2511.20893](https://arxiv.org/abs/2511.20893)  
**Code**: [github](https://github.com/aodongli/probabilistic-hash-embeddings)  
**Area**: Recommender System / Online Learning
**Keywords**: hash embedding, Bayesian online learning, categorical features, continual learning, variational inference

## TL;DR

This paper proposes Probabilistic Hash Embeddings (PHE), which models hash embedding tables as random variables and performs posterior inference via Bayesian online learning. PHE addresses the catastrophic forgetting problem caused by parameter sharing in deterministic hash embeddings under streaming data settings. It significantly outperforms deterministic baselines across classification, sequential modeling, and recommender system tasks, while requiring only 2%–4% of the memory needed by collision-free embedding tables.

## Background & Motivation

Categorical features are ubiquitous in high-value ML applications such as finance, fraud detection, and recommender systems. Feature hashing is the standard approach for handling large-scale categorical features: category values are mapped via hash functions to a fixed-size embedding table, achieving memory efficiency. However, existing hash embedding methods are designed for offline/batch settings, assuming a fixed vocabulary. In practice, data arrives as a stream—new products emerge continuously, new users register constantly, and IP addresses change dynamically.

The root cause of the problem is that deterministic hash embeddings suffer from *parameter interference* during online updates: when different categorical items share the same embedding row, updating the embedding for one item corrupts the representation of another, leading to catastrophic forgetting. Furthermore, the degree of forgetting depends on the order of data arrival, making performance unpredictable. The core idea of this paper is to treat hash embeddings as random variables and leverage the natural "previous posterior as next prior" mechanism of Bayesian online learning to mitigate forgetting.

## Method

### Overall Architecture

PHE serves as a plug-and-play module consisting of two components: a fixed hash function $h$ and a hash embedding table $E \in \mathbb{R}^{B \times d}$ with prior distribution $p(E)$. Given a categorical item $s$, $K$ hash functions produce $K$ hash values $\mathbf{h}_s = \{h^{(1)}_s, \dots, h^{(K)}_s\}$, which index $K$ embeddings that are then combined via an assembly function $g$ (e.g., sum or mean) into the final probabilistic embedding $E_{\mathbf{h}_s}$. The memory cost is $O(Bd)$, independent of the number of hash functions $K$.

### Key Designs

**1. Probabilistic Modeling and Variational Inference**

Each element of the embedding table $E$ is independently modeled as a Gaussian. The prior is $p(E_{bj}) = \mathcal{N}(0, 1)$, and the variational posterior is $q_\lambda(E) = \prod_{b,j} q_{\lambda_{bj}}(E_{bj})$. Variational parameters are learned by maximizing the ELBO:

$$\mathcal{L}(\lambda) = \mathbb{E}_{q_\lambda(E)}\left[\sum_{i=1}^N \log p(y_i | E_{\mathbf{h}_{s_i}})\right] - \sum_{b=1}^B \sum_{j=1}^d D_{\text{KL}}(q_{\lambda_{bj}}(E_{bj}) | p(E_{bj}))$$

The KL term acts as regularization, preventing the posterior from drifting too far from the prior and implicitly protecting previously learned knowledge.

**2. Bayesian Online Learning for Forgetting-Free Updates**

When a new dataset $\mathcal{D}_1$ arrives, the approximate posterior from the previous round $q_{\lambda_0^*}(E)$ is used as the new prior, and the following updated ELBO is maximized:

$$\mathcal{L}^{(1)}(\lambda; \lambda_0^*) = \mathbb{E}_{q_\lambda(E)}\left[\sum_{i=1}^{N_1} \log p(y_i | E_{\mathbf{h}_{s_i}})\right] - \sum_{b,j} D_{\text{KL}}(q_{\lambda_{bj}}(E_{bj}) | q_{\lambda_{0,bj}^*}(E_{bj}))$$

Only the embedding table parameters are updated; all other network parameters are frozen. Since each categorical item activates at most $K$ embedding rows, gradient updates are sparse, leading to fast convergence.

**3. Theoretical Guarantee: Equivalence to Batch Learning**

The paper proves that under exact Bayesian inference, for any data permutation $\boldsymbol{\pi}$, the online posterior $p(E|\mathcal{D}_{\boldsymbol{\pi}})$ equals the batch posterior $p_{\text{batch}}(E|\mathcal{D})$ everywhere (Proposition 3.1), meaning PHE's performance is invariant to the order of data arrival.

### Loss & Training

A variational EM algorithm is adopted to jointly optimize model parameters $\theta$ and variational parameters $\lambda$. During the online phase, $\theta^*$ and $\lambda_0^*$ are fixed, and only the embedding posterior is updated. The reparameterization trick and gradient-based methods are used for optimization. Universal hashing employs $K$ hash functions with different seeds sharing a single embedding table, reducing collision probability to $O(1/B^K)$.

## Key Experimental Results

### Main Results

Online learning classification results (accuracy ×100):

| Method | Adult ↑ | Bank ↑ | Mushroom ↑ | CoverType ↑ | Retail ↓ | MovieLens ↓ |
|--------|---------|--------|------------|-------------|----------|-------------|
| SlowAda | 82.2±0.7 | 89.7±0.1 | 97.7±0.7 | 63.5±0.5 | 49.1±82.9 | 15.3±0.1 |
| MedAda | 74.8±4.5 | 89.0±0.9 | 97.9±0.5 | 59.1±1.2 | 22.7±20.3 | 15.1±0.1 |
| FastAda | 71.1±4.0 | 86.9±1.6 | 98.3±0.3 | 55.3±1.2 | - | 15.1±0.1 |
| **PHE** | **84.1±0.2** | **89.6±0.0** | **98.8±0.0** | **64.3±0.2** | **3.0±0.2** | **14.7±0.0** |
| EE (collision-free) | 84.2±0.0 | 90.0±0.0 | 98.8±0.0 | 64.3±0.1 | 3.7±0.1 | 15.1±0.0 |
| P-EE (probabilistic collision-free) | 84.8±0.0 | 90.1±0.0 | 98.8±0.0 | 64.0±0.4 | 3.2±0.4 | - |

Memory compression ratio of PHE relative to P-EE:

| Dataset | Adult | Bank | Mushroom | CoverType | Retail | MovieLens |
|---------|-------|------|----------|-----------|--------|-----------|
| Ratio | 0.09 | 0.2 | 0.62 | 0.2 | 0.02 | 0.04 |

### Ablation Study

PHE as a plug-and-play module across different tasks:

| Setting | Backbone | PHE vs. Best Ada | PHE vs. P-EE Memory |
|---------|----------|-----------------|---------------------|
| Classification (Adult) | Logistic/NN | 84.1 vs. 82.2 (+1.9) | 9% |
| Sequential Modeling (Retail) | Deep Kalman Filter | 3.0 vs. 22.7 (−19.7) | 2% |
| Recommendation (MovieLens-32M) | Neural CF | 14.7 vs. 15.1 (−0.4) | 4% |

### Key Findings

- Ada-series methods exhibit a declining trend during online learning—even when re-learning previously seen categories—confirming catastrophic forgetting in deterministic hash embeddings.
- PHE significantly outperforms the collision-free P-EE on Retail sequential modeling (MAE 3.0 vs. 3.2), likely because P-EE must initialize new embeddings from scratch, causing slow cold-start.
- PHE surpasses all Ada baselines on every dataset using a single unified set of hyperparameters, whereas Ada performance is highly sensitive to the number of training epochs.
- On MovieLens-32M (87k movies, 200k users, 28 years of data), PHE exceeds all baselines using only 4% of the memory.

## Highlights & Insights

- The combination of Bayesian online learning with hash embeddings is a natural and elegant idea; the prior→posterior→prior iteration inherently protects previously learned knowledge.
- The theoretical proof of invariance to data arrival order is a strong guarantee that eliminates the most difficult-to-control source of uncertainty in online learning.
- The 2%–4% memory footprint demonstrates high practical utility, making the approach well-suited for resource-constrained deployment.
- PHE operates as a plug-and-play module compatible with diverse probabilistic model backbones (DKF, NCF, etc.), offering strong extensibility.

## Limitations & Future Work

- Variational inference introduces approximation error due to the mean-field assumption; theoretical guarantees hold only approximately in practice.
- Model parameters $\theta$ are assumed fixed during the online phase, which may be insufficient when data distributions shift dramatically.
- Only single-valued categorical features are considered; handling multi-valued features and high-order feature interactions remains to be explored.
- Although experiments cover a broad range of tasks, direct comparison with dedicated continual learning methods (e.g., EWC, PackNet) is absent.

## Related Work & Insights

- **vs. Deterministic Hash Embeddings (Ada)**: Ada suffers from catastrophic forgetting due to parameter sharing during online updates, with performance highly sensitive to hyperparameters. PHE naturally resists forgetting through probabilistic modeling and Bayesian updates.
- **vs. Expandable Embedding Tables (EE)**: EE adds new rows for each new category, leading to unbounded memory growth. PHE maintains a fixed memory footprint and even surpasses EE on Retail and MovieLens.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of Bayesian online learning and hash embeddings is proposed for the first time, with solid theoretical grounding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers classification, sequential modeling, and recommendation; uses public datasets; provides both theoretical and empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete theoretical derivations, and thorough explanation of motivation and intuition.
- **Value**: ⭐⭐⭐⭐ — Addresses a practical and widespread industrial problem; the plug-and-play design offers strong usability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data](multitab_a_scalable_foundation_for_multitask_learning_on_tabular_data.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](../../NeurIPS2025/recommender/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](../../ACL2026/recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)

</div>

<!-- RELATED:END -->
