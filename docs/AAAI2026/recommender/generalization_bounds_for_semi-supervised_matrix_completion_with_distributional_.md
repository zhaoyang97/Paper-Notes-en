---
title: >-
  [Paper Note] Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information
description: >-
  [AAAI 2026][Recommender Systems][Semi-supervised matrix completion] This paper proposes the first semi-supervised matrix completion learning paradigm: assuming that the sampling distribution $P$ and the ground-truth matr…
tags:
  - "AAAI 2026"
  - "Recommender Systems"
  - "Semi-supervised matrix completion"
  - "generalization bounds"
  - "low-rank subspace"
  - "implicit feedback"
  - "explicit feedback"
  - "nuclear norm constraint"
date: 2026-05-08
content_hash: e852f049e78c14e1
---

# Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information

**Conference**: AAAI 2026
**arXiv**: [2511.13049](https://arxiv.org/abs/2511.13049)
**Area**: Matrix Completion Theory / Recommender Systems
**Keywords**: Semi-supervised matrix completion, generalization bounds, low-rank subspace, implicit feedback, explicit feedback, nuclear norm constraint

## TL;DR

This paper proposes the first semi-supervised matrix completion learning paradigm: assuming that the sampling distribution $P$ and the ground-truth matrix $G$ share a low-rank subspace, and given a large amount of unlabeled data $M$ and a small amount of labeled data $N$, it proves that the generalization error can be decomposed into two independent terms $\tilde{O}(\sqrt{nd/M}) + \tilde{O}(\sqrt{dr/N})$, achieving significant improvements over explicit-feedback-only baselines on the Douban and MovieLens datasets.

## Background & Motivation

**Background**: Matrix completion is a core technique in recommender systems, aiming to recover a complete user–item rating matrix from limited observed entries. Classical theory (Candes & Recht et al.) proves that nuclear norm minimization can exactly recover a rank-$r$ matrix under uniform sampling, requiring $\tilde{O}(nr)$ observations. Inductive Matrix Completion (IMC) further reduces sample complexity by incorporating side information matrices $X, Y$.

**Limitations of Prior Work**:
(1) **All existing theoretical results assume fully labeled observations** — every observed entry $(i,j)$ is associated with a rating label, whereas in practice a large volume of interactions (browsing/clicking/purchasing, i.e., "implicit feedback") carry no ratings;
(2) **IMC assumes side information is given** — in practice, side information matrices must be estimated from auxiliary data (social graphs, molecular structures, etc.) rather than being directly available;
(3) **The relationship between implicit and explicit feedback lacks theoretical characterization** — while extensive empirical evidence shows that implicit feedback contains useful information for rating prediction, a formal learning-theoretic account is absent.

**Key Insight**: The paper assumes that the sampling distribution $P$ (the source of implicit feedback) and the ground-truth matrix $G$ (explicit ratings) share a low-rank subspace, uses a large number of unlabeled samples to estimate the shared subspace, and then performs matrix recovery within this subspace using a small number of labeled samples.

## Method

### Overall Architecture

The DAMC (Distributionally Aware Matrix Completion) algorithm proceeds in two steps:

1. **Step 1 — Subspace Estimation**: Construct the empirical distribution matrix from unlabeled data $\mathcal{H} = \frac{1}{M}\sum_{o=1}^M \mathbf{1}_{i_o, j_o}$, apply rank-$d$ truncated SVD to obtain $U, V$, and construct side information $X = \sqrt{m/d} \cdot U$, $Y = \sqrt{n/d} \cdot V$.

2. **Step 2 — Matrix Recovery**: Fix $X, Y$ and solve for the core matrix $\underline{M}$ via nuclear-norm-constrained empirical risk minimization:
$\min_{\|\underline{M}\|_* \leq \mathcal{M}} \frac{1}{N}\sum_{o=1}^N l((X\underline{M}Y^\top)_{\xi_o}, \tilde{G}_o)$

### Key Designs

1. **Shared Low-Rank Subspace Assumption (Assumption 2)**

    - **Function**: Formalizes the relationship between implicit and explicit feedback.
    - **Mechanism**: Assumes $P = U^* \Sigma^* [V^*]^\top$ (rank $d$) and $G = U^* \underline{M}^{*-} [V^*]^\top$ (rank $r \leq d$), i.e., the row/column spaces of both matrices are spanned by the same subspace.
    - **Design Motivation**: In recommender systems, users' interaction patterns (implicit feedback) and rating preferences (explicit feedback) are driven by the same latent factors. This is the simplest formalization, yet sufficient for theoretical analysis.

2. **Error Decomposition Theorem for Generalization Bounds (Theorem 1)**

    - **Function**: Proves that the generalization error of semi-supervised matrix completion decomposes into two independent error terms.
    - **Core Result**: Under assumptions of boundedness, Lipschitz continuity, shared subspace, incoherence, and uniform marginals, the generalization error bound consists of:
        - $\tilde{O}(\sqrt{(m+n)r/M})$: subspace estimation error, controlled by unlabeled samples $M$;
        - $\tilde{O}(\sqrt{dr/N})$: matrix recovery error, controlled by labeled samples $N$;
        - A higher-order cross term $\tilde{O}(\sqrt{\Gamma(m+n)r/(MN)})$: absorbable under mild conditions.
    - **Key Significance**: The sample complexity for labeled data is only $\tilde{O}(dr)$, far less than $\tilde{O}((m+n)r)$ without side information — even as the number of labeled samples per row/column approaches zero, accurate recovery remains possible as long as unlabeled samples are sufficient.

### Loss & Training

In practice, the nuclear norm constraint is reformulated in Lagrangian form:
$\min \frac{1}{N}\sum l((XAB^\top Y^\top)_{\xi_o}, \tilde{G}_o) + \lambda_\mathcal{M}[\|A\|_{Fr}^2 + \|B\|_{Fr}^2]$,
exploiting the classical lemma that $\|A\|_{Fr}^2 + \|B\|_{Fr}^2$ is equivalent to the nuclear norm. In real-data experiments, SVD is replaced by a nonlinear autoencoder to handle data nonlinearity.

## Key Experimental Results

### Synthetic Experiments — Validation of Error Decomposition

On a $200 \times 200$ matrix ($d=r=4$) with $M \in [10000, 100000]$, $N \in [50, 1000]$, and 30 independent runs, the generalization error is highly correlated with the decomposition estimate ($\text{GAP}(M, 1000) + \text{GAP}(100000, N)$), confirming that the two error terms combine additively and independently.

### Real-Data Experiments — RMSE Comparison

| Dataset | Method | p=0.0 | p=0.5 | p=0.9 | p=0.95 |
|---------|--------|-------|-------|-------|--------|
| ML-100K | UserKNN | 1.012 | 1.038 | 1.168 | 1.246 |
| | IGMC | 0.928 | 0.982 | 1.117 | 1.148 |
| | Soft Impute | 0.918 | 0.962 | 1.419 | 1.823 |
| | **DAMC** | **0.907** | **0.936** | **1.006** | **1.046** |
| Douban | UserKNN | 0.795 | 0.829 | 0.964 | 0.984 |
| | **DAMC** | **0.718** | **0.738** | **0.832** | **0.858** |

*$p$ denotes the proportion of removed labels; $p=0.9$ means only 10% of explicit ratings are retained.*

### Key Findings

- DAMC's advantage becomes more pronounced at high label removal rates ($p=0.9$, $0.95$) — Soft Impute's RMSE surges to 1.419 at $p=0.9$ while DAMC achieves only 1.006.
- Unlabeled samples (implicit feedback) do contain useful information for estimating the row/column subspace of the ground-truth matrix.
- In synthetic experiments, scatter plots of the decomposition estimates versus actual generalization errors fall almost exactly on the diagonal, strongly supporting the theoretical predictions.

## Highlights & Insights

1. **First formal characterization of implicit–explicit feedback relationship**: The shared low-rank subspace assumption is simple yet effective, providing a theoretical foundation for jointly modeling dual feedback in recommender systems.
2. **Elegant error decomposition**: The errors from the two data sources combine additively and independently, yielding precise trade-off relations between labeled and unlabeled data volumes.
3. **Broad practical implications**: The theory supports the empirical practice of building high-quality recommender systems with few ratings and abundant interaction logs.
4. **Clear contrast with related settings such as MNAR (Missing Not At Random)**: The paper provides a detailed discussion distinguishing the proposed framework from other non-uniform sampling theories.

## Limitations & Future Work

1. Assumption 7 (uniform upper bound on sampling probabilities) is strong; in real recommendation scenarios, popular items are sampled far more frequently than niche items.
2. The subspace dimension $d$ must be specified — in practice $d$ is unknown, and inappropriate selection degrades performance.
3. Only RMSE is evaluated; ranking metrics commonly used in recommender systems (NDCG, Hit Rate, etc.) are not assessed.
4. The substitution of SVD with a nonlinear autoencoder lacks theoretical guarantees.

## Related Work & Insights

- The semi-supervised matrix completion learning paradigm is generalizable to other matrix completion applications such as drug interaction prediction and chemical engineering.
- Treating the sampling distribution as an exploitable signal rather than a bias to be corrected represents a thought-provoking new perspective.
- The theoretical framework can be combined with graph neural network methods in collaborative filtering (e.g., IGMC) to simultaneously exploit structural and distributional information.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐⭐: First to introduce a semi-supervised paradigm in matrix completion with rigorous generalization bounds.
- **Experimental Thoroughness** ⭐⭐⭐: Synthetic experiments are carefully designed, but real-data experiments are relatively thin.
- **Writing Quality** ⭐⭐⭐⭐: Mathematical derivations are rigorous and clear; assumptions are thoroughly discussed.
- **Value** ⭐⭐⭐⭐: Provides a solid theoretical foundation for the joint exploitation of implicit and explicit feedback in recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[NeurIPS 2025\] PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders](../../NeurIPS2025/recommender/pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders.md)
- [\[ACL 2026\] Content Fuzzing for Escaping Information Cocoons on Social Media](../../ACL2026/recommender/content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)

</div>

<!-- RELATED:END -->
