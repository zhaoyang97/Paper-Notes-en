---
title: >-
  [Paper Note] Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss
description: >-
  [ICLR 2026][Image Restoration][Linear autoencoder] This paper generalizes the EDLAE recommendation model's objective to a Decoupled Expected Quadratic Loss (DEQL)…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "Linear autoencoder"
  - "recommender systems"
  - "collaborative filtering"
  - "expected quadratic loss"
  - "closed-form solution"
date: 2026-05-08
content_hash: e761c490f266cead
---

# Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss

**Conference**: ICLR 2026
**arXiv**: [2603.07402](https://arxiv.org/abs/2603.07402)  
**Code**: [https://github.com/coderaBruce/DEQL](https://github.com/coderaBruce/DEQL)  
**Area**: Image Restoration
**Keywords**: Linear autoencoder, recommender systems, collaborative filtering, expected quadratic loss, closed-form solution

## TL;DR
This paper generalizes the EDLAE recommendation model's objective to a Decoupled Expected Quadratic Loss (DEQL), derives closed-form solutions over a broader hyperparameter range ($b>0$), and reduces computational complexity from $O(n^4)$ to $O(n^3)$ via the Miller matrix inversion lemma, surpassing EDLAE and deep learning models on multiple benchmark datasets.

## Background & Motivation

**Background**: Linear autoencoder (LAE) recommendation models (e.g., EASE, EDLAE, ELSA) achieve performance comparable to or better than deep learning models in collaborative filtering tasks, owing to their simplicity and reproducibility via closed-form solutions. They learn an item-item similarity matrix $W \in \mathbb{R}^{n \times n}$ to reconstruct the user-item interaction matrix $R$.

**Limitations of Prior Work**: EDLAE improves training-test alignment by introducing dropout and an emphasis weight matrix $A$ (parameters $a$ and $b$). However, the original EDLAE derives a closed-form solution only for the special case $b=0$, limiting the model's expressive capacity.

**Key Challenge**: The solution space for $b>0$ may yield better solutions, but since the Hessian matrix $H^{(i)}$ varies with column index $i$, direct inversion incurs $O(n^4)$ complexity, making it computationally infeasible.

**Goal**: (a) Derive closed-form solutions for $b>0$; (b) reduce computational complexity to a practical level; (c) verify whether the expanded solution space enables discovery of better models.

**Key Insight**: The EDLAE objective is reformulated in expectation form, decoupling the optimization of each column $W_{*i}$ into independent expected quadratic loss subproblems, yielding a unified closed-form solution framework.

**Core Idea**: By generalizing EDLAE to the Decoupled Expected Quadratic Loss (DEQL), the paper derives closed-form solutions for all $b \geq 0$ in a unified manner, and applies the matrix inversion update theorem to reduce computation for $b>0$ to $O(n^3)$.

## Method

### Overall Architecture
The input is a binary user-item interaction matrix $R \in \{0,1\}^{m \times n}$; the output is an item similarity matrix $W \in \mathbb{R}^{n \times n}$, yielding predictions $\hat{R} = RW$. The training objective minimizes an expected quadratic loss with dropout and emphasis weights:

$$f(W) = \mathbb{E}_\Delta[\|A \odot (R - (\Delta \odot R)W)\|_F^2]$$

where $\Delta$ is a Bernoulli dropout matrix and $A$ is an emphasis matrix (dropped entries weighted by $a$, retained entries by $b$). The method proceeds in three steps: (1) derive the DEQL closed-form solution; (2) design an efficient algorithm to reduce complexity; (3) incorporate $L_2$ regularization and a zero-diagonal constraint.

### Key Designs

1. **Decoupled Expected Quadratic Loss (DEQL)**:

    - **Function**: Decouples the global objective by columns of $W$, enabling independent optimization of each column.
    - **Mechanism**: Exploits the column-wise decomposability of the Frobenius norm to write the objective as $l(W) = \sum_{i=1}^n h^i(W_{*i})$. Each $h^i$ is quadratic in $W_{*i}$, with closed-form solution $W_{*i}^* = \mathbb{E}[X^TX]^{-1}\mathbb{E}[X^TY_{*i}]$, i.e., the standard expected linear regression solution.
    - **Design Motivation**: This decoupling enables independent analysis of the optimal solution for each column, avoiding difficulties arising from coupling. It also reveals the theoretical property that when $b=0$, the solution is non-unique (diagonal elements are arbitrary).

2. **Closed-Form Solution and Uniqueness Theorem for $b>0$**:

    - **Function**: Proves that when $b>0$, $H^{(i)} = G^{(i)} \odot R^TR$ is positive definite, and the closed-form solution exists and is unique.
    - **Mechanism**: Explicitly computes the expectation expressions for $H^{(i)}$ and $v^{(i)}$ via Lemma 3.2. The entries of $G^{(i)}$ are polynomials in $a$, $b$, and $p$; when $b>0$, $G^{(i)}$ is positive definite, and hence $H^{(i)}$ is positive definite.
    - **Design Motivation**: Resolves the restriction of the original EDLAE to $b=0$. Notably, solutions exist even for $b > a$ (beyond the original EDLAE range), and experiments show this regime achieves the best performance on certain datasets.

3. **Miller Matrix Inversion Iterative Algorithm**:

    - **Function**: Reduces the computation of all $n$ inverses $H^{(i)^{-1}}$ from $O(n^4)$ to $O(n^3)$.
    - **Mechanism**: Decomposes $H^{(i)}$ as $H_0 + E_1^{(i)} + E_2^{(i)}$, where $H_0$ is independent of $i$ (inverted only once), and $E_1^{(i)}$, $E_2^{(i)}$ are each rank-1 matrices. Applying the Miller theorem's rank-1 update formula, each product $H^{(i)^{-1}}v^{(i)}$ requires only $O(n)$ vector operations.
    - **Design Motivation**: This is the key step that makes the $b>0$ formulation practically feasible. Without this optimization, the $O(n^4)$ complexity would be entirely impractical for large-scale recommendation datasets.

### Loss & Training
The final optimization objective supports three variants:
- **DEQL(plain)**: Pure DEQL, Equation (9).
- **DEQL(L2)**: $W^* = (H^{(i)} + \lambda I)^{-1}v^{(i)}$; replacing $H_0$ with $H_0 + \lambda I$ allows reuse of the same algorithm.
- **DEQL(L2+zero-diag)**: Adds a zero-diagonal constraint on top of L2, enforced via the projection in Equation (17).

## Key Experimental Results

### Main Results (Strong Generalization Setting — LAE Model Comparison)

| Dataset | Metric | DEQL(L2) | EDLAE | EASE | Gain |
|--------|------|----------|-------|------|------|
| Games | R@20 | **0.2891** | 0.2851 | 0.2733 | +1.4% |
| Beauty | R@20 | **0.1408** | 0.1324 | 0.1323 | +6.3% |
| Gowalla-1 | R@20 | **0.2298** | 0.2268 | 0.2230 | +1.3% |
| ML-20M | R@20 | **0.3940** | 0.3925 | 0.3905 | +0.4% |
| Netflix | R@20 | **0.3662** | 0.3656 | 0.3618 | +0.2% |
| MSD | R@20 | **0.3348** | 0.3336 | 0.3332 | +0.4% |

### Weak Generalization Setting (Deep Learning Model Comparison)

| Dataset | Metric | DEQL(L2) | SSM (Best DL) | Gain |
|--------|------|----------|-------------|------|
| Amazonbook | R@20 | **0.0629** | 0.0496 | +26.8% |
| Amazonbook | N@20 | **0.0362** | 0.0270 | +34.1% |
| Yelp2018 | R@20 | 0.0746 | **0.0765** | −2.5% |
| Gowalla | R@20 | 0.1824 | **0.1894** | −3.7% |

### Ablation Study

| Configuration | Games R@20 | Beauty R@20 | Notes |
|------|-----------|------------|------|
| DEQL(L2) | **0.2891** | **0.1408** | Full model, $b>0$ + L2 |
| DEQL(L2+zero-diag) | 0.2872 | 0.1388 | Slight drop after adding zero-diagonal constraint |
| DEQL(plain) | 0.2524 | 0.1093 | No L2 regularization; significant performance drop |
| EDLAE ($b=0$) | 0.2851 | 0.1324 | Original method |

### Key Findings
- $L_2$ regularization is critical; DEQL(plain) falls far short of DEQL(L2).
- The zero-diagonal constraint is not consistently beneficial; DEQL(L2) frequently outperforms DEQL(L2+zero-diag).
- On certain datasets (e.g., Games), the optimal $b/a$ ratio exceeds 1 (i.e., $b>a$), surpassing the $a \geq b$ range originally recommended by EDLAE.
- DEQL achieves a 27–34% advantage over deep learning models on Amazonbook, but slightly underperforms SSM on Yelp2018 and Gowalla.

## Highlights & Insights
- The **Miller matrix inversion iterative technique** for complexity reduction is particularly elegant: column-dependent terms in the matrix inverse are decomposed into two rank-1 perturbations, and the rank-1 update formula is applied column by column, reducing $O(n^4)$ to $O(n^3)$. This idea transfers to any setting where a matrix inverse varies with an index but the variation is low-rank.
- The **decoupled objective formulation** reveals the theoretical non-uniqueness of the solution when $b=0$ (diagonal elements are free), providing a theoretical explanation for the "relaxed zero-diagonal constraint" proposed by Moon et al. (2023).
- Simple linear models still hold considerable potential in recommender systems; the key lies in correctly designing the training objective to align with the test scenario.

## Limitations & Future Work
- Performance gains on large datasets are marginal (< 0.5% on Netflix and MSD), suggesting diminishing returns from expanding the solution space.
- The $b>0$ formulation introduces additional hyperparameters ($a$, $b$, $p$, $\lambda$), making grid search costly.
- Only implicit feedback (binary interaction matrices) is considered; extension to explicit rating scenarios is not explored.
- Under the weak generalization setting on Yelp2018 and Gowalla, DEQL underperforms deep models such as SSM, indicating the limitations of simple linear models in capturing graph-structured information.

## Related Work & Insights
- **vs. EDLAE (Steck 2020)**: EDLAE addresses only the special case $b=0$; this paper extends it to the full solution space $b \geq 0$. Performance improvements are modest but consistent.
- **vs. EASE (Steck 2019)**: EASE is a special case of DEQL with dropout rate $p=0$ and no emphasis weight mechanism.
- **vs. LightGCN**: Deep graph models have an advantage on dense interaction data but underperform DEQL on sparse data (e.g., Amazonbook).

## Rating
- **Novelty**: ⭐⭐⭐ The technical generalization is natural but essentially constitutes a hyperparameter space extension of an existing method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Nine datasets, multiple baselines, and statistical significance tests.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, though notation is heavy.
- **Value**: ⭐⭐⭐ Offers theoretical contributions to LAE recommendation models, but practical performance gains are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](../../NeurIPS2025/image_restoration/improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)
- [\[ICLR 2026\] Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ICLR 2026\] DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)

</div>

<!-- RELATED:END -->
