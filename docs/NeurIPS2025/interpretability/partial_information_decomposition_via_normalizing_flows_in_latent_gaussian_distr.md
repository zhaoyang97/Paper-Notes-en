---
title: >-
  [Paper Note] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions
description: >-
  [NeurIPS 2025][Partial Information Decomposition] Two complementary tools are proposed: Thin-PID is an efficient Gaussian PID algorithm (10× faster than existing methods), and Flow-PID applies normalizing flows to map arbitrary input distributions to Gaussian space before computing PID, addressing the infeasibility of PID on continuous high-dimensional data. The paper also resolves an open problem regarding whether the joint Gaussian solution is optimal.
tags:
  - NeurIPS 2025
  - Partial Information Decomposition
  - normalizing flow
  - Gaussian distribution
  - multimodal learning
  - mutual information
date: 2026-05-08
content_hash: 1f514dc9937353c1
---

# Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions

**Conference**: NeurIPS 2025
**arXiv**: [2510.04417](https://arxiv.org/abs/2510.04417)
**Code**: [https://github.com/warrenzha/flow-pid](https://github.com/warrenzha/flow-pid)
**Area**: Interpretability / Information Theory
**Keywords**: Partial Information Decomposition, normalizing flow, Gaussian distribution, multimodal learning, mutual information

## TL;DR

Two complementary tools are proposed: Thin-PID is an efficient Gaussian PID algorithm (10× faster than existing methods), and Flow-PID applies normalizing flows to map arbitrary input distributions to Gaussian space before computing PID, addressing the infeasibility of PID on continuous high-dimensional data. The paper also resolves an open problem regarding whether the joint Gaussian solution is optimal.

## Background & Motivation

**State of the Field**: Partial Information Decomposition (PID) is an information-theoretic framework for quantifying multi-source information interactions. It decomposes the total mutual information of two sources $X_1, X_2$ about a target $Y$ into four non-negative components: redundant information $R$ (shared by both), unique information $U_1, U_2$ (exclusive to each), and synergistic information $S$ (only available when both are combined). PID has been applied in multimodal learning to understand modality interactions.

**Limitations of Prior Work**: Computing PID requires solving an optimization problem over the set of joint distributions satisfying marginal constraints. This is feasible for discrete small-scale data (via CVX), but practically impossible for continuous high-dimensional data, where even estimating mutual information and entropy is extremely challenging. The BATCH method uses neural network parameterization but achieves poor accuracy. Tilde-PID is restricted to Gaussian distributions without proof of optimality.

**Root Cause**: A fundamental gap exists between the theoretical elegance of PID and its computational infeasibility — while PID can precisely quantify modality interactions in theory, in practice it is limited to discrete low-dimensional data.

**Paper Goals**: (1) Prove the optimality of the joint Gaussian solution in Gaussian PID; (2) Design an algorithm more efficient than Tilde-PID; (3) Generalize to non-Gaussian high-dimensional continuous data.

**Starting Point**: Two key insights — PID under Gaussian distributions admits a closed-form solution and can be computed efficiently; the invertibility of normalizing flows preserves mutual information, enabling transformation to Gaussian space prior to PID computation.

**Core Idea**: Transform data to Gaussian via normalizing flows, then efficiently compute PID in the Gaussian space.

## Method

### Overall Architecture

The framework operates at two levels: (1) **Thin-PID** handles Gaussian PID — reformulating the optimization objective as minimizing a function of the noise cross-covariance matrix and solving via projected gradient descent; (2) **Flow-PID** handles general distributions — training a Cartesian product normalizing flow $f_1 \times f_2 \times f_Y$ to map $(X_1, X_2, Y)$ to a Gaussian marginal space, then invoking Thin-PID.

### Key Designs

1. **Thin-PID: Efficient Gaussian PID Algorithm**

   - **Function**: Efficiently solves the PID optimization problem when the marginals are known to be Gaussian.
   - **Mechanism**: PID is reinterpreted via a Gaussian broadcast channel model — $Y$ is the transmitted signal, with $X_1 = H_1 Y + n_1$ and $X_2 = H_2 Y + n_2$. Synergistic information is equivalent to the cooperative gain under the worst-case noise correlation. The optimization variable is reduced to the noise cross-covariance matrix $\Sigma_{n_1 n_2}^{\text{off}}$ (of size $d_{X_1} \times d_{X_2}$), solved via projected gradient descent. Gradients have closed-form expressions (Proposition 3.4), and projection is implemented via SVD (truncating singular values to $[0,1]$). Complexity is $O(\min(d_{X_1}, d_{X_2})^3)$.
   - **Design Motivation**: Tilde-PID requires eigendecomposition of the full $(d_{X_1}+d_{X_2}) \times (d_{X_1}+d_{X_2})$ matrix, whereas Thin-PID only performs SVD on the $d_{X_1} \times d_{X_2}$ cross-covariance, yielding significant speedups when $d_{X_1} \gg d_{X_2}$.

2. **Proof of Joint Gaussian Optimality**

   - **Function**: Proves that the optimal joint distribution under the GPID definition is necessarily Gaussian, resolving an open problem.
   - **Mechanism**: The key lemma establishes that for any $q$, $h_q(Y|X_1,X_2) \leq h_{\hat{q}}(Y|X_1,X_2)$, where $\hat{q}$ is the Gaussian distribution sharing the same first and second moments as $q$ (via the Gaussian upper bound property of conditional entropy). Since the optimization objective is equivalent to maximizing $h_q(Y|X_1,X_2)$ and $\hat{q}$ preserves the marginal constraints, the Gaussian solution is necessarily optimal.
   - **Design Motivation**: The prior Tilde-PID merely assumed that the Gaussian solution was sufficiently good without formal proof. This result elevates a "heuristic approximation" to an "exact solution."

3. **Flow-PID: Normalizing Flow Encoder**

   - **Function**: Transforms non-Gaussian continuous data into a marginal Gaussian space, enabling Thin-PID.
   - **Mechanism**: Three independent normalizing flows $f_1, f_2, f_Y$ are trained so that the marginals of $(f_1(X_1), f_Y(Y))$ and $(f_2(X_2), f_Y(Y))$ approximate Gaussians. By Theorem 4.1, invertible mappings preserve total mutual information; Corollary 4.2 guarantees that the PID decomposition is likewise preserved. The training objective minimizes KL divergence to variational Gaussian marginals.
   - **Design Motivation**: Direct estimation of MI in high dimensions is extremely difficult, whereas Gaussian MI has a closed-form solution. The invertibility of the flows ensures that PID in the latent space is equivalent to PID in the original space.

### Loss & Training

The Flow-PID loss is the sum of Gaussian marginal regularization terms over the two marginal pairs: $\mathcal{L}_{\text{flow}} = \mathcal{L}_\mathcal{N}(\{(X_1, Y)\}) + \mathcal{L}_\mathcal{N}(\{(X_2, Y)\})$, which is equivalent to maximizing the Gaussian log-likelihood of the transformed samples plus the Jacobian term.

## Key Experimental Results

### Main Results: Non-Gaussian Synthetic Data

| Dimension | Method | R | U1 | U2 | S | Notes |
|-----------|--------|---|----|----|---|-------|
| (2,2,2) | Tilde-PID | 0.18 | 0.29 | 0.76 | 0.02 | Severe bias |
| (2,2,2) | **Flow-PID** | **0.62** | **0.91** | **0.50** | **0.11** | Close to ground truth |
| (2,2,2) | Ground Truth | 0.79 | 1.46 | 0.58 | 0.18 | — |
| (100,60,2) | Tilde-PID | 1.48 | 0 | 1.97 | 0.13 | Worse in high dimensions |
| (100,60,2) | **Flow-PID** | **4.34** | **0.36** | **0** | **0.25** | Close to ground truth |
| (100,60,2) | Ground Truth | 5.71 | 1.01 | 0 | 0.57 | — |

### Ablation Study: Computational Efficiency

| Method | Main Bottleneck | When $\min(d_{X_1},d_{X_2})>100$ |
|--------|----------------|----------------------------------|
| **Thin-PID** | SVD on $\min(d_{X_1},d_{X_2})$ | **10×+ faster than Tilde-PID** |
| Tilde-PID | ED on $d_{X_1}+d_{X_2}$ | Baseline |

### Key Findings

- **Thin-PID achieves very high precision**: absolute error $<10^{-12}$ on Gaussian synthetic data, compared to $>10^{-8}$ for Tilde-PID.
- **Flow-PID correctly recovers the interaction structure of non-Gaussian data**: Tilde-PID applied directly to sample covariances leads to entirely incorrect interaction types (e.g., misclassifying unique information as redundant), while Flow-PID correctly identifies the structure via learned inverse transformations.
- **Synergistic information is the hardest to estimate**: BATCH tends to overestimate redundancy and underestimate synergy; Flow-PID reduces this bias.
- **Real multimodal data application**: On 6 MultiBench datasets, Flow-PID estimates total mutual information far exceeding that of BATCH, yielding results more consistent with actual model performance.
- **Model selection accuracy**: PID estimates from Flow-PID achieve 96–100% accuracy in multimodal model selection tasks.

## Highlights & Insights

- **Theoretical contribution resolving an open problem**: Although the proof of joint Gaussian optimality under GPID is not technically complex, its significance is substantial — it elevates Tilde-PID from a "heuristic" to an "exact" method.
- **The broadcast channel reinterpretation is elegant**: Recasting PID as a worst-case noise optimization in a Gaussian broadcast channel establishes a beautiful bridge between information theory and multimodal learning.
- **Theoretical guarantee that flows preserve PID**: Rather than simply "training a good encoder and applying it," the framework provides rigorous mathematical guarantees that invertible mappings preserve the entire PID decomposition.

## Limitations & Future Work

- The accuracy of Flow-PID depends on how well the normalizing flow approximates the true distribution; complex distributions may require more expressive flow architectures.
- The current framework handles only two-source PID; extension to multiple sources is theoretically feasible but increases complexity.
- No ground-truth PID is available for real datasets, necessitating evaluation via indirect metrics.
- The whitening preprocessing in Thin-PID assumes independent noise, which may be too strong an assumption in certain settings.
- No direct comparison is made with neural mutual information estimators such as MINE.

## Related Work & Insights

- **vs. CVX/BATCH**: CVX is limited to discrete small-scale data; BATCH uses neural network parameterization but achieves poor accuracy (synergistic information is severely underestimated). Flow-PID is accurate and efficient on continuous high-dimensional data.
- **vs. Tilde-PID**: Shares the same Gaussian PID definition but is approximately 10× slower and does not prove the optimality of the Gaussian solution.
- **vs. MINE/NWJ**: These methods estimate MI only and cannot be directly applied to PID, which additionally requires optimization over a constrained distribution set.

## Rating

- Novelty: ⭐⭐⭐⭐ Resolving an open problem and the flow encoder design both demonstrate originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ground-truth validation on synthetic data; coverage of multiple real-world benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical exposition and clear paper organization.
- Value: ⭐⭐⭐⭐ Extends PID to practical multimodal scenarios with important implications for understanding modality interactions.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)
- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)

<!-- RELATED:END -->
