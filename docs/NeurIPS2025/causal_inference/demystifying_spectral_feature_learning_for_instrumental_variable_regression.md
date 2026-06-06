---
title: >-
  [Paper Note] Demystifying Spectral Feature Learning for Instrumental Variable Regression
description: >-
  [NeurIPS 2025][Causal Inference][Instrumental Variables] This paper establishes rigorous generalization error bounds for spectral feature-based nonparametric instrumental variable (NPIV) regression…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "Instrumental Variables"
  - "Spectral Features"
  - "Two-Stage Least Squares"
  - "Contrastive Learning"
  - "Causal Effect Estimation"
date: 2026-05-08
content_hash: 786ba94a8713bb9e
---

# Demystifying Spectral Feature Learning for Instrumental Variable Regression

**Conference**: NeurIPS 2025
**arXiv**: [2506.10899](https://arxiv.org/abs/2506.10899)  
**Code**: None  
**Area**: Causal Inference
**Keywords**: Instrumental Variables, Spectral Features, Two-Stage Least Squares, Contrastive Learning, Causal Effect Estimation

## TL;DR

This paper establishes rigorous generalization error bounds for spectral feature-based nonparametric instrumental variable (NPIV) regression, revealing that performance is jointly governed by two factors: **spectral alignment** between the structural function and the conditional expectation operator (approximation error) and the **rate of singular value decay** (estimation error). A Good-Bad-Ugly trichotomy is proposed along with data-driven diagnostic tools.

## Background & Motivation

**Background**: Nonparametric instrumental variable (NPIV) regression is a central method for causal effect estimation in the presence of hidden confounders. The classical approach is two-stage least squares (2SLS): first regress the treatment variable $X$ on features of the instrument $Z$, then regress the outcome $Y$ on the predicted features. In recent years, spectral feature methods—using the top $d$ singular functions of the conditional expectation operator $\mathcal{T}$ as features—have demonstrated strong empirical performance, yet lack theoretical understanding.

**Limitations of Prior Work**: (1) The spectral contrastive learning approach of [Xu et al.] performs well empirically, but its theoretical justification rests on an overly strong assumption (that the joint density admits an exact finite-rank decomposition) whose practical implications are unclear. (2) There is no systematic theoretical analysis of when spectral feature methods succeed or fail. (3) Practitioners have no principled way to determine whether spectral features or alternative methods are appropriate for a given problem.

**Key Challenge**: Spectral features minimize sieve ill-posedness (i.e., $\tau_{\varphi,d} = \sigma_d^{-1}$, which is optimal over all $d$-dimensional subspaces), but the structural function $h_0$ need not lie predominantly in the top eigenspace of $\mathcal{T}$. If $h_0$ is "misaligned" with the top $d$ singular functions, approximation error can be large, rendering the minimized estimation error insufficient.

**Goal**: (1) Derive rigorous generalization error bounds for spectral feature 2SLS; (2) Identify the key factors governing performance; (3) Provide diagnostic tools that can be estimated from data.

**Key Insight**: Starting from the classical sieve 2SLS generalization bounds of Blundell & Chen et al., the paper specializes these bounds to the spectral feature setting and exploits the exact structure of the singular value decomposition to derive tighter results.

**Core Idea**: The utility of spectral features is determined by two measurable quantities—spectral alignment and singular value decay rate—corresponding to the Good, Bad, and Ugly regimes.

## Method

### Overall Architecture
Consider the NPIV model $Y = h_0(X) + U$, $\mathbb{E}[U|Z] = 0$. The conditional expectation operator $\mathcal{T}: h \mapsto \mathbb{E}[h(X)|Z]$ is a Hilbert-Schmidt operator (under mild assumptions) with SVD $\mathcal{T} = \sum_i \sigma_i u_i \otimes v_i$. The core analytical pipeline is: (1) review generalization bounds for general sieve 2SLS; (2) prove that spectral features minimize sieve ill-posedness; (3) specialize the general bounds to spectral features to obtain refined expressions; (4) analyze the behavior of the two controlling terms (approximation error and estimation error); (5) connect the analysis to practical learning via the contrastive loss.

### Key Designs

1. **Optimality of Spectral Features (Proposition 1)**:

    - *Function*: Proves that spectral features achieve the minimum sieve ill-posedness among all possible $d$-dimensional feature sets.
    - *Mechanism*: Sieve ill-posedness is defined as $\tau_{\varphi,d} = \sup_{h \in \mathcal{H}_{\varphi,d}} \|h\|_{L_2} / \|\mathcal{T}h\|_{L_2}$, i.e., the norm of the inverse operator restricted to the subspace. Setting $\mathcal{H}_{\varphi,d} = \text{span}\{v_1,\ldots,v_d\}$ attains the minimum value $\sigma_d^{-1}$. Intuitively, the top $d$ right singular functions are mapped by $\mathcal{T}$ while maintaining the maximum "signal-to-noise ratio."
    - *Design Motivation*: This constitutes the central theoretical justification for spectral features—they guarantee the least ill-conditioned inverse problem.

2. **Good-Bad-Ugly Trichotomy (Corollary 1)**:

    - *Function*: Partitions all problem scenarios into three categories, each with clearly characterized performance expectations.
    - *Mechanism*: The generalization error bound decomposes into two terms—approximation error $\|(I - \Pi_{\mathcal{X},d})h_0\|$ and estimation error $\sqrt{d/(n\sigma_d^2)}$. **Good**: $h_0$ concentrates most of its energy in the top $d$ singular functions (strong spectral alignment) and $\sigma_d$ decays slowly (strong instruments); both terms are small, yielding optimal convergence. **Bad**: Good alignment but fast decay of $\sigma_d$ (weak instruments), leading to large estimation error and requiring exponentially more samples. **Ugly**: $h_0$ is misaligned with the top $d$ singular functions; approximation error remains large regardless of sample size, causing the method to fail entirely.
    - *Design Motivation*: Practitioners need a rapid way to assess whether spectral feature methods are suitable for a given problem.

3. **Contrastive Loss Equivalence to Best Rank-$d$ Hilbert-Schmidt Approximation (Theorem 2)**:

    - *Function*: Formally connects the spectral contrastive learning objective of [Xu et al.] to the truncated SVD of $\mathcal{T}$.
    - *Mechanism*: Consider the objective $\mathcal{L}_d(\varphi, \psi) = \|\sum_i \psi_i \otimes \varphi_i - \mathcal{T}\|_{HS}^2$. By the Eckart-Young-Mirsky theorem, minimizing this objective is equivalent to finding the best rank-$d$ approximation $\mathcal{T}_d$, yielding $\mathcal{H}_{\varphi,d} = \mathcal{V}_d$. Furthermore, this objective can be equivalently rewritten as a spectral contrastive loss: $\mathbb{E}_X\mathbb{E}_Z[(\varphi(X)^\top\psi(Z))^2] - 2\mathbb{E}_{X,Z}[\varphi(X)^\top\psi(Z)] + \text{const}$, which is directly estimable from samples.
    - *Design Motivation*: [Xu et al.] motivate the contrastive loss via an overly strong assumption (exact finite-rank density decomposition). This paper shows the contrastive loss is in fact performing HS approximation—a meaningful objective even when the assumption fails.

### Loss & Training
The empirical spectral contrastive loss is: $\hat{\mathcal{L}}_d = \frac{1}{m(m-1)}\sum_{i \neq j}(\varphi(\tilde{x}_i)^\top \psi(\tilde{z}_j))^2 - \frac{2}{m}\sum_i \varphi(\tilde{x}_i)^\top \psi(\tilde{z}_i)$. Features $\varphi, \psi$ are parameterized by neural networks and optimized via SGD. The learned features are then plugged into the standard two stages of 2SLS.

## Key Experimental Results

### Main Results

| Regime | Spectral Alignment | Decay Rate | Generalization Error | Spectral Feature Performance | Notes |
|--------|--------------------|------------|----------------------|------------------------------|-------|
| Good | Strong | Slow (polynomial) | Low | Optimal; matches or outperforms end-to-end | Error decreases rapidly with $n$ |
| Bad | Strong | Fast (exponential) | Moderate to high | Feasible but requires large data | Slow error decrease |
| Ugly | Weak | Arbitrary | High | Method fails | Approximation error dominates |

### Ablation Study

| Configuration | Good MSE | Bad MSE | Ugly MSE | Notes |
|---------------|----------|---------|----------|-------|
| Spectral features (proposed) | Lowest | Moderate | High | Validates trichotomy |
| Random features | High | High | High | No feature learning |
| End-to-end joint optimization | Comparable to spectral | Slightly better | Slightly better | Leverages $Y$ information |
| dSprites diagnostics | — | — | — | Successfully identifies Good regime |

### Key Findings
- **Spectral features are indeed optimal in the Good regime**: In synthetic experiments, spectral feature MSE matches theoretical bounds and outperforms other methods under strong alignment and slow decay.
- **Contrastive loss removes the overly strong assumption**: The experiments in [Xu et al.] operate in the Good regime, which explains the strong empirical performance—the new framework situates this finding in the correct theoretical context.
- **Data-driven diagnostics are feasible**: On the dSprites dataset, estimating the empirical singular value decay rate and spectral coefficients successfully identifies the problem as belonging to the Good regime, consistent with the algorithm's strong empirical performance.
- **End-to-end methods may be preferable in the Bad regime**: Because they can exploit information in $Y$ to mitigate weak instrument issues, at the cost of more complex non-convex optimization.

## Highlights & Insights
- **Theoretically elegant and practically useful**: A method that was empirically effective but theoretically opaque is placed within a three-regime analytical framework that achieves both mathematical rigor and actionable guidance. The "Good-Bad-Ugly" nomenclature (inspired by the classic Western film) enables concise communication of complex theory.
- **A new understanding of contrastive loss**: Proving that the spectral contrastive loss is equivalent to Hilbert-Schmidt best approximation removes the overly strong assumption of the original paper. This result has independent value for self-supervised learning theory—any contrastive method that learns conditional dependency structure can be analyzed within a similar framework.
- **Diagnostic tools operationalize the theory**: Rather than merely identifying when the method succeeds or fails, the paper provides concrete steps for making this determination from data, bridging the gap between theory and practice.

## Limitations & Future Work
- **No recovery strategy for Bad/Ugly regimes**: The theory identifies when spectral features fail but does not propose alternative strategies or fixes (e.g., how to select features to reduce approximation error in the Ugly regime).
- **Verifiability of theoretical assumptions**: Assumption 2 (joint density controlled by a product measure) and Assumption 3 (link condition) may be difficult to verify in practice.
- **Restricted to finite-dimensional/sieve framework**: The analysis does not extend to a complete treatment of infinite-dimensional settings such as kernel methods (RKHS).
- **Primarily synthetic experiments**: While dSprites experiments are included, validation on real-world causal inference problems (e.g., demand estimation in economics) is lacking.

## Related Work & Insights
- **vs. Xu et al. (2024)**: Proposed spectral contrastive learning but relied on an overly strong theoretical foundation (finite-rank density assumption). This paper rigorously proves equivalence to HS approximation and establishes conditions for success and failure.
- **vs. End-to-end IV methods (Hartford, DeepIV, etc.)**: End-to-end methods jointly optimize features and regression objectives and may perform better in the Bad regime, but face harder non-convex optimization. The advantage of spectral methods lies in decoupling—feature learning does not depend on $Y$, avoiding complex three-variable joint optimization.
- **vs. Adversarial/saddle-point methods (Dikkala, Lewis, etc.)**: The paper proves that the saddle-point formulation is equivalent to 2SLS in closed form ($\hat{\theta}_{bis} = \hat{\theta}$), so no additional saddle-point optimization complexity is needed.

## Rating
- Novelty: ⭐⭐⭐⭐ The Good-Bad-Ugly trichotomy offers a fresh perspective; the theoretical reinterpretation of contrastive loss has independent value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments precisely validate theoretical predictions; the dSprites diagnostic tool is practically useful.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, clearly written, with memorable nomenclature.
- Value: ⭐⭐⭐⭐ Provides theoretical guidance for feature selection in causal inference; diagnostic tools have practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)

</div>

<!-- RELATED:END -->
