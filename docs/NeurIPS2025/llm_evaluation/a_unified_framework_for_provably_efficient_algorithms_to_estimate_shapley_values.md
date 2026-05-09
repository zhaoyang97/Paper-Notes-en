---
title: >-
  [Paper Note] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values
description: >-
  [NEURIPS2025][LLM Evaluation][Shapley Values] This paper proposes a unified framework that subsumes KernelSHAP, LeverageSHAP, and related Shapley value estimators under a randomized sketching perspective, provides the first non-asymptotic theoretical guarantees for KernelSHAP, and extends these methods to high-dimensional datasets such as CIFAR-10 via algorithmic improvements including Poisson approximation.
tags:
  - NEURIPS2025
  - LLM Evaluation
  - Shapley Values
  - KernelSHAP
  - LeverageSHAP
  - Explainable AI
  - Randomized Numerical Linear Algebra
date: 2026-05-08
content_hash: 6bb03bc5d20cefd5
---

# A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values

**Conference**: NEURIPS2025
**arXiv**: [2506.05216](https://arxiv.org/abs/2506.05216)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: Shapley Values, KernelSHAP, LeverageSHAP, Explainable AI, Randomized Numerical Linear Algebra

## TL;DR

This paper proposes a unified framework that subsumes KernelSHAP, LeverageSHAP, and related Shapley value estimators under a randomized sketching perspective, provides the first non-asymptotic theoretical guarantees for KernelSHAP, and extends these methods to high-dimensional datasets such as CIFAR-10 via algorithmic improvements including Poisson approximation.

## Background & Motivation

**Central role of explainable AI**: Shapley values have become the dominant approach for explaining feature contributions in black-box models, and are critical in safety-sensitive domains such as finance, healthcare, and law.

**Intractability of exact computation**: Computing exact Shapley values requires evaluating $2^d$ subsets, which grows exponentially with the feature dimension $d$ and is infeasible for arbitrary models.

**Lack of theoretical guarantees for KernelSHAP**: KernelSHAP is the most widely used estimator in the SHAP library, yet rigorous convergence guarantees have been absent. Methods with existing theoretical guarantees (unbiased KernelSHAP, LeverageSHAP) perform worse in practice than KernelSHAP.

**Bottleneck for high-dimensional scaling**: Existing theoretically grounded methods face two obstacles—combinatorial overflow and large-support binomial sampling—that prevent scaling to real-world settings with $d > 100$.

**Absence of unified comparison across estimators**: The theoretical performance trade-offs among different estimators (regression-based vs. matrix-vector product) and different sampling strategies (with vs. without replacement) remain unclear.

**Core scientific question**: Can a unified framework be established that simultaneously covers KernelSHAP and related estimators, while providing provable sample complexity guarantees for all methods?

## Method

### Overall Architecture

```
Shapley Values → Constrained regression problem (Eq.1.2)
        ↓  Variable substitution (Theorem 2.1)
  Unconstrained problem: φ* = Q·U^T·b_λ + α·1
        ↓  Random sketch S
  ┌─────────────────┬──────────────────────┐
  │ Regression est. │ Matrix-vector est.   │
  │ φ^R (KernelSHAP)│ φ^M (unbiased KSHAP) │
  └─────────────────┴──────────────────────┘
        ↓  Sampling distribution choice
  ┌────────┬────────────┬──────────────┐
  │ Kernel │ Leverage   │ Modified ℓ₂  │
  │ weight │ Score/ℓ₂²  │ (geom. mean) │
  └────────┴────────────┴──────────────┘
```

### Key Designs

**Design 1: Constraint Elimination via Variable Substitution (Theorem 2.1)**

- An orthogonal matrix $Q$ is introduced whose columns form an orthonormal basis for the orthogonal complement of $\mathbf{1}$, transforming the constrained regression problem into an unconstrained form.
- Define $U = \sqrt{d/(d-1)} \cdot Z'Q$, where $U^T U = I$ (column orthogonality).
- Key degree of freedom: the parameter $\lambda$ can be chosen freely, with $b_\lambda = \sqrt{d/(d-1)}(b - \lambda Z'\mathbf{1})$.
- Unified expression: $\phi^* = Q \cdot U^T b_\lambda + \alpha \cdot \mathbf{1}$, where $\alpha = (v([d]) - v(\emptyset))/d$.

**Design 2: Sketching-Based Construction of Two Estimator Classes**

- **Regression estimator** $\phi_\lambda^R$: solves the sketched least-squares problem $\min_x \|S(Ux - b_\lambda)\|^2$, with computational complexity $O(md^2 + mT_v)$. KernelSHAP corresponds to the regression estimator with $\lambda = \alpha$ and Kernel-weight sampling.
- **Matrix-vector product estimator** $\phi_\lambda^M$: directly approximates $U^T b_\lambda$ as $U^T S^T S b_\lambda$, yielding an unbiased estimate with complexity $O(md + mT_v)$, though generally less accurate than the regression estimator.

**Design 3: Unified Parameterization of Three Sampling Distributions**

A weighted geometric mean $p_S^\tau \propto (k(S))^\tau (\|u_S\|^2)^{1-\tau}$ unifies three distributions:

- $\tau = 1$: Kernel-weight sampling (used by KernelSHAP)
- $\tau = 0$: Leverage score / $\ell_2^2$ sampling (used by LeverageSHAP)
- $\tau = 1/2$: Modified $\ell_2$ sampling (proposed in this paper)

### Core Theoretical Result (Theorem 2.2)

A unified sample complexity bound is provided such that $\Pr[\|\phi^* - \hat\phi\| < \varepsilon] > 1 - \delta$:

- Regression estimator: $m = O\left(\frac{\gamma(P_U b_\lambda)}{\delta \varepsilon^2} + \eta \log(d/\delta)\right)$
- Matrix-vector product estimator: $m = O\left(\frac{\gamma(b_\lambda)}{\delta \varepsilon^2}\right)$

Here $\eta$ and $\gamma$ depend on the choice of sampling distribution. The bound under Modified $\ell_2$ sampling is never worse than that under Leverage score, and can be up to $\sqrt{d}$ times better in the best case; the bound under Kernel-weight sampling can be up to $d/\log d$ times better in the best case.

## Key Experimental Results

### Main Results

| Setting | Details |
|---------|---------|
| Tabular data | 8 classical datasets ($d$ up to 101), XGBoost model |
| Image data | MNIST ($d=784$), CIFAR-10 ($d=3072$) |
| Ground truth | True Shapley values computed via TreeExplainer |
| Metrics | Normalized MSE, AUC (insertion/deletion), Spearman rank correlation |
| Random seeds | 100 seeds (0–99) |
| Sample size | $m = 10^3$ to $10^6$ |

### Key Findings

| Comparison Dimension | Conclusion |
|----------------------|-----------|
| Regression vs. matrix-vector | Regression estimators achieve higher accuracy on nearly all datasets, with larger gaps in high-dimensional settings |
| Three sampling distributions | $\ell_2^2$ (Leverage score) marginally outperforms Modified $\ell_2$, both significantly outperform Kernel-weight sampling |
| With vs. without replacement | Little difference for matrix-vector estimators; with-replacement sampling outperforms on some datasets for regression estimators |
| $\lambda = 0$ vs. $\lambda = \alpha$ | Matrix-vector estimator with $\lambda = \alpha$ consistently outperforms $\lambda = 0$ (i.e., unbiased KernelSHAP) |
| High-dimensional performance | Proposed methods significantly outperform the SHAP library's KernelSHAP implementation on MNIST/CIFAR-10 |
| Faithfulness | Methods converge on MNIST after 100k samples, with KernelSHAP slightly higher in rank correlation; on CIFAR-10, the proposed methods show notable improvement in rank correlation |

## Highlights & Insights

1. **First non-asymptotic theoretical guarantees for KernelSHAP**, resolving a long-standing open problem in the field.
2. **Elegant unified framework**: variable substitution combined with sketching brings multiple estimators under a single analytical system; the degree of freedom in $\lambda$ reveals hidden connections among different methods.
3. **Modified $\ell_2$ sampling** carries a "never worse" theoretical guarantee, making it a robust default choice.
4. **Practical high-dimensional algorithmic innovation**: Poisson approximation circumvents combinatorial overflow, extending theoretically grounded methods to $d = 3072$ (CIFAR-10) for the first time.

## Limitations & Future Work

1. Theoretical bounds depend on quantities such as $\|b_\lambda\|$ that cannot be efficiently computed in practice, preventing users from directly instantiating these bounds.
2. No clear selection criterion is provided among the multiple variants of without-replacement sampling.
3. Whether Kernel-weight and Modified $\ell_2$ sampling can genuinely outperform Leverage score on practical neural networks remains an open question.
4. Experiments are validated only on XGBoost/Decision Tree models and have not been evaluated on deep neural network models.
5. The convergence rate of $\sim 1/\sqrt{m}$ still requires large sample sizes for scenarios demanding very high precision.

## Related Work & Insights

| Method | Type | Theoretical Guarantee | Role in This Paper |
|--------|------|----------------------|--------------------|
| KernelSHAP [LL17] | Regression + Kernel weight | ❌ None (first provided here) | Special case in framework ($\lambda=\alpha$, regression) |
| Unbiased KernelSHAP [CL20] | Matrix-vector + Kernel weight | Asymptotic variance analysis | Special case in framework ($\lambda=0$, MV) |
| LeverageSHAP [MW25] | Regression + Leverage score | ✅ Non-asymptotic bound | Special case in framework ($\lambda=\alpha$, regression) |
| SimSHAP [Zha+24] | Parameterized method | Partial | High-dimensional alternative |
| LIME [RSG16] | Local linear approximation | — | XAI baseline |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel unified framework perspective; first theoretical coverage of KernelSHAP
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 tabular datasets + 2 image datasets, 100 seeds, multi-metric comparison
- **Writing Quality**: ⭐⭐⭐⭐ — Clear notation system, well-organized theory and experiments
- **Value**: ⭐⭐⭐⭐ — Provides theoretical grounding for the most widely used tool in the XAI community, with practical impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[NeurIPS 2025\] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](../../CVPR2026/llm_evaluation/unified_primitive_proxies_for_structured_shape_completion.md)

</div>

<!-- RELATED:END -->
