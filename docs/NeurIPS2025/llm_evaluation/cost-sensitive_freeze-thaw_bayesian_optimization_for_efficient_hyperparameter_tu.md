---
title: >-
  [Paper Note] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning
description: >-
  [NeurIPS 2025][LLM Evaluation][Hyperparameter Optimization] CFBO incorporates user-defined utility functions (cost–performance trade-offs) into the freeze-thaw Bayesian optimization framework…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Hyperparameter Optimization"
  - "Bayesian Optimization"
  - "Freeze-thaw"
  - "Cost-Sensitive"
  - "Learning Curve Extrapolation"
  - "Transfer Learning"
date: 2026-05-08
content_hash: f52cb70effc51c8a
---

# Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2510.21379](https://arxiv.org/abs/2510.21379)  
**Code**: [GitHub](https://github.com/db-Lee/CFBO)  
**Area**: LLM Evaluation
**Keywords**: Hyperparameter Optimization, Bayesian Optimization, Freeze-thaw, Cost-Sensitive, Learning Curve Extrapolation, Transfer Learning

## TL;DR
CFBO incorporates user-defined utility functions (cost–performance trade-offs) into the freeze-thaw Bayesian optimization framework, and combines an adaptive stopping criterion with LC mixup-based transfer learning to achieve optimal cost–performance trade-offs on multi-fidelity HPO benchmarks.

## Background & Motivation

**Background**: Hyperparameter optimization (HPO) for deep learning is computationally expensive. Multi-fidelity methods (Hyperband, DyHPO, ifBO) significantly improve efficiency by pruning configurations early via learning curve extrapolation.

**Limitations of Prior Work**: Traditional multi-fidelity methods assume sufficient budgets and target final performance maximization, without accounting for user preferences regarding cost–performance trade-offs — for instance, cloud computing users may prefer early stopping when credits are limited rather than running configurations to completion.

**Key Challenge**: How can the HPO process automatically terminate near the optimal utility point according to user preferences?

**Key Insight**: Define a utility function $U(b, \tilde{y}_b)$ to characterize the cost–performance trade-off, and design a matching acquisition function and stopping criterion.

**Core Idea**: Maximize user utility rather than asymptotic performance within the freeze-thaw BO framework, while using LC mixup transfer learning to improve early extrapolation accuracy.

## Method

### Utility Function

The user utility function $U(b, \tilde{y}_b): [B] \times [0,1] \to [0,1]$ is decreasing in budget $b$ and increasing in the best cumulative performance $\tilde{y}_b$. A typical form is:

$$U(b, \tilde{y}_b) = \tilde{y}_b - \alpha \left(\frac{b}{B}\right)^c$$

where $\alpha \in [0,1]$ controls penalty strength and $c \in \{0.5, 1, 2\}$ corresponds to square-root, linear, and quadratic penalties, respectively. Setting $\alpha=0$ reduces to the conventional penalty-free setting.

For users uncertain about their own preferences, the utility can be estimated from pairwise preference data via the Bradley-Terry model, requiring approximately 30 comparisons for recovery.

### EI-type Acquisition Function

Configuration $x_{n^*}$ is selected to maximize the expected improvement in utility:

$$A(n; U) = \max_{\Delta t \in [T-t_n]} \mathbb{E}_{y_{n,\cdot} \sim p_\theta} \left[\left[U(b+\Delta t, \tilde{y}_{b+\Delta t}) - U_p\right]^+\right]$$

- The remaining learning curve of configuration $x_n$ is extrapolated via a Prior-Fitted Network (PFN).
- The optimal target epoch $\Delta t$ is selected dynamically rather than fixed to the final epoch.
- The reference value $U_p$ is the utility at the most recent step (not the historical best), since budget expenditure is irreversible.

A key behavioral property: early in the search the acquisition favors long-horizon, high-performance configurations (exploration); as performance saturates and cost dominates, it shifts toward short-term greedy selection (exploitation).

### Adaptive Stopping Criterion

Based on an estimated regret value:

$$\hat{R}_b = \frac{\hat{U}_{\max} - U_p}{\hat{U}_{\max} - \hat{U}_{\min}} > \delta_b$$

The adaptive threshold incorporates the probability of improvement (PI):

$$\delta_b = \text{BetaCDF}(p_b; \beta, \beta)^\gamma$$

where $p_b$ is the probability that the selected configuration improves $U_p$ in the future. $\gamma$ controls the threshold when PI = 0.5, and $\beta$ controls the degree of PI incorporation ($\beta \to 0$: PI ignored; $\beta \to \infty$: pure PI decision). Defaults are $\beta=e^{-1}$ and $\gamma=\log_{0.5}0.2$.

### LC Mixup Transfer Learning

To improve early learning curve extrapolation accuracy, two-level mixup data augmentation is applied to the PFN surrogate model:

1. **Across datasets**: $L' = \lambda_1 L^{(m)} + (1-\lambda_1)L^{(m')}$
2. **Across configurations**: $l' = \lambda_2 l_n + (1-\lambda_2) l_{n'}$, $x' = \lambda_2 x_n + (1-\lambda_2) x_{n'}$

$\lambda_1, \lambda_2 \sim \text{Uniform}(0,1)$. This allows unlimited sampling of training data and reduces overfitting.

## Key Experimental Results

### Datasets

| Dataset | $d_x$ | $|\mathcal{X}|$ | $T$ | Train Tasks | Test Tasks |
|--------|-------|---------|-----|---------|---------|
| LCBench | 7 | 2000 | 51 | 20 | 15 |
| TaskSet | 8 | 1000 | 50 | 21 | 9 |
| PD1 | 4 | 240 | 50 | 16 | 7 |

### Main Results (Normalized Regret ×100, PD1 Benchmark)

| Method | α=0 | α=2⁻⁶ | α=2⁻⁴ | α=2⁻² |
|------|-----|--------|--------|--------|
| ifBO | 0.8±0.1 | 2.3±0.1 | 6.0±0.6 | 15.2±2.0 |
| Quick-Tune† | — | — | — | — |
| CFBO-NT | 0.2±0.0 | 1.5±0.0 | 4.5±0.0 | 8.5±0.0 |
| **CFBO** | **0.2±0.0** | **1.0±0.0** | **0.9±0.0** | **1.7±0.0** |

### Ablation Study (PD1)

| Adaptive Threshold | Acq. | Transfer | α=0 | α=2⁻⁴ | α=2⁻² |
|-----------|------|------|-----|--------|--------|
| ✗ | ✗ | ✗ | 0.8 | 6.0 | 15.2 |
| ✗ | ✗ | ✓ | 0.2 | 5.9 | 11.7 |
| ✗ | ✓ | ✓ | 0.2 | 4.5 | 8.5 |
| ✓ | ✓ | ✓ | **0.2** | **0.9** | **1.7** |

- Transfer learning yields the largest gain at α=0; the cost-sensitive acquisition function contributes most when α>0.
- The adaptive threshold delivers the greatest improvement under strong penalty (8.5→1.7).

### Runtime (Seconds per BO Step, A100)

| Method | LCBench | TaskSet | PD1 |
|------|---------|---------|-----|
| ifBO | 0.58 | 0.30 | 0.08 |
| CFBO | 1.52 | 0.78 | 0.23 |

The gap is negligible — network training time dominates BO step time in practice.

## Highlights & Insights
- **Elegant problem formulation**: Formalizing "when should the user stop" as utility maximization is considerably more practical than fixed-budget settings.
- **Self-adaptive acquisition behavior**: The method naturally transitions from exploration to exploitation, with $\Delta t$ decreasing as BO progresses.
- **Adaptive stopping criterion** substantially outperforms fixed thresholds and approaches the theoretically optimal stopping point.
- LC mixup data augmentation is simple yet effective, and generalizes to other PFN training scenarios.

## Limitations & Future Work
- The utility function form requires user priors (although Bradley-Terry estimation is provided, a functional family must still be specified).
- The configuration space is restricted to a finite set $\mathcal{X}$, precluding continuous HPO.
- Validation is limited to tabular, NLP, and visual classification learning curves; large-scale LLM training scenarios are not covered.
- Runtime is approximately 2–3× that of ifBO (though overall overhead remains negligible).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of cost-sensitive utility and adaptive stopping constitutes a novel HPO formalization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks × multiple utility functions × complete ablations × real-dataset validation.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Directly applicable to cloud-based and time-constrained HPO scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)
- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)

</div>

<!-- RELATED:END -->
