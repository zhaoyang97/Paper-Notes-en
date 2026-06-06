---
title: >-
  [Paper Note] ECSEL: Explainable Classification via Signomial Equation Learning
description: >-
  [ICML 2026][Causal Inference][signomial function] ECSEL employs "one signomial function (sum of power-law terms with real exponents) per class + softmax" as a classifier. Combined with L1 sparse regularization and multi-…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "signomial function"
  - "symbolic regression"
  - "explainable classification"
  - "L1 sparse regularization"
  - "closed-form attribution"
date: 2026-05-08
content_hash: 9b372743ec879bde
---

# ECSEL: Explainable Classification via Signomial Equation Learning

**Conference**: ICML 2026  
**arXiv**: [2601.21789](https://arxiv.org/abs/2601.21789)  
**Code**: https://github.com/AdiaLumadjeng/ecsel (Available)  
**Area**: Explainable Machine Learning / Symbolic Regression / Inherently Interpretable Classifiers  
**Keywords**: signomial function, symbolic regression, explainable classification, L1 sparse regularization, closed-form attribution  

## TL;DR
ECSEL employs "one signomial function (sum of power-law terms with real exponents) per class + softmax" as a classifier. Combined with L1 sparse regularization and multi-stage optimization, it recovers 95.86% of target equations on symbolic regression benchmarks like AI Feynman with significantly less compute than SOTA methods, while matching XGBoost/MLP performance across 11 classification datasets. All feature attributions are derived closed-form from model parameters.

## Background & Motivation

**Background**: Current Explainable AI (XAI) follows two main paradigms. The first is *post-hoc* explanation (LIME, SHAP, Integrated Gradients), which trains a surrogate model to explain black-box predictions. The second is *inherently interpretable* models (Decision Trees, GAM, sparse linear models), where the structure itself serves as the explanation. Symbolic Regression (SR) represents an extreme form of the latter, directly producing human-readable equations.

**Limitations of Prior Work**: General SR methods (GP, PySR, DGSR, NeSymRes) define the search space as "arbitrary functional forms," leading to two issues: (1) extreme computational cost (e.g., DGSR averages 612s per equation and frequently times out); (2) performance collapse on high-dimensional data. Meanwhile, post-hoc explanations are criticized by researchers like Rudin for being unreliable in high-stakes decision-making.

**Key Challenge**: The expressive capacity of general SR is *not realized in benchmarks*. The authors observed that 45 out of 100 physical equations in AI Feynman are essentially signomials (of the form $\sum_k \alpha_k \prod_{j} x_j^{\beta_{k,j}}$). Benchmarks already suggest a specific structure, yet general methods persist in blind-searching massive spaces.

**Goal**: (1) Establish signomials as a formal "model family" rather than just an optimization target; (2) Enable signomials for both SR and classification; (3) Derive "global/decision-boundary/local" explanations *closed-form* from model parameters, eliminating the need for sampling.

**Key Insight**: In log-space, a signomial term is a linear function ($\log z = \sum_j \beta_j \log x_j + \log\alpha$). Thus, the exponents $\beta_{k,j}$ directly encode the "elasticity of features relative to the output" (as defined in economics). This provides a natural "parameters-as-explanation" structure.

**Core Idea**: Replace deep classifiers with "one signomial per class + softmax + L1 regularization," trading training complexity for zero-cost interpretability.

## Method

### Overall Architecture

The input consists of an arbitrary feature vector $x \in \mathbb{R}^m$, initially mapped to $[1, 10]$ via affine transformation to satisfy the positivity requirement of signomials. For a $C$-class problem, each class $c$ learns a signomial score function composed of $K$ power-law terms:

$$z_c(x) = \sum_{k=1}^{K} \alpha_{c,k} \prod_{j=1}^{m} x_j^{\beta_{c,k,j}}$$

Specifically, $z_c(x) = \sum_{k=1}^{K} \alpha_{c,k} \prod_{j=1}^{m} x_j^{\beta_{c,k,j}}$, where parameters include coefficients $\alpha_{c,k} \in \mathbb{R}$ and exponents $\beta_{c,k,j} \in \mathbb{R}$. $K$ controls complexity: $K=1$ is a single power law, while $K>1$ is an additive combination. Probabilities are given via softmax (multi-class) or sigmoid (binary). The SR version replaces cross-entropy with MSE.

The authors prove a **Signomial Universal Approximation Theorem**: by mapping exponential-linear functions on the positive orthant via $\log$ transformations and applying Stone-Weierstrass, signomials are shown to be dense in continuous functions on compact subsets of $\mathbb{R}^m_{>0}$. This places signomials on par with neural networks as universal approximators, though with an inherent preference for multiplicative power-law relationships.

### Key Designs

1.  **Class-specific Signomial + L1 Exponent Sparsification**:
    - **Function**: Ensures the classifier's score function is a human-readable "multiplication/division" equation with automatic feature selection.
    - **Mechanism**: Each class $c$ has independent $\{\alpha_{c,k}, \beta_{c,k,j}\}$. The training objective is $\mathcal{L} = -\frac{1}{N}\sum_i \log p_{y_i}(x_i) + \lambda \sum_{c,k,j} |\beta_{c,k,j}|$. The L1 term acts only on the *exponents*, pushing $\beta$ of irrelevant features to 0. This is equivalent to removing $x_j^0 = 1$ from the term, producing sparse equations. This differs from coefficient $(\alpha)$ sparsification: sparsifying $\beta$ performs "feature selection," while sparsifying $\alpha$ performs "term selection."
    - **Design Motivation**: Traditional GAMs/linear models are additive and cannot capture multiplicative interactions (e.g., PageValue/ExitRate in e-commerce). Signomials naturally express these interactions, and since they are linear after a $\log$ transformation, they allow closed-form attribution.

2.  **Staged Optimization ($K=1$ vs $K>1$)**:
    - **Function**: Ensures reliable convergence in the non-convex exponent space, transforming "theoretically elegant signomials" into a "practically runnable" method.
    - **Mechanism**: For $K=1$, the objective is a low-dimensional smooth function solved via L-BFGS-B. For $K>1$, the space is high-dimensional and non-convex, requiring a three-stage strategy: ① Adam + strong L1 for "structure discovery" to differentiate terms; ② Adam with reduced L1 for "refinement"; ③ L-BFGS initialized from the best Adam point for final "polishing." Multi-start with random seeds is used, along with $\log$-domain transformations and feature scaling for numerical stability.
    - **Design Motivation**: Signomial exponents can be any real number (negative, fractional). Gradients with respect to $\beta$ involve $z_{c,k}(x) \cdot \log x_j$, which are prone to explosion. Direct Adam optimization often fails; the staged strategy uses noisy gradients to escape local optima before refining with second-order methods, increasing recovery rates from 59% (DGSR) to 95.86%.

3.  **Closed-form Tri-level Explanations (Global Elasticity / Decision Boundary / Local Attribution)**:
    - **Function**: Once trained, explanation queries are simple algebraic operations on parameters with zero additional computation.
    - **Mechanism**: (a) **Global Elasticity** $E_{c,j}(x) = \partial \log z_c / \partial \log x_j = \sum_k \frac{z_{c,k}(x)}{z_c(x)} \beta_{c,k,j}$, which simplifies to the constant $\beta_{c,j}$ when $K=1$; (b) **Counterfactuals**: the new score after scaling $x_j$ by $q$ is $z_c^{\text{new}}(x) = \sum_k q^{\beta_{c,k,j}} z_{c,k}(x)$ without re-prediction; (c) **Decision Boundary Sensitivity** $\partial(z_c - z_{c'})/\partial \log x_j$, which for $K=1$ is $z_c \beta_{c,j} - z_{c'} \beta_{c',j}$, directly showing which exponent difference drives class competition; (d) **Local Attribution** uses $\log z_{c,k}(x) = \log z_{c,k}(b) + \sum_j \beta_{c,k,j} \log(x_j/b_j)$, which is an *exact* SHAP-style decomposition for $K=1$ and a first-order linearization $\phi_j \approx G_{c,j}(x^*)(\log x_j - \log x_j^*)$ for $K>1$.
    - **Design Motivation**: SHAP/LIME are slow because they use Monte Carlo sampling to approximate quantities that should be closed-form. The $\log$-linear structure of signomials provides analytical forms—a "structural dividend." Theorem 3.2 formally proves that ECSEL satisfies seven axiomatic properties (G1-G3, D1-D2, L1-L2).

### Loss & Training

Classification uses cross-entropy with L1 on $\beta$: $\mathcal{L} = -\frac{1}{N}\sum_i \log p_{y_i}(x_i) + \lambda \sum_{c,k,j} |\beta_{c,k,j}|$; SR uses the MSE version $\mathcal{L}_{\text{SR}} = \frac{1}{N}\sum_i (y_i - z(x_i))^2 + \lambda \sum_{k,j}|\beta_{k,j}|$. $\lambda$ is a critical hyperparameter ($2 \times 10^4$ for PaySim). Optimization uses L-BFGS-B for $K=1$ and the Adam-Adam-LBFGS staged approach for $K>1$. Hyperparameters are searched via Optuna TPE within 30 trials.

## Key Experimental Results

### Main Results

**Symbolic Regression (45 AI Feynman signomial subset + Livermore/Jin/Korns/DGSR synthetic sets, 5 random seeds 42-46)**:

| Method | Symbolic Recovery Rate | Avg. Time (s/Eq) |
|------|-----------|--------------------|
| NeSymRes | 56% | 126.3 |
| NGGP | 58.54% | 468.7 |
| DGSR (Prev. SOTA) | 59.10% | 612.9 |
| **Ours (ECSEL)** | **95.86%** | **86.4** |

**Classification (11 binary/multi-class benchmarks, 5-fold CV, 3 representative datasets)**:

| Dataset | Method | Acc. | F1 | Minority Recall |
|--------|------|------|------|---------------|
| Ilpd | LR | 71.55 | 58.45 | 3.03 |
| Ilpd | XGBoost | 72.41 | 63.03 | 6.06 |
| Ilpd | **Ours** | **75.86** | **74.39** | **42.42** |
| Compas | XGBoost | 68.18 | 68.08 | 62.54 |
| Compas | **Ours** | **68.47** | **68.36** | **62.82** |
| Transfusion | XGBoost | 80.06 | 78.72 | 38.89 |
| Transfusion | **Ours** | 79.33 | 77.95 | **41.67** |

ECSEL ranked first in F1 on 4 out of 11 datasets (Seeds/Hearts/ILPD/Compas) and maintained a $<1\%$ gap with the best method on 9 datasets. On ILPD, F1 is 11.36 higher than XGBoost, with minority recall increasing by +36 points.

### Ablation Study / Explainer Comparison (OSI e-commerce dataset)

| Method | Explainer | Comp. Time (s) | Top-3 Features |
|------|--------|--------------|-----------|
| **Ours** | Exact Exponent | **0.1** | PVER, SI, PV |
| LR | LinearSHAP | 0.1 | PVER, Mo, PR |
| LR | LIME | 5.3 | PVER, Mo, PR |
| RF | TreeSHAP | 1.5 | PVER, PV, SI |
| RF | LIME | 32.0 | PVER, PV, ER |
| XGBoost | TreeSHAP | 0.1 | PVER, Mo, SI |
| XGBoost | LIME | 7.7 | PVER, PR, ER |
| MLP | KernelSHAP | 28.5 | PVER, PR, Mo |

### Key Findings

- **Structural Dividend**: While DGSR is SOTA on AI Feynman, its inability to constrain functional forms results in only 59% recovery. ECSEL's hardcoded signomial form increases recovery by 37 points and reduces time to 1/7.
- **Minority Recall Advantage**: On ILPD, XGBoost's minority recall is only 6%, whereas ECSEL reaches 42%. On the PaySim fraud detection dataset, ECSEL achieves 79.08% F1, surpassing the previous 78% (DSC), with precision reaching 94.27%.
- **Zero Amortized Explanation Cost**: Investing slightly more in training time (5.5s vs 0.1s for LR on OSI) eliminates the need for post-hoc inference. KernelSHAP takes 28.5s for test set explanations on MLP, while ECSEL takes 0.1s.
- **Domain-Meaningful Equations**: On PaySim, $\beta_{\text{OBO}} = 1.42$ reveals that "fraudsters target high-value accounts super-linearly"—an actionable insight missing from black-box models. On OSI, PVER (PageValue/ExitRate) naturally emerges as a dominant predictor.

## Highlights & Insights

- **Absolute "Parameters-as-Explanation"**: Many inherently interpretable models (e.g., GAM) still require partial dependence plots. ECSEL transforms elasticity, counterfactuals, decision boundaries, and local attributions into algebraic expressions of $\beta$ and $z_{c,k}$. Theorem 3.2 formalizes this into seven provable properties.
- **From Benchmark Observation to Algorithm Design**: The method is derived from an empirical observation (45/100 AI Feynman equations are signomials). This strategy of recognizing structural cues in benchmarks that general methods overlook is highly transferable.
- **L1 on Exponents, Not Coefficients**: This detail is crucial. $\beta_j = 0$ implies $x_j^0 = 1$, meaning the feature is absent from that term. Sparsifying $\beta$ acts as per-term feature selection, which is finer-grained than sparsifying $\alpha$ (which only eliminates entire terms).

## Limitations & Future Work

- $K$ must be pre-specified, which is a standard hyperparameter for classification but a constraint for SR. It still lags behind specialized methods on high-degree univariate polynomials (e.g., Nguyen).
- **Limitations**: (1) Requires all features $x > 0$, necessitating $[1, 10]$ mapping; handling negative or categorical features is currently suboptimal. (2) For $K > 1$, local attribution degrades to first-order linearization. (3) Multi-stage optimization hyperparameters (Adam steps, L1 annealing) affect equation "aesthetics," challenging reproducibility. (4) Lack of discussion on discrete/categorical features limits application beyond tabular data.
- **Future Work**: Make $K$ learnable (growth-on-demand), constrain exponents to rational subsets for exact recovery, explore group-L1 for shared feature selection across classes, and use Mixture-of-signomials for multimodal distributions.

## Related Work & Insights

- **vs DGSR/NeSymRes/gplearn**: These search arbitrary functions in massive spaces. ECSEL locks into the signomial subspace, trading non-power-law structures for a 37-point gain in recovery and 7x speedup.
- **vs GAM / Neural Additive Models (NAM)**: GAM/NAM are *additive* and fail to capture multiplicative interactions. ECSEL is *multiplicative*, naturally handling elasticity-based features. They are complementary and could be merged into "Generalized Additive + Multiplicative Models."
- **vs SHAP/LIME**: Post-hoc methods sample to estimate values; ECSEL provides them as analytical identities. This turns SHAP's "estimate" into a "property" of the signomial.
- **vs KAN (Kolmogorov-Arnold Networks)**: KAN uses learnable splines and symbolification for interpretability. ECSEL's signomial is a more constrained but naturally closed-form subspace.

## Rating
- Novelty: ⭐⭐⭐⭐ Reframing signomials from optimization targets to a model class is a key insight, though individual components are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 45 SR equations, 11 classification datasets, 2 case studies, and comparison with 4 baseline categories and 5 explainers.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rigorous indexing; equation numbering is slightly dense in the classification section.
- Value: ⭐⭐⭐⭐⭐ Provides a true "non-post-hoc" interpretable classifier for high-stakes scenarios (finance/healthcare).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression](outcome-aware_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[ICLR 2026\] Self-Supervised Learning from Structural Invariance](../../ICLR2026/causal_inference/self-supervised_learning_from_structural_invariance.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](../../ICLR2026/causal_inference/learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](../../CVPR2026/causal_inference/retrieving_counterfactuals_improves_visual_in-context_learning.md)

</div>

<!-- RELATED:END -->
