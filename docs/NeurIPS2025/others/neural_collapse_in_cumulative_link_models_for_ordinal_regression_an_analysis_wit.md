---
title: >-
  [Paper Note] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model
description: >-
  [NeurIPS 2025][Neural Collapse] This paper extends Neural Collapse (NC) theory to ordinal regression (OR) tasks based on cumulative link models (CLM). Under the unconstrained feature model (UFM) framework, three hallmark properties of Ordinal Neural Collapse (ONC) are formally proven: within-class mean collapse (ONC1), feature collapse onto a one-dimensional subspace (ONC2), and ordered arrangement of latent variables by class (ONC3). In the zero-regularization limit…
tags:
  - "NeurIPS 2025"
  - "Neural Collapse"
  - "ordinal regression"
  - "cumulative link models"
  - "unconstrained feature model"
  - "regularization"
  - "threshold models"
date: 2026-05-08
content_hash: a9a4cca60930ec87
---

# Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model

**Conference**: NeurIPS 2025
**arXiv**: [2506.05801](https://arxiv.org/abs/2506.05801)  
**Authors**: Chuang Ma, Tomoyuki Obuchi, Toshiyuki Tanaka (Kyoto University, RIKEN AIP)
**Code**: Not available  
**Area**: Other
**Keywords**: Neural Collapse, ordinal regression, cumulative link models, unconstrained feature model, regularization, threshold models

## TL;DR

This paper extends Neural Collapse (NC) theory to ordinal regression (OR) tasks based on cumulative link models (CLM). Under the unconstrained feature model (UFM) framework, three hallmark properties of Ordinal Neural Collapse (ONC) are formally proven: within-class mean collapse (ONC1), feature collapse onto a one-dimensional subspace (ONC2), and ordered arrangement of latent variables by class (ONC3). In the zero-regularization limit, a concise geometric relationship between latent variables and thresholds is revealed.

## Background & Motivation

### State of the Field
Neural Collapse (NC) is a key phenomenon observed in deep classification networks: after sufficient training, the penultimate-layer features and final classifier weights exhibit an extremely simple symmetric geometric structure (Simplex ETF). NC has been extended to multi-label classification, multivariate regression, diffusion models, and other tasks, but whether a similar phenomenon exists in ordinal regression (OR) has not been investigated.

### Limitations of Prior Work
- NC theory has been primarily developed for standard classification; its behavior in OR tasks is entirely unknown.
- The key distinction between OR and classification lies in the ordered structure of labels and the asymmetric cost of misclassification; unlike regression, OR labels carry only ordinal rather than quantitative information.
- Existing NC extensions (e.g., NRC, linguistic collapse) do not address the special structure of ordered labels.
- The role of latent variables in threshold models is analogous to logits in classification networks, but the inherent ordering constraints necessitate an entirely new theoretical framework for NC analysis.

### Root Cause
The paper investigates whether geometric collapse phenomena analogous to NC exist in deep OR tasks based on CLM, providing rigorous theoretical proofs and empirical validation within the UFM framework.

## Method

### Problem Formulation
- **Ordinal Regression**: Input space X, ordered label set Y={1,2,...,Q}, training set D={(x_i, y_i)}.
- **Cumulative Link Model (CLM)**: Class probabilities are modeled via latent variable $z$ and thresholds $b=(b_0,b_1,\ldots,b_Q)$ as $P(y \leq q \mid z) = g(b_q - z)$, where $g$ is a strictly monotonically increasing inverse link function.
- **DNN Feature Extraction**: $z = w^\top h_\theta(x)$, where $w$ is the classifier weight and $h_\theta$ is the feature extractor.
- **Loss Function**: Negative log-likelihood + L2 regularization.

### UFM Framework
The feature vectors $h_\theta(x_i)$ are treated as freely learnable variables $h_{q,i}$. Exploiting the invariance of the objective function under orthogonal transformations, the high-dimensional optimization problem is decomposed into a multi-stage optimization. The core technique is to fix the direction $a$ of $w$ and optimize only its magnitude $w$ and the per-class feature projections $z_q$.

### Three Properties of ONC (Theorem 4.2)

Assuming the inverse link function $g$ is differentiable with a log-concave derivative $g'$, under conditions $\lambda_w, \lambda_h > 0$:

1. **ONC1 (Within-Class Mean Collapse)**: All optimal features within the same class collapse to the same vector. The key to the proof is establishing the convexity of $L(z, a, b)$ with respect to $z$ via Theorem 4.1—if $p(x)$ is log-concave, then the integral $P(b-x) - P(a-x)$ is also log-concave. Combined with the strict convexity of the L2 regularization term, the uniqueness of the optimal solution for each subproblem is guaranteed.
2. **ONC2 (One-Dimensional Subspace Collapse)**: The per-class means $h_q^*$ are parallel to the classifier $w^*$. Proof: decomposing $h$ into parallel and orthogonal components shows that the loss depends only on the parallel component, so the optimal value of the orthogonal component is zero.
3. **ONC3 (Ordinal Structure Collapse)**: The optimal latent variables satisfy $z_1^* \leq z_2^* \leq \cdots \leq z_Q^*$, with strict inequalities when $g'$ is strictly log-concave.

### Equations of State (EOS) and Phase Transition (Theorem 4.3)
- When $w^* > 0$, the optimal solution satisfies a set of nonlinear equations (EOS).
- A phase transition exists: when $\lambda_h \lambda_w \geq C$, the trivial solution ($w^* = 0$) holds; when $\lambda_h \lambda_w < C$, a non-trivial solution exists.
- The phase transition boundary $C$ is precisely determined by the thresholds and class proportions.
- In the zero-regularization limit, symmetric inverse link functions (logit/probit) yield the concise relation: $z_q^* = (b_q + b_{q-1})/2$.
- Scaling law: the magnitude of $w^*$ is proportional to $(\lambda_h / \lambda_w)^{1/4}$.

### Threshold Treatment Strategies
- **Fixed Thresholds**: Boundary thresholds are symmetrically fixed ($-b_Q = b_0$), with interior thresholds uniformly spaced.
- **Learnable Thresholds**: Strict ordering is enforced via softplus parameterization, jointly optimized with model parameters.

## Key Experimental Results

### Experiment 1: Evolution of ONC Metrics During Training

ONC emergence is validated on ER (a tabular dataset with 5 ordinal classes) and UTKFace (facial age estimation with ResNet101, ages grouped by 5-year intervals):

| Metric | ER-Fixed | ER-Learnable | UTKFace-Fixed | UTKFace-Learnable |
|--------|----------|--------------|---------------|-------------------|
| ONC1 metric | Decreases continuously toward 0 | Decreases continuously toward 0 | Decreases continuously toward 0 | Decreases continuously toward 0 |
| ONC2-1 metric | Rapidly converges to 0 | Rapidly converges to 0 | Rapidly converges to 0 | Rapidly converges to 0 |
| ONC2-2 metric | Rapidly converges to 0 | Rapidly converges to 0 | Rapidly converges to 0 | Rapidly converges to 0 |
| ONC3 metric | Always small, continuously decreasing | Converges to a nonzero value | Always small, continuously decreasing | Converges to a nonzero value |
| MAE (train) | Approaches 0 | Approaches 0 | Decreases | Decreases but more slowly |
| Classification accuracy | Approaches 1 | Approaches 1 | Significantly higher | Lower |
| Minimum sensitivity | Approaches 1 | Approaches 1 | >0 | Always =0 |

Key finding: ONC1 and ONC2 emerge under both fixed and learnable thresholds; ONC3 emerges only under fixed thresholds, in the form $z_q^* = (b_q + b_{q-1})/2$.

### Experiment 2: Practical Performance Comparison — Fixed vs. Learnable Thresholds

Comparison on the UTKFace dataset reveals practical advantages of fixed thresholds:

| Performance Dimension | Fixed Thresholds | Learnable Thresholds |
|-----------------------|-----------------|----------------------|
| Training convergence speed | Faster and more stable | Slower |
| Classification accuracy | Significantly higher | Lower |
| Minimum sensitivity | >0 (all classes covered) | =0 (at least one class entirely ignored) |
| Minority class performance | Fairer latent space allocation | Minority collapse |
| ONC3 emergence | Yes | No |

Fixed thresholds provide fairer predictive probability allocation across all classes—including minority classes—through uniform partitioning of the latent space, substantially outperforming learnable thresholds in class-imbalanced settings. This finding carries important practical guidance for practitioners.

## Highlights & Insights

- **Theoretical Pioneering**: This is the first work to extend NC theory to ordinal regression, rigorously proving the three ONC properties under the UFM framework and filling the theoretical gap of NC in ordered-label tasks.
- **Elegant Zero-Regularization Limit**: The derivation of the concise relation $z_q^* = (b_q + b_{q-1})/2$ reveals a geometric symmetry between thresholds and latent variables.
- **Phase Transition Discovery**: The existence of a phase transition between trivial and non-trivial solutions in the regularization parameter space is proven, with the exact phase boundary characterized analytically.
- **Practical Value**: Fixed thresholds are found to be not merely a theoretical convenience but also yield significant gains in classification accuracy and minority class protection on imbalanced data.
- **Quantitative ONC Metrics**: Four measurable ONC metrics (ONC1, ONC2-1, ONC2-2, ONC3) are proposed, providing a systematic toolkit for empirical validation.

## Limitations & Future Work

- **Fixed Threshold Assumption**: The theoretical proofs rely on fixed thresholds; although experiments show that ONC1–2 also emerge under learnable thresholds, theoretical guarantees are lacking.
- **Two-Phase Assumption Not Fully Proven**: Theorem 4.3 assumes the continuity and monotonicity of $w^*$ with respect to regularization parameters; it has not been rigorously proven that the system admits only two phases.
- **ONC3 Metric Limitation**: The current ONC3 metric is only applicable to uniformly spaced fixed thresholds and is not suitable for learnable threshold settings.
- **UFM Simplification**: UFM treats features as free variables, ignoring the actual training dynamics of DNNs and data dependencies.
- **Limited Dataset Scale**: Validation is primarily conducted on small-to-medium-scale tabular datasets and UTKFace; verification on large-scale complex tasks is absent.
- **No Derived Algorithm**: ONC insights have not yet been translated into concrete regularization terms or loss function designs with empirical validation.

## Related Work & Insights

- **Papyan et al. (2020)**: Identified four hallmarks of NC in classification; this work extends them to OR and finds that ONC exhibits a different geometric structure (one-dimensional ordered arrangement vs. Simplex ETF).
- **Andriopoulos et al. (2024)**: Discovered NRC in multivariate regression, with features collapsing onto the target subspace; this work finds that features in OR collapse onto a one-dimensional subspace while preserving ordinal structure.
- **Thrampoulidis et al. (2022)**: Studied NC under class imbalance, finding that NC1 still holds but the global geometry changes; this work leverages fixed thresholds to protect minority classes in imbalanced OR settings.
- **Zhou et al. (2022)**: Demonstrated the universality of NC across various loss functions; this work proves the emergence of ONC under the CLM negative log-likelihood loss.
- **Dang et al. (2023, 2024)**: Studied imbalanced NC in deep linear and ReLU UFMs; this work establishes NC theory for OR in a single-layer UFM.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First extension of NC theory to ordinal regression with theoretical and conceptual innovations, though the framework follows the UFM tradition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic validation on tabular and image datasets with quantitative metrics designed; dataset scale and diversity could be further strengthened.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous structure, complete proofs, and a clear logical chain from theory to experiments.
- **Value**: ⭐⭐⭐⭐ — Fills the theoretical gap of NC in OR; the practical finding on fixed thresholds has applied value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[ICLR 2026\] GoR: A Unified and Extensible Generative Framework for Ordinal Regression](../../ICLR2026/others/gor_a_unified_and_extensible_generative_framework_for_ordinal_regression.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[NeurIPS 2025\] MaxSup: Overcoming Representation Collapse in Label Smoothing](maxsup_overcoming_representation_collapse_in_label_smoothing.md)

</div>

<!-- RELATED:END -->
