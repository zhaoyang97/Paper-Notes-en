---
title: >-
  [Paper Note] Adjustment for Confounding using Pre-Trained Representations
description: >-
  [ICML 2025][Optimization][Causal Inference] This paper investigates how to leverage latent representations from pre-trained neural networks to adjust for confounding in non-tabular data (e.g., images, text). It formalizes representation sufficiency conditions, proves that sparsity/additivity assumptions do not hold under Invertible Linear Transformations (ILTs), and establishes convergence rate theory for deep networks based on low intrinsic dimension and Hierarchical Composi…
tags:
  - "ICML 2025"
  - "Optimization"
  - "Causal Inference"
  - "Average Treatment Effect"
  - "Pre-trained Representations"
  - "Double Machine Learning"
  - "Intrinsic Dimension"
date: 2026-05-08
content_hash: 54cabf3dcd993cee
---

# Adjustment for Confounding using Pre-Trained Representations

**Conference**: ICML 2025  
**arXiv**: [2506.14329](https://arxiv.org/abs/2506.14329)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Causal Inference, Average Treatment Effect, Pre-trained Representations, Double Machine Learning, Intrinsic Dimension

## TL;DR

This paper investigates how to leverage latent representations from pre-trained neural networks to adjust for confounding in non-tabular data (e.g., images, text). It formalizes representation sufficiency conditions, proves that sparsity/additivity assumptions do not hold under Invertible Linear Transformations (ILTs), and establishes convergence rate theory for deep networks based on low intrinsic dimension and Hierarchical Compositional Models (HCMs), thereby guaranteeing valid inference of ATE estimation within the Double Machine Learning (DML) framework.

## Background & Motivation

### Problem Setup

In causal inference, the estimation of the **Average Treatment Effect (ATE)** is a core task. In observational data, confounders affect both the treatment variable $T$ and the outcome variable $Y$, leading to bias in naive estimators. Traditional methods typically handle tabular confounding variables, but in domains such as medicine, confounding information is often hidden in **non-tabular data**—for instance, the severity of a disease in a CT scan image simultaneously influences both the treatment choice and the prognosis.

### Limitations of Prior Work

- The **Double Machine Learning (DML) framework** (Chernozhukov et al., 2017) allows using ML methods to adjust for non-linear confounding effects, but was initially designed for tabular data.
- Directly feeding non-tabular data into DML faces the **curse of dimensionality**: image dimensions are extremely high, whereas sample sizes in medical scenarios are often limited.
- Using pre-trained models to extract latent representations $Z = \varphi(W)$ is a natural solution, but **lacks theoretical guarantees**.
- Key theoretical obstacle: the representation $Z$ is identifiable only up to an Invertible Linear Transformation (ILT), meaning $Z$ and $QZ$ (where $Q$ is an invertible matrix) are information-theoretically equivalent.

### Core Motivation

This paper aims to answer: **Under what conditions can pre-trained representations substitute for raw non-tabular data for confounding adjustment in ATE estimation?** This requires resolving the failure of structural assumptions caused by representation unidentifiability and establishing convergence rate theories that guarantee valid statistical inference.

## Method

### Overall Architecture

The theoretical framework of the paper is divided into three levels:

1. **Representation Sufficiency** (Section 3.1): When does the pre-trained representation contain sufficient confounding information?
2. **Convergence Rate Analysis** (Section 4-5): What assumptions guarantee fast convergence under ILT unidentifiability?
3. **DML Inference Validity** (Section 5.3): Integrating the above results into the DML estimator.

Given $n$ i.i.d. observations $(T, W, Y)$, the goal is to estimate:

$$\text{ATE} \coloneqq \mathbb{E}[\mathbb{E}[Y|T=1,W] - \mathbb{E}[Y|T=0,W]]$$

where $W$ is non-tabular confounding data, and $T \in \{0,1\}$ is a binary treatment variable.

### Key Designs

#### 1. Representation Sufficiency Conditions

The paper formalizes three progressive sufficiency conditions (Definition 3.1):

- **$P$-valid**: The weakest condition, which guarantees $Z$ is a valid adjustment set, i.e., $\mathbb{E}_P[\mathbb{E}_P[Y|T=t,\sigma(Z)]] = \mathbb{E}_P[\mathbb{E}_P[Y|T=t,W]]$
- **$P$-OMS (Outcome Mean Sufficient)**: Under which equality holds almost everywhere, i.e., $\mathbb{E}[Y|T=t,Z] = \mathbb{E}[Y|T=t,W]$
- **$P$-ODS (Outcome Distribution Sufficient)**: $Y \perp W | T, Z$, which is the strongest condition

Key Insight: **$P$-valid is a necessary and sufficient condition to guarantee correct ATE estimation**, without requiring the strongest ODS condition.

#### 2. Invariance Analysis under ILT

This is the most core theoretical contribution of this paper. The paper systematically analyzes the invariance of different structural assumptions under ILT:

| Structural Assumption | ILT Invariance | Corresponding Convergence Rate | Applicability |
|---------|-----------|------------|--------|
| Smoothness ($s$-smooth) | ✅ Preserved | $n^{-s/(2s+d)}$ | Reasonable but insufficient |
| Additivity (Additive) | ❌ Not Preserved | $n^{-s/(2s+1)}$ | Unreasonable |
| Sparsity (Sparse) | ❌ Not Preserved | $\sqrt{p\log(d/p)/n}$ | Unreasonable |
| Intrinsic Dimension ($d_\mathcal{M}$) | ✅ Preserved | $n^{-s/(2s+d_\mathcal{M})}$ | Reasonable |

**Lemma 4.2** proves that for almost all ILTs $Q$ under the Haar measure: if $f$ is additive with a non-linear component, then $f \circ Q^{-1}$ is not additive; if $f$ is sparse linear, then $f \circ Q^{-1}$ is not sparse.

This implies that methods relying on sparsity/additivity assumptions, such as Lasso, Random Forests, and coordinate-split-based tree methods, **cannot theoretically guarantee fast convergence in the pre-trained representation space**.

#### 3. Hierarchical Compositional Model + Manifold Structure (Assumption 5.2)

The paper proposes a new assumption combining the **manifold hypothesis** with the **Hierarchical Compositional Model (HCM)**:

The target function can be decomposed as $f_0 = f \circ \psi$, where:
- $\mathcal{M}$ is a compact $d_\mathcal{M}$-dimensional smooth manifold
- $\psi: \mathcal{M} \to \mathbb{R}^p$ is an $s_\psi$-smooth mapping (the embedding of the manifold into Euclidean space)
- $f$ is a $k$-level structure of HCM (Hierarchical Compositional Model)

The definition of HCM (Definition 5.1) is recursive:
- **Level 0**: $f(x) = x_j$ (selecting a coordinate)
- **Level $k$**: $f(x) = h(h_1(x), \ldots, h_p(x))$, where $h$ is an $s$-smooth function, and $h_i$ are level $k-1$ HCMs

**Lemma 5.3** proves that Assumption 5.2 is invariant under ILT—this is a key theoretical guarantee. The hierarchical structure of HCM naturally aligns with the layer-by-layer computation of Deep Neural Networks (DNNs), enabling DNNs to utilize this structure efficiently.

#### 4. DNN Convergence Rate (Theorem 5.5)

Under Assumption 5.2, there exists a feedforward DNN architecture such that:

$$\|\hat{f} - f_0\|_{L_2(P_Z)} = O_p\left(\max_{(s,p) \in \mathcal{P} \cup (s_\psi, d_\mathcal{M})} n^{-s/(2s+p)}\right)$$

The rate depends only on the **worst-case pair** in the constraint set $\mathcal{P}$ and the manifold embedding parameters $(s_\psi, d_\mathcal{M})$. Utilizing Whitney's Embedding Theorem, $p$ can be set to $2d_\mathcal{M}$ and $s_\psi = \infty$, so the rate is primarily controlled by the manifold's intrinsic dimension.

### Loss & Training

#### DML Estimator

A cross-fitting strategy is adopted: the samples are split into $K$ folds, where the nuisance functions for each fold are trained on the remaining folds. The final ATE estimator uses the orthogonalized score function:

$$\rho(T_i, Y_i, Z_i; g, m) = g(1, Z_i) - g(0, Z_i) + \frac{T_i(Y_i - g(1,Z_i))}{m(Z_i)} + \frac{(1-T_i)(Y_i - g(0,Z_i))}{1 - m(Z_i)}$$

where $g(t,z) = \mathbb{E}[Y|T=t, Z=z]$ is the outcome regression function, and $m(z) = \mathbb{P}[T=1|Z=z]$ is the propensity score.

#### DML Inference Validity (Theorem 5.7)

If $g$ and $m$ satisfy Assumption 5.2, along with the regularity condition:

$$\min_{(s,p) \in \mathcal{P}_g \cup (s_\psi, d_\mathcal{M})} \frac{s}{p} \times \min_{(s',p') \in \mathcal{P}_m \cup (s'_\psi, d_\mathcal{M})} \frac{s'}{p'} > \frac{1}{4}$$

then the DML estimator satisfies asymptotic normality: $\sqrt{n}(\widehat{\text{ATE}} - \text{ATE}) \to \mathcal{N}(0, \sigma^2)$. This condition characterizes the trade-off between smoothness and dimensionality—the input dimension of each composite function must be less than twice its smoothness.

## Key Experimental Results

### Main Results

The experiments simulate confounding scenarios using two types of non-tabular data:

| Dataset | Data Type | Pre-trained Model | Representation Dimension $d$ | Sample Size |
|-------|---------|-----------|------------|-------|
| IMDb Movie Reviews | Text | BERT (bert-base-uncased) | 768 | 50,000 |
| Chest X-ray (Kermany) | Image | DenseNet-121 (TorchXRayVision) | 1,024 | 5,863 |

**Label Confounding Experimental Results** (IMDb, 5 simulations):

| Estimator | Method Type | ATE Bias | 95% CI Coverage | Description |
|-------|---------|---------|------------|-----|
| Naive | Unadjusted | Strong negative bias | Not covered | Confounding ignored |
| DML (Lasso) | Sparsity assumption | Significant bias | Not covered | Sparsity fails under ILT |
| DML (RF) | Tree-based | Significant bias | Not covered | Coordinate splitting inapplicable |
| DML (Linear) | Unpenalized linear | **Unbiased** | **Covered** | ILT-invariant estimator |
| S-Learner | Single regression | Significant bias | Not covered | Lacks double robustness |
| Oracle | Adjust on true labels | Unbiased | Covered | Baseline upper bound |

### Ablation Study

**Complex Confounding Experiments** (X-ray, confounding constructed via autoencoders):

| Configuration | ATE Bias | CI Coverage | Description |
|-----|---------|--------|-----|
| DML + Neural Network | Small | High coverage | DNN adapts to low-dimensional manifold under HCM |
| DML + Random Forest | Large | Low coverage | Sparsity assumption fails under ILT |
| S-Learner + Neural Network | Small | Overly optimistic CI | Lacks double robustness |
| DML (Pre-trained) vs DML (CNN) | Pre-trained unbiased / CNN highly biased | Only pre-trained covered | Pre-training shows a clear advantage with 500 samples |

### Key Findings

1. **Failure of non-invariant ILT methods**: Lasso and Random Forests perform poorly on pre-trained representations because the sparsity/additivity assumptions do not hold under ILT—exactly as predicted by the theory.
2. **Intrinsic dimension is far lower than ambient dimension**: The intrinsic dimension of X-ray representations is $d_\mathcal{M} \approx 12$, whereas the ambient dimension is $d = 1024$, supporting the manifold hypothesis.
3. **Pre-training is crucial**: With a limited sample size (500 images), DML using pre-trained representations is unbiased, whereas DML training a CNN from scratch is biased.
4. **Double robustness of DML is important**: Even with the same neural network, S-Learner cannot guarantee valid inference; only the orthogonalized score of DML provides correct coverage.

## Highlights & Insights

1. **Close alignment between theory and practice**: Lemma 4.2 predicts that Lasso/RF will fail $\to$ perfectly validated by experiments; Theorem 5.5 predicts that DNN adapts to the low-dimensional structure $\to$ confirmed by experiments.
2. **A complete argument transition from "what fails" to "what works"**: It first proves that sparsity/additivity is unreasonable (negative result), and then establishes an alternative theory based on manifolds + HCM (positive result).
3. **Practical guidance**: When performing causal inference on pre-trained representations, neural networks (rather than Lasso/tree-based methods) should be preferred as nuisance estimators.
4. **Universality of HCM + Manifold Hypothesis**: The proposed framework is universally applicable, extending beyond ATE estimation to other semiparametric inference tasks.

## Limitations & Future Work

1. **Single-modality assumption**: Only confounding from a single non-tabular data source is considered; multimodal fusion scenarios (e.g., both images and text acting as confounders) are not discussed.
2. **Limited to ATE**: The paper does not address the inference of ATT (Average Treatment Effect on the Treated) or CATE (Conditional Average Treatment Effect).
3. **Assumption on pre-training quality**: The theory heavily relies on the pre-trained representation satisfying the $P$-valid condition, which is difficult to verify in practice.
4. **Validation of HCM structure**: It is difficult to directly test whether a nuisance function possesses an HCM structure, and selecting parameters for the constraint set $\mathcal{P}$ depends on domain knowledge.
5. **Computational cost**: The computational efficiency differences among different nuisance estimators are not discussed.

## Related Work & Insights

- **DML Framework** (Chernozhukov et al., 2017, 2018): The theoretical foundation of this work, extended to non-tabular data scenarios.
- **Representation Learning + Causal Inference** (Veitch et al., 2019, 2020): Directly uses non-tabular data but lacks convergence rate guarantees.
- **Manifold Hypothesis** (Fefferman et al., 2016): High-dimensional data concentrates on low-dimensional manifolds, utilized in this paper to establish convergence theory.
- **DNN Adapting to Intrinsic Dimension** (Chen et al., 2019; Schmidt-Hieber, 2019): DNNs can automatically adapt to manifold dimensions; this paper combines this finding with HCM.
- **Insights**: This theoretical framework can be extended to fields such as off-policy evaluation in reinforcement learning, multimodal causal discovery, etc.

## Rating

| Dimension | Score (1-10) | Description |
|-----|-----------|-----|
| Novelty | 8 | First systematic analysis of theoretical guarantees for pre-trained representations in causal inference |
| Theoretical Depth | 9 | Clear structure from invariance analysis to convergence rates to inference validity |
| Experimental Thoroughness | 7 | Ingenious experimental design, but scenarios are somewhat synthetic, lacking real-world causal tasks |
| Value | 7 | Provides clear guidance on method selection, though verifying conditions remains difficult |
| Writing Quality | 8 | Exceptionally clear logical flow, with tight alignment between theoretical motivations and experiments |
| **Overall Score** | **7.8** | A crossover work between causal inference and representation learning with solid theoretical contributions |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](../../ICLR2026/optimization/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)
- [\[NeurIPS 2025\] Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](../../NeurIPS2025/optimization/contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)
- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](../../NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](../../ICML2026/optimization/probing_neural_tsp_representations_for_prescriptive_decision_support.md)

</div>

<!-- RELATED:END -->
