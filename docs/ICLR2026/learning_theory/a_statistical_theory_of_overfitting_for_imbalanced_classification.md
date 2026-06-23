---
title: >-
  [Paper Note] A Statistical Theory of Overfitting for Imbalanced Classification
description: >-
  [ICLR 2026][learning_theory][margin rebalancing] This paper establishes a statistical theory for high-dimensional imbalanced linear classification: under a two-class Gaussian Mixture Model (2-GMM), test logits follow $N(0,1)$, while training logits converge to $\max\{\kappa, N(0,1)\}$ (rectified Gaussian). A variational problem characterizes how this "truncation" occ
tags:
  - ICLR 2026
  - learning_theory
  - margin rebalancing
date: 2026-05-08
content_hash: 6fbdcd1b24e9e95d
---
# A Statistical Theory of Overfitting for Imbalanced Classification

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=cKthi6QfUr](https://openreview.net/forum?id=cKthi6QfUr)  
**Code**: https://github.com/jlyu55/Imbalanced_Classification_iclr  
**Area**: Statistical Learning Theory  
**Keywords**: Imbalanced Classification, Overfitting, High-dimensional Asymptotics, Rectified Gaussian, margin rebalancing

## TL;DR
This paper establishes a statistical theory for high-dimensional imbalanced linear classification: under a two-class Gaussian Mixture Model (2-GMM), test logits follow $N(0,1)$, while training logits converge to $\max\{\kappa, N(0,1)\}$ (rectified Gaussian). A variational problem characterizes how this "truncation" occurs as a function of dimensionality, rigorously explaining why minority classes suffer more from overfitting, why margin rebalancing is effective, and how overfitting exacerbates confidence calibration.

## Background & Motivation
**Background**: In imbalanced classification (rare diseases, anomaly detection, long-tail populations), the minority class constitutes only a small fraction of training samples. In the deep learning era, a common practice is to freeze a pre-trained network as a feature extractor and retrain only the final linear classification head (linear probing), which essentially performs linear classification $f(x)=\langle x,\beta\rangle+\beta_0$ on high-dimensional features $x\in\mathbb{R}^d$.

**Limitations of Prior Work**: Classical statistical theory, built on large-sample asymptotics and finite-sample corrections, largely fails in high-dimensional scenarios where $d$ is comparable to $n$. Two phenomena are frequently observed but poorly understood: ① overfitting (the gap between training and test accuracy) is significantly more severe for the minority class; ② there is a lack of systematic characterization of how dimensionality, imbalance, and signal strength affect test accuracy and uncertainty quantification. Existing methods like reweighting, resampling, and margin-based losses remain ad hoc, offering little guidance for hyperparameter selection or feature interpretation.

**Key Challenge**: In high dimensions, data is often linearly separable, allowing SVM/logistic regression to drive training error to zero while test error remains non-zero—this train/test gap is overfitting. The problem is that viewing train/test accuracy alone is too coarse to reveal what overfitting does to the classes at a "distributional level" or why it hurts the minority class disproportionately.

**Goal**: ① Provide a precise characterization of overfitting at the logit distribution level; ② Quantify the monotonic impact of dimensionality, imbalance, and signal strength on test error and calibration; ③ Provide a theoretical explanation for the optimal hyperparameters in the commonly used margin rebalancing trick.

**Key Insight**: Instead of focusing solely on scalar test error, this work characterizes the entire **logit distribution**—the Empirical Logit Distribution (ELD) on the training set and the Test Logit Distribution (TLD). The authors find that the TLD is Gaussian, while the ELD is a rectified Gaussian "pushed" by the margin. This discrepancy serves as the fingerprint of overfitting.

**Core Idea**: By using Gordon’s Theorem from high-dimensional statistics, the max-margin training objective is simplified into a variational problem. It reveals that "overfitting = transporting overlapping TLD mass to the margin boundary through truncation." Since both classes share a finite "overfitting budget," the minority class ends up being truncated more severely.

## Method

### Overall Architecture
This is a statistical theory paper that "establishes theorems" for observed phenomena rather than proposing a new iterative method. It targets an analytically tractable toy model—two-class isotropic Gaussian Mixture (2-GMM): $P(y=+1)=\pi$ (minority), $P(y=-1)=1-\pi$, and $x\mid y\sim N(y\mu, I_d)$ with signal vector $\mu\in\mathbb{R}^d$. Standard (hard-margin) SVM or logistic regression is trained on this model, focusing on the parameters $(\hat\beta,\hat\beta_0,\hat\kappa)$ and logits $\hat f(x_i)=\langle x_i,\hat\beta\rangle+\hat\beta_0$.

The analysis follows proportional asymptotics $n/d\to\delta$ (where $\delta$ is the aspect ratio) as $n,d\to\infty$. The paper progresses through three sections: Section 2 characterizes overfitting (ELD vs. TLD truncation), Section 3 derives optimal hyperparameters for margin rebalancing, Section 4 extends the characterization to confidence calibration, and Section 5 generalizes conclusions to multi-class and non-isotropic covariances. The entire logic is driven by a single variational problem (Eq. 5).

### Key Designs

**1. ELD/TLD: Upgrading Overfitting from "Scalar Error" to "Distributional Truncation"**

Direct comparison of train/test accuracy confirms overfitting exists but not how it happens. Two objects are defined: the **Empirical Logit Distribution (ELD)** $\hat\nu_n^{\text{train}}=\frac1n\sum_i \delta_{(y_i,\hat f(x_i))}$ and the **Test Logit Distribution (TLD)** $\hat\nu_n^{\text{test}}=\mathrm{Law}(y_{\text{test}},\hat f(x_{\text{test}}))$. The key observation is that when the training set is linearly separable (margin $\hat\kappa_n=\min_i y_i\hat f(x_i)>0$), the TLD for each class is Gaussian, while the ELD is a **rectified Gaussian** $\max\{Z,\kappa\}$ truncated at the margin. Intuitively, mass in the TLD that would have contributed to test error is forcibly "pushed" to the margin boundary by the optimizer during training. This phenomenon is universal: the authors verified it on RNA-seq data, CIFAR-10 (ResNet-18 features), IMDb (BERT features), and Llama-3-8B activation probes on TruthfulQA.

**2. Variational Problem + Shared Overfitting Budget: Why the Minority Class is Subject to More Truncation (Main Theorem 2.1)**

This is the technical core. Max-margin SVM is rewritten as a min–max problem, and **Gordon’s Theorem** is used to reduce the random matrix problem to one involving only random vectors. The parameters $(\hat\rho,\hat\beta_0,\hat\kappa)$ converge to the unique solution of the following variational problem:

$$\max_{\rho\in[-1,1],\,\beta_0,\,\kappa>0,\,\xi\in L^2}\ \kappa \quad \text{s.t.}\quad \rho\|\mu\|_2+G+Y\beta_0+\sqrt{1-\rho^2}\,\xi\ge\kappa,\ \ \mathbb{E}[\xi^2]\le 1/\delta,$$

where $\rho=\langle\hat\beta/\|\hat\beta\|,\mu/\|\mu\|\rangle$, $(Y,G)\sim P_y\times N(0,1)$, and $\xi$ is a free random variable. The physical meaning of $\xi$ is crucial: while the first few terms represent "useful" components, the remaining non-signal dimensions in $d$-dimensional space provide "room for overfitting," encoded by $\xi$. The constraint $\mathbb{E}[\xi^2]\le1/\delta$ represents a finite **"overfitting budget"**. When $\delta$ is small (high dimensionality), the budget is loose, and the TLD is heavily distorted. The mapping $\sqrt{1-\rho^2}\,\xi=(\kappa-\rho\|\mu\|_2-G-Y\beta_0)_+$ represents the "transport map" moving TLD mass to the margin. The minority class suffers more because transporting its ELD mass "costs" less of the shared budget, leading the optimizer to truncate it more aggressively.

**3. Optimal $\tau$ for Margin Rebalancing and the Three-Phase Transition (Prop 3.1 / Theorem 3.2)**

Margin rebalancing introduces $\tau>0$ to scale the margin constraints for the minority class. Proposition 3.1 proves that the optimal $\tau^{\text{opt}}$ that minimizes balanced error ($\mathrm{Err}_b$) results in $\beta_0^*=0$ and $\mathrm{Err}_+^*=\mathrm{Err}_-^*$. Remarkably, $\tau^{\text{opt}}\asymp\sqrt{1/\pi}$ provides a clean rule of thumb. In the **high-imbalance regime**, Theorem 3.2 identifies three phases: ① **High Signal**: Rebalancing is optional as error is already $o(1)$; ② **Medium Signal**: Rebalancing is **essential**; otherwise, the minority error stays $1-o(1)$; ③ **Low Signal**: Rebalancing cannot save the model, and error remains near random guessing.

**4. Overfitting Exacerbates Confidence Calibration (Theorem 4.1)**

The same distributional characterization extends to calibration. For confidence $\hat p(x)=\sigma(\hat f(x))$, Theorem 4.1 proves that miscalibration metrics (ECE, MSE) have determined asymptotic limits. For instance, $\mathrm{MSE}^*$ decreases as $\pi, \|\mu\|_2,$ or $\delta$ increases. This reveals that factors increasing test error (higher dimensionality, higher imbalance) simultaneously inflate confidence and worsen calibration.

### Loss & Training
The analysis covers two standard convex problems: Logistic Regression and hard-margin SVM. Since the gradient descent iterates for logistic regression converge in direction to the max-margin solution (implicit bias), the two are closely linked. Margin rebalancing is implemented by incorporating $\tau$ into the margin constraints or loss function, which theoretically shifts the decision boundary.

## Key Experimental Results

### Main Results: Universality of the Truncation Phenomenon

| Data / Modality | Feature Extractor | Dimension $d$ | Imbalance $\pi$ | Observation |
|------|------|------|------|------|
| Synthetic 2-GMM | — | 4000 | 0.15 | ELD is rectified Gaussian; TLD is Gaussian. |
| IFNB RNA-seq (Table) | Raw | 2000 | 0.2 | ELD truncated at margin; minority class more heavily. |
| CIFAR-10 (Image) | ResNet-18 | 512 | 0.1 | Consistent rectified Gaussian pattern. |
| IMDb (Text) | BERT-base | 768 | 0.02 | Consistent pattern. |
| TruthfulQA (LLM Probe) | Llama-3-8B | — | 0.04 | Truncation in primary direction; suggests LLM probing overfitting. |

The consistent observation across tabular, image, text, and LLM activation data suggests that truncation is a universal law for high-dimensional linear heads.

### Monotonicity of Error and Calibration

| Parameter ↑ | $\mathrm{Err}^*$ (Test Error) | $\mathrm{CalErr}^*/\mathrm{MSE}^*$ |
|------|------|------|
| Imbalance $\pi\uparrow$ (More balanced) | ↓ Prop 3.1 | ↓ Thm 4.1 |
| Signal Strength $\|\mu\|_2\uparrow$ | ↓ Prop 3.1 | ↓ |
| Aspect Ratio $\delta=n/d\uparrow$ | ↓ Prop 3.1 | ↓ |

Numerical simulations ($n=100, d=200$) align perfectly with Theorem 2.1. Without rebalancing ($\tau=1$), as $\pi\downarrow$, $\mathrm{Err}_+$ approaches 1 while $\mathrm{Err}_-$ goes to 0. Optimal $\tau$ aligns the errors and significantly reduces balanced error.

### Key Findings
- **Overfitting = Truncation**: High-dimensional training logits are rectified Gaussians due to margin constraints, explaining the train/test gap without extra assumptions.
- **Root Cause for Minority Class**: The "shared overfitting budget" $\mathbb{E}[\xi^2]\le1/\delta$ makes it "cheaper" for the optimizer to truncate the minority class.
- **$\tau^{\text{opt}}\asymp\sqrt{1/\pi}$**: Margin rebalancing shifts the decision boundary without rotating the direction; it is essential in medium-signal regimes.
- **Calibration Side-effect**: Factors that raise test error (imbalance, high dimensionality) also inflate confidence, worsening calibration.

## Highlights & Insights
- **Distributional Concrete Definition**: Translating the vague concept of "overfitting" into a concrete distributional operation—$T^*(x)=\max\{\kappa^*,x\}$—is the paper's strongest insight.
- **Shared Budget Intuition**: The idea that two classes compete for a finite budget determined by $1/\delta$ is a powerful and transferable concept for any high-dimensional linear head.
- **Practical Rule for Rebalancing**: Providing $\tau\asymp\sqrt{1/\pi}$ as a rule of thumb for hyperparameter selection is more useful than trial-and-error.
- **Interpretability for LLM Probes**: Observations in Llama-3 suggest that imbalanced probing sets can lead to "false signals" due to over-memorization, which has implications for understanding LLM activations.

## Limitations & Future Work
- **Stylized Model**: The core theorems rely on the 2-GMM and linear classifiers. While Section 5 generalizes to multi-class and non-isotropic cases, the multi-class case relies on conjectures and numerical experiments.
- **Linear Probing Focus**: The analysis applies to frozen features. It does not account for feature learning or end-to-end training where features evolve.
- **Informal Versions**: Some results are presented in "informal" or "claim" formats, with full rigor provided only in the appendices.
- **Future Directions**: Extending truncation theory to feature learning, proving multi-class conjectures, and designing joint error-calibration rebalancing schemes.

## Related Work & Insights
- **vs High-Dimensional Asymptotics (Sur & Candès, etc.)**: While prior work uses Gordon’s Theorem for scalar errors, this work upgrades the analysis to the entire ELD/TLD distribution and specifically addresses class imbalance.
- **vs Benign Overfitting (Bartlett, etc.)**: Instead of explaining why overfitting allows for generalization, this work focuses on the "malign" side—how imbalance causes asymmetric harm and ruins calibration.
- **vs Margin-based Generalization Bounds**: Unlike distribution-agnostic conservative bounds, this work provides distribution-dependent exact asymptotics, identifying the "useful/useless/impossible" phases for rebalancing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to characterize imbalanced overfitting as ELD truncation and explain asymmetric harm via a shared budget.
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulations match theorems perfectly; cross-modal verification is strong, though real-data experiments are primarily for validation rather than benchmarking.
- Writing Quality: ⭐⭐⭐⭐ Clear theorem-intuition-simulation structure; well-organized monotonicity tables.
- Value: ⭐⭐⭐⭐⭐ Provides calculable theory and actionable rules for long-tail classification and LLM probing.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AI4SLT: Empirical Processes in Lean 4 for Formal Statistical Learning Theory](../../ICML2026/learning_theory/ai4slt_empirical_processes_in_lean_4_for_formal_statistical_learning_theory.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)
- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Statistical and Structural Identifiability in Representation Learning](statistical_and_structural_identifiability_in_representation_learning.md)
- [\[ICLR 2026\] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution](theoretical_analysis_of_contrastive_learning_under_imbalanced_data_from_training.md)

</div>

<!-- RELATED:END -->
