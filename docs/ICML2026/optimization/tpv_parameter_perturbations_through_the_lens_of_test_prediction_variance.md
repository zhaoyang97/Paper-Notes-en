---
title: >-
  [Paper Note] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] The authors formalize the "local prediction sensitivity of a trained model to parameter perturbations" as Test Prediction Variance (TPV). They prove that under a first-order approximation, TPV reduces to a trace form $\mathrm{Tr}(H_{\mathrm{eff}}C)$, unifying SGD noise, label noise, quantization, and pruning within a c
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 98ac0137a298e2fd
---
# TPV: Parameter Perturbations Through the Lens of Test Prediction Variance

**Conference**: ICML2026  
**arXiv**: [2512.11089](https://arxiv.org/abs/2512.11089)  
**Code**: Marked as "Code Available Here" in the paper (specific repository on the arXiv page)  
**Area**: Optimization  
**Keywords**: Prediction Variance, Parameter Perturbations, Benign Overfitting, Wide Minima, Training Set Model Selection  

## TL;DR
The authors formalize the "local prediction sensitivity of a trained model to parameter perturbations" as Test Prediction Variance (TPV). They prove that under a first-order approximation, TPV reduces to a trace form $\mathrm{Tr}(H_{\mathrm{eff}}C)$, unifying SGD noise, label noise, quantization, and pruning within a curvature-covariance framework. A stability theorem is provided to estimate TPV using only the training set, leading to the label-free pruning criterion JBR and model selection signals that do not require test labels.

## Background & Motivation
**Background**: Understanding the robustness of trained networks to "post-training noise" (SGD convergence noise, finite precision, label noise during fine-tuning, pruning masks) currently involves several theoretical branches: wide/flat minima theory (Keskar et al.), implicit optimization bias (Soudry et al.), benign overfitting (Bartlett et al.), and NTK theory (Jacot et al.).

**Limitations of Prior Work**: Each of these theories uses distinct tools to answer a similar question: "which $w^\star$ does the optimizer land on?" However, practical deployment concerns how much the output of a given $w^\star$ changes under real perturbations. This is a local, fixed-model problem that prior theories do not directly address with a unified metric.

**Key Challenge**: Existing perspectives model changes as "re-training to obtain a different $w$," treating it as a global variable. In contrast, real-world noise (small-step SGD jitter, quantization, mask-based pruning) acts on the neighborhood of a fixed $w^\star$, making it essentially a local parameter perturbation. The statistical meanings of these two perspectives differ, and conflating them can lead to errors.

**Goal**: Define a metric that directly characterizes "output sensitivity to parameter perturbations at a fixed $w^\star$"; prove it can be estimated using the training set; unify various noise mechanisms; and apply it to practical tasks like pruning and model selection.

**Key Insight**: Using a first-order Taylor expansion $f_{w^\star+\delta w}(x)\approx f_{w^\star}(x)+J(x)\delta w$, prediction variance naturally decomposes into the "geometry of the model Jacobian" and "perturbation covariance." This split corresponds to the physical structure of most post-training noise sources—where the noise mechanism determines $C$, and $H_{\mathrm{eff}}$ is a pure geometric object independent of the noise.

**Core Idea**: Use the trace form of first-order TPV, $\mathrm{Tr}(H_{\mathrm{eff}}C)$, as a unified lens to compress the robustness of various post-training perturbations into how the perturbation covariance $C$ couples with the Jacobian geometry $H_{\mathrm{eff}}$.

## Method

### Overall Architecture
TPV is an analytical framework rather than a new training algorithm, consisting of three main components:

1.  **Definition Layer**: Defines TPV as $\mathrm{TPV}:=\mathbb{E}_{x,\delta w}\bigl[\|f_{w^\star+\delta w}(x)-f_{w^\star}(x)\|^2\bigr]$. Under first-order approximation, this becomes $\mathrm{Tr}(H_{\mathrm{eff}}C)$, where $H_{\mathrm{eff}}:=\mathbb{E}_x[J(x)^\top J(x)]$ is the second moment of the model Jacobian and $C:=\mathbb{E}_R[\delta w\delta w^\top]$ is the perturbation covariance.
2.  **Stability Layer**: Proves that under over-parameterization and isotropic perturbations, the difference between $\mathrm{TPV}(w^\star;X_{\mathrm{tr}})$ and $\mathrm{TPV}(w^\star;X_{\mathrm{te}})$ is bounded by a small term independent of generalization, allowing the training set to estimate test set TPV.
3.  **Mechanism Layer**: Explicitly calculates the form of $C$ for various noise sources (label noise, SGD steady-state noise, quantization), expressing TPV as an interpretable function and recovering existing theoretical conclusions (benign overfitting, wide minima).

These layers support downstream applications: the JBR pruning criterion and training-set model selection.

### Key Designs

**1. Trace Decomposition $\mathrm{Tr}(H_{\mathrm{eff}}C)$ as a Unified Lens: Mapping Heterogeneous Perturbations to a Single Scalar**

Prior analyses often required separate mathematical tools for SGD noise, label noise, quantization, and pruning. TPV starts from the first-order Taylor expansion $f_{w^\star+\delta w}(x)\approx f_{w^\star}(x)+J(x)\delta w$. By writing $(J(x)\delta w)^2=\mathrm{Tr}(J(x)^\top J(x)\delta w\delta w^\top)$ for each $x$ and taking expectations over $x$ and $\delta w$ (assumed independent), one obtains:

$$\mathrm{TPV}\approx\mathrm{Tr}(H_{\mathrm{eff}}C),\quad H_{\mathrm{eff}}:=\mathbb{E}_x[J(x)^\top J(x)],\ C:=\mathbb{E}_R[\delta w\delta w^\top]$$

$H_{\mathrm{eff}}$ is a label-free pure geometric quantity, while $C$ is determined solely by the noise mechanism. Analyzing each noise type reduces to "calculating $C$": SGD noise $C\propto(\eta/b)\nabla^2 L(w^\star)$, quantization $C=\tfrac{\delta^2}{12}I$, and label noise $C$ relates to the pseudo-inverse of $J^\top J$. This template explains why wide minima hypotheses hold for SGD and quantization (both $C$ are proportional to the Hessian) but not for label noise (where $C$ follows the Jacobian direction, independent of the Hessian spectrum).

**2. TPV Trace Stability Theorem: Estimating Test TPV from the Training Set**

Sharpness literature has long used "training sharpness as an approximation for test sharpness" for label-free model selection without rigorous proof. Theorem 3.1 provides the first upper bound:

$$\big|\mathrm{TPV}(w^\star;X_{\mathrm{tr}})-\mathrm{TPV}(w^\star;X_{\mathrm{te}})\big|\le c_1\mathrm{Tr}(C),\quad c_1=\tfrac{n_{\mathrm{tr}}+n_{\mathrm{te}}}{p}\varepsilon_{\mathrm{NTK}}+o(1)$$

The proof relies on two points: NTK remains stable throughout training (Jacot/Allen-Zhu), and $H_{\mathrm{eff}}(w_0;X)$ at initialization concentrates to its population mean for both training and test sets (Law of Large Numbers). Combined, these show the difference in $H_{\mathrm{eff}}$—and thus $\mathrm{Tr}(H_{\mathrm{eff}}C)$—is minimal. Crucially, this bound is independent of the generalization gap $L_{test} - L_{train}$, contradicting the intuition that training set statistics only substitute for test statistics when a model generalizes well.

**3. Non-linear Benign Overfitting (Theorem 4.2): Extending Linear Regression Conclusions to Deep Networks**

Classic benign overfitting provides $\sigma_\varepsilon^2\mathrm{Tr}((XX^\top)^{-1})$ for linear regression. This work re-derives the case where training labels are corrupted by $\varepsilon_i$ and the minimum-norm solution is chosen, yielding a first-order linearization:

$$\mathrm{TPV}_{\mathrm{label}}\approx\sigma_\varepsilon^2\sum_{i=1}^r B_{ii}/s_i^2$$

where $s_i$ are non-zero singular values of the training Jacobian and $B_{ii}=(V^\top H_{\mathrm{eff}}V)_{ii}$ is the projection energy of the test distribution Jacobian onto the right singular vectors of the training Jacobian. This formula shows that label noise sensitivity in deep networks is dominated by whether test Jacobian energy exists in the directions of small singular values. NTK theory guarantees bounded minimum singular values for over-parameterized networks, suppressing $\sum B_{ii}/s_i^2$, which provides a geometric explanation for why over-parameterization reduces label noise sensitivity.

### Loss & Training
TPV itself is not a training objective. Theorem 4.3 notes that under squared loss, SGD steady-state noise TPV is approximately $\tfrac{\eta\sigma_\varepsilon^2}{2b}\mathrm{Tr}(\nabla_w^2 L(w^\star))$, i.e., "learning rate/batch size ratio × squared residual × Hessian trace," recovering the wide minima intuition. Quantization noise yields $\tfrac{\delta^2}{12}\mathrm{Tr}(\nabla_w^2 L(w^\star))$.

## Key Experimental Results

### Main Results
Experiments are divided into TPV stability verification and TPV–test loss correlation.

| Scenario | Scale | Key Observation | Significance |
| :--- | :--- | :--- | :--- |
| Synthetic TPV Stability | 324 configs × 2 noise sources, $n_{\mathrm{train}}=1000$ | TPV spans 5 orders of magnitude; points align with $y=x$ diagonal even at width = 1. | Stability exceeds theoretical requirements. |
| Low-sample Breakpoint | Same as above, width = 256, $n_{\mathrm{train}}\in\{10,1000\}$ | Significant deviation at $n=10$, alignment at $n=1000$. | Stability fails only with extremely few samples. |
| CIFAR-10 MobileNetV2 | Multiple width multipliers | Stability holds across architectures. | Valid on real data. |
| CIFAR-100 Logit Noise + FT | Increasing width MobileNetV2 | Width↑ → Training/Test TPV↓ → Test CE↓. | TPV correlates with generalization in low training loss regions. |

### Ablation Study

| Dimension | Setting | Conclusion |
| :--- | :--- | :--- |
| Noise Variance $\sigma_\varepsilon$ | Synthetic $\sigma=0.01$ vs $0.1$ | At $\sigma=0.1$, training TPV saturates to $\sigma^2$ while test TPV decreases with width. Stability fails under large noise. |
| Regularization Sweep | Weight decay / dropout / label smoothing on CIFAR-10 | TPV and test loss show U-shape: co-decrease in low loss region, diverge in high loss (underfitting). |
| Training Trajectory | ResNet-18 / CIFAR-100 / 30% label noise | Peak TPV coincides with the boundary between underfitting and epoch-wise double descent. |
| Pruning Comparison | JBR vs 7 baselines (Jacobian, L1, BN, etc.) | JBR matches or exceeds SOTA in all 4 settings. |
| Recipe Diagnosis | MLP over 7 weight decay values | Sharpness does not correlate with label noise sensitivity; TPV identifies the most robust recipe. |

### Key Findings
- TPV stability is **decoupled** from model generalization. Models with large generalization gaps still display TPV alignment, overturning the intuition that good generalization is a prerequisite for using training statistics.
- The U-shaped relationship between TPV and test loss holds across architectures and regularization methods. The "argmax TPV epoch" serves as an observable landmark to separate underfitting from double descent.
- The framework unifies the wide minima hypothesis and benign overfitting, which correspond to $C\propto$ Hessian and $C\propto J^\dagger J^{\dagger\top}$ perturbation covariances, respectively.

## Highlights & Insights
- **Unification**: A single formula $\mathrm{Tr}(H_{\mathrm{eff}}C)$ recovers four distinct areas of robustness theory (wide minima, benign overfitting, quantization, pruning) by simply "switching $C$".
- **Training Set Only**: TPV stability elevates label-free model selection from a rule of thumb to a theoretical conclusion, enabling model selection in label-scarce scenarios (e.g., medical, privacy-sensitive data).
- **U-shape Mechanism**: Attributes the monotonicity or U-shape of model selection curves to the relative dominance of bias and variance relative to $L_{\mathrm{train}}$, providing a phase partition for hyperparameter tuning.

## Limitations & Future Work
- Theorem 3.1 strictly holds for isotropic $C$ and over-parameterized limits; anisotropic perturbations (e.g., label noise where $C$ couples with Jacobian geometry) require empirical stability.
- First-order Taylor approximation fails under large perturbations or when $\sigma_\varepsilon$ pushes the solution outside the linearized neighborhood.
- TPV is defined within an MSE framework. Classification tasks use logit perturbations or fine-tuning; while the paper bridges this with empirical CE results, a formal definition is missing. Extension to structured outputs (detection, generation) remains open.

## Related Work & Insights
- **vs. Wide Minima (Keskar et al., 2017; SAM)**: Traditional theory suggests "small $\mathrm{Tr}(\nabla^2 L)$ leads to good generalization." TPV reformulates this as $H_{\mathrm{eff}}$ and $C$ jointly determining robustness, explaining why reparameterization counter-examples (Dinh et al.) do not truly invalidate wide minima—$C$ changes with the reparameterization.
- **vs. Benign Overfitting (Bartlett et al., 2020)**: Extends linear regression results to arbitrary non-linear deep networks by replacing the data matrix with the Jacobian.
- **vs. Finite-width Fluctuations (Bordelon & Pehlevan, 2023)**: They use DMFT for global variance of training dynamics; TPV is a local, post-training metric.
- **vs. Jacobian Criterion (Chen et al., 2025)**: JBR is derived from TPV geometry, yielding a score similar to JC but incorporating $\delta_u$ to preserve predictions on correctly classified training samples.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A single formula $\mathrm{Tr}(H_{\mathrm{eff}}C)$ unifies four independent theories.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive synthetic and real-world coverage, though some downstream results are relegated to the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and mapping between theorems and mechanisms, though dense formulas require the appendix for full derivation details.
- Value: ⭐⭐⭐⭐⭐ Practical utility for label-free model selection and theoretical unification for post-training robustness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](../../CVPR2025/optimization/test-time_augmentation_improves_efficiency_in_conformal_prediction.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICLR 2026\] Personalized Collaborative Learning with Affinity-Based Variance Reduction](../../ICLR2026/optimization/personalized_collaborative_learning_with_affinity-based_variance_reduction.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)
- [\[ICML 2026\] Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction](colorful_pinball_density-weighted_quantile_regression_for_conditional_guarantee_.md)

</div>

<!-- RELATED:END -->
